"""Binary autoencoder (BAE) architecture.

    I. Ben-Amor, J.-F. Bonastre, S. Mdhaffar, "Extraction of interpretable
    and shared speaker-specific speech attributes through binary
    auto-encoder", in Interspeech 2024, ISCA, sept. 2024, p. 3230-3234.
    doi: 10.21437/Interspeech.2024-1011.


"""
from __future__ import annotations

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class _StraightThroughBinarize(torch.autograd.Function):
    """Hard 0/1 threshold forward, hardtanh-clipped gradient backward.

    """

    @staticmethod
    def forward(ctx, x):
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


def _init_linear_kaiming(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in")


class BinaryAutoencoder(nn.Module):
    """Encoder -> straight-through binarization -> decoder.

    - encoder: Linear(input_dim, internal_dim) -> ReLU -> BatchNorm1d -> Linear(internal_dim, internal_dim) -> Tanh
    - binarization: straight-through threshold at 0, output in {0, 1}^internal_dim
    - decoder: Linear(internal_dim, internal_dim) -> Tanh -> Linear(internal_dim, input_dim)

    Linear layers are Kaiming-initialized (`fan_in` mode) on construction.

    Optional ("nested"/Matryoshka insprired) dropout on the binary
    bottleneck.

    - `dropout_law="uniform"`: `n ~ Uniform(dropout_n_min, internal_dim)` --
      no bit favored over another.
    - `dropout_law="geometric"`: `n ~ Geometric(dropout_rho)`, truncated to
      `[dropout_n_min, internal_dim]` -- short prefixes are drawn more often
      as `dropout_rho -> 0`.

    Args:
        input_dim: dimension of the input embeddings.
        internal_dim: dimension of the binary bottleneck (aka `binary_dim`).
        bit_dropout: enable structured dropout on the binary bottleneck
            during training. Off by default -- has no effect at eval time.
        dropout_law: `"uniform"` or `"geometric"`, see above.
        dropout_n_min: smallest number of leading bits ever kept active.
        dropout_rho: geometric distribution parameter (`dropout_law="geometric"`
            only), in `(0, 1)`. Close to 1: soft cutoff. Close to 0:
            aggressive cutoff (usually only `dropout_n_min` bits kept).
        dropout_seed: seed for the RNG that draws the per-example cutoff `n`.
            A plain `random.Random`, kept separate from the global RNG so the
            drop pattern is reproducible independently of anything else
            drawing random numbers during training.
    """

    def __init__(
        self,
        input_dim: int = 256,
        internal_dim: int = 512,
        bit_dropout: bool = False,
        dropout_law: str = "uniform",
        dropout_n_min: int = 1,
        dropout_rho: float = 0.9,
        dropout_seed: int = 1234,
    ):
        super().__init__()

        if dropout_law not in ("uniform", "geometric"):
            raise ValueError(f"dropout_law must be 'uniform' or 'geometric', got {dropout_law!r}")
        if dropout_law == "geometric" and not (0.0 < dropout_rho < 1.0):
            raise ValueError(f"dropout_rho must be in (0, 1), got {dropout_rho}")
        if not (1 <= dropout_n_min <= internal_dim):
            raise ValueError(f"dropout_n_min must be in [1, internal_dim={internal_dim}], got {dropout_n_min}")

        self.input_dim = input_dim
        self.internal_dim = internal_dim
        self.bit_dropout = bit_dropout
        self.dropout_law = dropout_law
        self.dropout_n_min = dropout_n_min
        self.dropout_rho = dropout_rho
        self._dropout_rng = random.Random(dropout_seed)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, internal_dim),
            nn.ReLU(),
            nn.BatchNorm1d(internal_dim),
            nn.Linear(internal_dim, internal_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(internal_dim, internal_dim),
            nn.Tanh(),
            nn.Linear(internal_dim, input_dim),
        )

        self.apply(_init_linear_kaiming)

    def _sample_dropout_n_active(self, batch_size: int) -> list[int]:
        """Per-example cutoff `n`: how many leading bits stay active."""
        K, n_min = self.internal_dim, self.dropout_n_min
        if self.dropout_law == "geometric":
            log_rho = math.log(self.dropout_rho)
            values = []
            for _ in range(batch_size):
                u = self._dropout_rng.random()
                u = u if u > 0.0 else 1e-12
                n = math.ceil(math.log(u) / log_rho)
                values.append(min(max(n, n_min), K))
            return values
        return [self._dropout_rng.randint(n_min, K) for _ in range(batch_size)]

    def _sample_dropout_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """(B, internal_dim) mask: 1 for the leading `n` (per-example) bits, 0 after."""
        n_active = torch.tensor(self._sample_dropout_n_active(batch_size), device=device)
        idx = torch.arange(self.internal_dim, device=device).unsqueeze(0)
        return (idx < n_active.unsqueeze(1)).float()

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, input_dim) input embeddings.

        Returns:
            dict with:
              - "reconstruction": (B, input_dim), decoder output.
              - "binary": (B, internal_dim), binarized bottleneck, in {0, 1}.
              - "latent": (B, internal_dim), pre-binarization encoder output
                (tanh-bounded, i.e. in [-1, 1]).
        """
        latent = self.encoder(x)
        binary = _StraightThroughBinarize.apply(latent)

        if self.training and self.bit_dropout:
            mask = self._sample_dropout_mask(binary.shape[0], binary.device)
            binary_for_decoder = binary * mask
        else:
            binary_for_decoder = binary

        reconstruction = self.decoder(binary_for_decoder)
        return {"reconstruction": reconstruction, "binary": binary, "latent": latent}
