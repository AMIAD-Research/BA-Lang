"""Language classification head: MHFA pooling + a classifier on top.

Two classifier variants are supported, selected by `use_balr_head`:

- Cosine-softmax (default): a standard AM-Softmax-style classifier —
  `logits = cosine_similarity(embedding, class_prototypes)`.
- BA-LR (Binary Attribute Likelihood Ratio) classifier (`use_balr_head=True`,
  requires `use_binary_encoder=True`): the embedding is first mapped to a
  binary attribute vector, and each class is scored by the Bernoulli
  log-likelihood of that binary vector under a per-class, per-attribute
  learned probability `sigmoid(ba_logits[class, attribute])`:

      score(x, c) = sum_k  x_k * log p[c,k] + (1 - x_k) * log(1 - p[c,k])

  This gives an interpretable, binarized embedding space alongside
  classification (each attribute is a soft yes/no learned per language).

  The BA-LR head optionally supports structured ("nested"/Matryoshka)
  dropout on the binary attributes during training (`bit_dropout=True`):
  per example, a cutoff `n` is drawn and only the leading `n` attributes
  (out of `binary_dim`) are used, so the attributes end up importance-ordered
  — the model works with any prefix of the embedding at inference, not just
  the full `binary_dim`. See `LanguageClassificationHead`'s docstring for the
  knobs (`dropout_law`, `dropout_rho`, `nested_dropout`, ...).
"""
from __future__ import annotations

import math
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from balr_lid.models.pooling import MultiHeadFactorizedAttentivePooling


class _StraightThroughBinarize(torch.autograd.Function):
    """Hard 0/1 threshold on the forward pass, identity gradient on the backward pass."""

    @staticmethod
    def forward(ctx, x):
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BinaryEncoder(nn.Module):
    """Small MLP + straight-through binarization: continuous embedding -> {0,1}^binary_dim."""

    def __init__(self, input_dim: int, binary_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, binary_dim),
            nn.ReLU(),
            nn.BatchNorm1d(binary_dim),
            nn.Linear(binary_dim, binary_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return _StraightThroughBinarize.apply(latent)


class LanguageClassificationHead(nn.Module):
    """MHFA pooling + (cosine-softmax | BA-LR) classifier.

    Args:
        num_layers: number of SSL backbone layers to pool over.
        input_dim: SSL backbone hidden size.
        embed_dim: pooled embedding dimension (MHFA output).
        num_classes: number of languages.
        num_heads: MHFA attention heads.
        use_binary_encoder: route the embedding through `BinaryEncoder`
            before classification.
        binary_dim: output dimension of `BinaryEncoder` (required if
            `use_binary_encoder=True`).
        use_balr_head: use the BA-LR Bernoulli classifier instead of the
            cosine-softmax classifier. Requires `use_binary_encoder=True`.
        bit_dropout: enable structured dropout on the binary attributes
            during training (BA-LR head only, `use_balr_head=True` required).
            Per training example, a cutoff `n` in `[dropout_n_min, binary_dim]`
            is drawn and attributes beyond it are dropped:
              - `dropout_law="uniform"`: `n ~ Uniform(dropout_n_min, binary_dim)`
                — classic nested dropout, no particular attribute is favored.
              - `dropout_law="geometric"`: `n ~ Geometric(dropout_rho)`,
                truncated to `[dropout_n_min, binary_dim]` — short prefixes
                are drawn more often as `dropout_rho -> 0` (Matryoshka-style:
                the leading attributes end up carrying the most information,
                since they're kept active most often).
            Has no effect at eval time (`model.eval()`) — the full embedding
            is always used for validation/inference.
        dropout_law: `"uniform"` or `"geometric"`, see `bit_dropout` above.
        dropout_n_min: smallest number of leading attributes ever kept active.
        dropout_rho: geometric distribution parameter (`dropout_law="geometric"`
            only), in `(0, 1)`. Close to 1: soft cutoff, most attributes stay
            active most of the time. Close to 0: aggressive cutoff, usually
            only `dropout_n_min` attributes are kept.
        nested_dropout: how dropped attributes are scored.
              - `False` (default, "Matryoshka"-style): dropped attributes are
                excluded from the score entirely, as if `binary_dim` were
                truncated to `n` for that example.
              - `True` ("nested dropout"-style): dropped attributes are set
                to 0 (observed absent) instead of excluded, and still count
                in the score via their `log(1 - p)` term.
        dropout_seed: seed for the RNG that draws the per-example cutoff `n`.
            A plain `random.Random`, kept separate from the global RNG so the
            drop pattern is reproducible independently of everything else
            drawing random numbers during training.
    """

    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        embed_dim: int,
        num_classes: int,
        num_heads: int = 8,
        use_binary_encoder: bool = False,
        binary_dim: Optional[int] = None,
        use_balr_head: bool = False,
        bit_dropout: bool = False,
        dropout_law: str = "uniform",
        dropout_n_min: int = 1,
        dropout_rho: float = 0.9,
        nested_dropout: bool = False,
        dropout_seed: int = 1234,
    ):
        super().__init__()

        if use_balr_head and not use_binary_encoder:
            raise ValueError("use_balr_head=True requires use_binary_encoder=True")
        if use_binary_encoder and binary_dim is None:
            raise ValueError("binary_dim must be set when use_binary_encoder=True")
        if bit_dropout and not use_balr_head:
            raise ValueError("bit_dropout=True requires use_balr_head=True")
        if dropout_law not in ("uniform", "geometric"):
            raise ValueError(f"dropout_law must be 'uniform' or 'geometric', got {dropout_law!r}")
        if dropout_law == "geometric" and not (0.0 < dropout_rho < 1.0):
            raise ValueError(f"dropout_rho must be in (0, 1), got {dropout_rho}")
        if bit_dropout and not (1 <= dropout_n_min <= binary_dim):
            raise ValueError(f"dropout_n_min must be in [1, binary_dim={binary_dim}], got {dropout_n_min}")

        self.bit_dropout = bit_dropout
        self.dropout_law = dropout_law
        self.dropout_n_min = dropout_n_min
        self.dropout_rho = dropout_rho
        self.nested_dropout = nested_dropout
        self._dropout_rng = random.Random(dropout_seed)

        self.pooling = MultiHeadFactorizedAttentivePooling(
            num_layers=num_layers, input_dim=input_dim, latent_dim=embed_dim, num_heads=num_heads
        )

        self.use_binary_encoder = use_binary_encoder
        self.use_balr_head = use_balr_head

        prototype_dim = binary_dim if use_binary_encoder else embed_dim
        self.class_prototypes = nn.Parameter(torch.empty(num_classes, prototype_dim))
        nn.init.xavier_uniform_(self.class_prototypes)

        if use_binary_encoder:
            self.binary_encoder = BinaryEncoder(input_dim=embed_dim, binary_dim=binary_dim)

        if use_balr_head:
            # ba_logits[c, k] parameterizes p(attribute k = 1 | class c) = sigmoid(ba_logits[c, k])
            self.ba_logits = nn.Parameter(torch.zeros(num_classes, binary_dim))

    def extract_embedding(self, prepooling: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            prepooling: (B, T, D, L) stacked backbone layer outputs.
            padding_mask: (B, T) bool, True = valid frame.

        Returns:
            (B, embed_dim) or (B, binary_dim) if use_binary_encoder=True.
        """
        features = prepooling.permute(0, 2, 1, 3)  # (B, D, T, L)
        embedding = self.pooling(features, padding_mask=padding_mask)
        embedding = F.normalize(embedding, dim=-1)

        if self.use_binary_encoder:
            embedding = self.binary_encoder(embedding)

        return embedding

    def _sample_dropout_n_active(self, batch_size: int) -> list[int]:
        """Per-example cutoff `n`: how many leading attributes stay active."""
        K, n_min = self.ba_logits.shape[1], self.dropout_n_min
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
        """(B, binary_dim) mask: 1 for the leading `n` (per-example) attributes, 0 after."""
        n_active = torch.tensor(self._sample_dropout_n_active(batch_size), device=device)
        idx = torch.arange(self.ba_logits.shape[1], device=device).unsqueeze(0)
        return (idx < n_active.unsqueeze(1)).float()

    def _balr_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Bernoulli log-likelihood score of binary embedding `x` under each class.

        If `bit_dropout` is on and the head is training, a structured dropout
        mask (see the class docstring) zeroes out a per-example suffix of `x`
        before scoring.
        """
        x = x.float()
        log_p = F.logsigmoid(self.ba_logits)
        log_1_minus_p = F.logsigmoid(-self.ba_logits)

        if not (self.training and self.bit_dropout):
            return x @ log_p.T + (1.0 - x) @ log_1_minus_p.T  # (B, num_classes)

        mask = self._sample_dropout_mask(x.shape[0], x.device)
        x_masked = x * mask
        if self.nested_dropout:
            # Dropped attributes are zeroed (x_masked) but still scored via
            # their log(1-p) term, as if observed absent — "nested dropout".
            return x_masked @ (log_p - log_1_minus_p).T + log_1_minus_p.sum(dim=1)
        # Dropped attributes are excluded from the score entirely, as if
        # binary_dim were truncated to n for that example — "Matryoshka".
        return x_masked @ (log_p - log_1_minus_p).T + mask @ log_1_minus_p.T

    def forward_classification(self, prepooling: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> dict:
        """
        Args:
            prepooling: (B, T, D, L) stacked backbone layer outputs.
            padding_mask: (B, T) bool, True = valid frame.

        Returns:
            dict with "logits" (B, num_classes) and "embeddings" (B, embed_dim).
        """
        embeddings = self.extract_embedding(prepooling, padding_mask=padding_mask)

        if self.use_balr_head:
            logits = self._balr_logits(embeddings)
        else:
            logits = F.linear(F.normalize(embeddings, dim=-1), F.normalize(self.class_prototypes, dim=-1))

        return {"logits": logits, "embeddings": embeddings}
