"""LightningModule for training a `BinaryAutoencoder` by reconstruction.

"""
from __future__ import annotations

import hydra
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from omegaconf import DictConfig


class BinaryAutoencoderModel(LightningModule):
    """
    Args:
        autoencoder: Hydra config.
        optimizer: Hydra config for an optimizer.
        normalize_input: L2-normalize embeddings (dim=-1) before feeding
            the autoencoder.
     
    """

    def __init__(
        self,
        autoencoder: DictConfig,
        optimizer: DictConfig,
        normalize_input: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self._optimizer_cfg = optimizer
        self.normalize_input = normalize_input
        self.autoencoder = hydra.utils.instantiate(autoencoder)

    def _normalize(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(embeddings, p=2, dim=-1) if self.normalize_input else embeddings

    def forward(self, embeddings: torch.Tensor) -> dict:
        return self.autoencoder(self._normalize(embeddings))

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        target = self._normalize(batch["embedding"])
        out = self.autoencoder(target)
        loss = F.mse_loss(out["reconstruction"], target)

        batch_size = target.shape[0]
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=(stage != "train"),
        )
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self):
        return hydra.utils.instantiate(self._optimizer_cfg, params=self.parameters())
