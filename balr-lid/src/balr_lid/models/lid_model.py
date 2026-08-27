"""LightningModule implementing the two-stage LID training recipe.

Both stages (head-only training and full SSL finetuning, see docs/recipe.md)
use this exact same module and forward pass — `backbone -> stack all SSL
layers -> MHFA pooling+classifier head -> cross-entropy`. What changes
between stages is purely config: which parameters are trainable
(`freeze_exclude_patterns`, see balr_lid/optim.py), the learning-rate
schedule, and the sampler's `balance_mode`.
"""
from __future__ import annotations

from typing import Optional

import hydra
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from omegaconf import DictConfig

from balr_lid.optim import freeze_by_pattern


class LanguageIdentificationModel(LightningModule):
    """
    Args:
        backbone: Hydra config for the SSL backbone (`balr_lid.models.backbone.SSLBackbone`).
        head: Hydra config for the classification head (`balr_lid.models.head.LanguageClassificationHead`).
        optimizer: Hydra config for the optimizer (e.g. `torch.optim.Adam`), missing `params`.
        scheduler: optional Hydra config for a `torch.optim.lr_scheduler`, missing `optimizer`.
            `None` means a constant learning rate (used in stage 1).
        freeze_exclude_patterns: regex patterns selecting which parameters stay
            trainable; see `balr_lid.optim.freeze_by_pattern`.
        backbone_checkpoint: optional path to a backbone-only state_dict to
            load before training (e.g. to start from a public SSL checkpoint
            saved outside the HF format).
        checkpoint: optional path to a full Lightning checkpoint (backbone +
            head) to warm-start from — this is how stage 2 continues from
            stage 1's output.
    """

    def __init__(
        self,
        backbone: DictConfig,
        head: DictConfig,
        optimizer: DictConfig,
        scheduler: Optional[DictConfig] = None,
        freeze_exclude_patterns: Optional[list] = None,
        backbone_checkpoint: Optional[str] = None,
        checkpoint: Optional[str] = None,
    ):
        super().__init__()
        # `backbone_checkpoint`/`checkpoint` are one-shot "warm start from this
        # path" instructions for the run that constructs this module; they are
        # deliberately excluded from the saved hparams so that later calling
        # `load_from_checkpoint(...)` on *this* run's own checkpoint doesn't
        # try to re-resolve a (possibly no-longer-existing) upstream path.
        self.save_hyperparameters(logger=False, ignore=["backbone_checkpoint", "checkpoint"])

        self._optimizer_cfg = optimizer
        self._scheduler_cfg = scheduler
        self._freeze_exclude_patterns = freeze_exclude_patterns

        # Built eagerly (not deferred to `setup()`) so that
        # `LanguageIdentificationModel.load_from_checkpoint(...)` — which
        # instantiates the module and loads its state_dict without going
        # through the Lightning Trainer / `setup()` hook — has submodules to
        # load weights into.
        self.backbone = hydra.utils.instantiate(backbone)
        self.head = hydra.utils.instantiate(head)

        if backbone_checkpoint:
            # weights_only=False: these are our own Lightning checkpoints
            # (may embed OmegaConf DictConfig objects in hparams), not
            # arbitrary untrusted files.
            state_dict = torch.load(backbone_checkpoint, map_location="cpu", weights_only=False)
            self.backbone.load_state_dict(state_dict, strict=False)

        if checkpoint:
            loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.load_state_dict(loaded["state_dict"], strict=False)

    def forward(self, audio: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> dict:
        backbone_out = self.backbone(audio, padding_mask=padding_mask)
        prepooling = torch.stack(backbone_out.layer_outputs, dim=-1)  # (B, T, D, L)
        return self.head.forward_classification(prepooling, padding_mask=backbone_out.padding_mask)

    @staticmethod
    def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (logits.argmax(dim=-1) == targets).float().mean()

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        out = self(batch["audio"], padding_mask=batch.get("padding_mask"))
        loss = F.cross_entropy(out["logits"], batch["language_index"])
        batch_size = batch["audio"].shape[0]

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train_acc", self._accuracy(out["logits"], batch["language_index"]), prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        out = self(batch["audio"], padding_mask=batch.get("padding_mask"))
        loss = F.cross_entropy(out["logits"], batch["language_index"])
        batch_size = batch["audio"].shape[0]

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("val_acc", self._accuracy(out["logits"], batch["language_index"]), prog_bar=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> dict:
        out = self(batch["audio"], padding_mask=batch.get("padding_mask"))
        probabilities = F.softmax(out["logits"], dim=-1)
        return {
            "id": batch["id"],
            "language": batch["language"],
            "predicted_index": out["logits"].argmax(dim=-1).cpu(),
            "probabilities": probabilities.cpu(),
            "embeddings": out["embeddings"].cpu(),
        }

    def configure_optimizers(self):
        freeze_by_pattern(self, self._freeze_exclude_patterns)
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        optimizer = hydra.utils.instantiate(self._optimizer_cfg, params=trainable_params)

        if self._scheduler_cfg is None:
            return optimizer

        scheduler = hydra.utils.instantiate(self._scheduler_cfg, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }
