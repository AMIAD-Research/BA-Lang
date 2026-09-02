"""Training entrypoint (Hydra) for the binary autoencoder (BAE) stage.

Usage:
    python -m balr_lid.train_bae \
        bae_data=fleurs output_dir=exp/fleurs-bae \
        bae_data.train_embedding_file=/path/to/train_embeddings.h5 \
        bae_data.val_embedding_file=/path/to/dev_embeddings.h5 \
        bae_model.input_dim=1280 bae_model.internal_dim=320

Resumes automatically from `<output_dir>/checkpoints/last.ckpt` if it
already exists.
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
from lightning.pytorch import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="bae")
def main(cfg: DictConfig) -> None:
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    seed_everything(cfg.seed, workers=True)

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    logger.info(
        "BAE: input_dim=%d, internal_dim=%d, bit_dropout=%s%s",
        cfg.bae_model.input_dim,
        cfg.bae_model.internal_dim,
        cfg.bae_model.bit_dropout,
        f" ({cfg.bae_model.dropout_law})" if cfg.bae_model.bit_dropout else "",
    )

    last_checkpoint = Path(cfg.output_dir) / "checkpoints" / "last.ckpt"
    ckpt_path = str(last_checkpoint) if last_checkpoint.exists() else None
    if ckpt_path:
        logger.info("Found an existing checkpoint, resuming from %s", ckpt_path)

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
