"""Training entrypoint (Hydra).

Usage:
    python -m balr_lid.train data=fleurs recipe=head_stage \
        output_dir=exp/fleurs-stage1 arch.backbone.pretrained_model_name_or_path=facebook/mms-1b

    python -m balr_lid.train data=fleurs recipe=finetune_stage \
        recipe.checkpoint=exp/fleurs-stage1/checkpoints/last.ckpt \
        output_dir=exp/fleurs-stage2 arch.backbone.pretrained_model_name_or_path=facebook/mms-1b

Resumes automatically from `<output_dir>/checkpoints/last.ckpt` if it
already exists, so a run interrupted mid-training (e.g. a SLURM job hitting
its time limit) can simply be resubmitted unchanged.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import hydra
import transformers
from lightning.pytorch import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# Lightning's own "srun is available but not used" hint (PossibleUserWarning)
# — irrelevant here, this repo's sbatch templates never launch training
# through `srun`.
warnings.filterwarnings("ignore", message=r".*srun.*")
# Lightning's "GPU available / TPU available / HPU available" accelerator
# banner (rank_zero_info, logged at INFO level) — replaced below by a single
# line naming the actual backbone/head being used.
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("lightning.fabric").setLevel(logging.WARNING)
# torchaudio.load()'s "will switch to torchaudio.load_with_torchcodec in 2.9"
# deprecation notice — harmless with the torchaudio version pinned here, and
# raised on every single call to LanguageIdDataset._load_audio (once per
# DataLoader worker, since each is a forked process with its own warnings
# state). Registered before any DataLoader worker is spawned so the `fork`
# start method carries this filter into every worker too.
warnings.filterwarnings("ignore", message=r".*torchaudio\.load_with_torchcodec.*")


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # Silences transformers' from_pretrained() load report (e.g. the
    # "lm_head.bias/weight UNEXPECTED" table) — expected noise every run
    # since we always load a full ASR/CTC checkpoint (with a head we don't
    # use) into the bare Wav2Vec2Model backbone.
    transformers.logging.set_verbosity_error()

    seed_everything(cfg.seed, workers=True)

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    head_type = "BA-LR (Bernoulli)" if cfg.arch.head.use_balr_head else "cosine-softmax"
    logger.info(
        "SSL backbone: %s (%d layers, hidden_size=%d)\n"
        "Head: MHFA(embed_dim=%d, num_heads=%d) -> %s classifier"
        "%s, num_classes=%d",
        cfg.arch.backbone.pretrained_model_name_or_path,
        cfg.arch.head.num_layers,
        cfg.arch.head.input_dim,
        cfg.arch.head.embed_dim,
        cfg.arch.head.num_heads,
        head_type,
        f" (binary_dim={cfg.arch.head.binary_dim})" if cfg.arch.head.use_binary_encoder else "",
        cfg.arch.head.num_classes,
    )

    last_checkpoint = Path(cfg.output_dir) / "checkpoints" / "last.ckpt"
    ckpt_path = str(last_checkpoint) if last_checkpoint.exists() else None
    if ckpt_path:
        logger.info("Found an existing checkpoint, resuming from %s", ckpt_path)

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
