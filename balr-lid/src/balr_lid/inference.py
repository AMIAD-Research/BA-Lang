"""Run a trained LID checkpoint over a manifest and write predictions.

Usage:
    python -m balr_lid.inference \\
        --checkpoint exp/fleurs-stage2/checkpoints/last.ckpt \\
        --manifest data/manifests/fleurs-test.csv \\
        --audio-root /path/to/fleurs/audio \\
        --language-list fra eng ... \\
        --output predictions.csv \\
        [--save-embeddings embeddings.h5]

`--language-list` must be given in the exact same order used at training
time (see configs/data/*.yaml `language_list`) — it defines what each output
class index means.

`--score-first-k`/`--score-last-k` (BA-LR head only, `use_balr_head=True`):
restrict scoring to a prefix/suffix of the attribute vector.
"""
from __future__ import annotations

import argparse
import logging
import warnings

import h5py
import numpy as np
import pandas as pd
import transformers
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from balr_lid.data.dataset import LanguageIdDataset, collate_fn
from balr_lid.models.lid_model import LanguageIdentificationModel

logger = logging.getLogger(__name__)


warnings.filterwarnings("ignore", message=r".*srun.*")
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("lightning.fabric").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=r".*torchaudio\.load_with_torchcodec.*")


def _segment_seconds(value: str) -> float | None:
    if value.strip().lower() in ("full", "none"):
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to a Lightning .ckpt file.")
    parser.add_argument("--manifest", required=True, help="Manifest CSV to run inference on.")
    parser.add_argument("--audio-root", required=True, help="Directory audio_path values are relative to.")
    parser.add_argument("--language-list", nargs="+", required=True, help="Ordered class labels used at training time.")
    parser.add_argument(
        "--segment-seconds",
        type=_segment_seconds,
        default=15.0,
        help="Fixed, center-cropped evaluation window in seconds. Pass 'full' (or 'none') to "
             "disable cropping and use each utterance's full length instead (batches padded to "
             "the longest item) — needed to reproduce a checkpoint's original eval numbers if it "
             "wasn't itself evaluated on a fixed-length crop.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output", required=True, help="CSV file predictions are written to.")
    parser.add_argument(
        "--save-embeddings",
        default=None,
        help="Optional HDF5 file to also dump pooled embeddings to (dataset 'ids' + 'embeddings').",
    )
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="1")

    score_dims = parser.add_mutually_exclusive_group()
    score_dims.add_argument(
        "--score-first-k",
        type=int,
        default=None,
        help="BA-LR head only: score using only the first K binary attributes "
             "(dims [0, K)) instead of the full binary_dim.",
    )
    score_dims.add_argument(
        "--score-last-k",
        type=int,
        default=None,
        help="BA-LR head only: score using only the last K binary attributes "
             "(dims [binary_dim - K, binary_dim)) instead of the full binary_dim.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    transformers.logging.set_verbosity_error()

    model = LanguageIdentificationModel.load_from_checkpoint(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model.eval()

    if args.score_first_k is not None or args.score_last_k is not None:
        if not model.head.use_balr_head:
            raise SystemExit(
                "--score-first-k/--score-last-k require a BA-LR head "
                "(use_balr_head=True in the checkpoint's training config)"
            )
        binary_dim = model.head.ba_logits.shape[1]
        if args.score_first_k is not None:
            model.head.set_score_dim_range(0, args.score_first_k)
        else:
            model.head.set_score_dim_range(binary_dim - args.score_last_k, binary_dim)

    dataset = LanguageIdDataset(
        manifest_csv=args.manifest,
        audio_root=args.audio_root,
        label_list=args.language_list,
        segment_seconds=args.segment_seconds,
        random_crop=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    trainer = Trainer(accelerator=args.accelerator, devices=args.devices, logger=False)
    batches = trainer.predict(model, dataloaders=loader)

    rows = []
    embeddings_by_id: dict[str, np.ndarray] = {}
    for batch in batches:
        predicted_languages = [args.language_list[i] for i in batch["predicted_index"].tolist()]
        for i, utt_id in enumerate(batch["id"]):
            rows.append(
                {
                    "id": utt_id,
                    "reference_language": batch["language"][i],
                    "predicted_language": predicted_languages[i],
                }
            )
            if args.save_embeddings:
                embeddings_by_id[utt_id] = batch["embeddings"][i].numpy()

    pd.DataFrame(rows).to_csv(args.output, index=False)
    logger.info("Wrote %d predictions to %s", len(rows), args.output)

    if args.save_embeddings:
        ids = list(embeddings_by_id.keys())
        embeddings = np.stack([embeddings_by_id[i] for i in ids])
        with h5py.File(args.save_embeddings, "w") as f:
            f.create_dataset("ids", data=np.array(ids, dtype=h5py.string_dtype(encoding="utf-8")))
            f.create_dataset("embeddings", data=embeddings)
        logger.info("Wrote embeddings for %d utterances to %s", len(ids), args.save_embeddings)


if __name__ == "__main__":
    main()
