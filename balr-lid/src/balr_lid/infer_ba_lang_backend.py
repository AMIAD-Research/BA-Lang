"""Score binary embeddings against a BA-Lang backend and write predictions.

Writes the *same* `id, reference_language, predicted_language` CSV format
as `balr_lid.inference` (see src/balr_lid/inference.py), so accuracy is
computed with :

    python scripts/compute_accuracy.py --predictions predictions.csv --confidence-intervals


Usage:
    python -m balr_lid.infer_ba_lang_backend \
        --enrollment-statistics exp/fleurs-balang/enrollment_statistics.csv \
        --binary-embeddings exp/fleurs-bae/binary/inference_fleurs_test.h5 \
        --manifest data/manifests/fleurs-test.csv \
        --output exp/fleurs-balang/predictions_fleurs_test.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from balr_lid.data.embedding_dataset import EmbeddingH5Dataset
from balr_lid.models.ba_lang_backend import BALangBackend

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--enrollment-statistics", required=True, help="CSV written by balr_lid.train_ba_lang_backend.")
    parser.add_argument("--binary-embeddings", required=True, help="Binary embeddings HDF5 file to score.")
    parser.add_argument("--manifest", required=True, help="Manifest CSV with 'id' and 'language' columns.")
    parser.add_argument("--output", required=True, help="CSV file predictions are written to.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=1e-3, help="Activation-probability clamp, see BALangBackend.")

    score_dims = parser.add_mutually_exclusive_group()
    score_dims.add_argument(
        "--k-first", type=int, default=None,
        help="Score using only the first K attributes (dims [0, K)) instead of the full vector.",
    )
    score_dims.add_argument(
        "--k-last", type=int, default=None,
        help="Score using only the last K attributes.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    backend = BALangBackend.from_csv(args.enrollment_statistics, epsilon=args.epsilon)
    dataset = EmbeddingH5Dataset(embedding_file=args.binary_embeddings, manifest_csv=args.manifest, label_key="language")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    logger.info(
        "Scoring %d utterances against %d languages (%d attributes%s)",
        len(dataset), len(backend.label_list), backend.n_attributes,
        f", k_first={args.k_first}" if args.k_first else (f", k_last={args.k_last}" if args.k_last else ""),
    )

    rows = []
    for batch in loader:
        predicted = backend.predict(batch["embedding"], k_first=args.k_first, k_last=args.k_last)
        for i, utt_id in enumerate(batch["id"]):
            rows.append(
                {
                    "id": utt_id,
                    "reference_language": batch["language"][i],
                    "predicted_language": predicted[i],
                }
            )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    logger.info("Wrote %d predictions to %s", len(rows), args.output)


if __name__ == "__main__":
    main()
