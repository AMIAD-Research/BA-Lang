"""Compute BA-Lang enrollment statistics from binary embeddings.

For each language, this is the empirical activation probability of every
binary attribute -- P(attribute k active | language) -- estimated from all
enrollment utterances of that language.


Usage:
    python -m balr_lid.train_ba_lang_backend \
        --binary-embeddings exp/fleurs-bae/binary/inference_fleurs_train.h5 \
        --manifest data/manifests/fleurs-train.csv \
        --output exp/fleurs-balang/enrollment_statistics.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from balr_lid.data.embedding_dataset import EmbeddingH5Dataset

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary-embeddings", required=True, help="Binary embeddings HDF5 file (e.g. from balr_lid.infer_bae).")
    parser.add_argument("--manifest", required=True, help="Manifest CSV with 'id' and 'language' columns.")
    parser.add_argument("--output", required=True, help="Output CSV: language, count, BA0, BA1, ...")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def compute_activation_probabilities(loader: DataLoader) -> pd.DataFrame:
    """Returns a `language, count, BA0, BA1, ...` DataFrame of activation probabilities."""
    activation_sum: dict[str, torch.Tensor] = {}
    count: dict[str, int] = {}

    for batch in loader:
        embeddings = batch["embedding"].float()
        for language, embedding in zip(batch["language"], embeddings):
            if language not in activation_sum:
                activation_sum[language] = torch.zeros(embeddings.shape[1])
            activation_sum[language] += embedding
            count[language] = count.get(language, 0) + 1

    rows = []
    for language in sorted(activation_sum):
        probabilities = (activation_sum[language] / count[language]).tolist()
        rows.append(
            {"language": language, "count": count[language]}
            | {f"BA{i}": p for i, p in enumerate(probabilities)}
        )
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    dataset = EmbeddingH5Dataset(embedding_file=args.binary_embeddings, manifest_csv=args.manifest, label_key="language")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    logger.info("Computing per-language activation probabilities over %d utterances", len(dataset))
    stats = compute_activation_probabilities(loader)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(args.output, index=False)
    logger.info("Wrote enrollment statistics for %d languages to %s", len(stats), args.output)


if __name__ == "__main__":
    main()
