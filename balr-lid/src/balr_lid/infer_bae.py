"""Run a trained BAE checkpoint over an embeddings HDF5 file and write out
binary embeddings.


Usage:
    python -m balr_lid.infer_bae \
        --checkpoint exp/fleurs-bae/checkpoints/last.ckpt \
        --h5-in embeddings/inference_fleurs_train.h5 \
        --h5-out exp/fleurs-bae/binary/inference_fleurs_train.h5

"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from balr_lid.data.embedding_dataset import EmbeddingH5Dataset
from balr_lid.models.bae_model import BinaryAutoencoderModel

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to a BAE Lightning .ckpt file.")
    parser.add_argument("--h5-in", required=True, help="Input embeddings HDF5 file.")
    parser.add_argument("--h5-out", required=True, help="Output HDF5 file for the binary embeddings.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    # weights_only=False: this is our own Lightning checkpoint, whose
    # hparams embed OmegaConf DictConfig objects (unsafe-by-default to
    # unpickle since PyTorch 2.6). Only load checkpoints you trust the
    # origin of -- same rationale as balr_lid.inference.
    model = BinaryAutoencoderModel.load_from_checkpoint(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    model.eval()

    dataset = EmbeddingH5Dataset(embedding_file=args.h5_in)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    logger.info(
        "Encoding %d embeddings (input_dim=%d -> internal_dim=%d, normalize_input=%s)",
        len(dataset),
        model.autoencoder.input_dim,
        model.autoencoder.internal_dim,
        model.normalize_input,
    )

    ids: list[str] = []
    binary_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["embedding"].to(args.device))
            ids.extend(batch["id"])
            binary_chunks.append(out["binary"].cpu().numpy())

    binary_embeddings = np.concatenate(binary_chunks, axis=0)

    Path(args.h5_out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.h5_out, "w") as f:
        f.create_dataset("ids", data=np.array(ids, dtype=h5py.string_dtype(encoding="utf-8")))
        f.create_dataset("embeddings", data=binary_embeddings)

    logger.info("Wrote %d binary embeddings to %s", len(ids), args.h5_out)


if __name__ == "__main__":
    main()
