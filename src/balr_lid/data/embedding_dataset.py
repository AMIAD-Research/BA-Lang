"""PyTorch Dataset for precomputed embeddings stored in HDF5.

Reads the flat schema written by `balr_lid.inference --save-embeddings`
and by `balr_lid.infer_bae`:

    ids         (N,)    vlen utf-8 strings
    embeddings  (N, D)  float32

"""
from __future__ import annotations

from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _read_embeddings_h5(h5_path: str) -> tuple[list[str], np.ndarray]:
    """Returns (ids, embeddings) from an embeddings HDF5 file.

    Supports three layouts :
      - flat: f["ids"], f["embeddings"]
      - f["ids"], f["embeddings"]["embedding"]["data"]
      - f["ids"], f["embedding"]["data"]
          
    """
    with h5py.File(h5_path, "r") as f:
        raw_ids = f["ids"][:]
        ids = [i.decode("utf-8") if isinstance(i, bytes) else str(i) for i in raw_ids]

        if "embeddings" in f:
            node = f["embeddings"]
            embeddings = node[:] if isinstance(node, h5py.Dataset) else np.stack(node["embedding"]["data"][:], axis=0)
        elif "embedding" in f:
            embeddings = np.asarray(f["embedding"]["data"][:])
        else:
            raise KeyError(
                f"{h5_path}: found neither 'embeddings' nor 'embedding' at the top level "
                f"(keys: {list(f.keys())})"
            )

    return ids, np.asarray(embeddings, dtype=np.float32)


class EmbeddingH5Dataset(Dataset):
    """Loads `{"id": str, "embedding": Tensor}` items from an embeddings HDF5 file.

    Args:
        embedding_file: path to the HDF5 file.
        manifest_csv: optional manifest CSV (the same files used by
            `balr_lid.data.dataset.LanguageIdDataset`, i.e. must have an
            `id` column). If given, the dataset is restricted to (and
            ordered by) the manifest's ids -- every manifest id must have a
            matching embedding, or a `ValueError` is raised. If `None`
            (default), every embedding in the file is used, in file order.
        label_key: optional manifest column to also return as `"language"`
            (unused by BAE training itself; only for analysis).
            Requires `manifest_csv`.
    """

    def __init__(
        self,
        embedding_file: str,
        manifest_csv: Optional[str] = None,
        label_key: Optional[str] = None,
    ):
        if label_key is not None and manifest_csv is None:
            raise ValueError("label_key requires manifest_csv to be set")

        ids, embeddings = _read_embeddings_h5(embedding_file)

        if manifest_csv is not None:
            id_to_index = {utt_id: i for i, utt_id in enumerate(ids)}
            manifest = pd.read_csv(manifest_csv, dtype={"id": str})
            if "id" not in manifest.columns:
                raise ValueError(f"Manifest {manifest_csv} is missing required column 'id'")

            missing = [utt_id for utt_id in manifest["id"] if utt_id not in id_to_index]
            if missing:
                raise ValueError(
                    f"{len(missing)} manifest id(s) from {manifest_csv} have no matching "
                    f"embedding in {embedding_file}, e.g. {missing[:5]}"
                )

            order = [id_to_index[utt_id] for utt_id in manifest["id"]]
            self.ids = list(manifest["id"])
            self.embeddings = embeddings[order]
            self.labels = list(manifest[label_key]) if label_key is not None else None
        else:
            self.ids = ids
            self.embeddings = embeddings
            self.labels = None

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict:
        item = {
            "id": self.ids[index],
            "embedding": torch.from_numpy(self.embeddings[index]),
        }
        if self.labels is not None:
            item["language"] = self.labels[index]
        return item
