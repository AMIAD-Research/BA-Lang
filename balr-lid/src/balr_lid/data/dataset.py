"""PyTorch Dataset for spoken language identification from a CSV manifest."""
from __future__ import annotations

from typing import Sequence

import torch
import torchaudio
from torch.utils.data import Dataset

from balr_lid.data.manifest import load_manifest


class LanguageIdDataset(Dataset):
    """Loads audio + language label pairs described by a manifest CSV.

    Training and validation use the same class with different `segment_seconds`
    / `random_crop` settings (see configs/model/*.yaml):
      - training: a short, randomly-positioned crop (e.g. 3-5s) so each epoch
        sees different sub-segments of long utterances.
      - validation/test: a longer, fixed (center-cropped) window (e.g. 15s)
        for stable, reproducible metrics.

    Args:
        manifest_csv: path to a manifest CSV (see manifest.py).
        audio_root: directory `audio_path` values are resolved against.
        label_list: ordered list of language codes; its index defines the
            class index used everywhere in the model (logits, checkpoints,
            ...). Must be identical across train/val/test for a given corpus.
        segment_seconds: fixed crop length in seconds. `None` keeps the full
            utterance (batches will then be padded to the longest item).
        random_crop: if True, crop position is random (training). If False,
            the crop is centered (evaluation).
        sample_rate: target sample rate; audio is resampled on load if the
            file's native sample rate differs.
        min_duration: optional minimum utterance duration filter, in seconds.
    """

    def __init__(
        self,
        manifest_csv: str,
        audio_root: str,
        label_list: Sequence[str],
        segment_seconds: float | None = None,
        random_crop: bool = True,
        sample_rate: int = 16000,
        min_duration: float | None = None,
    ):
        self.label_list = list(label_list)
        self.label_to_index = {label: i for i, label in enumerate(self.label_list)}

        self.table = load_manifest(
            manifest_csv,
            audio_root,
            min_duration=min_duration,
            languages=self.label_list,
        )

        self.segment_seconds = segment_seconds
        self.random_crop = random_crop
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.table)

    @property
    def labels(self) -> list[int]:
        """Class index for every row, in dataset order (used by BalancedBatchSampler)."""
        return [self.label_to_index[lang] for lang in self.table["language"]]

    def _load_audio(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        return waveform.squeeze(0)  # (T,)

    def _crop_or_pad(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.segment_seconds is None:
            return waveform

        target_len = int(self.segment_seconds * self.sample_rate)
        cur_len = waveform.shape[0]

        if cur_len == target_len:
            return waveform
        if cur_len > target_len:
            if self.random_crop:
                start = torch.randint(0, cur_len - target_len + 1, (1,)).item()
            else:
                start = (cur_len - target_len) // 2
            return waveform[start : start + target_len]

        # shorter than target: zero-pad (padding is masked out downstream)
        pad = torch.zeros(target_len - cur_len, dtype=waveform.dtype)
        return torch.cat([waveform, pad])

    def __getitem__(self, index: int) -> dict:
        row = self.table.iloc[index]
        waveform = self._load_audio(row["audio_abspath"])
        waveform = self._crop_or_pad(waveform)

        return {
            "id": row["id"],
            "audio": waveform,
            "audio_length": waveform.shape[0],
            "language": row["language"],
            "language_index": self.label_to_index[row["language"]],
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pads a list of `LanguageIdDataset` items into a batch.

    Produces a boolean padding mask (`True` = valid frame) matching what
    `balr_lid.models.backbone.SSLBackbone` expects.
    """
    max_len = max(item["audio"].shape[0] for item in batch)

    audio = torch.zeros(len(batch), max_len)
    padding_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        length = item["audio"].shape[0]
        audio[i, :length] = item["audio"]
        padding_mask[i, :length] = True

    return {
        "id": [item["id"] for item in batch],
        "audio": audio,
        "audio_length": torch.tensor([item["audio_length"] for item in batch], dtype=torch.long),
        "padding_mask": padding_mask,
        "language": [item["language"] for item in batch],
        "language_index": torch.tensor([item["language_index"] for item in batch], dtype=torch.long),
    }
