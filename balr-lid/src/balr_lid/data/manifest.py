"""CSV manifest loading for language-identification datasets.

A manifest is a flat CSV with one row per utterance and (at least) the
columns:

    id           unique utterance identifier
    audio_path   path to the audio file, relative to `audio_root`
    language     ISO 639-3 language code (must appear in the corpus's
                 label list, see configs/data/*.yaml)
    duration     utterance duration in seconds

Extra columns (speaker id, transcription, ...) are allowed and simply
ignored. See docs/data_preparation.md for how to build one of these files.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("id", "audio_path", "language", "duration")


def load_manifest(
    csv_path: str | Path,
    audio_root: str | Path,
    min_duration: float | None = None,
    languages: list[str] | None = None,
) -> pd.DataFrame:
    """Load and validate a manifest CSV.

    Args:
        csv_path: path to the manifest CSV.
        audio_root: directory `audio_path` values are resolved against.
            The returned dataframe gets an extra `audio_abspath` column
            with the fully resolved path.
        min_duration: if set, drop rows shorter than this (seconds).
            Mirrors the minimum-duration filter used in the original
            training recipe to discard unusably short clips.
        languages: if set, drop rows whose `language` is not in this list
            (e.g. to sanity-check a manifest against a corpus's label list
            before training).

    Returns:
        A validated `pandas.DataFrame`.
    """
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Manifest {csv_path} is missing required column(s): {missing}. "
            f"Required columns are: {REQUIRED_COLUMNS}."
        )

    audio_root = Path(audio_root)
    df["audio_abspath"] = df["audio_path"].apply(lambda p: str(audio_root / p))

    if min_duration is not None:
        df = df[df["duration"] >= min_duration]

    if languages is not None:
        unknown = sorted(set(df["language"]) - set(languages))
        if unknown:
            raise ValueError(
                f"Manifest {csv_path} contains language codes not present in "
                f"the configured label list: {unknown}"
            )
        df = df[df["language"].isin(languages)]

    df = df.reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"Manifest {csv_path} is empty after filtering.")

    return df
