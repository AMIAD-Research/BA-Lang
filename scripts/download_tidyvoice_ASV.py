#!/usr/bin/env python3
"""Download a Tidylang dataset (TidyVoiceX_ASV or TidyVoiceX2_ASV) from
Mozilla Data Collective as a single .tar.gz archive.

Requires an API key: make one at
https://datacollective.mozillafoundation.org/api-reference, then set it as
an environment variable :

    export API_KEY=<your key>

Usage:
    # TidyVoiceX_ASV (train/dev)
    python scripts/download_tidyvoice_ASV.py \
        --dataset-id cmihtsewu023so207xot1iqqw \
        --output-dir /path/to/TidyVoiceX_ASV

    # TidyVoiceX2_ASV (eval)
    python scripts/download_tidyvoice_ASV.py \
        --dataset-id cmkv32i5e02tumg07j79d3c35 \
        --output-dir /path/to/TidyVoiceX2_ASV

This only downloads the archive — extract it yourself afterwards (e.g.
`tar xzf <output-dir>/<dataset-id>.tar.gz -C <output-dir>`), then build the
manifests with scripts/build_tidylang_manifest.py (see docs/data_manifest.md).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests
from tqdm import tqdm

API_BASE_URL = "https://mozilladatacollective.com/api"

# TidyVoiceX_ASV (train/dev) by default; pass --dataset-id
# cmkv32i5e02tumg07j79d3c35 for TidyVoiceX2_ASV (eval) instead.
DEFAULT_DATASET_ID = "cmihtsewu023so207xot1iqqw"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dataset-id", default=DEFAULT_DATASET_ID,
        help=f"Default: {DEFAULT_DATASET_ID} (TidyVoiceX_ASV, train/dev)",
    )
    p.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory to save <dataset-id>.tar.gz into (created if missing)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise SystemExit(
            "set the API_KEY environment variable to your Mozilla Data Collective "
            "API key (make one at https://datacollective.mozillafoundation.org/api-reference)"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.dataset_id}.tar.gz"

    response = requests.post(
        f"{API_BASE_URL}/datasets/{args.dataset_id}/download",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    response.raise_for_status()
    download_url = response.json()["downloadUrl"]

    response = requests.get(download_url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"\nDone! Saved to: {output_path}")


if __name__ == "__main__":
    main()
