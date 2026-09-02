#!/usr/bin/env python3
"""Build balr-lid manifest CSVs for the Tidylang corpus.

Raw layout :

  <corpus_root>/
    id010001/
      en/
        en_30308892.wav
        ...
      cy/
        cy_30309019.wav
        ...
    id010002/
      ...

`train` subcommand — TidyVoiceX_ASV, alongside a `manifest.txt` with one row
per utterance, whitespace-separated :

    <split_flag>  <speaker>/<lang>/<file>.wav  <lang_code>

`split_flag == 1` marks the train pool (the only rows this script uses).
This script draws a dev split :
90/10 train/dev.


`eval` subcommand — TidyVoiceX2_ASV, alongside `tl26_lid.txt`, has
two columns :

    <speaker>/<lang>/<file>.wav  <lang_code>


Writes data/manifests/tidylang-{train,dev,test}.csv in the format
`balr_lid.data.manifest.load_manifest` expects (id,audio_path,language,
duration). `audio_path` is written relative to `--corpus-root`, so at
training time pass `data.audio_root=<corpus_root>` (the *_ASV directory).

Usage:
    python scripts/build_tidylang_manifest.py train \\
        --corpus-root /corpora/TidyVoiceX_ASV \\
        --output-dir data/manifests

    python scripts/build_tidylang_manifest.py eval \\
        --corpus-root /corpora/TidyVoiceX2_ASV \\
        --output-dir data/manifests
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import soundfile as sf

DEFAULT_DEV_FRACTION = 0.1
DEFAULT_SEED = 42


DEFAULT_TRAIN_SUBDIRS = ["TidyVoiceX_Train", "TidyVoiceX_Dev"]


def parse_subdirs(value: str) -> list[str]:
    return [s for s in value.split(",") if s] if value else []


class ProgressBar:
    """Minimal dependency-free progress bar (rows/s, ETA) for a known total."""

    def __init__(self, total: int, prefix: str = "", width: int = 30, min_interval: float = 0.2):
        self.total = total
        self.prefix = prefix
        self.width = width
        self.min_interval = min_interval
        self.n = 0
        self._start = time.time()
        self._last_render = 0.0

    def update(self, n: int = 1) -> None:
        self.n += n
        now = time.time()
        if now - self._last_render < self.min_interval and self.n < self.total:
            return
        self._last_render = now
        self._render()

    def _render(self) -> None:
        elapsed = time.time() - self._start
        frac = min(self.n / self.total, 1.0) if self.total else 1.0
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        rate = self.n / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.n) / rate if rate > 0 else 0.0
        sys.stdout.write(
            f"\r{self.prefix} [{bar}] {self.n}/{self.total} "
            f"({frac * 100:5.1f}%) {rate:6.0f} rows/s ETA {eta:5.0f}s"
        )
        sys.stdout.flush()

    def close(self) -> None:
        if self.n < self.total or self._last_render == 0.0:
            self._render()
        sys.stdout.write("\n")
        sys.stdout.flush()



# This is a superset of TARGET_LANGUAGES below: the raw corpus
# has more languages than we train on (e.g. abk, hau, hsb, mkd, yor), and
# rows for those are filtered out in build_rows rather than raising — only a
# code that's not in this dict at all (a real mapping gap) is fatal.
LANG_CODE_TO_ISO3 = {
    "ab": "abk",
    "ar": "ara",
    "ba": "bak",
    "be": "bel",
    "bn": "ben",
    "bg": "bul",
    "ca": "cat",
    "cv": "chv", "chv": "chv",
    "cy": "cym",
    "de": "deu",
    "dv": "div", "div": "div",
    "el": "ell",
    "en": "eng",
    "fa": "fas",
    "fr": "fra",
    "ha": "hau",
    "hi": "hin",
    "hsb": "hsb",
    "hy": "hye",
    "ja": "jpn",
    "ka": "kat",
    "lt": "lit",
    "lg": "lug",
    "ml": "mal",
    "mr": "mar",
    "mk": "mkd",
    "nl": "nld",
    "or": "ori",
    "pl": "pol",
    "pt": "por",
    "ru": "rus",
    "ta": "tam",
    "th": "tha",
    "tk": "tuk",
    "tr": "tur",
    "ug": "uig",
    "uz": "uzb",
    "yo": "yor",
    "yue": "yue", "zh-hk": "yue", "zh-yue": "yue",
    "zh": "zho", "zh-cn": "zho", "zh-tw": "zho", "cmn": "zho",
}


TARGET_LANGUAGES = {
    "ara", "bak", "bel", "ben", "bul", "cat", "chv", "cym", "deu", "div",
    "ell", "eng", "fas", "fra", "hin", "hye", "jpn", "kat", "lit", "lug",
    "mal", "mar", "nld", "ori", "pol", "por", "rus", "tam", "tha", "tuk",
    "tur", "uig", "uzb", "yue", "zho",
}
assert len(TARGET_LANGUAGES) == 35, f"expected 35 languages, got {len(TARGET_LANGUAGES)}"
assert TARGET_LANGUAGES <= set(LANG_CODE_TO_ISO3.values()), "TARGET_LANGUAGES has a code missing from LANG_CODE_TO_ISO3"


def resolve_language(raw_code: str) -> str | None:
    """Look up `raw_code` in LANG_CODE_TO_ISO3.

    Tries the full code first (needed for e.g. `zh-hk` vs `zh-cn`, which
    resolve to *different* target languages, yue vs zho). Falls back to just
    the primary subtag (`hy-am` -> `hy`) for any other BCP-47-style
    `<lang>-<region>` code the corpus uses — the region doesn't change the
    target language for anything except zh, and that's already covered by
    the explicit zh-* entries matched above.
    """
    code = raw_code.strip().lower()
    if code in LANG_CODE_TO_ISO3:
        return LANG_CODE_TO_ISO3[code]
    primary_subtag = code.split("-", 1)[0]
    return LANG_CODE_TO_ISO3.get(primary_subtag)


def resolve_audio_path(corpus_root: Path, audio_path: str, subdirs: list[str]) -> Path | None:
    """Find `audio_path` under `corpus_root`, trying `corpus_root/audio_path`
    first, then `corpus_root/<subdir>/audio_path` for each of `subdirs` in
    order. Returns the path relative to `corpus_root` that actually exists
    (this is what gets written to the manifest CSV), or None if none do.
    """
    if (corpus_root / audio_path).is_file():
        return Path(audio_path)
    for subdir in subdirs:
        candidate = Path(subdir) / audio_path
        if (corpus_root / candidate).is_file():
            return candidate
    return None


def build_rows(
    manifest_lines: list[tuple[str, str]],
    corpus_root: Path,
    split_name: str,
    subdirs: list[str] = (),
) -> list[dict]:
    """Turn (audio_path, raw_lang_code) pairs into manifest rows.

    Resolves each raw language code, drops rows outside TARGET_LANGUAGES
    (in-corpus but not trained on), locates the audio file (see
    `resolve_audio_path`) and reads its duration. Rows with a completely
    unrecognized language code or missing/unreadable audio are skipped with
    a warning.
    """
    unknown_codes: set[str] = set()
    n_out_of_scope = 0
    n_missing_audio = 0
    n_unreadable = 0
    seen_ids: dict[str, int] = defaultdict(int)
    rows = []

    bar = ProgressBar(len(manifest_lines), prefix=f"[{split_name}]")
    for audio_path, raw_code in manifest_lines:
        bar.update(1)

        iso3 = resolve_language(raw_code)
        if iso3 is None:
            unknown_codes.add(raw_code)
            continue
        if iso3 not in TARGET_LANGUAGES:
            n_out_of_scope += 1
            continue

        resolved_path = resolve_audio_path(corpus_root, audio_path, subdirs)
        if resolved_path is None:
            n_missing_audio += 1
            continue
        audio_path = str(resolved_path)
        audio_abspath = corpus_root / resolved_path

        try:
            info = sf.info(str(audio_abspath))
            duration = info.frames / info.samplerate
        except Exception as e:
            print(f"\n[warn] unreadable audio, skipping {audio_path}: {e}")
            n_unreadable += 1
            continue

        utt_id = Path(audio_path).stem
        seen_ids[utt_id] += 1
        if seen_ids[utt_id] > 1:
            utt_id = f"{utt_id}_{seen_ids[utt_id]}"

        rows.append({
            "id": utt_id,
            "audio_path": audio_path,
            "language": iso3,
            "duration": duration,
        })
    bar.close()

    if unknown_codes:
        raise SystemExit(
            f"Unrecognized language code(s) in the raw manifest: {sorted(unknown_codes)}. "
            f"Add them to LANG_CODE_TO_ISO3 in {__file__} (mapping to the matching "
            f"ISO 639-3 code — check first whether it belongs in TARGET_LANGUAGES too)."
        )
    if n_out_of_scope:
        print(f"[info] {split_name}: {n_out_of_scope} row(s) skipped, language not in TARGET_LANGUAGES")
    if n_missing_audio:
        tried = [str(corpus_root)] + [str(corpus_root / s) for s in subdirs]
        print(f"[warn] {split_name}: {n_missing_audio} row(s) skipped, audio file not found under any of {tried}")
    if n_unreadable:
        print(f"[warn] {split_name}: {n_unreadable} row(s) skipped, audio file unreadable")

    found_languages = {r["language"] for r in rows}
    missing_languages = sorted(TARGET_LANGUAGES - found_languages)
    if missing_languages:
        print(f"[warn] {split_name}: no utterances found for language(s): {missing_languages}")

    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "audio_path", "language", "duration"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")


def split_train_dev(
    rows: list[dict], dev_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Stratified split: each language contributes its own dev_fraction share."""
    by_language: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_language[row["language"]].append(row)

    rng = random.Random(seed)
    train_rows, dev_rows = [], []
    for language, lang_rows in sorted(by_language.items()):
        lang_rows = lang_rows[:]
        rng.shuffle(lang_rows)
        n = len(lang_rows)
        n_dev = round(n * dev_fraction)
        if n > 1:
            n_dev = min(max(n_dev, 1), n - 1)  # keep at least 1 on each side
        else:
            n_dev = 0
        dev_rows.extend(lang_rows[:n_dev])
        train_rows.extend(lang_rows[n_dev:])

    return train_rows, dev_rows


def read_manifest_rows(manifest_file: Path, split_flag: str | None) -> list[tuple[str, str]]:
    """Read a raw manifest.txt/tl26_lid.txt into (audio_path, raw_lang_code) pairs.

    `split_flag`: keep only rows whose first column equals this (train
    manifest, 3 columns). Pass None for the eval manifest (2 columns, no
    flag, every row is used).
    """
    pairs = []
    n_skipped_flag = 0
    with open(manifest_file, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            fields = line.split()
            if not fields:
                continue

            if split_flag is not None:
                if len(fields) != 3:
                    print(f"[warn] {manifest_file}:{lineno}: expected 3 fields, got {len(fields)}, skipping")
                    continue
                flag, audio_path, raw_code = fields
                if flag != split_flag:
                    n_skipped_flag += 1
                    continue
            else:
                if len(fields) != 2:
                    print(f"[warn] {manifest_file}:{lineno}: expected 2 fields, got {len(fields)}, skipping")
                    continue
                audio_path, raw_code = fields

            pairs.append((audio_path, raw_code))

    if split_flag is not None and n_skipped_flag:
        print(f"[info] {manifest_file}: skipped {n_skipped_flag} row(s) with split flag != {split_flag}")

    return pairs


def cmd_train(args: argparse.Namespace) -> None:
    manifest_file = args.manifest_file or (args.corpus_root / "training_manifest.txt")
    if not manifest_file.is_file():
        raise SystemExit(f"manifest file not found: {manifest_file}")

    pairs = read_manifest_rows(manifest_file, split_flag="1")
    if not pairs:
        raise SystemExit(f"no split_flag==1 rows found in {manifest_file}")

    rows = build_rows(pairs, args.corpus_root, split_name="train_pool", subdirs=args.subdirs)
    train_rows, dev_rows = split_train_dev(rows, args.dev_fraction, args.seed)

    write_csv(train_rows, args.output_dir / "tidylang-train.csv")
    write_csv(dev_rows, args.output_dir / "tidylang-dev.csv")


def cmd_eval(args: argparse.Namespace) -> None:
    manifest_file = args.manifest_file or (args.corpus_root / "tl26_lid.txt")
    if not manifest_file.is_file():
        raise SystemExit(f"manifest file not found: {manifest_file}")

    pairs = read_manifest_rows(manifest_file, split_flag=None)
    if not pairs:
        raise SystemExit(f"no rows found in {manifest_file}")

    rows = build_rows(pairs, args.corpus_root, split_name="test", subdirs=args.subdirs)
    write_csv(rows, args.output_dir / "tidylang-test.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Build tidylang-{train,dev}.csv from TidyVoiceX_ASV")
    p_train.add_argument("--corpus-root", required=True, type=Path, help="TidyVoiceX_ASV root directory")
    p_train.add_argument("--manifest-file", type=Path, default=None, help="Default: <corpus-root>/manifest.txt")
    p_train.add_argument("--output-dir", default=Path("data/manifests"), type=Path, help="Default: data/manifests")
    p_train.add_argument("--dev-fraction", default=DEFAULT_DEV_FRACTION, type=float, help="Default: 0.1")
    p_train.add_argument("--seed", default=DEFAULT_SEED, type=int, help="Default: 42")
    p_train.add_argument(
        "--subdirs", type=parse_subdirs, default=DEFAULT_TRAIN_SUBDIRS,
        help="Comma-separated subdirectories of --corpus-root to also search for each "
             f"audio file (default: {','.join(DEFAULT_TRAIN_SUBDIRS)}; pass an empty "
             "string for a flat corpus_root/<speaker>/... layout)",
    )
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="Build tidylang-test.csv from TidyVoiceX2_ASV")
    p_eval.add_argument("--corpus-root", required=True, type=Path, help="TidyVoiceX2_ASV root directory")
    p_eval.add_argument("--manifest-file", type=Path, default=None, help="Default: <corpus-root>/tl26_lid.txt")
    p_eval.add_argument("--output-dir", default=Path("data/manifests"), type=Path, help="Default: data/manifests")
    p_eval.add_argument(
        "--subdirs", type=parse_subdirs, default=[],
        help="Comma-separated subdirectories of --corpus-root to also search for each "
             "audio file (default: none, flat corpus_root/<speaker>/... layout)",
    )
    p_eval.set_defaults(func=cmd_eval)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.corpus_root.is_dir():
        raise SystemExit(f"--corpus-root {args.corpus_root} is not a directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
