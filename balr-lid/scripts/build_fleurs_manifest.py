#!/usr/bin/env python3
"""Build balr-lid manifest CSVs from a raw FLEURS-style corpus layout.

Expects a single root directory, one subdirectory per language locale, each
holding its own tsv metadata and its own audio subtree:

  corpus_root/
    ha_ng/
      train.tsv
      dev.tsv
      test.tsv
      audio/
        train/<filename>.wav
        dev/<filename>.wav
        test/<filename>.wav
    fr_fr/
      train.tsv
      ...
      audio/
        train/<filename>.wav
        ...

Each *.tsv is the raw FLEURS per-utterance metadata, tab-separated, no
header, 7 columns:
    id  filename  raw_transcription  normalized_transcription  phonemic  num_samples  gender

Writes data/manifests/fleurs-{train,dev,test}.csv in the format
`balr_lid.data.manifest.load_manifest` expects (id,audio_path,language,
duration) — no changes to the dataset/dataloader code are needed, only this
manifest. `duration` is derived from `num_samples / sample_rate` (default
16kHz) rather than re-reading every audio file, since FLEURS metadata already
carries the exact frame count.

`audio_path` in the generated manifest is written relative to `corpus_root`
(e.g. `ha_ng/audio/train/<filename>.wav`), so at training time pass
`data.audio_root=<corpus_root>`.

Usage:
    python scripts/build_fleurs_manifest.py \
        --corpus-root /corpora/fleurs \
        --output-dir data/manifests
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import soundfile as sf

SPLITS = ("train", "dev", "test")

# Some rows contain an unescaped literal newline inside a transcription
# field, which merges with following lines into one oversized field and
# trips csv's default 128KB-per-field limit; raise it well above anything
# a real utterance transcription could reach.
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


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

# google/fleurs locale directory name -> ISO 639-3 code used in
# configs/data/fleurs.yaml's `language_list`. Keys/values mirror that file
# 1:1 (102 languages); extend this dict if your corpus uses different
# locale directory names.
LOCALE_TO_ISO3 = {
    "af_za": "afr", "am_et": "amh", "ar_eg": "ara", "as_in": "asm",
    "ast_es": "ast", "az_az": "aze", "be_by": "bel", "bn_in": "ben",
    "bs_ba": "bos", "bg_bg": "bul", "ca_es": "cat", "ceb_ph": "ceb",
    "cs_cz": "ces", "ckb_iq": "ckb", "cmn_hans_cn": "cmn", "cy_gb": "cym",
    "da_dk": "dan", "de_de": "deu", "el_gr": "ell", "en_us": "eng",
    "et_ee": "est", "fa_ir": "fas", "fil_ph": "fil", "fi_fi": "fin",
    "fr_fr": "fra", "ff_sn": "ful", "ga_ie": "gle", "gl_es": "glg",
    "gu_in": "guj", "ha_ng": "hau", "he_il": "heb", "hi_in": "hin",
    "hr_hr": "hrv", "hu_hu": "hun", "hy_am": "hye", "ig_ng": "ibo",
    "id_id": "ind", "is_is": "isl", "it_it": "ita", "jv_id": "jav",
    "ja_jp": "jpn", "kam_ke": "kam", "kn_in": "kan", "ka_ge": "kat",
    "kk_kz": "kaz", "kea_cv": "kea", "km_kh": "khm", "ky_kg": "kir",
    "ko_kr": "kor", "lo_la": "lao", "lv_lv": "lav", "ln_cd": "lin",
    "lt_lt": "lit", "lb_lu": "ltz", "lg_ug": "lug", "luo_ke": "luo",
    "ml_in": "mal", "mr_in": "mar", "mk_mk": "mkd", "mt_mt": "mlt",
    "mn_mn": "mon", "mi_nz": "mri", "ms_my": "msa", "my_mm": "mya",
    "ne_np": "nep", "nl_nl": "nld", "nb_no": "nob", "nso_za": "nso",
    "ny_mw": "nya", "oc_fr": "oci", "or_in": "ori", "om_et": "orm",
    "pa_in": "pan", "pl_pl": "pol", "pt_br": "por", "ps_af": "pus",
    "ro_ro": "ron", "ru_ru": "rus", "sk_sk": "slk", "sl_si": "slv",
    "sn_zw": "sna", "sd_in": "snd", "so_so": "som", "es_419": "spa",
    "sr_rs": "srp", "sw_ke": "swa", "sv_se": "swe", "ta_in": "tam",
    "te_in": "tel", "tg_tj": "tgk", "th_th": "tha", "tr_tr": "tur",
    "uk_ua": "ukr", "umb_ao": "umb", "ur_pk": "urd", "uz_uz": "uzb",
    "vi_vn": "vie", "wo_sn": "wol", "xh_za": "xho", "yo_ng": "yor",
    "yue_hant_hk": "yue", "zu_za": "zul",
}

DEFAULT_SAMPLE_RATE = 16000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--corpus-root", required=True, type=Path,
        help="Root directory with one subdirectory per language locale (e.g. "
             "ha_ng, fr_fr), each containing {train,dev,test}.tsv and an "
             "audio/{train,dev,test}/ subtree",
    )
    p.add_argument(
        "--output-dir", default=Path("data/manifests"), type=Path,
        help="Where to write fleurs-{train,dev,test}.csv (default: data/manifests)",
    )
    p.add_argument(
        "--sample-rate", default=DEFAULT_SAMPLE_RATE, type=int,
        help="Sample rate used to convert the tsv's num_samples column to seconds "
             "(default: 16000)",
    )
    p.add_argument(
        "--skip-audio-check", action="store_true",
        help="Skip checking that every referenced audio file exists on disk "
             "(faster on very large corpora, but typos/missing files won't be caught here)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.corpus_root.is_dir():
        raise SystemExit(f"--corpus-root {args.corpus_root} is not a directory")

    lang_dirs = sorted(
        d for d in args.corpus_root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not lang_dirs:
        raise SystemExit(f"No language subdirectories found under {args.corpus_root}")

    unknown = [d.name for d in lang_dirs if d.name not in LOCALE_TO_ISO3]
    if unknown:
        raise SystemExit(
            f"Unrecognized language directory name(s): {unknown}. "
            f"Add them to LOCALE_TO_ISO3 in {__file__} (mapping to the ISO 639-3 "
            f"code used in configs/data/fleurs.yaml)."
        )

    # Cross-check against the full 102-language list: a locale present only as
    # a `.tar.gz` archive (not yet extracted) has no directory at all, and
    # would otherwise be silently skipped without any warning.
    found_locales = {d.name for d in lang_dirs}
    missing_locales = sorted(set(LOCALE_TO_ISO3) - found_locales)
    if missing_locales:
        for locale in missing_locales:
            archive = args.corpus_root / f"{locale}.tar.gz"
            hint = " (archive found but not extracted)" if archive.exists() else " (no directory and no .tar.gz found)"
            print(f"[warn] language '{locale}' ({LOCALE_TO_ISO3[locale]}) has no directory under {args.corpus_root}{hint}")
        print(
            f"[warn] {len(missing_locales)}/102 expected FLEURS language "
            f"director{'y is' if len(missing_locales) == 1 else 'ies are'} missing "
            f"— extract their .tar.gz archives first if you want them included "
            f"(e.g. `tar xzf {missing_locales[0]}.tar.gz`)."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        split_tsvs = [
            (lang_dir, lang_dir / f"{split}.tsv")
            for lang_dir in lang_dirs
            if (lang_dir / f"{split}.tsv").exists()
        ]
        missing_splits = set(lang_dirs) - {ld for ld, _ in split_tsvs}
        for lang_dir in sorted(missing_splits, key=lambda d: d.name):
            print(f"[warn] {lang_dir / f'{split}.tsv'} not found, skipping {lang_dir.name}/{split}")

        if not split_tsvs:
            print(f"[warn] no {split}.tsv found for any language, writing empty fleurs-{split}.csv")

        total_lines = sum(sum(1 for _ in open(tsv_path, encoding="utf-8")) for _, tsv_path in split_tsvs)
        bar = ProgressBar(total_lines, prefix=f"[{split}]")

        rows = []
        n_missing_audio = 0
        n_recovered = 0
        n_unrecoverable = 0
        for lang_dir, tsv_path in split_tsvs:
            iso3 = LOCALE_TO_ISO3[lang_dir.name]

            with open(tsv_path, newline="", encoding="utf-8") as f:
                # QUOTE_NONE: this is a raw tab-separated file, not a quoted CSV —
                # a stray `"` in a transcription must stay a literal character,
                # not trigger csv's quoted-field parsing (which otherwise
                # swallows tabs/newlines across multiple physical lines into a
                # single merged row, undercounting rows against total_lines).
                for lineno, line in enumerate(
                    csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE), start=1
                ):
                    bar.update(1)
                    if not line:
                        continue
                    # id and filename are always the first two columns; num_samples
                    # and gender are always the last two. The transcription columns
                    # in between can occasionally split unexpectedly (e.g. a raw
                    # literal newline inside a sentence), so anchor on both ends
                    # instead of fixed positional indices.
                    if len(line) < 2:
                        print(f"\n[warn] {tsv_path}:{lineno}: malformed row (only {len(line)} field(s)), no filename, skipping")
                        n_unrecoverable += 1
                        continue

                    filename = line[1]
                    # relative to --corpus-root, matches data.audio_root at train time
                    audio_path = f"{lang_dir.name}/audio/{split}/{filename}"
                    audio_abspath = args.corpus_root / audio_path

                    num_samples = None
                    if len(line) >= 4:
                        try:
                            num_samples = int(line[-2])
                        except ValueError:
                            num_samples = None

                    if num_samples is not None:
                        duration = num_samples / args.sample_rate
                        if not args.skip_audio_check and not audio_abspath.is_file():
                            n_missing_audio += 1
                            continue
                    else:
                        # tsv row is corrupted (e.g. an unescaped literal newline
                        # merged two columns together) — recover the utterance by
                        # reading the real duration off the audio file instead of
                        # dropping it, as long as that audio file is itself fine.
                        try:
                            info = sf.info(str(audio_abspath))
                            duration = info.frames / info.samplerate
                        except Exception as e:
                            print(f"\n[warn] {tsv_path}:{lineno}: corrupted metadata row AND audio unreadable ({audio_path}): {e}, skipping")
                            n_unrecoverable += 1
                            continue
                        n_recovered += 1

                    rows.append({
                        "id": Path(filename).stem,
                        "audio_path": audio_path,
                        "language": iso3,
                        "duration": duration,
                    })

        bar.close()

        if n_missing_audio:
            print(f"[warn] {split}: {n_missing_audio} row(s) skipped, audio file not found under {args.corpus_root}")
        if n_recovered:
            print(f"[info] {split}: {n_recovered} row(s) had a corrupted metadata row but were recovered from the audio file directly")
        if n_unrecoverable:
            print(f"[warn] {split}: {n_unrecoverable} row(s) skipped, metadata corrupted and no filename or no readable audio to recover from")

        out_path = args.output_dir / f"fleurs-{split}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "audio_path", "language", "duration"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
