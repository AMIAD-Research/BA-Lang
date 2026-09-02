#!/usr/bin/env python3
"""Compute micro/macro accuracy from a balr_lid.inference predictions CSV.

Expects the `reference_language`/`predicted_language` columns.


Usage:
    python scripts/compute_accuracy.py --predictions predictions.csv
    python scripts/compute_accuracy.py --predictions predictions.csv --per-language
    python scripts/compute_accuracy.py --predictions predictions.csv --confidence-intervals
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def compute_accuracies(df: pd.DataFrame) -> tuple[float, float, pd.Series]:
    """Returns (micro_accuracy, macro_accuracy, per_language_accuracy)."""
    correct = df["reference_language"] == df["predicted_language"]

    micro_accuracy = correct.mean()
    per_language_accuracy = correct.groupby(df["reference_language"]).mean().sort_index()
    macro_accuracy = per_language_accuracy.mean()

    return micro_accuracy, macro_accuracy, per_language_accuracy


def bootstrap_accuracy_cis(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Bootstrap percentile confidence intervals for micro/macro accuracy.

    Resampling is stratified by `reference_language`: each language is
    resampled with replacement to its own original count, independently of
    the other languages. This guarantees every bootstrap replicate contains
    every language present. Micro accuracy is computed from the
    same stratified replicates for consistency.

    Returns {"micro": (lo, hi), "macro": (lo, hi)}.
    """
    rng = np.random.default_rng(seed)
    correct = (df["reference_language"] == df["predicted_language"]).to_numpy()
    languages = df["reference_language"].to_numpy()
    correct_by_language = [correct[languages == lang] for lang in np.unique(languages)]

    micro_correct = np.zeros(n_bootstrap)
    macro_accuracy = np.zeros(n_bootstrap)
    total_n = 0
    for lang_correct in correct_by_language:
        n = len(lang_correct)
        resampled = rng.choice(lang_correct, size=(n_bootstrap, n), replace=True)
        micro_correct += resampled.sum(axis=1)
        macro_accuracy += resampled.mean(axis=1)
        total_n += n
    micro_accuracy = micro_correct / total_n
    macro_accuracy /= len(correct_by_language)

    alpha = 1.0 - confidence_level
    percentiles = [100 * alpha / 2, 100 * (1.0 - alpha / 2)]
    return {
        "micro": tuple(np.percentile(micro_accuracy, percentiles)),
        "macro": tuple(np.percentile(macro_accuracy, percentiles)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, help="predictions.csv written by balr_lid.inference")
    parser.add_argument(
        "--per-language",
        action="store_true",
        help="Also print accuracy broken down by reference_language",
    )
    parser.add_argument(
        "--confidence-intervals",
        action="store_true",
        help="Also report micro/macro accuracy with bootstrap confidence intervals. Macro "
             "accuracy is resampled stratified by reference_language so every language in "
             "the test set is guaranteed to appear in every bootstrap replicate "
             "(see bootstrap_accuracy_cis).",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=1000,
        help="Number of bootstrap resamples (only used with --confidence-intervals).",
    )
    parser.add_argument(
        "--confidence-level", type=float, default=0.95,
        help="Confidence level for the bootstrap interval (only used with --confidence-intervals).",
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=0,
        help="RNG seed for bootstrap resampling (only used with --confidence-intervals).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.predictions)

    missing = [c for c in ("reference_language", "predicted_language") if c not in df.columns]
    if missing:
        raise SystemExit(f"{args.predictions} is missing required column(s): {missing}")

    micro_accuracy, macro_accuracy, per_language_accuracy = compute_accuracies(df)

    print(f"rows:            {len(df)}")
    print(f"languages:       {per_language_accuracy.shape[0]}")

    if args.confidence_intervals:
        cis = bootstrap_accuracy_cis(
            df,
            n_bootstrap=args.n_bootstrap,
            confidence_level=args.confidence_level,
            seed=args.bootstrap_seed,
        )
        ci_pct = args.confidence_level * 100
        micro_lo, micro_hi = cis["micro"]
        macro_lo, macro_hi = cis["macro"]
        print(f"micro accuracy:  {micro_accuracy:.4f}  ({ci_pct:.0f}% CI: [{micro_lo:.4f}, {micro_hi:.4f}])")
        print(f"macro accuracy:  {macro_accuracy:.4f}  ({ci_pct:.0f}% CI: [{macro_lo:.4f}, {macro_hi:.4f}])")
    else:
        print(f"micro accuracy:  {micro_accuracy:.4f}")
        print(f"macro accuracy:  {macro_accuracy:.4f}")

    if args.per_language:
        print("\nper-language accuracy:")
        counts = df.groupby("reference_language").size()
        for lang, acc in per_language_accuracy.items():
            print(f"  {lang:>4s}  {acc:.4f}  (n={counts[lang]})")


if __name__ == "__main__":
    main()
