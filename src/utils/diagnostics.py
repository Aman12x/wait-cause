"""
diagnostics.py — Reusable diagnostic utilities for IV and model validation.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def print_iv_diagnostics(summary: dict) -> None:
    """Pretty-print IV diagnostic table."""
    print("\n" + "="*55)
    print("IV DIAGNOSTIC SUMMARY")
    print("="*55)

    checks = [
        ("First Stage F-stat", summary["first_stage_f"],
         "> 10", summary["first_stage_f"] > 10),
        ("Endogenous treatment?", summary["endogenous"],
         "True (Hausman p<0.05)", summary["endogenous"]),
        ("Placebo test passed?", summary["placebo_passed"],
         "True", summary["placebo_passed"]),
        ("2SLS p-value", summary["iv_2sls_pval"],
         "< 0.05", summary["iv_2sls_pval"] < 0.05),
    ]

    for name, value, threshold, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status}  {name:<30} {str(value):<12}  (threshold: {threshold})")

    print("="*55)
    print(f"  LATE estimate:  {summary['iv_2sls_coef']:.5f}")
    print(f"  95% CI:         [{summary['iv_2sls_ci_low']:.5f}, {summary['iv_2sls_ci_high']:.5f}]")
    print(f"  N:              {summary['n']:,}")
    print("="*55 + "\n")


def check_overlap(df: pd.DataFrame, treatment_col: str, feature_cols: list) -> pd.DataFrame:
    """
    Check covariate overlap between high-wait and low-wait groups.
    Flags features with poor overlap (potential external validity issues).
    """
    median_wait = df[treatment_col].median()
    high = df[df[treatment_col] >= median_wait]
    low = df[df[treatment_col] < median_wait]

    results = []
    for col in feature_cols:
        if df[col].dtype in [float, int]:
            diff = high[col].mean() - low[col].mean()
            pooled_std = df[col].std()
            smd = diff / pooled_std if pooled_std > 0 else 0  # standardized mean diff
            results.append({
                "feature": col,
                "mean_high": high[col].mean(),
                "mean_low": low[col].mean(),
                "smd": abs(smd),
                "balance_ok": abs(smd) < 0.1
            })

    return pd.DataFrame(results).sort_values("smd", ascending=False)


def describe_instrument(df: pd.DataFrame, instrument_col: str = "rain_intensity_mm") -> None:
    """Print descriptive stats for the instrument."""
    col = df[instrument_col]
    print(f"\nInstrument ({instrument_col}) Summary:")
    print(f"  Mean:        {col.mean():.3f}")
    print(f"  Std:         {col.std():.3f}")
    print(f"  % Zero:      {(col == 0).mean() * 100:.1f}%")
    print(f"  % Rain > 0.5mm: {(col > 0.5).mean() * 100:.1f}%")
    print(f"  Null rate:   {col.isna().mean() * 100:.2f}%\n")
