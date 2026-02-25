"""
ols_baseline.py — Naive OLS and OLS-with-controls as benchmarks.
Samples 200K rows for speed/memory — sufficient for coefficient comparison.
"""

import logging
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (
    DATA_PROCESSED,
    OUTPUTS_TABLES,
    TREATMENT_COL,
    OUTCOME_COL,
    RANDOM_STATE,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

SAMPLE_N = 200_000  # sufficient for stable coefficient estimates


def _sample(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) > SAMPLE_N:
        logger.info(f"Sampling {SAMPLE_N:,} rows from {len(df):,} for OLS baseline")
        return df.sample(SAMPLE_N, random_state=RANDOM_STATE)
    return df


def run_naive_ols(df: pd.DataFrame) -> dict:
    df = _sample(df)
    logger.info("Running naive OLS...")
    model = smf.ols(f"{OUTCOME_COL} ~ {TREATMENT_COL}", data=df).fit(cov_type="HC1")

    coef = model.params[TREATMENT_COL]
    se = model.bse[TREATMENT_COL]
    pval = model.pvalues[TREATMENT_COL]
    ci = model.conf_int().loc[TREATMENT_COL]
    logger.info(f"Naive OLS: β={coef:.4f}, SE={se:.4f}, p={pval:.4f}")
    return {
        "model": "Naive OLS",
        "coef": coef,
        "se": se,
        "pval": pval,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "r2": model.rsquared,
        "n": int(model.nobs),
    }


def run_ols_with_controls(df: pd.DataFrame) -> dict:
    df = _sample(df.dropna(subset=["borough", "surge_proxy"]))
    logger.info("Running OLS with controls...")

    # Within-group demean for hour + borough FE
    for col in [OUTCOME_COL, TREATMENT_COL, "surge_proxy", "is_weekend", "is_holiday"]:
        grp = df.groupby(["hour_of_day", "borough"])[col].transform("mean")
        df[f"{col}_dm"] = df[col] - grp

    model = smf.ols(
        f"{OUTCOME_COL}_dm ~ {TREATMENT_COL}_dm + surge_proxy_dm + is_weekend_dm + is_holiday_dm - 1",
        data=df,
    ).fit(cov_type="HC1")

    coef = model.params[f"{TREATMENT_COL}_dm"]
    se = model.bse[f"{TREATMENT_COL}_dm"]
    pval = model.pvalues[f"{TREATMENT_COL}_dm"]
    ci = model.conf_int().loc[f"{TREATMENT_COL}_dm"]
    logger.info(f"OLS w/ controls: β={coef:.4f}, SE={se:.4f}, p={pval:.4f}")
    return {
        "model": "OLS + Controls",
        "coef": coef,
        "se": se,
        "pval": pval,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "r2": model.rsquared,
        "n": int(model.nobs),
    }


def run_baselines(df: pd.DataFrame = None, save: bool = True) -> pd.DataFrame:
    if df is None:
        path = DATA_PROCESSED / "master.parquet"
        if not path.exists():
            raise FileNotFoundError("master.parquet not found. Run join.py first.")
        df = pd.read_parquet(path)

    results_df = pd.DataFrame([run_naive_ols(df), run_ols_with_controls(df)])
    logger.info("\n" + results_df[["model", "coef", "se", "pval"]].to_string())

    if save:
        dest = OUTPUTS_TABLES / "ols_baseline.csv"
        results_df.to_csv(dest, index=False)
        logger.info(f"Saved: {dest}")

    return results_df


if __name__ == "__main__":
    run_baselines()
