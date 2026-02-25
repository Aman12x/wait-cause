"""
iv_2sls.py — Two-stage least squares (2SLS) IV estimation.
Instrument: rain_intensity_mm (NOAA hourly precipitation)
Treatment: wait_time_mins
Outcome: cancelled

Includes all required diagnostics:
  - First stage F-statistic (weak instrument test)
  - Hausman test for endogeneity
  - Reduced form
  - Wald estimator
  - Robustness: multiple instruments
"""

import logging
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.gmm import IV2SLS
from linearmodels.iv import IV2SLS as LMIV2SLS
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (
    DATA_PROCESSED, OUTPUTS_TABLES,
    INSTRUMENT_COLS, TREATMENT_COL, OUTCOME_COL,
    WEAK_INSTRUMENT_F_THRESHOLD
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_first_stage(df: pd.DataFrame) -> dict:
    """
    First stage: wait_time_mins ~ rain_intensity_mm + controls
    Key diagnostic: F-statistic on excluded instrument must be > 10.
    """
    logger.info("── First Stage ──")
    formula = (
        "wait_time_mins ~ rain_intensity_mm + wind_speed_ms + "
        "surge_proxy + is_weekend + is_holiday + "
        "C(hour_of_day) + C(borough)"
    )
    model = smf.ols(formula, data=df.dropna()).fit(cov_type="HC3")

    # F-stat on EXCLUDED instrument only (rain_intensity_mm)
    instrument_coef = model.params.get("rain_intensity_mm", np.nan)
    instrument_tstat = model.tvalues.get("rain_intensity_mm", np.nan)
    f_stat_excluded = instrument_tstat ** 2  # Approximate F for single instrument

    logger.info(f"First stage: rain → wait_time coef = {instrument_coef:.4f}")
    logger.info(f"First stage F-stat (excluded instrument): {f_stat_excluded:.2f}")

    if f_stat_excluded < WEAK_INSTRUMENT_F_THRESHOLD:
        logger.warning(
            f"WEAK INSTRUMENT: F={f_stat_excluded:.2f} < {WEAK_INSTRUMENT_F_THRESHOLD}. "
            "IV estimates unreliable."
        )
    else:
        logger.info(f"Instrument relevance confirmed: F={f_stat_excluded:.2f} ✓")

    # Save fitted values for manual 2SLS
    df = df.copy()
    df["wait_time_hat"] = model.predict(df)

    return {
        "coef_rain": instrument_coef,
        "f_stat": f_stat_excluded,
        "r2": model.rsquared,
        "n": int(model.nobs),
        "df_with_hat": df,
        "model": model,
    }


def run_reduced_form(df: pd.DataFrame) -> dict:
    """
    Reduced form: cancelled ~ rain_intensity_mm + controls
    Direct effect of instrument on outcome (sanity check).
    Should be significant — if not, something is wrong.
    """
    logger.info("── Reduced Form ──")
    formula = (
        "cancelled ~ rain_intensity_mm + wind_speed_ms + "
        "surge_proxy + is_weekend + is_holiday + "
        "C(hour_of_day) + C(borough)"
    )
    model = smf.ols(formula, data=df.dropna()).fit(cov_type="HC3")

    coef = model.params.get("rain_intensity_mm", np.nan)
    pval = model.pvalues.get("rain_intensity_mm", np.nan)
    logger.info(f"Reduced form: rain → cancel coef = {coef:.5f}, p={pval:.4f}")

    return {
        "coef_rain_on_cancel": coef,
        "pval": pval,
        "model": model,
    }


def compute_wald_estimator(first_stage: dict, reduced_form: dict) -> float:
    """
    Wald estimator: LATE = (reduced form) / (first stage)
    This is the IV estimate for a single binary instrument.
    """
    late = reduced_form["coef_rain_on_cancel"] / first_stage["coef_rain"]
    logger.info(f"Wald LATE estimate: {late:.5f}")
    logger.info(
        f"  = {reduced_form['coef_rain_on_cancel']:.5f} (reduced form) "
        f"/ {first_stage['coef_rain']:.4f} (first stage)"
    )
    return late


def run_2sls(df: pd.DataFrame) -> dict:
    """
    Full 2SLS using linearmodels.IV2SLS.
    Provides proper heteroskedasticity-robust SEs.
    """
    logger.info("── 2SLS (linearmodels) ──")
    df_clean = df.dropna(subset=[
        "cancelled", "wait_time_mins", "rain_intensity_mm",
        "wind_speed_ms", "surge_proxy", "borough_encoded",
        "hour_of_day", "is_weekend", "is_holiday"
    ]).copy()

    # Create hour dummies manually (linearmodels needs explicit dummies)
    hour_dummies = pd.get_dummies(df_clean["hour_of_day"], prefix="hour", drop_first=True)
    borough_dummies = pd.get_dummies(df_clean["borough"], prefix="borough", drop_first=True)
    df_model = pd.concat([df_clean, hour_dummies, borough_dummies], axis=1)

    # Build column lists
    exog_cols = (
        ["surge_proxy", "is_weekend", "is_holiday"] +
        list(hour_dummies.columns) +
        list(borough_dummies.columns)
    )
    instrument_cols = ["rain_intensity_mm", "wind_speed_ms"]

    # linearmodels IV2SLS requires: dependent, exog, endog, instruments
    import statsmodels.api as sm
    dependent = df_model["cancelled"].astype(float)
    exog = sm.add_constant(df_model[exog_cols].astype(float))
    endog = df_model[["wait_time_mins"]].astype(float)
    instruments = df_model[instrument_cols].astype(float)

    model = LMIV2SLS(dependent, exog, endog, instruments).fit(cov_type="robust")

    coef = float(model.params["wait_time_mins"])
    se = float(model.std_errors["wait_time_mins"])
    pval = float(model.pvalues["wait_time_mins"])
    ci = model.conf_int().loc["wait_time_mins"]

    logger.info(f"2SLS LATE: β={coef:.5f}, SE={se:.5f}, p={pval:.4f}")
    logger.info(f"95% CI: [{float(ci.iloc[0]):.5f}, {float(ci.iloc[1]):.5f}]")

    return {
        "model": "IV 2SLS",
        "coef": coef,
        "se": se,
        "pval": pval,
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "n": int(model.nobs),
        "fitted_model": model,
    }


def run_hausman_test(df: pd.DataFrame, first_stage_result: dict) -> dict:
    """
    Hausman test for endogeneity of wait_time_mins.
    If p < 0.05, treatment is endogenous → IV is necessary.
    Method: augmented regression (include first stage residuals in OLS).
    """
    logger.info("── Hausman Endogeneity Test ──")
    df = df.copy()
    df["v_hat"] = df["wait_time_mins"] - first_stage_result["df_with_hat"]["wait_time_hat"]

    formula = (
        "cancelled ~ wait_time_mins + v_hat + "
        "surge_proxy + is_weekend + is_holiday + "
        "C(hour_of_day) + C(borough)"
    )
    model = smf.ols(formula, data=df.dropna()).fit(cov_type="HC3")
    coef_vhat = model.params.get("v_hat", np.nan)
    pval_vhat = model.pvalues.get("v_hat", np.nan)

    endogenous = pval_vhat < 0.05
    logger.info(
        f"Hausman test: v_hat coef={coef_vhat:.5f}, p={pval_vhat:.4f} → "
        f"{'ENDOGENOUS ✓ (IV justified)' if endogenous else 'Cannot reject exogeneity'}"
    )

    return {
        "coef_vhat": coef_vhat,
        "pval_vhat": pval_vhat,
        "endogenous": endogenous,
    }


def run_placebo_instrument_test(df: pd.DataFrame) -> dict:
    """
    Placebo test: use next-day rain as instrument.
    Should have NO effect on today's cancellations (exclusion restriction check).
    """
    logger.info("── Placebo Instrument Test ──")
    df = df.copy().sort_values("date_hour")

    # Shift rain by 24 hours
    df["rain_placebo"] = df["rain_intensity_mm"].shift(24)
    df_test = df.dropna(subset=["rain_placebo"])

    formula = "cancelled ~ rain_placebo + surge_proxy + is_weekend + C(hour_of_day) + C(borough)"
    model = smf.ols(formula, data=df_test.dropna()).fit(cov_type="HC3")

    coef = model.params.get("rain_placebo", np.nan)
    pval = model.pvalues.get("rain_placebo", np.nan)

    passed = pval > 0.05
    logger.info(
        f"Placebo test (next-day rain): coef={coef:.5f}, p={pval:.4f} → "
        f"{'PASSED ✓ (no future rain effect)' if passed else 'FAILED (suspicious — check data)'}"
    )

    return {"coef_placebo": coef, "pval_placebo": pval, "passed": passed}


def run_iv_analysis(df: pd.DataFrame = None, save: bool = True) -> dict:
    """Full IV analysis pipeline. Returns all results."""
    if df is None:
        master_path = DATA_PROCESSED / "master.parquet"
        if not master_path.exists():
            raise FileNotFoundError("master.parquet not found. Run join.py first.")
        df = pd.read_parquet(master_path)

    logger.info(f"Running IV analysis on {len(df):,} observations...")

    first_stage = run_first_stage(df)
    reduced_form = run_reduced_form(df)
    wald = compute_wald_estimator(first_stage, reduced_form)
    iv_result = run_2sls(df)
    hausman = run_hausman_test(df, first_stage)
    placebo = run_placebo_instrument_test(df)

    summary = {
        "first_stage_f": first_stage["f_stat"],
        "first_stage_coef_rain": first_stage["coef_rain"],
        "reduced_form_coef": reduced_form["coef_rain_on_cancel"],
        "wald_late": wald,
        "iv_2sls_coef": iv_result["coef"],
        "iv_2sls_se": iv_result["se"],
        "iv_2sls_pval": iv_result["pval"],
        "iv_2sls_ci_low": iv_result["ci_low"],
        "iv_2sls_ci_high": iv_result["ci_high"],
        "hausman_pval": hausman["pval_vhat"],
        "endogenous": hausman["endogenous"],
        "placebo_passed": placebo["passed"],
        "n": iv_result["n"],
    }

    logger.info("\n── IV Summary ──")
    for k, v in summary.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.5f}")
        else:
            logger.info(f"  {k}: {v}")

    if save:
        dest = OUTPUTS_TABLES / "iv_results.csv"
        pd.DataFrame([summary]).to_csv(dest, index=False)
        logger.info(f"Saved IV results: {dest}")

    return {
        "summary": summary,
        "first_stage": first_stage,
        "reduced_form": reduced_form,
        "iv_result": iv_result,
        "hausman": hausman,
        "placebo": placebo,
    }


if __name__ == "__main__":
    run_iv_analysis()
