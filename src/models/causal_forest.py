"""
causal_forest.py — Heterogeneous Treatment Effects via Causal Forest (EconML).
Estimates CATE: how does the effect of wait_time on cancellation vary
across boroughs, times of day, and rider characteristics?
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (
    DATA_PROCESSED, OUTPUTS_TABLES,
    CF_N_ESTIMATORS, CF_MIN_SAMPLES_LEAF, CF_TEST_SIZE, RANDOM_STATE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_cf_data(df: pd.DataFrame) -> tuple:
    """
    Prepare arrays for causal forest.
    Returns: Y, T, X, feature_names
    """
    df = df.dropna(subset=[
        "cancelled", "wait_time_mins", "rain_intensity_mm",
        "surge_proxy", "hour_of_day", "is_weekend", "borough"
    ]).copy()

    # Encode borough
    le = LabelEncoder()
    df["borough_code"] = le.fit_transform(df["borough"].fillna("Unknown"))

    feature_cols = [
        "surge_proxy", "hour_of_day", "is_weekend", "is_holiday",
        "borough_code", "rain_intensity_mm", "wind_speed_ms"
    ]

    Y = df["cancelled"].astype(float).values
    T = df["wait_time_mins"].astype(float).values
    X = df[feature_cols].astype(float).values

    return Y, T, X, feature_cols, df


def run_causal_forest(df: pd.DataFrame = None, save: bool = True) -> dict:
    """
    Fit CausalForestDML from EconML.
    Uses rain as instrument within the forest framework.
    """
    try:
        from econml.dml import CausalForestDML
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    except ImportError:
        logger.error("EconML not installed. Run: pip install econml")
        return {}

    if df is None:
        master_path = DATA_PROCESSED / "master.parquet"
        if not master_path.exists():
            raise FileNotFoundError("master.parquet not found.")
        df = pd.read_parquet(master_path)

    logger.info(f"Fitting causal forest on {len(df):,} observations...")
    Y, T, X, feature_cols, df_clean = prepare_cf_data(df)

    # Train/test split
    idx = np.arange(len(Y))
    idx_train, idx_test = train_test_split(
        idx, test_size=CF_TEST_SIZE, random_state=RANDOM_STATE
    )

    # Causal Forest DML
    # Uses cross-fitting: estimates nuisance functions (E[Y|X] and E[T|X]) via ML,
    # then fits forest on residuals — combining Double ML with GRF
    cf = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=4),
        model_t=GradientBoostingRegressor(n_estimators=100, max_depth=4),
        n_estimators=CF_N_ESTIMATORS,
        min_samples_leaf=CF_MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        verbose=0,
    )

    logger.info("Fitting model (this may take a few minutes)...")
    cf.fit(Y[idx_train], T[idx_train], X=X[idx_train])
    logger.info("Causal forest fitted ✓")

    # CATE estimates on test set
    cate = cf.effect(X[idx_test])
    cate_interval = cf.effect_interval(X[idx_test], alpha=0.05)

    logger.info(f"Mean CATE: {cate.mean():.5f}")
    logger.info(f"CATE std: {cate.std():.5f}")
    logger.info(f"CATE range: [{cate.min():.5f}, {cate.max():.5f}]")

    # Attach CATE to test set rows
    df_test = df_clean.iloc[idx_test].copy()
    df_test["cate"] = cate
    df_test["cate_lower"] = cate_interval[0]
    df_test["cate_upper"] = cate_interval[1]

    # HTE by borough
    borough_hte = (
        df_test.groupby("borough")["cate"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_cate", "std": "std_cate", "count": "n"})
        .sort_values("mean_cate", ascending=False)
    )
    logger.info("\nHTE by Borough:\n" + borough_hte.to_string())

    # HTE by hour bucket
    df_test["hour_bucket"] = pd.cut(
        df_test["hour_of_day"],
        bins=[0, 6, 12, 18, 24],
        labels=["Late Night (0-6)", "Morning (6-12)", "Afternoon (12-18)", "Evening (18-24)"],
        right=False
    )
    hour_hte = (
        df_test.groupby("hour_bucket", observed=True)["cate"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_cate", "std": "std_cate", "count": "n"})
    )
    logger.info("\nHTE by Hour Bucket:\n" + hour_hte.to_string())

    # Feature importance (heterogeneity drivers)
    feat_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": cf.feature_importances_
    }).sort_values("importance", ascending=False)
    logger.info("\nFeature Importance (heterogeneity drivers):\n" + feat_importance.to_string())

    # Best linear projection of CATE
    blp = cf.ate(X[idx_test])
    logger.info(f"\nAverage Treatment Effect (causal forest ATE): {blp:.5f}")

    if save:
        borough_hte.to_csv(OUTPUTS_TABLES / "hte_by_borough.csv", index=False)
        hour_hte.to_csv(OUTPUTS_TABLES / "hte_by_hour.csv", index=False)
        feat_importance.to_csv(OUTPUTS_TABLES / "feature_importance.csv", index=False)
        df_test[["pu_location_id", "borough", "hour_of_day", "cate", "cate_lower", "cate_upper"]].to_parquet(
            DATA_PROCESSED / "cate_estimates.parquet", index=False
        )
        logger.info("Saved HTE tables.")

    return {
        "model": cf,
        "cate": cate,
        "df_test": df_test,
        "borough_hte": borough_hte,
        "hour_hte": hour_hte,
        "feat_importance": feat_importance,
        "ate": float(blp),
    }


if __name__ == "__main__":
    run_causal_forest()
