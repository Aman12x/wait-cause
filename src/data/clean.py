"""
clean.py — Clean raw TLC FHVHV parquet files.
Single responsibility: filter, validate, derive core fields.

Output: trips_clean.parquet
"""

import logging
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (
    TLC_DIR,
    DATA_PROCESSED,
    WAIT_TIME_MIN,
    WAIT_TIME_MAX,
    CANCELLATION_WINDOW,
    FARE_PER_MILE_MIN,
    FARE_PER_MILE_MAX,
    MIN_ROWS_AFTER_CLEANING,
    MIN_CANCELLATION_RATE,
    TLC_MONTHS,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_raw_trips(months: list[tuple] = None) -> pd.DataFrame:
    """
    Load FHVHV parquet files using DuckDB (memory efficient for large files).
    Returns raw dataframe with only columns we need.
    """
    months = months or TLC_MONTHS
    files = [str(TLC_DIR / f"fhvhv_tripdata_{y}-{m}.parquet") for y, m in months]

    # Filter to files that actually exist
    existing = [f for f in files if Path(f).exists()]
    if not existing:
        raise FileNotFoundError(
            f"No TLC parquet files found in {TLC_DIR}. Run download.py first."
        )
    logger.info(f"Loading {len(existing)} parquet file(s) via DuckDB...")

    file_list = ", ".join(f"'{f}'" for f in existing)
    query = f"""
        SELECT
            request_datetime,
            on_scene_datetime,
            pickup_datetime,
            dropoff_datetime,
            PULocationID,
            DOLocationID,
            base_passenger_fare,
            driver_pay,
            trip_miles,
            trip_time
        FROM read_parquet([{file_list}])
        WHERE request_datetime IS NOT NULL
    """
    con = duckdb.connect()
    df = con.execute(query).df()
    con.close()

    logger.info(f"Loaded {len(df):,} raw rows")
    return df


def derive_wait_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct wait_time_mins. For cancelled trips (null on_scene),
    use pickup_datetime as fallback, else leave null — filtered out downstream.
    """
    df = df.copy()
    df["request_datetime"] = pd.to_datetime(df["request_datetime"])
    df["on_scene_datetime"] = pd.to_datetime(df["on_scene_datetime"])
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    # For completed trips: on_scene - request
    df["wait_time_mins"] = (
        df["on_scene_datetime"] - df["request_datetime"]
    ).dt.total_seconds() / 60
    # For cancelled trips: pickup - request as proxy (will be filtered out anyway
    # since cancelled=1 rows get excluded from the IV regression, but we keep them
    # in the dataset for descriptive stats)
    mask = df["wait_time_mins"].isna() & df["pickup_datetime"].notna()
    df.loc[mask, "wait_time_mins"] = (
        df.loc[mask, "pickup_datetime"] - df.loc[mask, "request_datetime"]
    ).dt.total_seconds() / 60
    return df


def derive_cancellation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define cancellation: on_scene_datetime is null (driver never arrived/dispatched).
    TLC FHVHV data doesn't reliably record post-dispatch cancellations — those trips
    simply don't appear. Null on_scene is the cleanest signal available.
    """
    df = df.copy()
    df["cancelled"] = df["on_scene_datetime"].isna().astype(int)
    return df


def derive_surge_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Surge proxy = fare per mile. High values indicate surge pricing."""
    df = df.copy()
    df["surge_proxy"] = df["base_passenger_fare"] / df["trip_miles"].replace(0, np.nan)
    return df


def derive_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day-of-week, weekend, month features."""
    df = df.copy()
    dt = df["request_datetime"]
    df["hour_of_day"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek  # 0=Monday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    df["date"] = dt.dt.date
    df["date_hour"] = dt.dt.floor("h")  # for weather join
    return df


def apply_cleaning_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all data quality filters. Log row counts at each step."""
    n_start = len(df)
    logger.info(f"Starting rows: {n_start:,}")

    # Must have wait time
    df = df[df["wait_time_mins"].notna()]
    logger.info(
        f"After removing null wait_time: {len(df):,} ({n_start - len(df):,} dropped)"
    )

    # Wait time bounds
    df = df[df["wait_time_mins"].between(WAIT_TIME_MIN, WAIT_TIME_MAX)]
    logger.info(
        f"After wait_time bounds [{WAIT_TIME_MIN}, {WAIT_TIME_MAX}]: {len(df):,}"
    )

    # Valid zone IDs
    df = df[df["PULocationID"].between(1, 263)]
    logger.info(f"After zone filter: {len(df):,}")

    # Surge proxy bounds
    df = df[df["surge_proxy"].between(FARE_PER_MILE_MIN, FARE_PER_MILE_MAX)]
    logger.info(f"After surge proxy filter: {len(df):,}")

    # Must have pickup location
    df = df[df["PULocationID"].notna()]

    n_final = len(df)
    pct_kept = 100 * n_final / n_start
    logger.info(f"Final rows: {n_final:,} ({pct_kept:.1f}% of original)")
    return df


def validate_output(df: pd.DataFrame) -> None:
    """Assertions on cleaned data. Raises ValueError on failure."""
    logger.info("Running validation assertions...")

    assert (
        df["wait_time_mins"].between(WAIT_TIME_MIN, WAIT_TIME_MAX).all()
    ), "wait_time_mins out of bounds"
    assert (
        df["cancelled"].isin([0, 1]).all()
    ), "cancelled column has values other than 0/1"
    assert df["PULocationID"].between(1, 263).all(), "PULocationID out of valid range"
    assert (
        df["surge_proxy"].between(FARE_PER_MILE_MIN, FARE_PER_MILE_MAX).all()
    ), "surge_proxy out of bounds"
    assert (
        len(df) >= MIN_ROWS_AFTER_CLEANING
    ), f"Too few rows after cleaning: {len(df):,} (min: {MIN_ROWS_AFTER_CLEANING:,})"

    cancel_rate = df["cancelled"].mean()
    assert (
        cancel_rate >= MIN_CANCELLATION_RATE
    ), f"Cancellation rate too low: {cancel_rate:.4f} (min: {MIN_CANCELLATION_RATE})"

    null_rates = df.isnull().mean()
    high_null = null_rates[null_rates > 0.1]
    if len(high_null) > 0:
        logger.warning(f"High null rate columns: {high_null.to_dict()}")

    logger.info(f"Validation passed. Cancellation rate: {cancel_rate:.3f}")


def run_cleaning(months: list[tuple] = None, save: bool = True) -> pd.DataFrame:
    """Full cleaning pipeline. Returns cleaned DataFrame."""
    df = load_raw_trips(months)
    df = derive_wait_time(df)
    df = derive_cancellation(df)
    df = derive_surge_proxy(df)
    df = derive_time_features(df)
    df = apply_cleaning_filters(df)
    validate_output(df)

    # Rename for consistency
    df = df.rename(columns={"PULocationID": "pu_location_id"})

    # Select final columns
    keep_cols = [
        "request_datetime",
        "date_hour",
        "date",
        "pu_location_id",
        "wait_time_mins",
        "cancelled",
        "surge_proxy",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "month",
        "year",
    ]
    df = df[keep_cols]

    if save:
        dest = DATA_PROCESSED / "trips_clean.parquet"
        df.to_parquet(dest, index=False)
        logger.info(f"Saved cleaned trips: {dest}")

    return df


if __name__ == "__main__":
    run_cleaning()
