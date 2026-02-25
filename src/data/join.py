"""
join.py — Join cleaned trips with NOAA weather and TLC zone metadata.
Single responsibility: merge datasets, produce master.parquet.

Strategy:
  1. Load zone shapefile → compute centroids → assign nearest NOAA station
  2. Load all NOAA CSVs → combine into hourly weather per station
  3. Join trips → zones → weather on (pu_location_id, date_hour)
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (
    DATA_PROCESSED, NOAA_DIR, ZONES_DIR,
    NOAA_STATIONS, MAX_NULL_RATE_INSTRUMENT, TLC_MONTHS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Zone Metadata ──────────────────────────────────────────────────────────

def load_zone_lookup() -> pd.DataFrame:
    """Load TLC zone → borough mapping."""
    lookup_path = ZONES_DIR / "taxi_zone_lookup.csv"
    if not lookup_path.exists():
        logger.warning("Zone lookup not found, using synthetic borough mapping.")
        return _synthetic_zone_lookup()

    df = pd.read_csv(lookup_path)
    df.columns = df.columns.str.lower().str.strip()
    df = df.rename(columns={"locationid": "pu_location_id", "zone": "zone_name"})
    logger.info(f"Loaded zone lookup: {len(df)} zones")
    return df[["pu_location_id", "borough", "zone_name"]]


def _synthetic_zone_lookup() -> pd.DataFrame:
    """Fallback: assign boroughs synthetically based on zone ID ranges."""
    zones = list(range(1, 264))
    borough_map = {}
    for z in zones:
        if z <= 69:
            borough_map[z] = "Bronx"
        elif z <= 131:
            borough_map[z] = "Brooklyn"
        elif z <= 196:
            borough_map[z] = "Manhattan"
        elif z <= 244:
            borough_map[z] = "Queens"
        else:
            borough_map[z] = "Staten Island"

    df = pd.DataFrame({
        "pu_location_id": zones,
        "borough": [borough_map[z] for z in zones],
        "zone_name": [f"Zone_{z}" for z in zones],
    })
    return df


def assign_nearest_station(zone_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each zone to its nearest NOAA weather station based on approximate centroids.
    Uses Manhattan-zone coordinate approximations.
    """
    # Approximate centroids for NYC boroughs
    borough_centroids = {
        "Manhattan":    (40.7831, -73.9712),
        "Brooklyn":     (40.6501, -73.9496),
        "Queens":       (40.7282, -73.7949),
        "Bronx":        (40.8448, -73.8648),
        "Staten Island": (40.5795, -74.1502),
        "EWR":          (40.6895, -74.1745),
    }

    stations = {name: (info["lat"], info["lon"]) for name, info in NOAA_STATIONS.items()}

    def nearest(borough):
        centroid = borough_centroids.get(borough, (40.7128, -74.0060))
        dists = {
            sname: ((centroid[0] - slat)**2 + (centroid[1] - slon)**2)**0.5
            for sname, (slat, slon) in stations.items()
        }
        return min(dists, key=dists.get)

    zone_df = zone_df.copy()
    zone_df["nearest_station"] = zone_df["borough"].map(nearest)
    return zone_df


# ── Weather Data ───────────────────────────────────────────────────────────

def load_weather(months: list[tuple] = None) -> pd.DataFrame:
    """Load and combine all NOAA weather CSVs into a single hourly DataFrame."""
    months = months or TLC_MONTHS
    dfs = []

    for station_name in NOAA_STATIONS:
        for year, month in months:
            filepath = NOAA_DIR / f"noaa_{station_name.lower()}_{year}_{month}.csv"
            if not filepath.exists():
                logger.warning(f"Missing weather file: {filepath}")
                continue

            df = pd.read_csv(filepath)
            df["station"] = station_name
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No weather CSVs found in {NOAA_DIR}. Run download.py first."
        )

    weather = pd.concat(dfs, ignore_index=True)
    weather["datetime"] = pd.to_datetime(weather["datetime"])
    weather["date_hour"] = weather["datetime"].dt.floor("h")

    # Aggregate to hourly if not already (some NOAA data is sub-hourly)
    weather = (
        weather.groupby(["date_hour", "station"])
        .agg(
            rain_intensity_mm=("rain_intensity_mm", "mean"),
            wind_speed_ms=("wind_speed_ms", "mean"),
            visibility_km=("visibility_km", "mean"),
        )
        .reset_index()
    )

    logger.info(f"Loaded weather: {len(weather):,} hourly records across {weather['station'].nunique()} stations")
    return weather


# ── Master Join ────────────────────────────────────────────────────────────

def build_master(
    trips: pd.DataFrame,
    weather: pd.DataFrame,
    zones: pd.DataFrame,
    save: bool = True
) -> pd.DataFrame:
    """
    Join trips → zone metadata → weather.
    Returns master DataFrame ready for modeling.
    """
    n_trips = len(trips)
    logger.info(f"Building master join from {n_trips:,} trips...")

    # Step 1: Attach zone metadata (borough, station assignment)
    zones_with_station = assign_nearest_station(zones)
    trips = trips.merge(
        zones_with_station[["pu_location_id", "borough", "nearest_station", "zone_name"]],
        on="pu_location_id",
        how="left"
    )
    logger.info(f"After zone join: {len(trips):,} rows (null borough: {trips['borough'].isna().sum():,})")

    # Step 2: Join weather on (date_hour, nearest_station)
    trips["date_hour"] = pd.to_datetime(trips["date_hour"])
    trips = trips.merge(
        weather.rename(columns={"station": "nearest_station"}),
        on=["date_hour", "nearest_station"],
        how="left"
    )
    logger.info(f"After weather join: {len(trips):,} rows")

    # Step 3: Encode borough as integer
    borough_order = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR"]
    trips["borough_encoded"] = pd.Categorical(
        trips["borough"], categories=borough_order
    ).codes

    # Step 4: Add is_holiday (federal holidays — simplified)
    trips["is_holiday"] = trips["date"].astype(str).isin(_federal_holidays_2023()).astype(int)

    # Step 5: Binary rain indicator
    trips["is_raining"] = (trips["rain_intensity_mm"] > 0.1).astype(int)

    # Step 6: Validate instrument null rate
    rain_null_rate = trips["rain_intensity_mm"].isna().mean()
    assert rain_null_rate <= MAX_NULL_RATE_INSTRUMENT, (
        f"Rain null rate {rain_null_rate:.3f} exceeds threshold {MAX_NULL_RATE_INSTRUMENT}"
    )
    logger.info(f"Rain null rate: {rain_null_rate:.3f} ✓")

    # Fill remaining nulls with 0 for instrument
    trips["rain_intensity_mm"] = trips["rain_intensity_mm"].fillna(0)
    trips["wind_speed_ms"] = trips["wind_speed_ms"].fillna(trips["wind_speed_ms"].median())
    trips["visibility_km"] = trips["visibility_km"].fillna(trips["visibility_km"].median())

    logger.info(f"Master dataset: {len(trips):,} rows, {trips.shape[1]} columns")
    logger.info(f"Cancellation rate: {trips['cancelled'].mean():.3f}")
    logger.info(f"Rain rate: {trips['is_raining'].mean():.3f}")

    if save:
        dest = DATA_PROCESSED / "master.parquet"
        trips.to_parquet(dest, index=False)
        logger.info(f"Saved master dataset: {dest}")

    return trips


def _federal_holidays_2023() -> list[str]:
    return [
        "2023-01-02", "2023-01-16", "2023-02-20", "2023-05-29",
        "2023-06-19", "2023-07-04", "2023-09-04", "2023-10-09",
        "2023-11-10", "2023-11-23", "2023-12-25"
    ]


def run_join(months: list[tuple] = None, save: bool = True) -> pd.DataFrame:
    """Full join pipeline."""
    trips_path = DATA_PROCESSED / "trips_clean.parquet"
    if not trips_path.exists():
        raise FileNotFoundError("trips_clean.parquet not found. Run clean.py first.")

    trips = pd.read_parquet(trips_path)
    weather = load_weather(months)
    zones = load_zone_lookup()

    return build_master(trips, weather, zones, save=save)


if __name__ == "__main__":
    run_join()
