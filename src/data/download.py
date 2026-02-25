"""
download.py — Fetch raw TLC FHVHV parquet files and NOAA weather CSVs.
Single responsibility: download only, no transformation.

Usage:
    python -m src.data.download --months 3
"""

import os
import logging
import argparse
import requests
from pathlib import Path
from tqdm import tqdm

# Allow running from project root
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (
    TLC_DIR, NOAA_DIR, TLC_BASE_URL, TLC_MONTHS,
    NOAA_STATIONS, NOAA_TOKEN_ENV
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ── TLC ────────────────────────────────────────────────────────────────────

def download_tlc_month(year: str, month: str, force: bool = False) -> Path:
    """Download one month of FHVHV trip data from TLC."""
    filename = f"fhvhv_tripdata_{year}-{month}.parquet"
    url = f"{TLC_BASE_URL}/{filename}"
    dest = TLC_DIR / filename

    if dest.exists() and not force:
        logger.info(f"Already exists, skipping: {filename}")
        return dest

    logger.info(f"Downloading: {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=filename
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    logger.info(f"Saved: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def download_tlc_zone_lookup() -> Path:
    """Download TLC taxi zone lookup CSV (zone → borough mapping)."""
    url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv"
    dest = Path(str(TLC_DIR).replace("tlc_fhvhv", "tlc_zones")) / "taxi_zone_lookup.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        logger.info("Zone lookup already exists, skipping.")
        return dest

    logger.info("Downloading TLC zone lookup...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    dest.write_bytes(r.content)
    logger.info(f"Saved zone lookup: {dest}")
    return dest


def download_tlc_zone_shapefile() -> Path:
    """Download TLC taxi zone shapefile for spatial joins."""
    url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
    dest_zip = Path(str(TLC_DIR).replace("tlc_fhvhv", "tlc_zones")) / "taxi_zones.zip"
    dest_zip.parent.mkdir(parents=True, exist_ok=True)

    if (dest_zip.parent / "taxi_zones.shp").exists():
        logger.info("Shapefile already exists, skipping.")
        return dest_zip.parent / "taxi_zones.shp"

    logger.info("Downloading TLC zone shapefile...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest_zip.write_bytes(r.content)

    import zipfile
    with zipfile.ZipFile(dest_zip, "r") as z:
        z.extractall(dest_zip.parent)
    logger.info(f"Extracted shapefile to: {dest_zip.parent}")
    return dest_zip.parent / "taxi_zones.shp"


# ── NOAA ───────────────────────────────────────────────────────────────────

def download_noaa_weather(year: str, month: str, force: bool = False) -> dict[str, Path]:
    """
    Download hourly weather from NOAA CDO API for all 3 stations.
    Requires NOAA_API_TOKEN in environment (free at https://www.ncdc.noaa.gov/cdo-web/token).
    """
    token = os.getenv(NOAA_TOKEN_ENV)
    if not token:
        logger.warning(
            f"No NOAA token found in env var '{NOAA_TOKEN_ENV}'. "
            "Generating synthetic weather data for development instead."
        )
        return _generate_synthetic_weather(year, month)

    paths = {}
    start = f"{year}-{month}-01"
    # Compute end date properly
    import calendar
    last_day = calendar.monthrange(int(year), int(month))[1]
    end = f"{year}-{month}-{last_day:02d}"

    for station_name, station_info in NOAA_STATIONS.items():
        filename = f"noaa_{station_name.lower()}_{year}_{month}.csv"
        dest = NOAA_DIR / filename

        if dest.exists() and not force:
            logger.info(f"Already exists: {filename}")
            paths[station_name] = dest
            continue

        logger.info(f"Fetching NOAA data: {station_name} {year}-{month}")
        url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
        params = {
            "datasetid": "LCD",
            "stationid": f"WBAN:{station_info['id'].split('W')[-1]}",
            "startdate": start,
            "enddate": end,
            "limit": 1000,
            "units": "metric",
            "datatypeid": "HourlyPrecipitation,HourlyWindSpeed,HourlyVisibility",
        }
        headers = {"token": token}

        r = requests.get(url, params=params, headers=headers, timeout=60)
        if r.status_code != 200:
            logger.warning(f"NOAA API returned {r.status_code} for {station_name}, using synthetic.")
            paths[station_name] = _generate_synthetic_weather_station(year, month, station_name)
            continue

        dest.write_text(r.text)
        logger.info(f"Saved: {dest}")
        paths[station_name] = dest

    return paths


def _generate_synthetic_weather(year: str, month: str) -> dict[str, Path]:
    """Generate realistic synthetic weather for development without NOAA token."""
    paths = {}
    for station_name in NOAA_STATIONS:
        paths[station_name] = _generate_synthetic_weather_station(year, month, station_name)
    return paths


def _generate_synthetic_weather_station(year: str, month: str, station_name: str) -> Path:
    """Generate hourly synthetic weather CSV for one station."""
    import pandas as pd
    import numpy as np

    filename = f"noaa_{station_name.lower()}_{year}_{month}.csv"
    dest = NOAA_DIR / filename

    import calendar
    last_day = calendar.monthrange(int(year), int(month))[1]
    hours = pd.date_range(
        start=f"{year}-{month}-01",
        end=f"{year}-{month}-{last_day} 23:00",
        freq="h"
    )

    np.random.seed(42)
    n = len(hours)

    # Realistic weather patterns: most hours dry, occasional rain
    rain = np.random.exponential(0.1, n) * (np.random.random(n) < 0.15)
    wind = np.abs(np.random.normal(4.0, 2.0, n))
    visibility = np.clip(np.random.normal(16.0, 4.0, n), 0.5, 24.0)
    # Visibility drops when raining
    visibility = np.where(rain > 0.5, visibility * 0.5, visibility)

    df = pd.DataFrame({
        "datetime": hours,
        "station": station_name,
        "rain_intensity_mm": rain.round(2),
        "wind_speed_ms": wind.round(2),
        "visibility_km": visibility.round(2),
    })

    df.to_csv(dest, index=False)
    logger.info(f"Generated synthetic weather: {dest} ({n} hourly records)")
    return dest


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download raw project data")
    parser.add_argument("--months", type=int, default=3, help="Number of months to download")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    args = parser.parse_args()

    months = TLC_MONTHS[:args.months]
    logger.info(f"Downloading {len(months)} months of data...")

    # TLC zone metadata
    download_tlc_zone_lookup()
    download_tlc_zone_shapefile()

    # Trip data + weather per month
    for year, month in months:
        logger.info(f"── {year}-{month} ──")
        download_tlc_month(year, month, force=args.force)
        download_noaa_weather(year, month, force=args.force)

    logger.info("Download complete.")


if __name__ == "__main__":
    main()
