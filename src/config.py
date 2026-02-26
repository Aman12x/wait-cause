"""
Project-wide configuration and constants.
All thresholds, paths, and parameters live here — never hardcode in scripts.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS_FIGURES = ROOT / "outputs" / "figures"
OUTPUTS_TABLES = ROOT / "outputs" / "tables"
TLC_DIR = DATA_RAW / "tlc_fhvhv"
NOAA_DIR = DATA_RAW / "noaa_weather"
ZONES_DIR = DATA_RAW / "tlc_zones"

# Create dirs if they don't exist
for d in [
    TLC_DIR,
    NOAA_DIR,
    ZONES_DIR,
    DATA_PROCESSED,
    OUTPUTS_FIGURES,
    OUTPUTS_TABLES,
]:
    d.mkdir(parents=True, exist_ok=True)

# ── Data Schema ────────────────────────────────────────────────────────────
SCHEMA = {
    "wait_time_mins": "float, 0–30, constructed from on_scene - request",
    "cancelled": "int 0/1, 1 if no trip completion within 15min of request",
    "rain_intensity_mm": "float >=0, hourly precip from nearest NOAA station",
    "pu_location_id": "int, 1–263, NYC TLC zone",
    "borough": "str, one of 5 NYC boroughs",
    "hour_of_day": "int 0–23",
    "is_weekend": "int 0/1",
    "surge_proxy": "float, fare_amount / trip_miles",
}

# ── Cleaning Thresholds ────────────────────────────────────────────────────
WAIT_TIME_MIN = 0.5  # minutes — filter out near-instant pickups
WAIT_TIME_MAX = 30.0  # minutes — filter extreme outliers
CANCELLATION_WINDOW = 15  # minutes — if no pickup within this, treat as cancelled
FARE_PER_MILE_MIN = 0.5  # surge proxy lower bound
FARE_PER_MILE_MAX = 50.0  # surge proxy upper bound

# ── NOAA Weather Stations ──────────────────────────────────────────────────
NOAA_STATIONS = {
    "JFK": {"id": "USW00094789", "lat": 40.6413, "lon": -73.7781},
    "LGA": {"id": "USW00014732", "lat": 40.7769, "lon": -73.8740},
    "CENTRAL_PARK": {"id": "USW00094728", "lat": 40.7789, "lon": -73.9692},
}
NOAA_TOKEN_ENV = "NOAA_API_TOKEN"  # set in .env file

# ── TLC Data ───────────────────────────────────────────────────────────────
TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
# Default: pull 3 months for dev, override for full analysis
TLC_MONTHS = [
    ("2023", "06"),
    ("2023", "07"),
    ("2023", "08"),
]

# ── Modeling ───────────────────────────────────────────────────────────────
INSTRUMENT_COLS = ["rain_intensity_mm", "wind_speed_ms", "visibility_km"]
CONTROL_COLS = [
    "hour_of_day",
    "is_weekend",
    "is_holiday",
    "surge_proxy",
    "borough_encoded",
]
TREATMENT_COL = "wait_time_mins"
OUTCOME_COL = "cancelled"

# IV diagnostics
WEAK_INSTRUMENT_F_THRESHOLD = 10.0

# Causal forest
CF_N_ESTIMATORS = 2000
CF_MIN_SAMPLES_LEAF = 50
CF_TEST_SIZE = 0.2
RANDOM_STATE = 42

# ── Validation Thresholds ──────────────────────────────────────────────────
MAX_NULL_RATE_INSTRUMENT = 0.05  # 5% max nulls on rain variable
MIN_CANCELLATION_RATE = 0.0001  # sanity check
MIN_ROWS_AFTER_CLEANING = 100_000
