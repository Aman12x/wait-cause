"""The placebo instrument must be the same station's rain exactly 24 hours later."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from src.models.iv_2sls import attach_next_day_rain


def _weather():
    hours = pd.date_range("2023-06-01", periods=72, freq="h")
    rows = []
    for station, offset in (("JFK", 0.0), ("LGA", 1000.0)):
        for i, h in enumerate(hours):
            rows.append({"date_hour": h, "station": station, "rain_intensity_mm": offset + i})
    return pd.DataFrame(rows)


def test_placebo_is_same_station_rain_24_hours_later():
    # Many trips per hour, deliberately unsorted: a row shift would pick the wrong value
    rng = np.random.default_rng(0)
    hours = pd.date_range("2023-06-01", periods=48, freq="h")
    trips = pd.DataFrame({
        "date_hour": rng.choice(hours, size=2000),
        "nearest_station": rng.choice(["JFK", "LGA"], size=2000),
    })
    out = attach_next_day_rain(trips, _weather())

    w = _weather().set_index(["date_hour", "station"])["rain_intensity_mm"]
    expected = [w[(h + pd.Timedelta(hours=24), s)] for h, s in zip(out["date_hour"], out["nearest_station"])]
    assert len(out) == len(trips)
    assert np.allclose(out["rain_placebo"], expected)


def test_placebo_is_missing_when_no_next_day_reading_exists():
    trips = pd.DataFrame({"date_hour": [pd.Timestamp("2023-06-03 12:00")], "nearest_station": ["JFK"]})
    assert attach_next_day_rain(trips, _weather())["rain_placebo"].isna().all()
