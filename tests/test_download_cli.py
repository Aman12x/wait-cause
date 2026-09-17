"""pipeline.py --step download must not crash on the pipeline's own arguments."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.data import download


def _stub_network(monkeypatch):
    calls = []
    monkeypatch.setattr(download, "download_tlc_zone_lookup", lambda: None)
    monkeypatch.setattr(download, "download_tlc_zone_shapefile", lambda: None)
    monkeypatch.setattr(download, "download_tlc_month", lambda y, m, force=False: calls.append((y, m)))
    monkeypatch.setattr(download, "download_noaa_weather", lambda y, m, force=False: None)
    return calls


def test_main_ignores_the_pipeline_command_line(monkeypatch):
    calls = _stub_network(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["pipeline.py", "--step", "download", "--sample"])
    download.main(months=[("2023", "06")])
    assert calls == [("2023", "06")]


def test_standalone_cli_still_parses_months(monkeypatch):
    calls = _stub_network(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["download.py", "--months", "2"])
    download.main()
    assert len(calls) == 2
