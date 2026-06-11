"""
test_extractor.py — Unit tests for extractor.py (no GEE connection required)
Run: conservation_env/Scripts/python.exe -m pytest test_extractor.py -v
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from pathlib import Path
from shapely.geometry import Point

# Patch ee before importing extractor so tests run without credentials
import sys
from unittest.mock import MagicMock
sys.modules.setdefault("ee", MagicMock())

from extractor import (
    load_and_validate,
    interpolate_missing,
    attach_quality_flags,
    _clamp,
    _date_window,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_csv(tmp_path):
    p = tmp_path / "fixes.csv"
    p.write_text(
        "animal_id,timestamp,latitude,longitude\n"
        "ELE_001,2023-01-15T06:00:00Z,0.245,37.102\n"
        "ELE_001,2023-01-16T08:00:00Z,0.248,37.115\n"
        "ELE_002,2023-02-01T07:00:00Z,0.380,36.923\n"
    )
    return p


@pytest.fixture
def sample_gdf():
    df = pd.DataFrame({
        "animal_id": ["ELE_001"] * 5,
        "timestamp": pd.to_datetime([
            "2023-01-10", "2023-01-11", "2023-01-12",
            "2023-01-13", "2023-01-14",
        ], utc=True),
        "latitude":  [0.24, 0.25, 0.26, 0.27, 0.28],
        "longitude": [37.1, 37.1, 37.1, 37.1, 37.1],
        "ndvi":      [0.60,  None, None,  0.65,  0.70],
        "rainfall_mm": [5.0,  None,  8.0,  None, 10.0],
    })
    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestLoadAndValidate:
    def test_valid_csv_loads_correctly(self, valid_csv):
        gdf = load_and_validate(valid_csv)
        assert len(gdf) == 3
        assert "geometry" in gdf.columns
        assert gdf.crs.to_epsg() == 4326

    def test_timestamps_are_utc_aware(self, valid_csv):
        gdf = load_and_validate(valid_csv)
        assert gdf["timestamp"].dt.tz is not None

    def test_missing_column_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("animal_id,timestamp,latitude\n001,2023-01-01,0.1\n")
        with pytest.raises(ValueError, match="missing required columns"):
            load_and_validate(p)

    def test_out_of_range_coords_are_dropped(self, tmp_path):
        p = tmp_path / "oob.csv"
        p.write_text(
            "animal_id,timestamp,latitude,longitude\n"
            "A,2023-01-01T00:00:00Z,0.2,37.1\n"    # valid
            "B,2023-01-02T00:00:00Z,999.0,37.1\n"  # bad lat
        )
        gdf = load_and_validate(p)
        assert len(gdf) == 1


class TestInterpolateMissing:
    def test_nulls_are_filled(self, sample_gdf):
        result = interpolate_missing(sample_gdf, ["ndvi", "rainfall_mm"])
        assert result["ndvi"].isna().sum() == 0
        assert result["rainfall_mm"].isna().sum() == 0

    def test_existing_values_are_preserved(self, sample_gdf):
        result = interpolate_missing(sample_gdf, ["ndvi"])
        valid_before = sample_gdf["ndvi"].dropna().values
        for val in valid_before:
            assert val in result["ndvi"].values

    def test_interpolated_ndvi_in_valid_range(self, sample_gdf):
        result = interpolate_missing(sample_gdf, ["ndvi"])
        assert result["ndvi"].between(-0.2, 1.0).all()


class TestHelpers:
    def test_clamp_within_range(self):
        assert _clamp(0.5, 0.0, 1.0) == 0.5

    def test_clamp_below_min(self):
        assert _clamp(-5.0, 0.0, 1.0) == 0.0

    def test_clamp_above_max(self):
        assert _clamp(1.5, 0.0, 1.0) == 1.0

    def test_date_window_symmetric(self):
        start, end = _date_window("2023-06-15", 5)
        assert start == "2023-06-10"
        assert end   == "2023-06-21"

    def test_date_window_zero_delta(self):
        start, end = _date_window("2023-06-15", 0)
        assert start == "2023-06-15"
        assert end   == "2023-06-16"


class TestQualityFlags:
    def test_flag_columns_created(self, sample_gdf):
        filled = interpolate_missing(sample_gdf, ["ndvi", "rainfall_mm"])
        flagged = attach_quality_flags(filled)
        assert "ndvi_interpolated"        in flagged.columns
        assert "rainfall_mm_interpolated" in flagged.columns

    def test_flag_columns_are_boolean(self, sample_gdf):
        filled  = interpolate_missing(sample_gdf, ["ndvi", "rainfall_mm"])
        flagged = attach_quality_flags(filled)
        assert flagged["ndvi_interpolated"].dtype == bool
