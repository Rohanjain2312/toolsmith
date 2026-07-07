"""Tests for the distance_calc sandbox tool."""

import pytest
from pydantic import ValidationError

from toolsmith.tools.sandbox.distance_calc import DistanceCalcArgs, distance_calc


def test_happy_path_paris_to_london() -> None:
    # Paris (48.8566, 2.3522) to London (51.5074, -0.1278); real distance ~344 km.
    args = DistanceCalcArgs(lat1=48.8566, lon1=2.3522, lat2=51.5074, lon2=-0.1278)
    result = distance_calc(args)
    assert 300.0 < result.distance_km < 400.0


def test_identical_points_return_zero_distance() -> None:
    args = DistanceCalcArgs(lat1=40.7128, lon1=-74.0060, lat2=40.7128, lon2=-74.0060)
    result = distance_calc(args)
    assert abs(result.distance_km) < 1e-6


def test_pole_to_pole_returns_half_circumference() -> None:
    args = DistanceCalcArgs(lat1=90.0, lon1=0.0, lat2=-90.0, lon2=0.0)
    result = distance_calc(args)
    assert result.distance_km == pytest.approx(20015.0, abs=5.0)


def test_invalid_latitude_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        DistanceCalcArgs(lat1=999, lon1=0, lat2=0, lon2=0)


def test_deterministic_repeated_calls_match() -> None:
    args = DistanceCalcArgs(lat1=35.6895, lon1=139.6917, lat2=34.0522, lon2=-118.2437)
    result_a = distance_calc(args)
    result_b = distance_calc(args)
    assert result_a == result_b
