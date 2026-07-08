"""Tests for shared cross-subpackage helpers, notably the haversine great-circle distance."""

from __future__ import annotations

import pytest

from toolsmith.utils import haversine_km


def test_haversine_km_known_quarter_circumference() -> None:
    # Equator, 90 degrees of longitude apart: exactly a quarter of Earth's circumference.
    distance = haversine_km(0.0, 0.0, 0.0, 90.0)

    assert distance == pytest.approx(10007.5, rel=1e-3)


def test_haversine_km_same_point_is_zero() -> None:
    assert haversine_km(48.8566, 2.3522, 48.8566, 2.3522) == pytest.approx(0.0, abs=1e-9)


def test_haversine_km_near_antipodal_does_not_raise() -> None:
    # Regression test for BUGFIX-T05: floating-point error can push the intermediate `a` term
    # fractionally above 1.0 for near-antipodal points, which raised ValueError from
    # math.sqrt(1 - a) before this fix.
    distance = haversine_km(
        40.628064952348524, 144.24374679103528, -40.62806495144124, -35.75625320896472
    )

    assert distance == pytest.approx(20015.09, rel=1e-3)  # ~half Earth's circumference


def test_haversine_km_exact_antipodes() -> None:
    distance = haversine_km(0.0, 0.0, 0.0, 180.0)

    assert distance == pytest.approx(20015.09, rel=1e-3)
