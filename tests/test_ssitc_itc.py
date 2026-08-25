"""Tests for the above-canopy σ_w/u* ITC reference (library/writeups/itc_sigmaw.md)."""

import numpy as np
import pytest

from utespac.calc_ssitc_flags import _above_canopy_sigmaw, coriolis_parameter


def test_unstable_side_foken_table():
    # Foken 2008 Table 2.11: 1.3 for -0.032 < z/L < 0; 2.0 (-z/L)^(1/8) below
    assert _above_canopy_sigmaw(-0.01) == 1.3
    assert _above_canopy_sigmaw(-0.5) == pytest.approx(2.0 * 0.5 ** 0.125)
    assert _above_canopy_sigmaw(-1e-9) == 1.3
    # the two forms meet at |z/L| = 0.0319 (continuity)
    assert _above_canopy_sigmaw(-0.032) == pytest.approx(1.3, rel=1e-3)


def test_stable_side_thomas_foken_with_latitude():
    # 0.21 ln(z+ f / u*) + 3.1, z+ = 1 m; at 41.15 N, u* = 0.3 m/s
    lat, ustar = 41.15, 0.3
    f = coriolis_parameter(lat)
    expect = 0.21 * np.log(1.0 * f / ustar) + 3.1
    assert _above_canopy_sigmaw(0.1, ustar, lat) == pytest.approx(expect)
    assert 1.0 < expect < 1.6           # ~1.4 near neutral, plausible magnitude
    # independent of zeta inside 0 <= z/L <= 0.4
    assert _above_canopy_sigmaw(0.3, ustar, lat) == pytest.approx(expect)


def test_stable_pahlow_beyond_04_and_without_latitude():
    # Pahlow et al. 2001 eq. 14: 1.1 + 0.9 (z/L)^0.6
    assert _above_canopy_sigmaw(1.0, 0.3, 41.15) == pytest.approx(2.0)
    assert _above_canopy_sigmaw(0.2) == pytest.approx(1.1 + 0.9 * 0.2 ** 0.6)
    assert _above_canopy_sigmaw(0.0) == pytest.approx(1.1)           # no latitude, neutral limit
    assert _above_canopy_sigmaw(10.0, 0.3, 41.15) == pytest.approx(1.1 + 0.9 * 10 ** 0.6)


def test_coriolis_parameter():
    assert coriolis_parameter(90.0) == pytest.approx(2 * 7.2921e-5)
    assert coriolis_parameter(0.0) == pytest.approx(0.0)
    assert np.isnan(_above_canopy_sigmaw(np.nan))
