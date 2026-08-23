"""utespac.wind_stats primitives: direction/speed conventions and the shadow sector."""

import numpy as np
import pytest

from utespac.wind_stats import (MANUFACTURER_CAMPBELL, MANUFACTURER_GILL, MANUFACTURER_RMYOUNG,
                                shadow_flag, shadow_sector, wind_direction_speed)


def test_campbell_direction_is_where_wind_comes_from():
    # wind blowing toward -u (into the head) with bearing 0: comes from 0 deg
    d, s = wind_direction_speed(np.array([1.0]), np.array([0.0]), 0.0, MANUFACTURER_CAMPBELL)
    assert d[0] == pytest.approx(0.0)
    assert s[0] == pytest.approx(1.0)
    d, _ = wind_direction_speed(np.array([0.0]), np.array([1.0]), 0.0, MANUFACTURER_CAMPBELL)
    assert d[0] == pytest.approx(270.0)


def test_bearing_adds_and_wraps():
    d, _ = wind_direction_speed(np.array([0.0]), np.array([1.0]), 215.0)
    assert d[0] == pytest.approx(125.0)


def test_manufacturer_conventions_agree_on_speed_and_map_axes():
    u, v = np.array([1.0]), np.array([2.0])
    for m in (MANUFACTURER_CAMPBELL, MANUFACTURER_RMYOUNG, MANUFACTURER_GILL):
        _, s = wind_direction_speed(u, v, 0.0, m)
        assert s[0] == pytest.approx(np.sqrt(5.0))
    d_c, _ = wind_direction_speed(u, v, 0.0, MANUFACTURER_CAMPBELL)
    d_g, _ = wind_direction_speed(u, v, 0.0, MANUFACTURER_GILL)
    assert (d_g[0] - d_c[0]) % 360.0 == pytest.approx(180.0)
    d_r, _ = wind_direction_speed(u, v, 0.0, MANUFACTURER_RMYOUNG)
    d_c2, _ = wind_direction_speed(v, -u, 0.0, MANUFACTURER_CAMPBELL)
    assert d_r[0] == pytest.approx(d_c2[0])


def test_shadow_sector_wraps_through_north():
    assert shadow_sector(180.0, 215.0, 20.0) == (15.0, 55.0)       # VAC001: centre 35
    lo, hi = shadow_sector(0.0, 350.0, 20.0)                        # centre 350 -> 330..10
    assert (lo, hi) == (330.0, 10.0)
    flag, _, _ = shadow_flag(np.array([340.0, 5.0, 100.0]), 0.0, 350.0, 20.0)
    assert flag.tolist() == [1.0, 1.0, 0.0]
    flag, _, _ = shadow_flag(np.array([20.0, 50.0, 60.0]), 180.0, 215.0, 20.0)
    assert flag.tolist() == [1.0, 1.0, 0.0]
