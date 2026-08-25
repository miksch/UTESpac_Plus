"""Tests for the datetime64 boundary shims in utespac.campbell_date."""

from datetime import datetime

import numpy as np

from utespac.campbell_date import (
    MATLAB_EPOCH, datetime64_to_matlab_datenum, datetime_to_matlab_datenum,
    matlab_datenum_to_datetime, matlab_datenum_to_datetime64,
)


def test_datenum_to_datetime64_matches_scalar_path():
    t = np.array([739404.0 + 0.05 / 86400 * k for k in range(5)] + [739404.5, 739419.0])
    d64 = matlab_datenum_to_datetime64(t)
    assert d64.dtype == np.dtype("datetime64[ms]")
    for md, d in zip(t, d64):
        py = matlab_datenum_to_datetime(md)
        assert abs((d.astype("datetime64[us]").astype(datetime) - py).total_seconds()) < 1e-3
    assert d64[1] - d64[0] == np.timedelta64(50, "ms")       # 20 Hz spacing survives rounding
    assert str(d64[-1]) == "2024-06-16T00:00:00.000"


def test_nan_and_nat():
    d64 = matlab_datenum_to_datetime64([np.nan, MATLAB_EPOCH])
    assert np.isnat(d64[0]) and d64[1] == np.datetime64("1970-01-01T00:00:00", "ms")
    back = datetime64_to_matlab_datenum(d64)
    assert np.isnan(back[0]) and back[1] == MATLAB_EPOCH


def test_round_trip_within_a_millisecond():
    t = 739404.0208333334 + np.arange(96) * (30 / 1440)        # 30-min period ends
    back = datetime64_to_matlab_datenum(matlab_datenum_to_datetime64(t))
    assert np.max(np.abs(back - t)) * 86400 < 1e-3
    d = datetime(2024, 6, 1, 0, 30)
    assert datetime64_to_matlab_datenum(np.datetime64(d)) == datetime_to_matlab_datenum(d)
