"""utespac.averaging: period arithmetic and block reductions shared by the stages."""

import numpy as np
import pytest

from utespac.averaging import (block_average, block_last, block_mean, block_mean_rows,
                               n_periods, period_bounds, vector_mean_direction)
from utespac.campbell_date import MATLAB_EPOCH
from utespac.simple_avg import simple_avg

DAY = 739406.0   # a datenum midnight


def _day_grid(hz, days=1):
    n = int(hz * 86400 * days)
    return DAY + np.arange(1, n + 1) / (hz * 86400.0)


def test_n_periods_whole_day_grid():
    t = _day_grid(1)
    assert n_periods(t, 30) == 48
    assert n_periods(t, 30, whole_days=False) == 48
    assert n_periods(_day_grid(1, 2), 30) == 96
    assert n_periods(t, 5) == 288


def test_n_periods_partial_file_conventions_differ():
    t = DAY + np.arange(1, 3601) / 86400.0          # one hour of 1 Hz data
    assert n_periods(t, 30) == 48                    # whole-day span (MATLAB completeTable)
    assert n_periods(t, 30, whole_days=False) == 2   # actual span


def test_period_bounds_equal_blocks():
    bp = period_bounds(86400, 48)
    assert bp[0] == 0 and bp[-1] == 86400
    assert np.all(np.diff(bp) == 1800)


def test_block_mean_and_last_skip_empty_blocks():
    vals = np.arange(10, dtype=float)[:, None]
    bounds = np.array([0, 4, 4, 10])
    m = block_mean(vals, bounds)
    assert m[0, 0] == pytest.approx(1.5)
    assert np.isnan(m[1, 0])
    assert m[2, 0] == pytest.approx(6.5)
    last = block_last(np.arange(10.0), bounds)
    assert last[0] == 3 and np.isnan(last[1]) and last[2] == 9


def test_block_mean_nan_aware_and_all_nan_block_stays_nan():
    vals = np.array([[1.0, np.nan], [3.0, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
    m = block_mean(vals, np.array([0, 2, 4]))
    assert m[0, 0] == 2.0 and np.isnan(m[0, 1]) and np.isnan(m[1, 0])


def test_vector_mean_direction_across_north():
    d = np.array([350.0, 10.0])
    s = np.ones(2)
    assert np.cos(np.radians(vector_mean_direction(d, s))) == pytest.approx(1.0, abs=1e-12)
    m = block_mean(np.column_stack([d, s]), np.array([0, 2]), vector_cols=[(0, 1)])
    assert np.cos(np.radians(m[0, 0])) == pytest.approx(1.0, abs=1e-12)
    assert m[0, 1] == 1.0


def test_block_average_matches_simple_avg():
    rng = np.random.default_rng(1)
    t = _day_grid(1)
    x = rng.normal(size=(t.size, 2))
    x[1000:5000, 0] = np.nan
    t_end, means = block_average(t, x, 30)
    legacy = simple_avg(np.column_stack([x, t]), 30)
    assert np.array_equal(t_end, legacy[:, 2])
    assert np.array_equal(means, legacy[:, :2], equal_nan=True)


def test_block_mean_rows_drops_remainder():
    out = block_mean_rows(np.arange(7.0), 3)
    assert out.shape == (2, 1)
    assert out[:, 0].tolist() == [1.0, 4.0]
