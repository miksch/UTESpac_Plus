"""Equal-length periods and clock-grid sub-period statistics.

Covers :func:`utespac.averaging.validate_avg_min`,
:func:`~utespac.averaging.period_bounds_uniform`,
:func:`~utespac.averaging.period_blocks` and
:func:`~utespac.averaging.sub_period_stats` on synthetic stamps.
"""

import numpy as np
import pytest

from utespac.averaging import (epoch_s, period_blocks, period_bounds_uniform,
                               sub_period_stats, validate_avg_min)

pd = pytest.importorskip("pandas")

FS = 20.0
GRID_MIN = 30.0
N_ROWS = int(GRID_MIN * 60 * FS)
SPAN_MIN = 420.0


# ── requested averaging periods ──────────────────────────────────────────────

def test_periods_that_tile_the_span_are_accepted_and_deduplicated():
    assert validate_avg_min((30.0, 10.0, 1.0), SPAN_MIN) == [30.0, 10.0, 1.0]
    assert validate_avg_min([30, 30.0, 10], SPAN_MIN) == [30.0, 10.0]


@pytest.mark.parametrize("avg_min", [8.0, 9.0, 11.0, 45.0, 400.0])
def test_a_period_that_does_not_divide_the_span_is_rejected(avg_min):
    with pytest.raises(ValueError, match="does not divide"):
        validate_avg_min([avg_min], SPAN_MIN)


@pytest.mark.parametrize("avg_min", [0.0, -5.0, 0.5, 2.5])
def test_a_period_that_is_not_a_positive_whole_minute_is_rejected(avg_min):
    with pytest.raises(ValueError, match="whole number of minutes"):
        validate_avg_min([avg_min], SPAN_MIN)


# ── epoch seconds ────────────────────────────────────────────────────────────

def test_epoch_s_counts_seconds_from_1970():
    stamps = np.array(["1970-01-01T00:00:00", "1970-01-02T00:00:00.250",
                       "2024-06-01T12:00:00"], dtype="datetime64[ms]")
    assert epoch_s(stamps).tolist() == [0.0, 86400.25, 1717243200.0]


def test_epoch_s_accepts_a_column_and_an_index():
    stamps = pd.to_datetime(["2024-06-01 12:00:00.00", "2024-06-01 12:00:00.25"])
    expected = pytest.approx([1717243200.0, 1717243200.25], abs=1e-6)
    assert epoch_s(pd.Series(stamps)).tolist() == expected
    assert epoch_s(pd.DatetimeIndex(stamps)).tolist() == expected


# ── uniform period bounds ────────────────────────────────────────────────────

def _grid_days(n=N_ROWS):
    t0 = np.datetime64("2024-06-01T12:00:00", "ns")
    grid = t0 + (np.arange(n) * (1e9 / FS)).astype("timedelta64[ns]")
    return epoch_s(grid) / 86400.0


@pytest.mark.parametrize("avg_min", [30.0, 15.0, 10.0, 5.0, 2.0, 1.0])
def test_period_bounds_are_uniform(avg_min):
    bounds, slices = period_bounds_uniform(_grid_days(), N_ROWS, avg_min)
    widths = np.diff(bounds)
    assert len(slices) == int(GRID_MIN / avg_min)
    assert widths.min() == widths.max() == int(avg_min * 60 * FS)
    assert bounds[0] == 0 and bounds[-1] == N_ROWS
    assert slices[0] == slice(0, int(avg_min * 60 * FS))


def test_period_bounds_reject_a_grid_the_periods_do_not_tile():
    with pytest.raises(ValueError, match="do not tile"):
        period_bounds_uniform(_grid_days(), N_ROWS + 1, 1.0)


# ── clock-grid period blocks ─────────────────────────────────────────────────

def _block(minutes=30, fs=FS, start="2024-06-01T14:00:00"):
    n = int(minutes * 60 * fs)
    t0 = np.datetime64(start, "ns")
    return t0 + (np.arange(n) * (1e9 / fs)).astype("timedelta64[ns]"), n


def test_period_blocks_cut_on_the_clock_grid():
    ts, n = _block(minutes=60, start="2024-06-01T13:50:00")
    starts, bounds = period_blocks(ts, 30.0)
    assert list(starts) == list(pd.to_datetime(["2024-06-01 13:30", "2024-06-01 14:00",
                                                "2024-06-01 14:30"]))
    per = int(10 * 60 * FS)
    assert bounds.tolist() == [0, per, per + int(30 * 60 * FS), n]


def test_period_blocks_keep_a_backward_clock_step_inside_one_block():
    """A stamp that steps back across a period edge stays with its neighbours."""
    ts, n = _block(minutes=2, fs=1.0, start="2024-06-01T14:00:00")
    ts = ts.copy()
    ts[61] = ts[61] - np.timedelta64(3, "s")     # 14:01:01 -> 14:00:58
    starts, bounds = period_blocks(ts, 1.0)
    assert bounds.tolist() == [0, 60, n]
    assert len(starts) == 2


def test_period_blocks_skip_empty_periods():
    a, _ = _block(minutes=1, fs=1.0, start="2024-06-01T14:00:00")
    b, _ = _block(minutes=1, fs=1.0, start="2024-06-01T14:05:00")
    starts, bounds = period_blocks(np.r_[a, b], 1.0)
    assert [str(s) for s in starts] == ["2024-06-01 14:00:00", "2024-06-01 14:05:00"]
    assert bounds.tolist() == [0, 60, 120]


# ── sub-period statistics ────────────────────────────────────────────────────

def test_sub_period_stats_splits_a_block_and_blanks_one_sparse_minute():
    ts, n = _block()
    rng = np.random.default_rng(1)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    b[3 * 1200:3 * 1200 + 900] = np.nan          # minute 3 keeps 25 %
    out = sub_period_stats(ts, 1, {"a": a, "b": b}, {"a": 0.5, "b": 0.5}, FS)
    assert len(out) == 30
    expect = pd.date_range("2024-06-01 14:00", periods=30, freq="1min")
    assert (pd.DatetimeIndex(out["period_start_lst"]) == expect).all()
    assert (out["n_scans"] == 1200).all()
    np.testing.assert_allclose(out["a_mean"], a.reshape(30, 1200).mean(axis=1))
    np.testing.assert_allclose(out["a_sd"], a.reshape(30, 1200).std(axis=1))
    assert out["b_frac"].iloc[3] == pytest.approx(0.25)
    assert np.isnan(out["b_mean"].iloc[3]) and np.isnan(out["b_sd"].iloc[3])
    assert out["b_mean"].drop(index=3).notna().all()
    assert out["a_mean"].notna().all()


def test_sub_period_means_recombine_to_the_block_mean():
    ts, n = _block()
    x = np.random.default_rng(2).standard_normal(n) + 0.3
    x[::7] = np.nan
    for avg in (1, 10):
        out = sub_period_stats(ts, avg, {"x": x}, 0.5, FS)
        w = out["x_frac"].to_numpy()
        assert np.sum(w * out["x_mean"]) / np.sum(w) == pytest.approx(
            np.nanmean(x), rel=1e-12)


def test_sub_period_stats_of_an_empty_series_has_the_columns_only():
    out = sub_period_stats(np.array([], dtype="datetime64[ns]"), 1,
                           {"x": np.array([])}, 0.5, FS)
    assert len(out) == 0
    assert list(out.columns) == ["period_start_lst", "n_scans",
                                 "x_mean", "x_sd", "x_frac"]
