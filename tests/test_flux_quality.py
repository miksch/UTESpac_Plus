"""utespac.flux.quality: the Foken and Wichura (1996) stationarity statistic
and classes, the Mahrt et al. (1998) flux intermittency, the Finkelstein and
Sims (2001) random error, and the sliding RSSI statistics."""

import warnings

import numpy as np
import pytest

from utespac.flux.quality import (FOKEN_CLASS_EDGES, MEAN_W_FLAG_EDGES,
                                  flux_intermittency, foken_class, mean_w_flag,
                                  random_error_fs2001, rssi_threshold_below,
                                  sliding_rssi_snr, stationarity_fw96,
                                  wind_constancy)


# ── stationarity ─────────────────────────────────────────────────────────────

def test_stationarity_is_the_relative_departure_in_percent():
    assert stationarity_fw96(np.array([0.9, 1.1, 1.3]), 1.0) == pytest.approx(10.0)
    assert stationarity_fw96(np.array([-0.5, -0.5]), -1.0) == pytest.approx(50.0)


def test_stationarity_ignores_missing_sub_covariances():
    assert stationarity_fw96(np.array([1.2, np.nan, 1.2]), 1.0) == pytest.approx(20.0)


@pytest.mark.parametrize("total", [0.0, np.nan, np.inf])
def test_stationarity_is_nan_for_an_undefined_total(total):
    assert np.isnan(stationarity_fw96(np.array([1.0, 2.0]), total))


def test_stationarity_is_nan_when_no_sub_covariance_is_finite():
    with pytest.warns(RuntimeWarning):
        assert np.isnan(stationarity_fw96(np.full(3, np.nan), 1.0))


def test_foken_class_edges_are_the_three_class_boundaries():
    assert FOKEN_CLASS_EDGES == (30.0, 100.0)


@pytest.mark.parametrize("stat,expected", [
    (0.0, 1), (29.9, 1), (30.0, 1),        # closed on the class-1 edge
    (30.1, 2), (99.9, 2), (100.0, 2),      # closed on the class-2 edge
    (100.1, 3), (1e4, 3), (np.nan, 0),
])
def test_foken_class_boundaries(stat, expected):
    assert foken_class(stat) == expected


def test_foken_class_accepts_other_edges():
    assert foken_class(40.0, edges=(50.0, 200.0)) == 1
    assert foken_class(250.0, edges=(50.0, 200.0)) == 3


# ── flux intermittency ───────────────────────────────────────────────────────

# Sub-blocks 1, 2, 3, 4: mean 2.5, squared deviations 2.25 + 0.25 + 0.25 +
# 2.25 = 5, sample variance 5 / 3; over a total of 2 the index is
# sqrt(5 / 3) / 2 = 0.6455.
FI_HAND = np.sqrt(5.0 / 3.0) / 2.0


def test_flux_intermittency_is_zero_for_identical_sub_blocks():
    assert flux_intermittency(np.full(6, 0.5), 0.5) == 0.0


@pytest.mark.parametrize("sub", [
    [1.0, 2.0, 3.0, 4.0],
    [1.0, np.nan, 2.0, 3.0, np.nan, 4.0],     # missing sub-blocks are skipped
])
def test_flux_intermittency_hand_case(sub):
    # float round-off on four terms
    assert flux_intermittency(np.array(sub), 2.0) == \
        pytest.approx(FI_HAND, rel=1e-12)


@pytest.mark.parametrize("total", [2.0, -2.0])
def test_flux_intermittency_uses_the_magnitude_of_the_total(total):
    # float round-off on four terms
    assert flux_intermittency(np.array([1.0, 2.0, 3.0, 4.0]), total) == \
        pytest.approx(FI_HAND, rel=1e-12)


@pytest.mark.parametrize("sub,total,expected", [
    ([1.0, 2.0, 3.0, 4.0], 2.0, FI_HAND),
    ([1.0, 2.0, 3.0, 4.0], 0.0, np.nan),
    ([1.0, 2.0, 3.0, 4.0], np.nan, np.nan),
    ([1.0, 2.0], np.inf, np.nan),
    ([1.0, np.nan, 3.0], 2.0, np.sqrt(2.0) / 2.0),   # two finite sub-blocks
    ([1.0, np.inf, 3.0], 2.0, np.sqrt(2.0) / 2.0),   # inf sub-block skipped
    ([1.0, np.nan, np.nan], 2.0, np.nan),     # one finite sub-block
    ([np.nan, np.nan], 2.0, np.nan),
])
def test_flux_intermittency_is_nan_without_warning_for_an_undefined_index(
        sub, total, expected):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fi = flux_intermittency(np.array(sub), total)
    # float round-off on four terms
    assert fi == pytest.approx(expected, rel=1e-12, nan_ok=True)
    assert not caught


@pytest.mark.filterwarnings("error")
def test_flux_intermittency_is_inf_for_a_subnormal_total():
    assert flux_intermittency(np.array([1.0, 2.0]), 5e-324) == np.inf


# ── mean vertical wind and wind constancy ────────────────────────────────────

def test_mean_w_flag_edges_are_the_mauder_2013_thresholds():
    assert MEAN_W_FLAG_EDGES == (0.10, 0.15)


@pytest.mark.parametrize("w,expected", [
    (0.0, 0), (0.05, 0), (0.10, 0),        # open on the 0.10 edge
    (0.1001, 1), (0.12, 1), (0.15, 1),     # open on the 0.15 edge
    (0.1501, 2), (0.5, 2),
    (-0.12, 1), (-0.2, 2),                 # magnitude of the mean
    (np.nan, 2),
])
def test_mean_w_flag_boundaries(w, expected):
    assert mean_w_flag(w) == expected


def test_mean_w_flag_accepts_other_edges():
    assert mean_w_flag(0.25, edges=(0.2, 0.3)) == 1


def _wind_means(direction_deg, speed):
    """Vector-mean magnitude and scalar mean speed of a horizontal wind."""
    rad = np.deg2rad(direction_deg)
    u, v = speed * np.cos(rad), speed * np.sin(rad)
    return np.hypot(u.mean(), v.mean()), np.hypot(u, v).mean()


@pytest.mark.parametrize("direction,expected", [
    (np.full(600, 30.0), 1.0),                       # fixed direction
    (np.r_[np.full(300, 30.0), np.full(300, 210.0)], 0.0),   # reversal
    # quarter turn
    (np.r_[np.full(300, 0.0), np.full(300, 90.0)], np.sqrt(0.5)),
])
def test_wind_constancy_closed_forms(direction, expected):
    vec, spd = _wind_means(direction, np.full(direction.size, 3.0))
    # float round-off on sums of 600 equal terms
    assert wind_constancy(vec, spd) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("vec,spd,expected", [
    (0.0, 0.0, np.nan), (1.0, np.nan, np.nan), (1.0, np.inf, np.nan),
    (np.inf, np.inf, np.nan), (np.inf, 2.0, np.nan),
    (2.0, 4.0, 0.5), (0.0, 2.0, 0.0),
])
def test_wind_constancy_is_nan_only_for_an_undefined_ratio(vec, spd, expected):
    assert wind_constancy(vec, spd) == pytest.approx(expected, nan_ok=True)


@pytest.mark.filterwarnings("error")
def test_wind_constancy_is_inf_on_overflow():
    assert wind_constancy(np.float64(1e308), np.float64(1e-308)) == np.inf


# ── random error ─────────────────────────────────────────────────────────────

def test_random_error_matches_the_white_noise_expectation():
    """For independent white series the FS2001 variance reduces to var(w)var(c)/n."""
    rng = np.random.default_rng(101)
    n = 40000
    w = rng.standard_normal(n) * 0.2
    c = rng.standard_normal(n) * 3.0
    err = random_error_fs2001(w - w.mean(), c - c.mean(), 20.0, 20.0)
    assert err == pytest.approx(np.std(w) * np.std(c) / np.sqrt(n), rel=0.15)


def test_random_error_grows_with_autocorrelation():
    """Two correlated series carry fewer independent samples, so the error rises.

    Only the product of the two auto-covariances enters, so both series have
    to be reddened: one white series collapses the lag sum to its zero-lag
    term.
    """
    rng = np.random.default_rng(102)
    n = 40000
    win = np.ones(200) / 200.0

    def red():
        v = np.convolve(rng.standard_normal(n + len(win) - 1), win, mode="valid")
        return (v - v.mean()) / v.std()

    a_w, b_w = rng.standard_normal(n), rng.standard_normal(n)
    a_r, b_r = red(), red()
    e_white = random_error_fs2001(a_w - a_w.mean(), b_w - b_w.mean(), 20.0, 60.0)
    e_red = random_error_fs2001(a_r, b_r, 20.0, 60.0)
    assert e_red > 3.0 * e_white


def test_random_error_is_nan_for_a_short_series_or_integration_range():
    assert np.isnan(random_error_fs2001(np.zeros(100), np.zeros(100), 20.0, 200.0))
    rng = np.random.default_rng(103)
    x = rng.standard_normal(5000)
    assert np.isnan(random_error_fs2001(x, x, 20.0, 0.1))


# ── the sliding RSSI statistics ──────────────────────────────────────────────

def test_sliding_rssi_snr_tracks_a_built_in_step():
    rssi = np.r_[np.full(60, 12.0), np.full(60, 30.0)]
    snr = np.r_[np.full(60, 1.0), np.full(60, 5.0)]
    centres, med, cnt = sliding_rssi_snr(rssi, snr, 4.0, 0.5, 8)
    low = np.isfinite(med) & (centres < 20)
    high = np.isfinite(med) & (centres > 25)
    assert med[low].max() == pytest.approx(1.0)
    assert med[high].min() == pytest.approx(5.0)
    assert cnt[low].min() >= 8
    # both extremes of the observed range sit inside a populated bin
    assert np.isfinite(med[0]) and np.isfinite(med[-1])


def test_sliding_rssi_snr_is_empty_when_there_is_too_little_data():
    centres, med, cnt = sliding_rssi_snr([1.0, 2.0], [1.0, 2.0], 4.0, 0.5, 8)
    assert centres.size == 0 and med.size == 0 and cnt.size == 0


def test_rssi_threshold_picks_the_highest_failing_bin():
    centres = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    values = np.array([1.0, 1.5, 1.9, 2.5, 3.0])
    counts = np.full(5, 20)
    assert rssi_threshold_below(centres, values, 2.0, 8, counts) == 14.0


def test_rssi_threshold_is_nan_when_nothing_fails():
    centres = np.array([10.0, 12.0])
    values = np.array([3.0, 4.0])
    assert np.isnan(rssi_threshold_below(centres, values, 2.0, 8,
                                         np.full(2, 20)))


def test_rssi_threshold_ignores_underpopulated_bins():
    centres = np.array([10.0, 12.0, 14.0])
    values = np.array([1.0, 1.0, 5.0])
    counts = np.array([20, 2, 20])     # the 12.0 bin is too thin to count
    assert rssi_threshold_below(centres, values, 2.0, 8, counts) == 10.0


def test_rssi_threshold_without_counts_uses_every_finite_bin():
    centres = np.array([10.0, 12.0, 14.0])
    values = np.array([1.0, 1.0, np.nan])
    assert rssi_threshold_below(centres, values, 2.0) == 12.0
