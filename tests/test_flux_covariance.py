"""utespac.flux.covariance: block anomalies, the lag search, the despike
screen and the Lenschow, Mann and Kristensen (2000) noise variance."""

import numpy as np
import pytest

from utespac.flux.covariance import (LMK_EXPONENT, despike_mask, detrend_block,
                                     lagged_covariances, noise_variance_lmk,
                                     peak_lag)


def _cov(a, b):
    """Covariance of two complete series about their own means."""
    return float(np.mean((a - a.mean()) * (b - b.mean())))


def _red_noise(n, tau_samples, rng):
    """A smooth, autocorrelated series standing in for the atmospheric signal.

    The kernel is Gaussian, so the autocovariance is smooth at lag zero and
    any sane extrapolation returns the signal variance.
    """
    half = 4 * tau_samples
    white = rng.standard_normal(n + 2 * half)
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / tau_samples) ** 2)
    k /= np.sqrt((k ** 2).sum())
    out = np.convolve(white, k, mode="same")[half:half + n]
    return (out - out.mean()) / out.std()


# ── detrend_block ────────────────────────────────────────────────────────────

def test_detrend_block_zeros_missing_samples_and_returns_the_finite_mean():
    x = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
    out, mean, ok = detrend_block(x)
    assert mean == pytest.approx(3.0)
    np.testing.assert_array_equal(ok, [True, False, True, False, True])
    np.testing.assert_array_equal(out, [-2.0, 0.0, 0.0, 0.0, 2.0])


def test_detrend_block_of_an_empty_block_has_a_nan_mean():
    out, mean, ok = detrend_block(np.full(4, np.nan))
    assert np.isnan(mean)
    assert not ok.any()
    np.testing.assert_array_equal(out, np.zeros(4))


# ── the lag search ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("shift", [-17, -5, 0, 3, 24])
def test_lag_search_recovers_a_known_shift(shift):
    """A scalar delayed by *shift* samples peaks at exactly that lag."""
    rng = np.random.default_rng(3)
    n = 12000
    w = rng.standard_normal(n)
    c = np.roll(w, shift) + 0.05 * rng.standard_normal(n)
    pad = abs(shift) + 1
    w[:pad] = np.nan
    w[-pad:] = np.nan

    wd, _, w_ok = detrend_block(w)
    cd, _, c_ok = detrend_block(c)
    lags, cov, sub = lagged_covariances(wd, w_ok, cd, c_ok, 40)
    assert sub is None
    found, ratio = peak_lag(lags, cov)
    assert found == shift
    assert ratio > 5.0


def test_lagged_covariances_match_a_direct_covariance_at_zero_lag():
    rng = np.random.default_rng(5)
    a = rng.standard_normal(4000)
    b = 0.4 * a + rng.standard_normal(4000)
    ad, _, a_ok = detrend_block(a)
    bd, _, b_ok = detrend_block(b)
    lags, cov, _ = lagged_covariances(ad, a_ok, bd, b_ok, 5)
    assert cov[lags == 0][0] == pytest.approx(_cov(a, b), rel=1e-12)


def test_sub_covariances_are_taken_about_their_own_means():
    """Each Foken and Wichura sub-interval is detrended by its own mean.

    Detrending the sub-intervals by the whole-period mean instead would make
    their average identically equal to the period covariance, and the
    stationarity statistic identically zero.
    """
    rng = np.random.default_rng(7)
    n = 3600
    a = rng.standard_normal(n)
    b = rng.standard_normal(n) + np.linspace(0.0, 4.0, n)     # a strong trend
    ad, _, a_ok = detrend_block(a)
    bd, _, b_ok = detrend_block(b)
    bounds = np.round(np.linspace(0, n, 7)).astype(int)
    lags, cov, sub = lagged_covariances(ad, a_ok, bd, b_ok, 0, bounds)
    assert sub.shape == (6, 1)
    for j in range(6):
        s = slice(int(bounds[j]), int(bounds[j + 1]))
        assert sub[j, 0] == pytest.approx(_cov(a[s], b[s]), rel=1e-10, abs=1e-14)
    assert float(np.mean(sub[:, 0])) != pytest.approx(cov[0], rel=1e-6)


def test_non_increasing_bounds_skip_the_sub_covariances():
    rng = np.random.default_rng(8)
    a = rng.standard_normal(600)
    ad, _, a_ok = detrend_block(a)
    _, _, sub = lagged_covariances(ad, a_ok, ad, a_ok, 2, np.array([0, 300, 300, 600]))
    assert sub is None


def test_peak_lag_is_nan_when_no_covariance_is_finite():
    lag, ratio = peak_lag(np.arange(-3, 4), np.full(7, np.nan))
    assert np.isnan(lag) and np.isnan(ratio)


def test_peak_lag_ratio_is_infinite_over_a_zero_background():
    lags = np.arange(-3, 4)
    cov = np.array([0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0])
    lag, ratio = peak_lag(lags, cov)
    assert lag == 0
    assert np.isinf(ratio)


# ── despike_mask ─────────────────────────────────────────────────────────────

def test_despike_mask_flags_isolated_spikes_only():
    pytest.importorskip("pandas")
    rng = np.random.default_rng(12)
    x = rng.standard_normal(2000)
    spikes = [150, 900, 1700]
    x[spikes] += 40.0
    mask = despike_mask(x, 101, 6.0)
    assert set(np.flatnonzero(mask)) == set(spikes)


def test_despike_mask_never_flags_missing_samples():
    pytest.importorskip("pandas")
    x = np.zeros(500)
    x[::7] = 1.0
    x[250] = np.nan
    mask = despike_mask(x, 101, 6.0)
    assert not mask[250]


# ── Lenschow, Mann and Kristensen noise variance ─────────────────────────────

def test_lmk_exponent_is_the_inertial_subrange_value():
    assert LMK_EXPONENT == 2.0 / 3.0


@pytest.mark.parametrize("noise_sd", [0.2, 0.35, 0.5])
def test_lmk_recovers_a_known_white_noise_variance(noise_sd):
    """A smooth signal plus known white noise returns that noise variance."""
    rng = np.random.default_rng(4)
    n = 36000
    signal = _red_noise(n, 40, rng)
    noise = noise_sd * rng.standard_normal(n)
    est, total = noise_variance_lmk(signal + noise, 20.0, (1, 10))
    assert total == pytest.approx(np.var(signal + noise), rel=0.02)
    assert est == pytest.approx(noise_sd ** 2, rel=0.25)


def test_lmk_cannot_resolve_noise_far_below_a_percent_of_the_variance():
    """At 0.25% of the total variance the estimate collapses to the zero clip."""
    rng = np.random.default_rng(4)
    signal = _red_noise(36000, 40, rng)
    est, total = noise_variance_lmk(signal + 0.05 * rng.standard_normal(36000),
                                    20.0, (1, 10))
    assert 0.05 ** 2 / total < 0.005
    assert est < 0.01 * total


def test_lmk_returns_near_zero_for_a_noise_free_signal():
    rng = np.random.default_rng(5)
    signal = _red_noise(36000, 40, rng)
    est, total = noise_variance_lmk(signal, 20.0, (1, 10))
    assert est < 0.02 * total


def test_lmk_noise_estimate_grows_with_the_added_noise():
    rng = np.random.default_rng(6)
    signal = _red_noise(36000, 40, rng)
    base_noise = rng.standard_normal(36000)
    a, _ = noise_variance_lmk(signal + 0.1 * base_noise, 20.0, (1, 10))
    b, _ = noise_variance_lmk(signal + 0.4 * base_noise, 20.0, (1, 10))
    assert b > 10.0 * a


def test_lmk_is_nan_for_a_short_or_empty_series_or_a_bad_fit_range():
    assert np.isnan(noise_variance_lmk(np.zeros(100), 20.0, (1, 10))[0])
    assert np.isnan(noise_variance_lmk(np.full(36000, np.nan), 20.0, (1, 10))[0])
    rng = np.random.default_rng(9)
    assert np.isnan(noise_variance_lmk(rng.standard_normal(5000), 20.0, (10, 10))[0])


def test_lmk_never_returns_a_negative_variance():
    """A signal whose autocovariance rises with lag clips at zero, not below."""
    t = np.arange(36000) / 20.0
    est, _ = noise_variance_lmk(np.sin(2 * np.pi * 0.01 * t), 20.0, (1, 10))
    assert est >= 0.0
