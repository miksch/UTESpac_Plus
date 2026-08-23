"""ec_coherent.preprocess: gap policy, detrend semantics, Taylor helpers, diagnostics."""

import numpy as np
import pytest

from ec_coherent import preprocess as pp


def test_fill_gaps_interpolates_and_reports_fraction():
    x = np.array([1.0, np.nan, 3.0, 4.0, np.nan, np.nan, 7.0, 8.0, 9.0, 10.0])
    g = pp.fill_gaps(x, max_frac=0.5)
    assert g.accepted and g.nan_frac == pytest.approx(0.3)
    assert np.allclose(g.x, np.arange(1, 11, dtype=float))


def test_fill_gaps_rejects_above_threshold_and_edges():
    x = np.array([np.nan, np.nan, 3.0, 4.0, 5.0])
    assert not pp.fill_gaps(x, max_frac=0.1).accepted
    g = pp.fill_gaps(x, max_frac=0.5)
    assert g.accepted and g.x[0] == 3.0 and g.x[1] == 3.0   # leading NaNs take the nearest value
    assert not pp.fill_gaps(np.full(5, np.nan), max_frac=1.0).accepted


@pytest.mark.parametrize("method", pp.DETREND_METHODS)
def test_detrend_preserves_nan_and_removes_mean(method):
    rng = np.random.default_rng(0)
    n = 6000
    x = 2.0 + 0.001 * np.arange(n) + rng.normal(size=n)
    x[100:110] = np.nan
    y = pp.detrend(x, method, fs=20.0, tau_s=60.0)
    assert np.isnan(y[100:110]).all()
    assert np.isfinite(y[~np.isnan(x)]).all()
    tol = {"recursive": 1.5, "butterworth": 0.3}.get(method, 0.05)   # causal RC filter lags the ramp by tau
    assert abs(np.nanmean(y)) < tol


def test_detrend_linear_removes_slope_block_keeps_it():
    n = 2000
    t = np.arange(n, dtype=float)
    x = 0.01 * t + 5.0
    lin = pp.detrend(x, "linear")
    blk = pp.detrend(x, "block")
    assert np.allclose(lin, 0.0, atol=1e-9)
    assert np.allclose(blk, x - x.mean())


def test_detrend_mostly_nan_returns_all_nan_like_utespac_nandetrend():
    from utespac.nandetrend import nandetrend
    x = np.full(100, np.nan)
    x[:5] = [1, 2, 3, 4, 5]
    assert np.isnan(pp.detrend(x, "linear")).all()
    assert np.isnan(nandetrend(x, 1)).all()
    x2 = np.arange(100, dtype=float)
    x2[::3] = np.nan
    assert np.allclose(pp.detrend(x2, "linear"), nandetrend(x2, 1), equal_nan=True)
    assert np.allclose(pp.detrend(x2, "block"), nandetrend(x2, 0), equal_nan=True)


def test_recursive_filter_is_moncrieff_eq_2_23():
    fs, tau = 20.0, 40.0
    a = np.exp(-1.0 / (fs * tau))
    x = np.array([1.0, 3.0, 2.0, 5.0])
    y = pp.recursive_filter(x, fs, tau, init=0.0)
    ref, prev = [], 0.0
    for v in x:
        prev = a * prev + (1 - a) * v
        ref.append(prev)
    assert np.allclose(y, ref)
    # a step input relaxes with time constant tau: after tau seconds 1 - 1/e of the step
    step = np.ones(int(10 * tau * fs))
    yt = pp.recursive_filter(step, fs, tau, init=0.0)
    assert yt[int(tau * fs) - 1] == pytest.approx(1 - np.exp(-1), rel=2e-3)
    # default start is the series mean; a constant series has no fluctuation
    assert np.allclose(pp.detrend(np.full(100, 7.0), "recursive", fs=fs, tau_s=tau), 0.0)


def test_detrend_filter_needs_fs_and_unknown_method_raises():
    with pytest.raises(ValueError):
        pp.detrend(np.ones(10), "recursive")
    with pytest.raises(ValueError):
        pp.detrend(np.ones(10), "butterworth")
    with pytest.raises(ValueError):
        pp.detrend(np.ones(10), "spline")


def test_taylor_helpers():
    f = np.array([0.1, 1.0, 10.0])
    assert np.allclose(pp.taylor_wavelength(f, 5.0), [50.0, 5.0, 0.5])
    assert np.allclose(pp.taylor_wavenumber(f, 5.0), 2 * np.pi * f / 5.0)
    assert pp.taylor_wavelength(np.array([0.0]), 5.0)[0] == np.inf


def test_taylor_and_rotation_checks():
    rng = np.random.default_rng(1)
    n = 36000
    u = 6.0 + 0.6 * rng.normal(size=n)
    v = 0.0 + 0.5 * rng.normal(size=n)
    w = 0.02 + 0.4 * rng.normal(size=n)
    t = pp.taylor_check(u, v, 0.5)
    assert t["U_mean"] == pytest.approx(6.0, abs=0.05)
    assert t["taylor_ratio"] < 0.2 and t["taylor_ok"] == 1.0
    r = pp.rotation_check(u, v, w)
    assert r["w_mean"] == pytest.approx(0.02, abs=0.01)
    assert abs(r["w_mean_over_sigma_w"]) < 0.1
