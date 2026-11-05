"""MRD module: Vickers & Mahrt worked example, sum closure, FHT identity, gap, module run.

Invariants (library/writeups/ec_mrd.md): sum of D(m) over scales equals the
(co)variance about the record mean exactly (Howell & Mahrt 1997 eq. 6); the
Vickers & Mahrt 2003 Table 1 example reproduces to machine precision.
"""

import warnings

import numpy as np
import pytest

from ec_coherent import mrd
from ec_coherent.config import ECConfig
from ec_coherent.io import open_hf

from ec_helpers import make_hf

FS = 10.0
VM_SERIES = np.array([1.0, 3.0, 2.0, 5.0, 1.0, 2.0, 1.0, 3.0])


def test_vickers_table1_worked_example():
    # Vickers & Mahrt 2003 Table 1: D(1)=1.125, D(2)=0.3125, D(3)=0.25 [CITED]
    D, s, counts = mrd.mr_spectrum(VM_SERIES)
    assert D == pytest.approx([1.125, 0.3125, 0.25], abs=1e-14)
    assert counts.tolist() == [4, 2, 1]
    assert np.isnan(s[-1])                          # one sample at m = M
    var = np.mean((VM_SERIES - VM_SERIES.mean()) ** 2)
    assert D.sum() == pytest.approx(var, abs=1e-14)


def test_fht_matches_direct_block_means():
    # eq. 15 (FHT products) equals eq. 7 (half-window minus parent-window means)
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=256), rng.normal(size=256)
    D, _, _ = mrd.mr_spectrum(x, y)
    M = 8
    for m in range(1, M + 1):
        w = 2 ** m
        parents_x = x.reshape(-1, w).mean(axis=1)
        parents_y = y.reshape(-1, w).mean(axis=1)
        halves_x = x.reshape(-1, w // 2).mean(axis=1)[1::2]   # second half of each parent
        halves_y = y.reshape(-1, w // 2).mean(axis=1)[1::2]
        direct = np.mean((halves_x - parents_x) * (halves_y - parents_y))
        assert D[m - 1] == pytest.approx(direct, abs=1e-12), m


def test_sum_closure_cospectrum():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=1024), rng.normal(size=1024)
    D, _, _ = mrd.mr_spectrum(x, y)
    cov = np.mean((x - x.mean()) * (y - y.mean()))
    assert D.sum() == pytest.approx(cov, abs=1e-12)


def test_scale_localisation():
    # a square wave alternating every 8 samples puts its variance at the 16-sample mode
    x = np.tile(np.r_[np.ones(8), -np.ones(8)], 64)
    D, _, _ = mrd.mr_spectrum(x)
    assert np.argmax(D) == 3                        # m = 4 -> 2^4 = 16 samples
    assert D[3] / D.sum() > 0.99


def test_to_grid_modes():
    x = np.arange(3000, dtype=float)
    t, mult = mrd.to_grid(x, "trim")
    assert len(t) == 2048 and mult == 1.0 and t[-1] == 2047.0
    g, mult = mrd.to_grid(x, "interp")
    assert len(g) == 2048
    assert mult == pytest.approx(2999.0 / 2047.0)   # Vickers 2003 eq. 8
    assert g[0] == 0.0 and g[-1] == pytest.approx(2999.0)  # all points spanned
    p, mult = mrd.to_grid(np.arange(512, dtype=float), "interp")
    assert len(p) == 512 and mult == 1.0            # power of two passes through
    with pytest.raises(ValueError):
        mrd.to_grid(x, "nope")


def test_gap_scale_detects_constructed_gap():
    # turbulence peak at m=3, near-zero gap modes, mesoscale rise at the end
    tau = 2.0 ** np.arange(1, 11)
    D = np.array([0.5, 1.0, 2.0, 1.0, 0.4, 0.001, 0.001, 0.001, 1.5, 2.0])
    g = mrd.gap_scale(tau, D, level_frac=0.01)
    assert np.isfinite(g)
    assert g <= tau[8]                              # fires at/before the mesoscale rise
    # monotone rise to the last scale: no gap
    assert np.isnan(mrd.gap_scale(tau, np.arange(1.0, 11.0)))


GAP_TAU = 2.0 ** np.arange(1, 11)
GAP_D = np.array([0.5, 1.0, 2.0, 1.0, 0.4, 0.001, 0.001, 0.001, 1.5, 2.0])
NOISY_D = np.array([3.0, 0.5, 1.0, 2.0, 1.0, 0.4, 0.001, 0.001, 1.5, 2.0])


@pytest.mark.parametrize("D, tau_min, expected", [
    (GAP_D, None, 64.0),
    # by hand: smoothed peak at tau 8, accumulative flux levels off at tau 64
    (GAP_D, 2.0, 64.0),
    # tau_min at the shortest scale excludes nothing
    (NOISY_D, None, 16.0),
    # noise peak at tau 2 latches; smoothed |D| rises again at tau 16
    (NOISY_D, 8.0, 128.0),
    # scan from tau 8: turbulence peak at tau 16, level-off at tau 128
    (NOISY_D, 4.0, 16.0),
    # smoothing carries the tau 2 noise into tau 4, whose flank peaks
    (np.array([3.0, 0.5, 1.0, 2.0, 1.0, 0.4, 0.06, 0.001, 1.5, 2.0]),
     8.0, 128.0),
    # accumulative flux includes the excluded scales: level-off at tau 128,
    # not 256
])
def test_gap_scale_tau_min_skips_shortest_scales(D, tau_min, expected):
    # exact: tau values are powers of two and the scan returns one of them
    out = mrd.gap_scale(GAP_TAU, D, level_frac=0.01, tau_min=tau_min)
    assert out == expected


# Smoothed |D| (Ds) and accumulative flux (C) worked by hand in each
# comment; tau = 2, 4, 8, ...
@pytest.mark.parametrize("D, level_frac, expected", [
    ([2.0, 1.0, 4.0, 0.0, 0.0, 0.0], 0.01, 16.0),
    # Ds = 2, 2, 2.25, 1, 0, 0: equal neighbours are no peak; peak at tau 8,
    # C flat at tau 16
    ([1.0, 4.0, 1.0, 1.0, 1.0, 1.0], 0.01, np.nan),
    # Ds = 1, 2.5, 1.75, 1, 1, 1: peak at tau 4; flat Ds is no rise and C
    # grows by 1 per scale
    ([2.0, 4.0, 3.0, 0.0, 0.0, 0.0], 0.5, 8.0),
    # Ds = 2, 3.25, 2.5, ...; C = 2, 6, 9: at tau 8 the change 3 equals
    # 0.5 * 6, a level-off
    ([-2.0, -4.0, -3.0, 0.0, 0.0, 0.0], 0.5, 8.0),
    # negative flux: Ds = 2, 3.25, 2.5, ...; C = -2, -6, -9: at tau 8 the
    # change 3 equals 0.5 * |-6|, a level-off
    ([2.0, 4.0, 4.0, 0.0, 0.0, 0.0], 0.5, 16.0),
    # Ds = 2, 3.5, 3, 1, ...; C = 2, 6, 10, 10: change 4 > 0.5 * 6 at tau 8,
    # level-off at tau 16
    ([4.0, 1.0, 2.0, 3.0, 3.0, 3.0], 0.01, 16.0),
    # Ds = 4, 2, 2, 2.75, 3, 3: peak at the shortest scale, tau 2; Ds rises
    # at tau 16 (C = 4, 5, 7, 10 never levels off)
    ([-2.0, 4.0, -2.0, 0.0, 2.0, 0.0], 0.01, 32.0),
    # Ds = 2, 1, 0, 0, 1, 0; C = -2, 2, 0, 0: no level-off measured against
    # C = 0 at tau 16, Ds rises at tau 32
    ([1.0, 2.0, 4.0, 0.0], 0.01, 16.0),
    # Ds = 1, 2.25, 2.5, 0: peak at the second-last scale, C flat at the
    # last scale
    ([1.0, 3.0, 1.0, -1.0, 1.0, 0.0], 0.01, 32.0),
    # signed: smoothing before |.| gives Ds = 1, 2, 1, 0, 0.25, 0 and a rise
    # at tau 32; |.| before smoothing would give 1, 2, 1.5, 1, 0.75, 0 and
    # a level-off at tau 64
    ([2.0, 4.0, np.inf, 0.0, 0.0, 0.0], 0.01, 8.0),
    # inf counts as zero: Ds = 2, 2.5, 1, 0, 0, 0; C = 2, 6, 6: peak at
    # tau 4, C flat at tau 8
    ([2.0, 4.0, np.inf, np.inf, 0.0, 0.0], 0.01, 8.0),
    # two infinite scales, both zero: as above, no overflow in the sum
])
def test_gap_scale_scan_rules(D, level_frac, expected):
    tau = 2.0 ** np.arange(1, len(D) + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = mrd.gap_scale(tau, np.asarray(D), level_frac=level_frac)
    np.testing.assert_equal(out, expected)


@pytest.mark.parametrize("kw, expected", [
    ({}, 32.0),
    # default level_frac 0.01; C = 200, 600, 900, 909.5, 918.1: change 9.5
    # > 9 at tau 16, change 8.6 <= 9.095 at tau 32
    ({"level_frac": 0.011}, 16.0),      # 9.5 <= 9.9: level-off at tau 16
    ({"level_frac": 0.009}, 64.0),      # 8.6 > 8.19 at tau 32; C flat at 64
])
def test_gap_scale_default_level_frac(kw, expected):
    # Ds = 200, 325, 252.4, 81.9, 6.7, 0: peak at tau 4, no later rise
    D = np.array([200.0, 400.0, 300.0, 9.5, 8.6, 0.0])
    tau = 2.0 ** np.arange(1, 7)
    np.testing.assert_equal(mrd.gap_scale(tau, D, **kw), expected)


@pytest.mark.parametrize("tau, D, kw, match", [
    (GAP_TAU, np.tile(GAP_D, (2, 1)), {}, "needs a 1-D D"),
    (GAP_TAU[:-1], GAP_D, {}, "does not match D shape"),
    (GAP_TAU[[0, 2, 1, 3, 4, 5, 6, 7, 8, 9]], GAP_D, {},
     "not strictly ascending"),
    (GAP_TAU, GAP_D, {"level_frac": 1.0}, r"outside \[0, 1\)"),
    (GAP_TAU, GAP_D, {"level_frac": -0.01}, r"outside \[0, 1\)"),
])
def test_gap_scale_rejects(tau, D, kw, match):
    with pytest.raises(ValueError, match=match):
        mrd.gap_scale(tau, D, **kw)


@pytest.mark.parametrize("gaps, expected", [
    ([np.inf, 30.0, 60.0, 90.0], 60.0),
    # infinite gap counts as missing
    ([60.0, np.nan, 30.0, 120.0, 300.0], 90.0),
    # median of four finite values
    ([30.0, 60.0, 90.0], 60.0),
    # exactly three records
    ([60.0, np.nan, 30.0, np.nan], np.nan),
    # two finite records
    ([[30.0, np.nan], [60.0, 10.0], [90.0, 20.0], [np.nan, np.nan]],
     [60.0, np.nan]),
    # per height
    ([[1.0, np.nan]] * 4, [1.0, np.nan]),
    # all-NaN height: NaN without a RuntimeWarning
])
def test_gap_scale_median(gaps, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = mrd.gap_scale_median(gaps)
    np.testing.assert_equal(np.asarray(out), np.asarray(expected))
    assert isinstance(out, float if np.ndim(expected) == 0 else np.ndarray)


def test_relative_error_definition():
    # eq. 12: (std of per-window products / D) / sqrt(count) -- checked at one scale
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=64), rng.normal(size=64)
    D, s, counts = mrd.mr_spectrum(x, y)
    hx = mrd.fht(x.copy())
    hy = mrd.fht(y.copy())
    pos = 1 + np.arange(32) * 2                     # scale m = 1 details
    prods = hx[pos] * hy[pos]
    assert s[0] == pytest.approx(prods.std(ddof=1), abs=1e-12)
    assert counts[0] == 32


def test_run_on_synthetic_file(tmp_path):
    path, _ = make_hf(tmp_path / "hf.nc", fs=FS, window_s=300.0, n_records=3)
    cfg = ECConfig.from_config({})
    with open_hf(path) as hf:
        ds = mrd.run(hf, cfg)
    assert ds.attrs["M"] == 11 and ds.attrs["n_used"] == 2048
    assert ds["mr_scale"].values[0] == pytest.approx(2.0 / FS)
    # closure: cov_uw = sum of D_uw over scales, and negative (helper builds -u'w')
    D_sum = ds["D_uw"].values.sum(axis=-1)
    cov = ds["cov_uw"].values
    fin = np.isfinite(cov)
    assert fin.any()
    assert np.allclose(D_sum[fin], cov[fin], atol=1e-9)
    assert (cov[fin] < 0).all()
    for name in ("D_u_u", "D_w_w", "D_ts_ts", "D_wTs", "D_vw", "err_uw", "gap_wTs", "gap_uw"):
        assert name in ds


def test_run_two_hour_windows(tmp_path):
    # rulings 2026-08-23: longer periods (up to ~2 h) may be analyzed; M follows the window
    path, _ = make_hf(tmp_path / "hf_2h.nc", fs=2.0, window_s=7200.0, n_records=1)
    cfg = ECConfig.from_config({})
    with open_hf(path) as hf:
        ds = mrd.run(hf, cfg)
    assert ds.attrs["M"] == 13 and ds.attrs["n_used"] == 8192   # 14400 samples -> 2^13
    assert ds["mr_scale"].values[-1] == pytest.approx(8192 / 2.0)
    cov = ds["cov_uw"].values
    fin = np.isfinite(cov)
    assert fin.any()
    assert np.allclose(ds["D_uw"].values.sum(axis=-1)[fin], cov[fin], atol=1e-9)


@pytest.mark.parametrize("name", ["gap_wTs", "gap_uw"])
def test_run_passes_gap_tau_min_from_config(tmp_path, name):
    # tau_min above the longest scale leaves no scale to scan: every gap is NaN
    path, _ = make_hf(tmp_path / "hf.nc", fs=FS, window_s=300.0, n_records=3)
    with open_hf(path) as hf:
        free = mrd.run(hf, ECConfig.from_config({}))[name].values
        cfg = ECConfig.from_config({}, mrd={"gap_tau_min": 1e9})
        capped = mrd.run(hf, cfg)[name].values
    assert np.isfinite(free).any()
    assert np.isnan(capped).all()


def test_run_interp_grid(tmp_path):
    path, _ = make_hf(tmp_path / "hf2.nc", fs=FS, window_s=300.0, n_records=2)
    cfg = ECConfig.from_config({}, mrd={"grid": "interp"})
    with open_hf(path) as hf:
        ds = mrd.run(hf, cfg)
    mult = ds.attrs["dt_multiplier"]
    assert mult == pytest.approx(2999.0 / 2047.0)
    assert ds["mr_scale"].values[0] == pytest.approx(2.0 / FS * mult)
    cov = ds["cov_uw"].values
    assert (cov[np.isfinite(cov)] < 0).all()
