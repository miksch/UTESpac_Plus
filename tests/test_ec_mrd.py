"""MRD module: Vickers & Mahrt worked example, sum closure, FHT identity, gap, module run.

Invariants (library/writeups/ec_mrd.md): sum of D(m) over scales equals the
(co)variance about the record mean exactly (Howell & Mahrt 1997 eq. 6); the
Vickers & Mahrt 2003 Table 1 example reproduces to machine precision.
"""

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
    for name in ("D_uu", "D_ww", "D_TsTs", "D_wTs", "D_vw", "err_uw", "gap_wTs", "gap_uw"):
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
