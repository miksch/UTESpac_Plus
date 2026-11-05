"""ec_coherent.spectra: closure identities, ogive, log-binning, Kaimal curves, module run."""

import warnings

import numpy as np
import pytest

from ec_coherent import ECConfig, io as ecio, spectra as spx
from ec_coherent.cli import run_file
from ec_helpers import make_hf


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(3)
    n = 36000
    w = rng.normal(size=n)
    for i in range(1, n):
        w[i] = 0.9 * w[i - 1] + rng.normal()
    u = -0.4 * w + rng.normal(size=n)
    return u - u.mean(), w - w.mean()


def test_periodogram_closes_parseval_exactly_with_boxcar(series):
    u, w = series
    sp = spx.spectrum(u, 20.0, taper="boxcar")
    assert spx.integrate(sp.f, sp.S) == pytest.approx(np.var(u), rel=1e-6)
    cs = spx.spectrum(u, 20.0, y=w, taper="boxcar")
    assert spx.integrate(cs.f, cs.S) == pytest.approx(np.mean(u * w), rel=1e-6)
    assert cs.S.dtype.kind == "c"                       # cross: Co = real, Qu = imag
    assert sp.f[0] == 0.0 and sp.f[-1] == pytest.approx(10.0)
    assert sp.f[1] == pytest.approx(1.0 / 1800.0)       # the 1/T line survives


def test_hann_and_welch_close_within_tolerance(series):
    u, _ = series
    sp = spx.spectrum(u, 20.0, taper="hann")
    assert spx.integrate(sp.f, sp.S) == pytest.approx(np.var(u), rel=0.1)
    sw = spx.spectrum(u, 20.0, taper="hann", nperseg=4096)
    assert spx.integrate(sw.f, sw.S) == pytest.approx(np.var(u), rel=0.1)
    assert sw.f[1] == pytest.approx(20.0 / 4096)


def test_ogive_runs_from_the_nyquist_bin_to_the_covariance(series):
    u, w = series
    cs = spx.spectrum(u, 20.0, y=w, taper="boxcar")
    og = spx.ogive(cs.f, cs.S)
    assert og[-1] == pytest.approx(np.real(cs.S[-1]) * (cs.f[1] - cs.f[0]))
    assert og[0] == pytest.approx(np.mean(u * w), rel=1e-6)
    assert og[0] == pytest.approx(spx.integrate(cs.f, cs.S), rel=1e-9)


DF = 1.0 / 7200.0               # 2-h record
F_OG = np.arange(3601) * DF
F_AVG = 4 * DF                  # 30-min averaging period


def _cospectrum(below=0.0, inside=0.0):
    """Cospectrum with *below* on the lines under F_AVG and *inside* on 300
    lines above it."""
    co = np.zeros_like(F_OG)
    co[1:4] = below
    co[100:400] = inside
    return co


@pytest.mark.parametrize(
    "co, abs_floor, tol, ratio, converged, undetermined", [
        (_cospectrum(inside=1e-2), 1e-6, 0.1, 1.0, True, False),
        # all covariance above F_AVG
        (-_cospectrum(inside=1e-2), 1e-6, 0.1, 1.0, True, False),
        # negative flux clears the floor
        (_cospectrum(below=1.0, inside=1e-2), None, 0.1, 0.5, False, False),
        # 3 lines x 1 = 300 lines x 0.01
        (_cospectrum(below=1.0, inside=1e-2), None, 0.6, 0.5, True, False),
        # |1 - 0.5| within tol 0.6
        (_cospectrum(inside=1e-12), 1e-6, 0.1, np.nan, False, True),
        # near-zero flux under the floor
        (_cospectrum(), None, 0.1, np.nan, False, True),
        # zero ogive, no floor
    ])
def test_ogive_convergence(co, abs_floor, tol, ratio, converged,
                           undetermined):
    out = spx.ogive_convergence(F_OG, spx.ogive(F_OG, co), F_AVG, tol=tol,
                                abs_floor=abs_floor)
    # constructed sums are exact to rounding; 1e-12 is far below a 10 %
    # change
    np.testing.assert_allclose(out[0], ratio, atol=1e-12)
    assert out[1:] == (converged, undetermined)


def test_ogive_convergence_non_finite_at_f_avg_is_undetermined():
    og = spx.ogive(F_OG, _cospectrum(inside=1e-2))
    og[4] = np.nan          # F_AVG sits on line 4; Og(f_min) stays finite
    ratio, converged, undetermined = spx.ogive_convergence(
        F_OG, og, F_AVG, abs_floor=1e-6)
    assert np.isnan(ratio) and (converged, undetermined) == (False, True)


F5 = np.arange(5.0)


@pytest.mark.parametrize(
    "og, f_avg, kw, ratio, converged, undetermined", [
        ([2.0, 2.0, 1.0, 1.0, 0.0], 2.0, {"tol": 0.5}, 0.5, True, False),
        # |1 - ratio| == tol
        ([1.0, 1.0, 2.0, 2.0, 0.0], 2.0, {}, 2.0, False, False),
        # ratio above 1 + tol
        ([16.0, 16.0, 14.0, 0.0, 0.0], 2.0, {}, 0.875, False, False),
        # 0.125 off: outside tol 0.1
        ([16.0, 16.0, 15.0, 0.0, 0.0], 2.0, {}, 0.9375, True, False),
        # 0.0625 off: inside tol 0.1
        ([4.0, 4.0, 4.0, 2.0, 0.0], 2.5, {}, 0.75, False, False),
        # linear between lines 2 and 3
        ([4.0, 4.0, 4.0, 2.0, 0.0], 0.0, {}, 1.0, True, False),
        # f_avg on the first line
        ([4.0, 4.0, 4.0, 2.0, 0.0], 4.0, {}, 0.0, False, False),
        # f_avg on the last line
        ([1.0, 1.0, 1.0, 0.0, 0.0], 1.0, {"abs_floor": 1.0}, 1.0, True,
         False),
        # |Og(f_min)| at the floor
        ([np.nan, 1.0, 1.0, 1.0, 0.0], 2.0, {}, np.nan, False, True),
        # non-finite Og(f_min)
        (np.array([2.0, 2.0, 1.0, 1.0, 0.0]) + 5j, 2.0, {}, 0.5, False,
         False),
        # real part only
    ])
def test_ogive_convergence_on_a_constructed_ogive(og, f_avg, kw, ratio,
                                                  converged, undetermined):
    # dyadic inputs: every ratio and comparison is exact in floating point
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = spx.ogive_convergence(F5, og, f_avg, **kw)
    np.testing.assert_equal(out[0], ratio)
    assert out[1:] == (converged, undetermined)


AXIS = "strictly ascending 1-D axis matching Og"


@pytest.mark.parametrize("f, og, f_avg, match", [
    (F_OG, None, 2 * F_OG[-1], "outside the ogive frequency axis"),
    (F_OG, None, -DF, "outside the ogive frequency axis"),
    (F_OG[[0, 1, 3, 2, 4, 5]], None, F_AVG, AXIS),   # interior lines swapped
    (F_OG[[0, 1, 1, 2, 3, 4]], None, F_AVG, AXIS),   # repeated line
    (np.r_[F_OG[:3], np.nan, F_OG[4:6]], None, F_AVG, AXIS),  # NaN line
    (np.array([]), None, F_AVG, AXIS),                # empty axis
    (F5, np.ones((2, 5)), 2.0, AXIS),                 # 2-D ogive
])
def test_ogive_convergence_rejects(f, og, f_avg, match):
    og = np.linspace(1.0, 0.0, len(f)) if og is None else og
    with pytest.raises(ValueError, match=match):
        spx.ogive_convergence(f, og, f_avg)


def test_log_bin_averages_and_conserves_the_sum_of_counts(series):
    u, _ = series
    sp = spx.spectrum(u, 20.0, taper="boxcar")
    edges = spx.log_bins(sp.f, 10)
    fc, Sb, cnt = spx.log_bin(sp.f, sp.S, edges)
    assert cnt.sum() == (sp.f > 0).sum() and cnt.min() >= 1
    assert np.all(np.diff(fc) > 0) and fc[0] == pytest.approx(sp.f[1])
    # edges sit on the half-line grid, so the band sum of the binned density is exact
    assert np.allclose(np.diff(edges) / (sp.f[1] - sp.f[0]), cnt)
    assert np.sum(Sb * np.diff(edges)) == pytest.approx(np.var(u), rel=1e-6)


def test_kaimal_curves_match_the_printed_values():
    # eq. 21a-g, Kaimal et al. 1972 p. 579, evaluated at f = 1: hand values
    f = np.array([1.0])
    assert spx.kaimal_neutral(f, "u")[0] == pytest.approx(105 / 34 ** (5 / 3))
    assert spx.kaimal_neutral(f, "v")[0] == pytest.approx(17 / 10.5 ** (5 / 3))
    assert spx.kaimal_neutral(f, "w")[0] == pytest.approx(2 / 6.3)
    assert spx.kaimal_neutral(np.array([0.1]), "T")[0] == pytest.approx(5.34 / 3.4 ** (5 / 3))
    assert spx.kaimal_neutral(np.array([0.2]), "T")[0] == pytest.approx(4.88 / 3.5 ** (5 / 3))
    assert spx.kaimal_neutral(f, "uw")[0] == pytest.approx(14 / 10.6 ** 2.4)
    assert spx.kaimal_neutral(np.array([0.5]), "wT")[0] == pytest.approx(5.5 / 7.65 ** 1.75)
    assert spx.kaimal_neutral(np.array([2.0]), "wT")[0] == pytest.approx(8.8 / 8.6 ** 2.4)
    assert spx.kaimal_neutral(f, "uT")[0] == pytest.approx(40 / 15 ** 2.6)
    with pytest.raises(ValueError):
        spx.kaimal_neutral(f, "q")


def test_module_run_on_synthetic_file_closes_and_writes(tmp_path):
    path, truth = make_hf(tmp_path / "SYN_hf_GPF_ConstDet_2024_06_01.nc", nan_block=(2, 0, 2000))
    cfg = ECConfig.from_config()
    out = run_file(path, cfg)
    ds = ecio.read_group(out, "spectra")
    assert ds.attrs["taper"] == "boxcar" and ds.attrs["nperseg"] == 0
    assert ds["S_u"].dims == ("record", "height", "frequency")
    n_win = truth["n_win"]
    for r in (0, 1, 3):
        for ih in (0, 1):
            u = truth["u_pf"][r * n_win:(r + 1) * n_win, ih]
            w = truth["w_pf"][r * n_win:(r + 1) * n_win, ih]
            up, wp = u - u.mean(), w - w.mean()
            assert float(ds["var_u"][r, ih]) == pytest.approx(np.var(up), rel=1e-4)
            assert float(ds["cov_uw"][r, ih]) == pytest.approx(np.mean(up * wp), rel=1e-3)
            # ogive at the lowest bin recovers the covariance to the bin interpolation
            assert float(ds["ogive_uw"][r, ih, 0]) == pytest.approx(np.mean(up * wp), rel=0.05)
            assert float(ds["U_mean"][r, ih]) == pytest.approx(u.mean(), rel=1e-3)
    # record 2 has 2/3 of u missing: rejected by the NaN policy, w still processed
    assert np.isnan(ds["S_u"][2, 0]).all() and np.isnan(ds["cov_uw"][2, 0])
    assert np.isfinite(ds["S_w"][2, 0]).any()
    assert float(ds["nan_filled_frac"][2, 0]) == pytest.approx(2000 / n_win)
    # scalars on their own height dim came through the nearest-height map
    assert np.isfinite(ds["S_rho_h2o"][0, 1]).any() and np.isfinite(ds["Co_wTs"][0, 0]).any()
