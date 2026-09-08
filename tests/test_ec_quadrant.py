"""Quadrant/octant module: H=0 closure, hole conventions, UTESpac cross-check, module run.

Invariants (library/writeups/ec_quadrant.md): quadrant flux fractions sum to
1 at H = 0 (Raupach 1981 eq. 4); octant fractions sum to 1 per component; the
sign-aware delta_S / delta_D / eta reproduce UTESpac's find_delta_flux /
find_delta_time / find_eta exactly (integration notes A.3).
"""

import numpy as np
import pytest

from ec_coherent import quadrant as qd
from ec_coherent.config import ECConfig
from ec_coherent.io import open_hf
from utespac.find_delta_flux import find_delta_flux
from utespac.find_delta_time import find_delta_time
from utespac.find_eta import find_eta

from ec_helpers import make_hf

FS = 10.0


def _correlated(n=4000, seed=0, r=-0.4):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = r * a + np.sqrt(1 - r * r) * rng.normal(size=n)
    return a, b


def test_h0_fractions_sum_to_one():
    u, w = _correlated()
    st = qd.quadrant_stats(u, w, [0.0, 2.0], "rms")
    assert st["S"][:, 0].sum() == pytest.approx(1.0, abs=1e-12)   # Raupach 1981 eq. 4
    assert st["T"][:, 0].sum() == pytest.approx(1.0, abs=1e-12)
    assert st["count"][:, 0].sum() == 4000
    # hole removes low-|p| samples: time fractions shrink, fractions no longer sum to 1
    assert st["T"][:, 1].sum() < st["T"][:, 0].sum()


def test_known_quadrant_content():
    # all samples in Q2 of the (u, w) plane: u < 0, w > 0
    u = -np.abs(np.random.default_rng(1).normal(size=100)) - 0.1
    w = np.abs(np.random.default_rng(2).normal(size=100)) + 0.1
    st = qd.quadrant_stats(u, w, [0.0], "rms")
    assert st["S"][1, 0] == pytest.approx(1.0)
    assert st["T"][1, 0] == 1.0
    assert st["S"][[0, 2, 3], 0].sum() == 0.0


def test_hole_norm_equivalence():
    # Raupach 1981 p. 365: H_flux = |rho| * H_rms selects the same samples
    u, w = _correlated(seed=3)
    p = u * w
    rho = abs(np.mean(p) / (u.std() * w.std()))
    st_rms = qd.quadrant_stats(u, w, [2.0], "rms")
    st_flux = qd.quadrant_stats(u, w, [2.0 / rho], "flux")
    assert np.allclose(st_rms["S"][:, 0], st_flux["S"][:, 0], atol=1e-12)
    assert (st_rms["count"][:, 0] == st_flux["count"][:, 0]).all()
    with pytest.raises(ValueError):
        qd.quadrant_stats(u, w, [0.0], "nope")


def test_large_hole_leaves_only_extremes():
    u, w = _correlated(seed=4)
    st = qd.quadrant_stats(u, w, [0.0, 50.0], "rms")
    assert st["count"][:, 1].sum() == 0                 # nothing survives H = 50
    assert np.all(st["S"][:, 1] == 0.0)


def test_derived_matches_utespac():
    # A.3 cross-check, both flux signs (advective sites flip H's sign)
    for seed, r in ((5, -0.4), (6, +0.35)):
        w, c = _correlated(seed=seed, r=r)
        d = qd.derived_h0(w, c, w_is=0)
        assert d["delta_S"] == pytest.approx(find_delta_flux(w, c), abs=1e-12)
        assert d["delta_D"] == pytest.approx(find_delta_time(w, c), abs=1e-12)
        assert d["eta"] == pytest.approx(find_eta(w, c), abs=1e-12)
        assert d["exuberance"] == pytest.approx(d["eta"] - 1.0, abs=1e-12)
    # uw plane: w is the second axis
    u, w = _correlated(seed=7)
    d = qd.derived_h0(u, w, w_is=1)
    assert d["delta_S"] == pytest.approx(find_delta_flux(w, u), abs=1e-12)


def test_derived_with_nans():
    w, c = _correlated(seed=8)
    w[::10] = np.nan
    d = qd.derived_h0(w, c, w_is=0)
    assert d["delta_S"] == pytest.approx(find_delta_flux(w, c), abs=1e-12)
    assert d["delta_D"] == pytest.approx(find_delta_time(w, c), abs=1e-12)


def test_octant_fractions_sum_to_one():
    rng = np.random.default_rng(9)
    u, w, c = rng.normal(size=(3, 5000))
    st = qd.octant_stats(u, w, c)
    for k in ("F_uw", "F_uc", "F_wc"):
        assert st[k].sum() == pytest.approx(1.0, abs=1e-12)
    assert st["T"].sum() == pytest.approx(1.0, abs=1e-12)
    assert st["count"].sum() == 5000


def test_octant_o2_o8_dominate_sheared_heated():
    # -u'w' shear with warm updrafts: hot ejections (O2) and cold sweeps (O8) carry u'w'
    rng = np.random.default_rng(10)
    w = rng.normal(size=8000)
    u = -0.6 * w + 0.5 * rng.normal(size=8000)
    c = 0.6 * w + 0.5 * rng.normal(size=8000)
    st = qd.octant_stats(u, w, c)
    F = st["F_uw"]
    assert F[1] + F[7] > 0.8 * F.sum()                  # O2 + O8 dominant (Li & Bo 2019)


def test_run_on_synthetic_file(tmp_path):
    path, _ = make_hf(tmp_path / "hf.nc", fs=FS, window_s=300.0, n_records=3)
    cfg = ECConfig.from_config({})
    with open_hf(path) as hf:
        ds = qd.run(hf, cfg)
        do = qd.run_octant(hf, cfg)
    s0 = ds["S_frac_uw"].isel(hole=0).values.sum(axis=-1)
    fin = np.isfinite(s0)
    assert fin.any()
    assert np.allclose(s0[fin], 1.0, atol=1e-9)
    # helper builds negative u'w' and positive w'Ts': ejections dominate both, sign-aware
    dS = ds["delta_S_uw"].values
    assert np.isfinite(dS).any()
    eta = ds["eta_uw"].values
    assert ((eta[np.isfinite(eta)] > 0) & (eta[np.isfinite(eta)] <= 1)).all()
    ex = ds["exuberance_uw"].values
    assert np.allclose(ex[np.isfinite(ex)], eta[np.isfinite(eta)] - 1.0, atol=1e-12)
    assert ds.attrs["hole_norm"] == "rms"
    # octants: both default triplets, tagged variables, closure per component
    assert do.attrs["octant_triplets"] == "u,w,ts; w,ts,rho_h2o"
    for name in ("flux_frac_u_w_ts_uw", "flux_frac_u_w_ts_uts", "flux_frac_u_w_ts_wts",
                 "flux_frac_w_ts_rho_h2o_wts", "flux_frac_w_ts_rho_h2o_wrho_h2o",
                 "flux_frac_w_ts_rho_h2o_tsrho_h2o"):
        f = do[name].values.sum(axis=-1)
        assert np.allclose(f[np.isfinite(f)], 1.0, atol=1e-9), name


def test_run_octant_single_triplet_and_missing_signal(tmp_path):
    path, _ = make_hf(tmp_path / "hf2.nc", fs=FS, window_s=300.0, n_records=2,
                      with_scalars=False)                 # no rho_h2o in the file
    cfg = ECConfig.from_config({})
    with open_hf(path) as hf:
        do = qd.run_octant(hf, cfg)
    assert do.attrs["octant_triplets"] == "u,w,ts"   # rho_h2o triplet skipped
    assert "flux_frac_u_w_ts_uw" in do and "flux_frac_w_ts_rho_h2o_wts" not in do
