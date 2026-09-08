"""Scale-separation module: band edges, exact band-fraction closure, module run.

Invariant (library/writeups/ec_scales.md): the three band fractions of variance
and covariance sum to 1 exactly (full-record boxcar periodogram band sums).
"""

import numpy as np
import pytest

from ec_coherent import scales, spectra as sp
from ec_coherent.config import ECConfig, ScalesConfig
from ec_coherent.io import open_hf

from ec_helpers import make_hf

FS = 10.0
N = 3000


def test_band_edges_modes():
    x = np.random.default_rng(0).normal(size=N)
    l1, l2, src = scales.band_edges(ScalesConfig(cutoff_mode="delta", delta_m=1000.0),
                                    x, FS, 5.0, 10.0)
    assert l2 == pytest.approx(np.pi * 1000.0)          # Balakumar & Adrian: pi delta
    assert l1 == pytest.approx(l2 / 10.0)               # one decade below
    assert src == 2
    l1, l2, src = scales.band_edges(ScalesConfig(cutoff_mode="scaled", z_mult_small=5.0,
                                                 z_mult_vlsm=50.0), x, FS, 5.0, 10.0)
    assert (l1, l2, src) == (50.0, 500.0, 3)
    with pytest.raises(ValueError):
        scales.band_edges(ScalesConfig(cutoff_mode="nope"), x, FS, 5.0, 10.0)


def test_band_fractions_sum_to_one_and_localise_known_lines():
    t = np.arange(N) / FS
    U = 5.0
    # two sines: lambda = U/f = 500 m (vlsm side) and 5 m (small side)
    x = 2.0 * np.sin(2 * np.pi * 0.01 * t) + 1.0 * np.sin(2 * np.pi * 1.0 * t)
    spec = sp.spectrum(x, FS)
    pos = spec.f > 0
    frac, tot = scales.band_fractions(spec.f[pos], spec.S[pos], U, 50.0, 100.0)
    assert np.sum(frac) == pytest.approx(1.0, abs=1e-12)
    # variance ratio 4:1 -> fractions 0.8 (vlsm) and 0.2 (small)
    assert frac[2] == pytest.approx(0.8, abs=0.01)
    assert frac[0] == pytest.approx(0.2, abs=0.01)
    assert frac[1] == pytest.approx(0.0, abs=0.01)


def test_run_closure_on_synthetic_file(tmp_path):
    path, _ = make_hf(tmp_path / "hf.nc", fs=FS, window_s=300.0, n_records=3)
    cfg = ECConfig.from_config({}, scales={"cutoff_mode": "scaled",
                                           "z_mult_small": 3.0, "z_mult_vlsm": 30.0})
    with open_hf(path) as hf:
        ds = scales.run(hf, cfg)
    for name in ("var_frac_u", "var_frac_w", "var_frac_ts", "flux_frac_uw", "flux_frac_wTs"):
        arr = ds[name].values
        s = arr.sum(axis=-1)
        assert np.allclose(s[np.isfinite(s)], 1.0, atol=1e-9), name
    # totals match the perturbation variance/covariance closure targets
    assert (ds["var_u"].values > 0).all()
    assert (ds["cov_uw"].values < 0).all()               # helper builds negative u'w'
    assert (ds["cutoff_source"].values == 3).all()
    z = np.array([3.0, 10.0])
    assert np.allclose(ds["lambda_small_lsm"].values, 3.0 * z[None, :])
    assert np.allclose(ds["lambda_lsm_vlsm"].values, 30.0 * z[None, :])


def test_run_delta_mode_edges(tmp_path):
    path, _ = make_hf(tmp_path / "hf2.nc", fs=FS, window_s=300.0, n_records=2)
    cfg = ECConfig.from_config({}, scales={"cutoff_mode": "delta", "delta_m": 200.0})
    with open_hf(path) as hf:
        ds = scales.run(hf, cfg)
    assert np.allclose(ds["lambda_lsm_vlsm"].values, np.pi * 200.0)
    assert np.allclose(ds["lambda_small_lsm"].values, np.pi * 20.0)
    assert (ds["cutoff_source"].values == 2).all()
