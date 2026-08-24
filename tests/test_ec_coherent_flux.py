"""ec_coherent.coherent_flux: conditional averages, triple-decomposition terms, quadrant estimator.

Validation targets from the sources (library/writeups/ec_coherent_flux.md):
the conditional average recovers the embedded pattern (CB 1993 II eq. 3);
F_ej + F_sw = F_cs exactly (TF 2007 p. 322); a periodic coherent train plus
uncorrelated noise gives F_cs equal to the coherent covariance and passes
the 0.8-1.2 representativeness gate (TF 2007 §4.2); the quadrant estimator
reproduces the sign-aware delta_S at H = 0.
"""

import numpy as np
import pytest

from ec_coherent import ECConfig, coherent_flux as cflux, io as ecio
from ec_coherent.cli import run_file
from ec_coherent.quadrant import derived_h0
from ec_helpers import make_hf

FS = 20.0


def coherent_pair(P=20.0, n_periods=60, noise=0.3, seed=0, fs=FS):
    """Sawtooth coherent parts x_l = -y_l plus independent white noise; events at the resets."""
    rng = np.random.default_rng(seed)
    n = int(P * fs * n_periods)
    t = np.arange(n) / fs
    xl = (t % P) / P - 0.5
    x = xl + noise * rng.normal(size=n)
    y = -xl + noise * rng.normal(size=n)
    times = np.arange(1, n_periods) * P
    return x - x.mean(), y - y.mean(), xl, times


def test_conditional_average_recovers_embedded_pattern():
    pat = np.sin(np.linspace(-np.pi, np.pi, 41))
    x = np.zeros(4000)
    times = np.array([50.0, 100.0, 150.0])
    for ti in times:
        i0 = int(ti * FS)
        x[i0 - 20: i0 + 21] += pat
    lag, cx = cflux.conditional_average(x, FS, times, 1.0)
    assert lag.size == 41 and lag[0] == -1.0 and lag[-1] == 1.0
    assert np.allclose(cx, pat)
    # an event too close to the record start: missing lags are NaN, not zero-filled
    _, edge = cflux.conditional_average(x, FS, np.array([0.5]), 1.0)
    assert np.isnan(edge[:10]).all() and np.isfinite(edge[10:]).all()


def test_flux_contribution_recovers_coherent_covariance():
    P = 20.0
    x, y, xl, times = coherent_pair(P)
    fc = cflux.flux_contribution(x, y, FS, times, P / 2)
    cov_l = float(np.mean(xl * -xl))
    assert fc["F_cs"] == pytest.approx(cov_l, rel=0.05)
    assert fc["F_ej"] + fc["F_sw"] == pytest.approx(fc["F_cs"], abs=1e-12)
    assert fc["F_t"] == pytest.approx(fc["F_tot"] - fc["F_cs"], abs=1e-12)
    assert 0.8 <= fc["ratio"] <= 1.2 and fc["valid"]
    assert fc["n"] == times.size
    # eq. 12b with block-detrended series collapses to -n/(n-1) dx dy
    dx = -np.mean(cflux.conditional_average(x, FS, times, P / 2)[1])
    dy = -np.mean(cflux.conditional_average(y, FS, times, P / 2)[1])
    n = x.size
    assert fc["flux_error"] == pytest.approx(-n / (n - 1) * dx * dy, abs=1e-12)


def test_flux_contribution_empty_events_is_nan():
    x, y, _, _ = coherent_pair()
    fc = cflux.flux_contribution(x, y, FS, np.array([]), 10.0)
    assert np.isnan(fc["F_cs"]) and not fc["valid"] and fc["n"] == 0


def test_quadrant_fraction_matches_sign_aware_delta_s():
    rng = np.random.default_rng(3)
    w = rng.normal(size=5000)
    u = -0.5 * w + rng.normal(size=5000)              # downgradient stress, total < 0
    q = cflux.quadrant_fraction(u, w, [0.0], w_is=1)
    d = derived_h0(u, w, w_is=1)
    assert q["F_ej"][0] - q["F_sw"][0] == pytest.approx(d["delta_S"])
    assert q["F_coh"][0] == pytest.approx(q["F_ej"][0] + q["F_sw"][0])
    c = 0.5 * w + rng.normal(size=5000)               # scalar plane, total > 0
    q2 = cflux.quadrant_fraction(w, c, [0.0, 1.0], w_is=0)
    d2 = derived_h0(w, c, w_is=0)
    assert q2["F_ej"][0] - q2["F_sw"][0] == pytest.approx(d2["delta_S"])
    assert q2["F_coh"][1] <= q2["F_coh"][0]           # a larger hole passes fewer samples


def test_haar_reconstruction_and_split_closure():
    rng = np.random.default_rng(4)
    n = 8192
    x = rng.normal(size=n)
    for i in range(1, n):
        x[i] = 0.95 * x[i - 1] + x[i]
    x = x - x.mean()
    W, valid = cflux.haar_coefficients(x)
    assert W.shape == (13, n) and valid[0, :-2].all() and not valid[-1, -1]
    xr = cflux.haar_reconstruct(W)
    core = slice(1024, -1024)                        # away from the zero-padded edges
    assert np.corrcoef(xr[core], x[core])[0, 1] > 0.995
    assert np.var(xr[core]) / np.var(x[core]) == pytest.approx(1.0, abs=0.05)
    xs, xw, xr2 = cflux.turner_split(x, K=4.0)
    assert np.allclose(xs + xw, xr2) and np.allclose(xr2, xr)
    # K = 0 puts every coefficient in the strong set; a huge K none
    s0, w0, r0 = cflux.turner_split(x, K=0.0)
    assert np.allclose(s0, r0) and np.allclose(w0, 0.0)
    s9, _, _ = cflux.turner_split(x, K=1e9)
    assert np.allclose(s9, 0.0)


def test_turner_fraction_separates_isolated_events_from_noise():
    """Turner's target case: strong localized features in weak background (their fig. 2)."""
    rng = np.random.default_rng(5)
    n = 8192
    pulse = np.r_[np.linspace(0, 1, 64), np.linspace(1, 0, 64)]    # isolated triangular event
    xl = np.zeros(n)
    for i0 in (1000, 2500, 4200, 6100, 7300):
        xl[i0: i0 + 128] += pulse
    xl -= xl.mean()
    x = xl + 0.05 * rng.normal(size=n)
    y = xl + 0.05 * rng.normal(size=n)
    tf = cflux.turner_fraction(x, y, K=1.0)
    assert 0.9 <= tf["ratio"] <= 1.1                 # reconstruction holds the covariance
    cov_frac_l = np.mean(xl * xl) / np.mean(x * y)   # the events' share of the covariance
    assert tf["frac"] == pytest.approx(cov_frac_l, abs=0.1)
    # the fraction shrinks monotonically as the threshold rises (their figs. 3-4 sensitivity)
    fracs = [cflux.turner_fraction(x, y, K)["frac"] for K in (0.5, 1.0, 2.0, 4.0)]
    assert all(a >= b for a, b in zip(fracs, fracs[1:]))
    # a correlated but structureless pair scores far lower at the same threshold
    # (K = 2: structured events keep ~0.56 here, Gaussian coefficients only ~0.09)
    a, b = rng.normal(size=n), rng.normal(size=n)
    t2 = cflux.turner_fraction(x, y, K=2.0)
    tn = cflux.turner_fraction(a, b + 0.3 * a, K=2.0)
    assert abs(tn["frac"]) < 0.5 * t2["frac"]


def test_tke_event_set_is_rejected():
    cfg = ECConfig.from_config(modules=("coherent_flux",),
                               coherent_flux={"event_signals": ("u", "e")})
    with pytest.raises(ValueError, match="TKE"):
        cflux.run(None, cfg)


def test_module_run_writes_the_coherent_flux_group(tmp_path):
    path, truth = make_hf(tmp_path / "SYN_hf_GPF_ConstDet_2023_07_06.nc")
    cfg = ECConfig.from_config(modules=("coherent_flux",),
                               ramps={"a_max_s": 100.0, "D_min_s": 0.0})
    out = run_file(path, cfg)
    ds = ecio.read_group(out, "coherent_flux")
    assert ds.attrs["event_signals"] == "u,Ts" and ds.attrs["window"] == "duration"
    assert (ds["n_structures_u"] > 0).all()
    assert np.isfinite(ds["half_window_u"]).all()
    for k in ("uw", "wTs", "wrhov"):
        assert ds[f"F_frac_{k}_u"].dims == ("record", "height")
        assert ds[f"F_ej_quad_{k}"].dims == ("record", "height", "hole")
        assert ds[f"{'F_tot'}_{k}_u"].attrs["method"] == "wavelet"
        assert ds[f"F_coh_quad_{k}"].attrs["method"] == "quadrant"
    assert list(ds["hole"].values) == [0.0, 0.5, 1.0]
    # identities where the wavelet estimator ran
    for s in ("u", "Ts"):
        cs, ej, sw = (ds[f"{f}_wTs_{s}"].values for f in ("F_cs", "F_ej", "F_sw"))
        fin = np.isfinite(cs)
        assert fin.any()
        assert np.allclose((ej + sw)[fin], cs[fin], atol=1e-9)
        frac = ds[f"F_frac_wTs_{s}"].values
        assert np.allclose(frac[fin], (cs / ds[f"F_tot_wTs_{s}"].values)[fin], equal_nan=True)
    # quadrant estimator at H = 0 matches the quadrant module's sign-aware split
    qds = ecio.read_group(run_file(path, cfg.with_(modules=("quadrant",))), "quadrant")
    dS = qds["delta_S_uw"].values
    dq = ds["F_ej_quad_uw"].values[:, :, 0] - ds["F_sw_quad_uw"].values[:, :, 0]
    fin = np.isfinite(dS) & np.isfinite(dq)
    assert fin.any() and np.allclose(dq[fin], dS[fin], atol=1e-6)


def test_module_run_fixed_window_and_turner(tmp_path):
    path, _ = make_hf(tmp_path / "SYN_hf_GPF_ConstDet_2023_07_06.nc", n_records=2)
    cfg = ECConfig.from_config(modules=("coherent_flux",),
                               ramps={"a_max_s": 100.0, "D_min_s": 0.0},
                               coherent_flux={"window": "fixed", "window_s": 20.0,
                                              "event_signals": ("u",), "pairs": ("uw",),
                                              "turner": True, "turner_K": 2.0})
    ds = ecio.read_group(run_file(path, cfg), "coherent_flux")
    hw = ds["half_window_u"].values
    assert np.all(hw[np.isfinite(hw)] == 10.0)
    assert ds.attrs["window"] == "fixed" and ds.attrs["window_s"] == 20.0
    assert ds.attrs["turner"] == 1 and ds.attrs["turner_K"] == 2.0
    assert ds["F_frac_turner_uw"].attrs["method"] == "turner"
    assert np.isfinite(ds["ratio_turner_uw"]).all()
    # the default config carries no turner variables
    cfg0 = ECConfig.from_config(modules=("coherent_flux",),
                                ramps={"a_max_s": 100.0, "D_min_s": 0.0},
                                coherent_flux={"event_signals": ("u",), "pairs": ("uw",)})
    ds0 = ecio.read_group(run_file(path, cfg0), "coherent_flux")
    assert ds0.attrs["turner"] == 0 and "F_frac_turner_uw" not in ds0
