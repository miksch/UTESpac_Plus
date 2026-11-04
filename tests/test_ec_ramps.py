"""ec_coherent.ramps: MHAT transform, wavelet variance, duration scale, zero-crossing detection."""

import numpy as np
import pytest

from ec_coherent import ECConfig, io as ecio, ramps
from ec_coherent.cli import run_file
from ec_helpers import make_hf


def ramp_train(fs=20.0, ramp_s=30.0, n_periods=40, noise=0.0, seed=0, sign=-1.0):
    """Collineau & Brunet Table II geometry: ramps of length *ramp_s* separated by flat
    intervals of the same length (period 2 ramp_s). sign=-1: gradual rise, sharp drop at
    the end of each ramp (unstable temperature ramp); sign=+1 the stable mirror image."""
    rng = np.random.default_rng(seed)
    P = 2 * ramp_s
    n = int(P * fs * n_periods)
    t = np.arange(n) / fs
    phase = t % P
    x = np.where(phase < ramp_s, phase / ramp_s, 0.0)     # rises over ramp_s, drops to 0, flat
    x = -sign * x
    x = x + noise * rng.normal(size=n)
    return t, x - x.mean()


def test_mhat_matches_table_I_and_cwt_is_the_covariance_form():
    assert ramps.mhat(np.array([0.0]))[0] == 1.0
    assert ramps.mhat(np.array([1.0, -1.0])).tolist() == [0.0, 0.0]
    assert ramps.D_G["mhat"] == pytest.approx(np.pi / np.sqrt(2))
    # a step at t0 gives a zero-crossing of T_1 at t0 for every scale (CB 1993 I §3.1)
    fs = 20.0
    x = np.r_[np.zeros(2000), np.ones(2000)]
    T = ramps.cwt(x, np.array([2.0, 5.0, 10.0]), fs)
    for row in T:
        zc = ramps.zero_crossings(row, fs, "positive")
        assert np.any(np.abs(zc - 2000 / fs) < 1.0 / fs)
    # eq. 4 for a unit step at t0: T_1(a, b) = -y exp(-y^2/2), y = (t0 - b)/a  (analytic)
    a, b, t0 = 3.0, 95.0, 2000 / fs
    y = (t0 - b) / a
    T3 = ramps.cwt(x, np.array([a]), fs)[0]
    assert T3[int(b * fs)] == pytest.approx(-y * np.exp(-0.5 * y * y), rel=1e-2)   # half-sample edge discretisation
    # RAMP and HAAR (first derivative-like) peak at the step: |T_1| maximal at t0
    for wname in ("ramp", "haar"):
        Tw = ramps.cwt(x, np.array([a]), fs, wname)[0][:3900]      # the zero-padded end is a step too
        assert abs(int(np.argmax(np.abs(Tw))) - 2000) <= 1


def test_duration_scale_recovers_ramp_length_within_cb_table_II_tolerance():
    fs, L = 20.0, 30.0
    t, x = ramp_train(fs, L, n_periods=40)
    scales = ramps.log_scales(1.0, 300.0, 16)
    T = ramps.cwt(x, scales, fs)
    W = ramps.wavelet_variance(T, fs)
    a0, D, i = ramps.duration_scale(W, scales, peak="global")
    assert D == pytest.approx(L, rel=0.1)             # CB Table II: D within 7 % of the ramp length
    a0s, Ds, _ = ramps.duration_scale(W, scales, peak="smallest_scale")
    assert Ds == pytest.approx(L, rel=0.1)
    # a back-to-back sawtooth (ramp length = period) gives D = half the period: one
    # "elementary event" per half cycle (CB 1993 I p. 369)
    saw = -(((t % L) / L) - 0.5)
    Wsaw = ramps.wavelet_variance(ramps.cwt(saw - saw.mean(), scales, fs), fs)
    assert ramps.duration_scale(Wsaw, scales, peak="global")[1] == pytest.approx(L / 2, rel=0.1)
    # D_min excludes small scales from the search; a monotone variance has no interior peak
    assert ramps.duration_scale(W, scales, D_min_s=L * 1.5)[0] is np.nan or np.isnan(
        ramps.duration_scale(W, scales, D_min_s=L * 1.5)[0])
    assert np.isnan(ramps.duration_scale(np.arange(len(scales), dtype=float), scales)[0])
    with pytest.raises(ValueError):
        ramps.duration_scale(W, scales, peak="largest")


def test_detect_finds_every_microfront_with_the_right_slope():
    fs, L, n_per = 20.0, 30.0, 40
    P = 2 * L
    t, x = ramp_train(fs, L, n_periods=n_per, noise=0.05, seed=1)
    scales = ramps.log_scales(1.0, 300.0, 16)
    ev = ramps.detect(x, fs, scales, slope="negative")
    expect = np.arange(0, n_per) * P + L                  # microfront at the end of each ramp
    lo, hi = 3 * ev.a0, x.size / fs - 3 * ev.a0
    expect = expect[(expect >= lo) & (expect <= hi)]
    assert ev.n == len(expect)
    # the MHAT zero-crossing at a0 sits ~0.35 a0 after the microfront on this ideal train
    # (ec_ramps.md, "Time localization"); refine="ramp" puts it on the front
    off = np.sort(ev.times) - expect
    assert np.all(off > 0) and np.allclose(off, 0.35 * ev.a0, atol=0.1 * ev.a0)
    assert ev.mean_spacing == pytest.approx(P, rel=0.02)
    evr = ramps.detect(x, fs, scales, slope="negative", refine="ramp")
    assert evr.n == len(expect) and np.allclose(np.sort(evr.times), expect, atol=2.0 / fs)
    evh = ramps.detect(x, fs, scales, slope="negative", refine="haar")
    assert np.allclose(np.sort(evh.times), expect, atol=2.0 / fs)
    # the opposite slope sign finds nothing at the microfronts of an unstable ramp
    ev2 = ramps.detect(x, fs, scales, slope="positive")
    assert not np.any(np.min(np.abs(ev2.times[:, None] - expect[None, :]), axis=0) < 0.5) if ev2.n else True
    # an inverse (stable) ramp flips the slope sign
    t, xi = ramp_train(fs, L, n_periods=n_per, noise=0.05, seed=2, sign=+1.0)
    evi = ramps.detect(xi, fs, scales, slope="positive")
    assert evi.n == len(expect)


def test_zero_crossings_interpolate_and_validate_slope():
    T = np.array([1.0, 0.5, -0.5, -1.0, 0.0, 1.0])
    zc = ramps.zero_crossings(T, 1.0, "negative")
    assert zc.tolist() == [1.5]
    assert ramps.zero_crossings(T, 1.0, "positive").size == 0   # exact zero at index 4 is not a sign change pair (0 sign)
    assert ramps.zero_crossings(T, 1.0, "both").tolist() == [1.5]
    with pytest.raises(ValueError):
        ramps.zero_crossings(T, 1.0, "up")


def test_slope_auto_follows_the_heat_flux_sign():
    assert ramps._slope_for("ts", "auto", 0.1) == "negative"
    assert ramps._slope_for("ts", "auto", -0.1) == "positive"
    assert ramps._slope_for("ts", "auto", np.nan) == "negative"
    assert ramps._slope_for("u_pf", "auto", -0.1) == "positive"
    assert ramps._slope_for("ts", "both", 0.1) == "both"


def sr_train(fs=20.0, a=1.2, d=20.0, s=10.0, n_periods=200, sign=-1.0, noise=0.0, seed=0):
    """Van Atta Fig. 1 geometry: ramps of amplitude *a* and duration *d*, quiet gaps *s*.
    sign=-1: gradual rise, instantaneous drop (unstable); sign=+1 the stable mirror."""
    rng = np.random.default_rng(seed)
    P = d + s
    n = int(P * fs * n_periods)
    t = np.arange(n) / fs
    phase = t % P
    x = np.where(phase < d, a * phase / d, 0.0)
    x = -sign * x + noise * rng.normal(size=n)
    return x - x.mean()


def test_structure_function_moments_match_the_derived_ramp_polynomials():
    # Paw U et al. 2005 eqs. 2a-c (re-derived in ec_ramps.md; Van Atta's printed 2.11
    # coefficients beyond the linear term disagree -- the derivation wins on brute force)
    fs, a, d, s = 20.0, 1.5, 20.0, 10.0
    x = sr_train(fs, a, d, s)
    for r in (0.5, 1.0, 2.0):
        S = ramps.structure_function(x, int(r * fs))
        v = r / d
        P2 = 1 - v**2 / 3
        P3 = 1 - 1.5 * v + 0.5 * v**3
        P5 = 1 - 2.5 * v + (10 / 3) * v**2 - 2.5 * v**3 + (2 / 3) * v**5
        assert S[2] == pytest.approx(a**2 * r / (d + s) * P2, rel=0.02)
        assert S[3] == pytest.approx(-(a**3) * r / (d + s) * P3, rel=0.02)
        assert S[5] == pytest.approx(-(a**5) * r / (d + s) * P5, rel=0.02)
        # Van Atta's printed n=3 polynomial (-1 + 5/2 v - 2 v^2 + v^3/2) does not fit
        va = -(a**3) * r / (d + s) * (1 - 2.5 * v + 2 * v**2 - 0.5 * v**3)
        if v >= 0.05:
            assert abs(S[3] - va) > 5 * abs(S[3] - (-(a**3) * r / (d + s) * P3))
    with pytest.raises(ValueError):
        ramps.structure_function(x, 0)


def test_vanatta_recovers_amplitude_period_and_the_two_lag_d_s_split():
    fs, a, d, s = 20.0, 1.2, 20.0, 10.0
    x = sr_train(fs, a, d, s)
    sm = ramps.vanatta(x, fs, 0.5)
    assert sm.a == pytest.approx(a, rel=0.05)             # linearized, v = 0.025
    assert sm.period == pytest.approx(d + s, rel=0.05)
    assert sm.S3_rate == pytest.approx(ramps.structure_function(x, 10)[3] / 0.5)
    # two-lag split (Paw U et al. 2005): d and s separately, P-corrected amplitude
    assert sm.d == pytest.approx(d, rel=0.1)
    assert sm.s == pytest.approx(s, rel=0.3)
    assert sm.a2 == pytest.approx(a, rel=0.05)
    # stable mirror image: the amplitude changes sign, the period does not
    xi = sr_train(fs, a, d, s, sign=+1.0)
    si = ramps.vanatta(xi, fs, 0.5)
    assert si.a == pytest.approx(-a, rel=0.05) and si.period == pytest.approx(d + s, rel=0.05)
    # a no-ramp signal fails the l+s >= 10 r constraint or the cubic -> NaN
    rng = np.random.default_rng(3)
    flat = ramps.vanatta(rng.normal(size=36000), fs, 0.5)
    assert np.isnan(flat.a) or flat.period >= 5.0


def test_sr_flux_is_alpha_a_z_over_period():
    assert ramps.sr_flux(1.2, 30.0, 10.85) == pytest.approx(1.2 * 10.85 / 30.0)
    assert ramps.sr_flux(-0.8, 20.0, 3.0, alpha=0.5) == pytest.approx(-0.5 * 0.8 * 3.0 / 20.0)


def test_phi_h_and_alpha_castellvi():
    # Hogstrom's function via Castellvi & Snyder 2009 eq. 5 ("116" misprint read as 11.6)
    assert ramps.phi_h(0.0) == pytest.approx(0.95)
    assert ramps.phi_h(0.5) == pytest.approx(0.95 + 7.8 * 0.5)
    assert ramps.phi_h(-1.0) == pytest.approx(0.95 / np.sqrt(12.6))
    assert np.isnan(ramps.phi_h(1.5)) and np.isnan(ramps.phi_h(-3.0))
    # eq. 3, inertial branch: alpha = sqrt(k/pi * (z-d)/z^2 * tau * u* / phi_h)
    a = ramps.alpha_castellvi(29.0, 1.0, 0.0, 10.85)
    assert a == pytest.approx(np.sqrt(0.4 / np.pi * (1 / 10.85) * 29.0 / 0.95))
    assert ramps.alpha_castellvi(29.0, 1.0, 0.0, 10.85, d=2.0) < a
    assert np.isnan(ramps.alpha_castellvi(29.0, 1.0, 2.0, 10.85))   # zeta out of range


def test_tke_trigger_finds_synthetic_bursts_and_iqa_is_the_cumulative_path():
    fs, T = 20.0, 1800.0
    n = int(fs * T)
    t = np.arange(n) / fs
    rng = np.random.default_rng(4)
    centers = np.arange(150.0, T - 100.0, 200.0)          # 8 bursts, well inside the edges
    env = 0.15 + sum(2.0 * np.exp(-0.5 * ((t - c) / 8.0) ** 2) for c in centers)
    u, v, w = (env * rng.normal(size=n) for _ in range(3))
    ev = ramps.detect_tke(u, v, w, fs, a_s=10.0, lp_s=10.0, thresh=1.25)
    assert ev.n == len(centers)
    # the trigger is the coefficient minimum preceding the burst (weak ejection phase)
    for c in centers:
        dt_c = ev.times - c
        assert np.any((dt_c > -40.0) & (dt_c < 10.0))
    assert np.all((ev.sweep_frac >= 0) & (ev.sweep_frac <= 1))
    assert ev.A_mean > 0 and ev.mean_spacing == pytest.approx(200.0, rel=0.1)
    # u_TKE and IQA definitions
    assert ramps.utke(np.array([3.0]), np.array([0.0]), np.array([4.0]))[0] == pytest.approx(5.0)
    X, Y, Z = ramps.iqa(np.ones(4), -np.ones(4), np.array([1.0, -2.0, 1.0, 1.0]), 2.0)
    assert X.tolist() == [0.5, 1.0, 1.5, 2.0]
    assert Z.tolist() == [0.5, -0.5, 0.0, 0.5]
    # the threshold is relative, so homogeneous noise still triggers (Mangan 2022 p. 54:
    # "may not work well under low u_TKE periods"); triggers stay inside the edge margin
    quiet = ramps.detect_tke(*(0.01 * rng.normal(size=(3, n))), fs)
    if quiet.n:
        assert quiet.times.min() >= 30.0 and quiet.times.max() <= T - 30.0


def test_module_run_writes_the_ramps_group(tmp_path):
    path, truth = make_hf(tmp_path / "SYN_hf_GPF_ConstDet_2024_06_01.nc")
    cfg = ECConfig.from_config(modules=("ramps",), ramps={"a_max_s": 100.0, "D_min_s": 0.0})
    out = run_file(path, cfg)
    ds = ecio.read_group(out, "ramps")
    assert ds.attrs["wavelet"] == "mhat" and ds.attrs["signals"] == "ts,u"
    assert ds["event_time_ts"].dims == ("record", "height", "event")
    assert np.isfinite(ds["D_ts"]).all() and np.isfinite(ds["D_u"]).all()
    assert (ds["n_events_ts"] > 0).all()
    n = int(ds["n_events_ts"][0, 0])
    times = ds["event_time_ts"][0, 0].values
    assert np.isfinite(times[:n]).all() and np.isnan(times[n:]).all()
    assert np.all(np.diff(times[:n]) > 0)
    assert ds["W_ts"].dims == ("record", "height", "scale")
    assert set(np.unique(ds["slope_u"].values)) == {1.0}
    # structure-function outputs on the sr_lag axis
    assert ds.attrs["sr_signals"] == "ts,u" and ds.attrs["sr_alpha"] == 1.0
    assert ds["sr_a_ts"].dims == ("record", "height", "sr_lag")
    assert list(ds["sr_lag"].values) == [0.25, 0.5, 0.75, 1.0]
    fin = np.isfinite(ds["sr_period_ts"].values)
    assert np.all(ds["sr_period_ts"].values[fin] >= 10.0 * np.broadcast_to(
        ds["sr_lag"].values, fin.shape)[fin])             # Spano's l+s >= 10 r constraint
    assert np.isfinite(ds["sr_S3_rate_u"]).any()
    # TKE trigger outputs
    assert ds["n_events_e"].dims == ("record", "height")
    ne0 = int(ds["n_events_e"][0, 0])
    te = ds["event_time_e"][0, 0].values
    assert np.isfinite(te[:ne0]).all() and np.isnan(te[ne0:]).all()
    sf = ds["sweep_frac_e"].values
    sfin = sf[np.isfinite(sf)]
    assert np.all((sfin >= 0) & (sfin <= 1))
    assert ds.attrs["tke_a_s"] == 10.0 and ds.attrs["tke_thresh"] == 1.25
    # sr_alpha modes (ruling 2026-08-23): fixed is the default, castellvi and fit selectable
    assert ds.attrs["sr_alpha_mode"] == "fixed"
    al = ds["sr_alpha_ts"].values
    assert np.all(al[np.isfinite(al)] == 1.0) and np.isfinite(al).all()
    hgt = ds.height.values
    F0 = ds["sr_a_ts"].values * hgt[None, :, None] / ds["sr_period_ts"].values
    assert np.allclose(ds["sr_flux_ts"].values, F0, equal_nan=True)
    dsc = ecio.read_group(run_file(path, ECConfig.from_config(
        modules=("ramps",), ramps={"a_max_s": 100.0, "D_min_s": 0.0,
                                   "sr_alpha_mode": "castellvi"})), "ramps")
    ac = dsc["sr_alpha_ts"].values
    fin = np.isfinite(dsc["sr_period_ts"].values)
    assert np.isfinite(ac[fin]).all() and np.all(ac[fin] > 0)
    assert not np.allclose(ac[fin], ac[fin].flat[0])      # per record/height, not constant
    dsf = ecio.read_group(run_file(path, ECConfig.from_config(
        modules=("ramps",), ramps={"a_max_s": 100.0, "D_min_s": 0.0, "sr_alpha_mode": "fit"})), "ramps")
    af = dsf["sr_alpha_ts"].values
    for il in range(af.shape[2]):                         # one fitted scalar per lag
        vals = af[:, :, il][np.isfinite(af[:, :, il])]
        if vals.size:
            assert np.allclose(vals, vals[0])
    with pytest.raises(ValueError):
        run_file(path, ECConfig.from_config(modules=("ramps",),
                                            ramps={"a_max_s": 100.0, "sr_alpha_mode": "bogus"}))


# ---------------------------------------------------------------- Thomas & Foken 2005 detector

TF_FS = 2.0
TF_T = np.arange(0.0, 1800.0, 1.0 / TF_FS)
TF05_GRID = ramps.tf_duration_grid(8.0, 140.0, 0.5)        # TF05 p. 97 grid
TF05_RAMP_GAPS = np.array([60, 340, 110, 280, 70, 112])     # gaps spanning 60-340 s, TF05 p. 100


def _tf_prepared(x, grid):
    pad = ramps.tf_pad_width(TF_FS, grid.max())
    return ramps.bior_lowpass(np.pad(ramps.tf_normalise(x), pad)), pad


def test_tf2005_triple_sine_matches_table3():
    x = sum(np.sin(2 * np.pi * TF_T / p) for p in (10.0, 60.0, 230.0))
    xf, pad = _tf_prepared(x, TF05_GRID)
    W = ramps.tf_wavelet_variance(xf, TF_FS, TF05_GRID, pad)
    D_e, maxima = ramps.tf_event_duration(W, TF05_GRID)
    # TF05 Table 3 targets; tolerances are the reported precision of the published values
    assert D_e == pytest.approx(30.5, abs=1.0)
    assert maxima.size == 2
    assert maxima[1] == pytest.approx(118.0, abs=4.0)
    events = ramps.tf_crossings(xf, TF_FS, -1, D_e, pad)
    T_x, sigma = ramps.tf_spacing_stats(events)
    assert events.size == 30
    assert T_x == pytest.approx(59.9, abs=0.5)
    assert sigma == pytest.approx(1.7, abs=0.5)


def test_tf2005_seven_ramps_detected_at_the_drops():
    starts = 64.0 + np.concatenate([[0.0], np.cumsum(TF05_RAMP_GAPS + 100.0)])
    x = np.zeros_like(TF_T)
    for s0 in starts:
        m = (TF_T >= s0) & (TF_T < s0 + 100.0)
        x[m] = (TF_T[m] - s0) / 100.0
    drops = starts + 100.0
    xf, pad = _tf_prepared(x, TF05_GRID)
    D_e, _ = ramps.tf_event_duration(ramps.tf_wavelet_variance(xf, TF_FS, TF05_GRID, pad), TF05_GRID)
    # TF05 Table 3: D_e ~ 100 s for 100 s ramps, T_x = mean gap + ramp length = 262 s
    assert D_e == pytest.approx(99.5, abs=6.0)
    events = ramps.tf_crossings(xf, TF_FS, -1, D_e, pad)
    assert events.size == 7
    assert np.all(np.abs(events - drops) < 20.0)           # within a fifth of the ramp length
    assert ramps.tf_spacing_stats(events)[0] == pytest.approx(262.0, abs=5.0)


@pytest.mark.parametrize("r, sign", [(0.4, -1), (-0.4, +1)])
def test_tf2005_table2_sign_selects_the_crossing_direction(r, sign):
    assert ramps.tf_crossing_sign(r) == sign
    t = np.arange(0.0, 60.0, 0.5)
    step_down = np.where(t < 30.0, 1.0, -1.0)
    found = ramps.tf_crossings(step_down, TF_FS, sign, 10.0)
    if sign < 0:
        assert found.size == 1 and found[0] == pytest.approx(30.0, abs=0.5)   # one sample
    else:
        assert np.all(np.abs(found - 30.0) > 5.0)


def test_tf2005_block_average_20_to_2_hz():
    x = np.arange(65536, dtype=float)
    y = ramps.block_average(x, 20.0, 2.0)
    assert y.size == 6553
    assert y[0] == pytest.approx(4.5)
    assert y[-1] == pytest.approx(np.mean(x[65520:65530]))
    with pytest.raises(ValueError, match="positive integer"):
        ramps.block_average(x, 20.0, 3.0)


def test_tf2005_conditional_average_of_exact_ramp_and_flux_split():
    D_e = 40.0
    t = np.arange(0.0, 2000.0, 1.0 / TF_FS)
    times = np.arange(100.0, 1900.0, 2 * D_e + 17.0)
    x = np.zeros_like(t)
    for ti in times:
        m = (t >= ti - D_e) & (t < ti)
        x[m] = (t[m] - (ti - D_e)) / D_e
    binned, fine, n_used = ramps.tf_conditional_average(x, TF_FS, times, D_e)
    assert n_used == times.size
    lag = ramps.tf_lag_grid()
    expected = np.where(lag < 0, lag + 1.0, 0.0)            # planted ramp over [-D_e, 0], zero after
    away = np.abs(lag * D_e) > 1.0 / TF_FS                  # interpolation across the drop
    np.testing.assert_allclose(fine[away], expected[away], atol=0.02)   # 0.02 of the ramp height
    np.testing.assert_allclose(binned[:11], np.linspace(1 / 24, 21 / 24, 11), atol=0.02)
    f = ramps.tf_flux_split(x, 0.5 * x, TF_FS, times, D_e)
    assert f["F_cs"] == pytest.approx(f["F_ej"] + f["F_sw"], abs=1e-12)   # Z11 eq. 3 identity
    # closed form: F_ej = (1 / 2) 0.5 int_0^1 s^2 ds = 1/12, F_sw = 0; 2e-3 covers the lag
    # points within one sample of the drop
    assert f["F_ej"] == pytest.approx(1.0 / 12.0, abs=2e-3)
    assert f["F_sw"] == pytest.approx(0.0, abs=2e-3)


@pytest.mark.parametrize("with_q", [True, False])
def test_detect_tf2005_on_a_20_hz_record(with_q):
    rng = np.random.default_rng(1)
    n = 65536
    t = np.arange(n) / 20.0
    T = np.sin(2 * np.pi * t / 60.0) + 0.3 * rng.normal(size=n)
    w = 0.5 * np.sin(2 * np.pi * t / 60.0) + rng.normal(size=n)
    q = T + rng.normal(size=n) if with_q else None
    ev = ramps.detect_tf2005(T, w, 20.0, others={"rho_h2o": q},
                             pairs={"wTs": ("w_pf", "ts"), "wq": ("w_pf", "rho_h2o")})
    assert ev.r_Tw > 0 and ev.sign == -1
    assert ev.D_e == pytest.approx(30.0, rel=0.15)          # half the planted 60 s period
    assert ev.n > 40
    f = ev.flux["wTs"]
    assert f["F_cs"] == pytest.approx(f["F_ej"] + f["F_sw"], abs=1e-12)
    assert ev.f_tc == pytest.approx(ev.n * ev.D_e / ev.length_s)
    assert np.isfinite(ev.comp["rho_h2o"]).all() == with_q
    assert np.isfinite(ev.flux["wq"]["F_cs"]) == with_q
