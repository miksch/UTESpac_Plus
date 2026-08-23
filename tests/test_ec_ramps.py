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
    assert ramps._slope_for("Ts", "auto", 0.1) == "negative"
    assert ramps._slope_for("Ts", "auto", -0.1) == "positive"
    assert ramps._slope_for("Ts", "auto", np.nan) == "negative"
    assert ramps._slope_for("u", "auto", -0.1) == "positive"
    assert ramps._slope_for("Ts", "both", 0.1) == "both"


def test_module_run_writes_the_ramps_group(tmp_path):
    path, truth = make_hf(tmp_path / "SYN_hf_GPF_ConstDet_2023_07_06.nc")
    cfg = ECConfig.from_config(modules=("ramps",), ramps={"a_max_s": 100.0, "D_min_s": 0.0})
    out = run_file(path, cfg)
    ds = ecio.read_group(out, "ramps")
    assert ds.attrs["wavelet"] == "mhat" and ds.attrs["signals"] == "Ts,u"
    assert ds["event_time_Ts"].dims == ("record", "height", "event")
    assert np.isfinite(ds["D_Ts"]).all() and np.isfinite(ds["D_u"]).all()
    assert (ds["n_events_Ts"] > 0).all()
    n = int(ds["n_events_Ts"][0, 0])
    times = ds["event_time_Ts"][0, 0].values
    assert np.isfinite(times[:n]).all() and np.isnan(times[n:]).all()
    assert np.all(np.diff(times[:n]) > 0)
    assert ds["W_Ts"].dims == ("record", "height", "scale")
    assert set(np.unique(ds["slope_u"].values)) == {1.0}
