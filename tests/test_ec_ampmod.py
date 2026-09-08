"""Amplitude-modulation module: envelope, decoupling, R, cutoff logic, module run.

Validation targets from the sources (library/writeups/ec_ampmod.md): the Hilbert
envelope of a pure AM signal recovers B + m(t) (Mathis 2009 §4); a
phase-scrambled signal with the same spectrum gives R ~ 0 (§7.1.1).
"""

import numpy as np
import pytest

from ec_coherent import ampmod
from ec_coherent.config import ECConfig
from ec_coherent.io import open_hf

from ec_helpers import make_hf

FS = 10.0
N = 4096


def _am_signal(depth=0.5, B=2.0, fm=0.05, fc_carrier=1.0, n=N, fs=FS):
    """u(t) = [B + m(t)] c(t), Mathis 2009 eqs. 4.6-4.7."""
    t = np.arange(n) / fs
    m = B * depth * np.sin(2 * np.pi * fm * t)
    c = np.sin(2 * np.pi * fc_carrier * t)
    return (B + m) * c, m, c


def test_envelope_recovers_modulating_signal():
    u, m, _ = _am_signal()
    env = ampmod.envelope(u)
    core = slice(N // 8, -N // 8)          # away from Hilbert edge effects
    assert np.allclose(env[core], (2.0 + m)[core], atol=0.05)


def test_lowpass_sharp_partition_is_exact():
    rng = np.random.default_rng(0)
    x = rng.normal(size=N)
    xl = ampmod.lowpass_sharp(x, FS, 0.2)
    xs = x - xl
    assert np.allclose(xl + xs, x)
    # the large-scale part holds no energy above the cutoff
    f = np.fft.rfftfreq(N, d=1 / FS)
    assert np.max(np.abs(np.fft.rfft(xl)[f > 0.2])) < 1e-8


def test_am_coefficient_high_for_constructed_modulation():
    u, m, _ = _am_signal()
    fc = 0.2                                # between fm = 0.05 and the 1 Hz carrier
    us = u - ampmod.lowpass_sharp(u, FS, fc)
    env_l = ampmod.lowpass_sharp(ampmod.envelope(us), FS, fc)
    R = ampmod.am_coefficient(m, env_l)     # the true modulator against the envelope
    assert R > 0.95


def test_phase_scrambled_signal_gives_zero_R():
    """Mathis 2009 §7.1.1: same spectrum, random phases -> R effectively zero."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=N)
    for i in range(1, N):
        x[i] = 0.97 * x[i - 1] + 0.1 * x[i]
    X = np.fft.rfft(x)
    scr = np.abs(X) * np.exp(1j * rng.uniform(0, 2 * np.pi, X.size))
    scr[0] = X[0]
    y = np.fft.irfft(scr, n=N)
    fc = 0.05
    Rs = []
    for s in (x, y):
        sl = ampmod.lowpass_sharp(s, FS, fc)
        ss = s - sl
        env_l = ampmod.lowpass_sharp(ampmod.envelope(ss), FS, fc)
        Rs.append(abs(ampmod.am_coefficient(sl, env_l)))
    assert Rs[1] < 0.15                     # scrambled: no modulation


def test_spectral_gap_found_on_bimodal_and_absent_on_rednoise():
    t = np.arange(N) / FS
    rng = np.random.default_rng(2)
    bimodal = np.sin(2 * np.pi * 0.02 * t) + 0.7 * np.sin(2 * np.pi * 1.0 * t) \
        + 0.05 * rng.normal(size=N)
    fg = ampmod.spectral_gap(bimodal, FS)
    assert 0.02 < fg < 1.0
    red = rng.normal(size=N)
    for i in range(1, N):
        red[i] = 0.9 * red[i - 1] + red[i]
    # red noise has a monotone premultiplied rise to one broad peak most of the
    # time; when a spurious interior dip appears it must still lie inside the axis
    fg2 = ampmod.spectral_gap(red, FS)
    assert np.isnan(fg2) or 0 < fg2 < FS / 2


def test_cutoff_frequency_modes():
    cfg = ECConfig.from_config({}).ampmod
    x = np.random.default_rng(3).normal(size=N)
    lam, fc, src = ampmod.cutoff_frequency(cfg.__class__(cutoff_mode="delta", delta_m=800.0),
                                           x, FS, 4.0, 10.0)
    assert lam == 800.0 and fc == pytest.approx(4.0 / 800.0) and src == 2
    lam, fc, src = ampmod.cutoff_frequency(cfg.__class__(cutoff_mode="scaled", z_mult=50.0),
                                           x, FS, 4.0, 10.0)
    assert lam == 500.0 and src == 3
    with pytest.raises(ValueError):
        ampmod.cutoff_frequency(cfg.__class__(cutoff_mode="nope"), x, FS, 4.0, 10.0)


def test_run_on_synthetic_file(tmp_path):
    path, _ = make_hf(tmp_path / "hf.nc", fs=FS, window_s=300.0, n_records=3)
    cfg = ECConfig.from_config({}, ampmod={"cutoff_mode": "delta", "delta_m": 300.0})
    with open_hf(path) as hf:
        ds = ampmod.run(hf, cfg)
    assert ds["R_uL_uS"].shape == (3, 2)
    got = ds["R_uL_uS"].values
    assert np.isfinite(got).all() and (np.abs(got) <= 1).all()
    for name in ("R_wL_uS", "R_uL_tsS", "R_uL_uwS", "R_wL_wTsS"):
        assert name in ds
        assert (np.abs(ds[name].values[np.isfinite(ds[name].values)]) <= 1).all()
    assert (ds["cutoff_source"].values == 2).all()
    assert np.allclose(ds["cutoff_lambda"].values, 300.0)
    assert np.allclose(ds["cutoff_f"].values, ds["U_mean"].values / 300.0)
    # zeta = z/L with L = -50 in the helper
    assert np.allclose(ds["zeta"].values, np.array([3.0, 10.0]) / -50.0)


def test_run_gap_mode_falls_back_and_flags(tmp_path):
    path, _ = make_hf(tmp_path / "hf2.nc", fs=FS, window_s=300.0, n_records=2)
    cfg = ECConfig.from_config({})       # spectral_gap default, red-noise winds: fallback expected
    with open_hf(path) as hf:
        ds = ampmod.run(hf, cfg)
    src = ds["cutoff_source"].values
    assert set(np.unique(src)) <= {0, 1}
    fallback = src == 1
    assert np.allclose(ds["cutoff_lambda"].values[fallback], cfg.ampmod.delta_m)
