"""Tests for the platform-frame conversion, gap-limited interpolation and
motion-diagnostic helpers in utespac.motion (synthetic data only)."""

import numpy as np
import pandas as pd
import pytest

from utespac.motion import (G, band_variance, bandpass, bounded_peak_lag, euler_from_T,
                            euler_T, heave_velocity, interp_max_gap, pair_on_grid,
                            to_platform_frame, xcorr)
from utespac.site_config import IMUInfo

FS = 20.0


def _channels(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.normal(0, 0.1, (n, 2)), 1 + rng.normal(0, 0.05, n),
                            rng.normal(0, 5, (n, 3)), rng.normal(0, 8, (n, 2)),
                            rng.uniform(-170, 170, n)])


# ── to_platform_frame ────────────────────────────────────────────────────────

def test_to_platform_frame_units_and_signs():
    ch = _channels()
    cfg = IMUInfo(leverArm=[0, 0, 1], accelUnits="g", accelSigns=[1, -1, -1],
                  gyroSigns=[-1, 1, 1], attitudeSigns=[1, 1, -1])
    acc, gyro, att = to_platform_frame(ch, cfg)
    np.testing.assert_allclose(acc, ch[:, 0:3] * [1, -1, -1] * G, rtol=0, atol=0)
    np.testing.assert_allclose(gyro, np.deg2rad(ch[:, 3:6]) * [-1, 1, 1], rtol=0, atol=0)
    np.testing.assert_allclose(att, np.deg2rad(ch[:, 6:9]) * [1, 1, -1], rtol=0, atol=0)


def test_to_platform_frame_ms2_passthrough():
    ch = _channels()
    acc, _, _ = to_platform_frame(ch, IMUInfo(leverArm=[0, 0, 1]))
    np.testing.assert_array_equal(acc, ch[:, 0:3])


def test_to_platform_frame_mount_tilt_is_undone():
    """Rotating level channels by the mount tilt and converting back recovers them."""
    n = 50
    roll, pitch = np.deg2rad(2.0), np.deg2rad(-3.0)
    cfg = IMUInfo(leverArm=[0, 0, 1], mountRoll=2.0, mountPitch=-3.0)
    Rm = euler_T(roll, pitch, 0.0)
    acc_true = np.tile([0.0, 0.0, G], (n, 1))
    acc_imu = acc_true @ Rm           # platform -> IMU axes is Rm.T applied as row @ Rm
    ch = np.column_stack([acc_imu, np.zeros((n, 6))])
    acc, gyro, att = to_platform_frame(ch, cfg)
    np.testing.assert_allclose(acc, acc_true, atol=1e-12)
    np.testing.assert_allclose(gyro, 0.0, atol=1e-15)
    r, p, y = euler_from_T(np.broadcast_to(Rm.T, (n, 3, 3)))
    np.testing.assert_allclose(att, np.column_stack([r, p, y]), atol=1e-12)


# ── interp_max_gap ───────────────────────────────────────────────────────────

def test_interp_max_gap_linear_and_masked():
    t = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
    ch = np.column_stack([t * 2.0, -t])
    tgt = np.array([-0.5, 0.5, 1.5, 3.0, 5.5, 6.0, 6.5])
    out = interp_max_gap(t, ch, tgt, max_gap_s=1.0)
    expect = np.array([[np.nan, np.nan], [1.0, -0.5], [3.0, -1.5], [np.nan, np.nan],
                       [11.0, -5.5], [12.0, -6.0], [np.nan, np.nan]])
    np.testing.assert_array_equal(out, expect)


def test_interp_max_gap_wider_gap_bridged():
    t = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
    out = interp_max_gap(t, t[:, None], np.array([3.0]), max_gap_s=3.0)
    np.testing.assert_allclose(out, [[3.0]])


def test_interp_max_gap_too_few_samples():
    out = interp_max_gap(np.array([1.0]), np.ones((1, 3)), np.arange(4.0), 1.0)
    assert out.shape == (4, 3) and np.isnan(out).all()


# ── band-pass diagnostics ────────────────────────────────────────────────────

def _tone(f, n=12000, amp=1.0, phase=0.0):
    return amp * np.sin(2 * np.pi * f * np.arange(n) / FS + phase)


def test_bandpass_keeps_inband_rejects_outband():
    x = _tone(0.9) + _tone(0.05, amp=5.0) + _tone(5.0, amp=2.0) + 3.0
    y = bandpass(x, 0.6, 1.2, FS)
    core = slice(2000, -2000)
    np.testing.assert_allclose(y[core], _tone(0.9)[core], atol=0.05)


def test_band_variance_of_tone():
    x = _tone(1.0, amp=2.0) + _tone(0.02, amp=10.0)
    assert band_variance(x, (0.5, 2.0), FS) == pytest.approx(2.0, rel=0.02)


def test_band_variance_fills_gaps_and_rejects_short():
    x = _tone(1.0, amp=2.0)
    x[5000:5010] = np.nan
    assert np.isfinite(band_variance(x, (0.5, 2.0), FS))
    short = np.full(1000, np.nan)
    short[:199] = 1.0
    assert np.isnan(band_variance(short, (0.5, 2.0), FS))


def test_heave_velocity_of_sinusoidal_heave():
    """az = 1 + a/g sin(wt) integrates to v = -a/w cos(wt).

    The running sum integrates to the end of each sample interval, half a
    sample after the sample time.
    """
    f, a = 0.9, 0.3
    n = 12000
    t = np.arange(n) / FS
    az = 1.0 + a / G * np.sin(2 * np.pi * f * t)
    v = heave_velocity(az, FS, 0.6, 1.2)
    expect = -a / (2 * np.pi * f) * np.cos(2 * np.pi * f * (t + 0.5 / FS))
    core = slice(2000, -2000)
    np.testing.assert_allclose(v[core], expect[core], atol=0.01 * a)


# ── cross-correlation lag ────────────────────────────────────────────────────

def test_xcorr_positive_lag_when_x_leads():
    rng = np.random.default_rng(3)
    x = rng.normal(size=4000)
    y = np.roll(x, 4)                 # y[n + 4] = x[n]: x leads by 0.2 s
    lags, r = xcorr(x, y, 1.0, FS)
    assert lags.min() == -1.0 and lags.max() == 1.0
    assert lags[np.argmax(r)] == pytest.approx(0.2)
    assert r.max() == pytest.approx(1.0, abs=0.01)


def test_bounded_peak_lag_picks_peak_inside_bound():
    """A periodic signal peaks at every period; the bound selects the near one."""
    rng = np.random.default_rng(4)
    x = bandpass(rng.normal(size=12000), 0.9, 1.1, FS)
    y = np.roll(x, -2)                # y leads by 0.1 s
    r, lag = bounded_peak_lag(x, y, 1.5, FS, 0.4)
    assert lag == pytest.approx(-0.1)
    assert r == pytest.approx(1.0, abs=0.01)


# ── pair_on_grid ─────────────────────────────────────────────────────────────

def _records(seed=0, spikes=0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2024-06-01 23:00")
    n = 20000
    ti = t0 - pd.Timedelta("30s") + pd.to_timedelta(np.arange(n) * 0.1, unit="s")
    az = 1 + 0.01 * np.sin(np.arange(n) * 0.1)
    imu = pd.DataFrame({"time": ti, "ax": 0.0, "ay": 0.0, "az": az, "gx": 0.0,
                        "gy": 0.0, "gz": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0})
    if spikes:
        imu.loc[rng.integers(0, n, spikes), "az"] = 3.0
    ts = t0 + pd.to_timedelta(np.arange(0, 1800, 0.05), unit="s")
    son = pd.DataFrame({"TIMESTAMP": ts, "Ux": 3.0, "Uy": 4.0,
                        "Uz": rng.normal(0, 0.1, len(ts))})
    return imu, son, t0, t0 + pd.Timedelta("30min")


LIMITS = dict(pad_s=10, min_records=500, trim_s=1, min_overlap_s=600)


def test_pair_on_grid_common_grid():
    imu, son, t0, t1 = _records(spikes=50)
    p = pair_on_grid(imu, son, t0, t1, FS, **LIMITS)
    assert p is not None
    step = np.diff(p["grid"])
    np.testing.assert_allclose(step, 1 / FS, atol=1e-6)
    assert p["grid"][0] == pytest.approx(t0.value / 1e9 + 1)
    assert p["spd"] == pytest.approx(5.0)
    assert np.all(np.abs(p["az"] - 1) <= 0.0101)      # spikes screened out
    assert p["rate"] == pytest.approx(10.0, rel=0.01)
    assert {len(p[k]) for k in ("az", "u", "v", "w")} == {len(p["grid"])}


def test_pair_on_grid_short_overlap_is_none():
    imu, son, t0, t1 = _records()
    son = son.iloc[:5000]             # 250 s of sonic
    assert pair_on_grid(imu, son, t0, t1, FS, **LIMITS) is None
    assert pair_on_grid(imu.iloc[:100], son, t0, t1, FS, **LIMITS) is None
