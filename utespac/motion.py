"""Platform-motion correction of sonic winds (moored floating platform).

The sonic on a floating platform measures wind in the platform frame; the
earth-frame wind is

    u_true = T u_obs + T (omega x r) + v_plat        (Edson et al. 1998, eq. 4)

with T the platform-to-earth rotation built from the IMU attitude, omega
the platform-frame angular-rate vector, r the lever arm from the IMU to
the sonic head, and v_plat the platform translational velocity from the
rotated, high-pass-filtered, integrated accelerometers. Angular-rate
integration and the complementary attitude filter follow Miller et al.
(2008) (fourth-order Butterworth, applied forward and backward); the
moored-platform case with no mean translation is Anctil et al. (1994);
filter-constant practice per Landwehr et al. (2015). Reference
implementation: ``fluxer.eddycov.flux.wind3D_correct`` (flux_capacitor,
UofM-CEOS), a port of S. Miller's MATLAB ``motion`` routine.

Conventions (kernel level, SI units):

- Platform frame is right-handed with z up (x nominal forward, y left);
  the earth frame is z-up. All angles in radians, rates in rad/s,
  accelerations in m/s^2. Unit and axis-sign conversion from logger
  conventions happens in :func:`utespac.stages.motion` (and, for plain
  channel arrays, in :func:`to_platform_frame`), not in the kernels.
- ``T = Rz(yaw) Ry(pitch) Rx(roll)`` maps platform vectors to earth
  (ZYX Euler, ``v_e = T v_p``); positive roll rotates +y toward +z,
  positive pitch rotates +z toward +x.
- Accelerometers measure specific force: at rest the z channel reads +g.

For a moored platform the absolute heading is deliberately left out by
default (``yaw_handling="demean"``): the winds come out earth-level but
in the mean-heading horizontal frame, so the downstream wind-direction
(sonic orientation bearing) and yaw-into-mean-wind stages keep their
meaning unchanged.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

G = 9.80665      # [m/s^2] standard gravity

__all__ = ["MotionParams", "MotionResult", "correct_wind", "euler_T", "euler_from_T",
           "accel_tilt", "complementary_attitude", "platform_velocity",
           "highpass", "lowpass", "integrate", "fill_gaps", "G",
           "to_platform_frame", "interp_max_gap", "bandpass", "band_variance",
           "heave_velocity", "xcorr", "bounded_peak_lag", "pair_on_grid"]


# ── configuration and result ─────────────────────────────────────────────────

@dataclass(frozen=True)
class MotionParams:
    """Knobs of the motion correction (the ``wind3D_correct`` signature)."""
    fs: float                                  # [Hz] scan rate
    lever_arm: Tuple[float, float, float]      # [m] sonic head rel. to IMU, platform frame
    Tcf: float = 20.0        # [s] complementary-filter cutoff period (attitude)
    Ta: float = 20.0         # [s] high-pass cutoff period for the accel integration
    order: int = 4           # Butterworth order (Miller et al. 2008 use 4)
    yaw_handling: str = "demean"   # "demean" | "full" | "zero"

    def __post_init__(self):
        if len(self.lever_arm) != 3:
            raise ValueError("lever_arm must have three components (x, y, z)")
        if self.yaw_handling not in ("demean", "full", "zero"):
            raise ValueError(f"yaw_handling must be 'demean', 'full' or 'zero', "
                             f"got {self.yaw_handling!r}")


@dataclass
class MotionResult:
    """Corrected winds plus the intermediate terms for diagnostics."""
    uvw: np.ndarray                  # (n, 3) corrected earth-frame winds
    roll: np.ndarray                 # (n,) [rad] attitude used
    pitch: np.ndarray                # (n,)
    yaw: np.ndarray                  # (n,) after yaw_handling
    v_platform: np.ndarray           # (n, 3) translational velocity term
    angular_term: np.ndarray         # (n, 3) T (omega x r)
    imu_nan_frac: dict = field(default_factory=dict)   # channel -> gap fraction filled


# ── small numerics ───────────────────────────────────────────────────────────

def fill_gaps(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """Linearly interpolate NaN gaps of a 1-D series (edges held at the
    nearest finite value); returns ``(filled copy, nan fraction)``.
    Raises ValueError when the series has no finite value at all."""
    x = np.asarray(x, dtype=float)
    bad = ~np.isfinite(x)
    frac = float(bad.mean())
    if not bad.any():
        return x.copy(), 0.0
    if bad.all():
        raise ValueError("IMU channel is entirely NaN")
    idx = np.arange(len(x))
    out = x.copy()
    out[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return out, frac


def _sos(fs: float, T_cutoff: float, btype: str, order: int):
    from scipy.signal import butter
    wn = (1.0 / T_cutoff) / (fs / 2.0)
    if not 0 < wn < 1:
        raise ValueError(f"cutoff period {T_cutoff}s is outside (2/fs, inf) at fs={fs} Hz")
    return butter(order, wn, btype=btype, output="sos")


def highpass(x: np.ndarray, fs: float, T_cutoff: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth high-pass, cutoff period *T_cutoff* [s]."""
    from scipy.signal import sosfiltfilt
    return sosfiltfilt(_sos(fs, T_cutoff, "highpass", order), x, axis=0)


def lowpass(x: np.ndarray, fs: float, T_cutoff: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass, cutoff period *T_cutoff* [s]."""
    from scipy.signal import sosfiltfilt
    return sosfiltfilt(_sos(fs, T_cutoff, "lowpass", order), x, axis=0)


def integrate(x: np.ndarray, fs: float) -> np.ndarray:
    """Cumulative trapezoid integral along axis 0, starting at zero."""
    x = np.asarray(x, dtype=float)
    steps = (x[1:] + x[:-1]) / (2.0 * fs)
    zero = np.zeros((1,) + x.shape[1:])
    return np.concatenate([zero, np.cumsum(steps, axis=0)])


# ── attitude ─────────────────────────────────────────────────────────────────

def euler_T(roll, pitch, yaw) -> np.ndarray:
    """Platform-to-earth rotation matrices ``(n, 3, 3)``: T = Rz Ry Rx."""
    cf, sf = np.cos(roll), np.sin(roll)
    ct, st = np.cos(pitch), np.sin(pitch)
    cp, sp = np.cos(yaw), np.sin(yaw)
    n = np.broadcast(cf, ct, cp).shape
    T = np.empty(n + (3, 3))
    T[..., 0, 0] = cp * ct
    T[..., 0, 1] = cp * st * sf - sp * cf
    T[..., 0, 2] = cp * st * cf + sp * sf
    T[..., 1, 0] = sp * ct
    T[..., 1, 1] = sp * st * sf + cp * cf
    T[..., 1, 2] = sp * st * cf - cp * sf
    T[..., 2, 0] = -st
    T[..., 2, 1] = ct * sf
    T[..., 2, 2] = ct * cf
    return T


def euler_from_T(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ZYX Euler angles (roll, pitch, yaw) back out of rotation matrices."""
    roll = np.arctan2(T[..., 2, 1], T[..., 2, 2])
    pitch = -np.arcsin(np.clip(T[..., 2, 0], -1.0, 1.0))
    yaw = np.arctan2(T[..., 1, 0], T[..., 0, 0])
    return roll, pitch, yaw


def accel_tilt(acc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Roll and pitch [rad] from the accelerometer specific force ``(n, 3)``
    (gravity reference; valid at frequencies well below the wave band)."""
    ax, ay, az = acc[:, 0], acc[:, 1], acc[:, 2]
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.hypot(ay, az))
    return roll, pitch


def _euler_rates(roll, pitch, gyro) -> np.ndarray:
    """Euler-angle rates (n, 3) from platform-frame rates ``gyro = (p, q, r)``."""
    p, q, r = gyro[:, 0], gyro[:, 1], gyro[:, 2]
    cf, sf = np.cos(roll), np.sin(roll)
    tt, ct = np.tan(pitch), np.cos(pitch)
    return np.column_stack([p + tt * (q * sf + r * cf),
                            q * cf - r * sf,
                            (q * sf + r * cf) / ct])


def _body_rates(roll, pitch, euler_dot) -> np.ndarray:
    """Platform-frame rates (p, q, r) from Euler-angle rates (the inverse of
    :func:`_euler_rates`)."""
    dphi, dtheta, dpsi = euler_dot[:, 0], euler_dot[:, 1], euler_dot[:, 2]
    cf, sf = np.cos(roll), np.sin(roll)
    st, ct = np.sin(pitch), np.cos(pitch)
    return np.column_stack([dphi - dpsi * st,
                            dtheta * cf + dpsi * ct * sf,
                            -dtheta * sf + dpsi * ct * cf])


def complementary_attitude(acc: np.ndarray, gyro: np.ndarray, fs: float,
                           Tcf: float = 20.0, order: int = 4) -> np.ndarray:
    """Attitude ``(n, 3)`` [rad] from raw accel + gyro (Anctil et al. 1994 /
    Miller et al. 2008): low-passed accelerometer tilt plus high-passed
    integrated angular rates; yaw is high-passed rate integration only
    (no low-frequency heading reference on a moored platform)."""
    roll_a, pitch_a = accel_tilt(acc)
    roll_lp = lowpass(roll_a, fs, Tcf, order)
    pitch_lp = lowpass(pitch_a, fs, Tcf, order)
    rates = _euler_rates(roll_lp, pitch_lp, gyro)
    hp = highpass(integrate(rates, fs), fs, Tcf, order)
    return np.column_stack([roll_lp + hp[:, 0], pitch_lp + hp[:, 1], hp[:, 2]])


# ── translational velocity ───────────────────────────────────────────────────

def platform_velocity(acc: np.ndarray, T: np.ndarray, fs: float,
                      Ta: float = 20.0, order: int = 4) -> np.ndarray:
    """Earth-frame platform velocity ``(n, 3)`` from the specific force:
    rotate to earth, remove gravity, high-pass, integrate, high-pass again
    (Edson et al. 1998 sect. 4; Miller et al. 2008 sect. 2d)."""
    a_e = np.einsum("nij,nj->ni", T, acc)
    a_e[:, 2] -= G
    a_hp = highpass(a_e, fs, Ta, order)
    return highpass(integrate(a_hp, fs), fs, Ta, order)


# ── the correction ───────────────────────────────────────────────────────────

def correct_wind(uvw: np.ndarray, params: MotionParams,
                 acc: Optional[np.ndarray] = None,
                 gyro: Optional[np.ndarray] = None,
                 attitude: Optional[np.ndarray] = None) -> MotionResult:
    """Motion-correct sonic winds ``uvw`` (n, 3), platform frame.

    Parameters
    ----------
    acc : (n, 3), optional
        Accelerometer specific force [m/s^2]. Without it the translational
        term is skipped (tilt + lever-arm correction only).
    gyro : (n, 3), optional
        Angular rates [rad/s], platform frame. Without it the rates are
        differentiated from *attitude*.
    attitude : (n, 3), optional
        Fused (roll, pitch, yaw) [rad]. Without it the attitude comes from
        the complementary filter, which needs both *acc* and *gyro*.

    NaN gaps in the IMU channels are linearly interpolated (fractions
    reported on the result); NaN winds stay NaN.
    """
    uvw = np.asarray(uvw, dtype=float)
    if uvw.ndim != 2 or uvw.shape[1] != 3:
        raise ValueError("uvw must be (n, 3)")
    n = uvw.shape[0]
    nan_frac = {}

    def _fill(arr, name):
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=float)
        if arr.shape != (n, 3):
            raise ValueError(f"{name} must be (n, 3) matching uvw")
        out = np.empty_like(arr)
        for j, ax in enumerate("xyz"):
            out[:, j], nan_frac[f"{name}_{ax}"] = fill_gaps(arr[:, j])
        return out

    acc = _fill(acc, "acc")
    gyro = _fill(gyro, "gyro")
    attitude = _fill(attitude, "att")

    if attitude is not None:
        roll, pitch = attitude[:, 0], attitude[:, 1]
        yaw = np.unwrap(attitude[:, 2])
    elif acc is not None and gyro is not None:
        att = complementary_attitude(acc, gyro, params.fs, params.Tcf, params.order)
        roll, pitch, yaw = att[:, 0], att[:, 1], att[:, 2]
    else:
        raise ValueError("correct_wind needs a fused attitude, or accel + gyro "
                         "for the complementary filter")

    if params.yaw_handling == "zero":
        yaw = np.zeros(n)
    elif params.yaw_handling == "demean":
        yaw = yaw - np.mean(yaw)

    T = euler_T(roll, pitch, yaw)

    if gyro is None:
        euler_dot = np.column_stack([np.gradient(roll) * params.fs,
                                     np.gradient(pitch) * params.fs,
                                     np.gradient(yaw) * params.fs])
        gyro = _body_rates(roll, pitch, euler_dot)

    r = np.asarray(params.lever_arm, dtype=float)
    angular = np.einsum("nij,nj->ni", T, np.cross(gyro, np.broadcast_to(r, (n, 3))))

    if acc is not None:
        v_plat = platform_velocity(acc, T, params.fs, params.Ta, params.order)
    else:
        v_plat = np.zeros((n, 3))

    corrected = np.einsum("nij,nj->ni", T, uvw) + angular + v_plat
    return MotionResult(corrected, roll, pitch, yaw, v_plat, angular, nan_frac)


# ── logger channels to the platform frame ────────────────────────────────────

def to_platform_frame(channels, cfg):
    """Accelerations, angular rates and attitude in the SI platform frame.

    Applies the axis signs and accelerometer units declared in the site's
    ``[imu]`` block, converts the angular channels from degrees, and rotates
    all three triplets through the mounting tilt when one is declared.

    Parameters
    ----------
    channels : array_like, shape (n, 9)
        Raw IMU channels in the column order ``ax, ay, az, gx, gy, gz,
        roll, pitch, yaw``: accelerometer in ``cfg.accelUnits``, angular
        rate in deg/s and attitude in deg.
    cfg : utespac.site_config.IMUInfo
        Axis signs (``accelSigns``, ``gyroSigns``, ``attitudeSigns``),
        ``accelUnits`` and the mounting tilt ``mountRoll``, ``mountPitch``,
        ``mountYaw`` [deg]. ``gyroUnits`` and ``attitudeUnits`` are not
        consulted: the angular channels are always taken as degrees.

    Returns
    -------
    acc : numpy.ndarray, shape (n, 3)
        Specific force [m/s^2].
    gyro : numpy.ndarray, shape (n, 3)
        Angular rate [rad/s].
    att : numpy.ndarray, shape (n, 3)
        Roll, pitch and yaw [rad].

    See Also
    --------
    utespac.stages.motion : The same conversion on a Run, honouring
        ``gyroUnits`` and ``attitudeUnits``.
    """
    c = np.asarray(channels, dtype=float)
    acc = c[:, 0:3] * np.asarray(cfg.accelSigns, dtype=float)
    if cfg.accelUnits == "g":
        acc = acc * G
    gyro = np.deg2rad(c[:, 3:6]) * np.asarray(cfg.gyroSigns, dtype=float)
    att = np.deg2rad(c[:, 6:9]) * np.asarray(cfg.attitudeSigns, dtype=float)
    if cfg.mountRoll or cfg.mountPitch or cfg.mountYaw:
        Rm = euler_T(*np.deg2rad([cfg.mountRoll, cfg.mountPitch, cfg.mountYaw]))
        acc = acc @ Rm.T
        gyro = gyro @ Rm.T
        att = np.column_stack(
            euler_from_T(euler_T(att[:, 0], att[:, 1], att[:, 2]) @ Rm.T))
    return acc, gyro, att


def interp_max_gap(imu_t, imu_ch, target_s, max_gap_s):
    """Channels linearly interpolated onto target times across short gaps.

    Target times whose enclosing sample interval is longer than
    *max_gap_s*, or which fall outside the record, are returned as NaN.

    Parameters
    ----------
    imu_t : numpy.ndarray, shape (m,)
        Sample times [s], increasing.
    imu_ch : numpy.ndarray, shape (m, k)
        Channels sampled at *imu_t*.
    target_s : numpy.ndarray, shape (n,)
        Target times [s], on the same scale as *imu_t*.
    max_gap_s : float
        Longest sample interval [s] interpolated across.

    Returns
    -------
    numpy.ndarray, shape (n, k)
        Interpolated channels; all NaN when fewer than two samples exist.

    See Also
    --------
    utespac.raw_processing.imu.align_imu : Nearest-sample reindexing onto a
        uniform grid, without interpolation.
    """
    out = np.full((len(target_s), imu_ch.shape[1]), np.nan)
    if len(imu_t) < 2:
        return out
    idx = np.clip(np.searchsorted(imu_t, target_s, side="right") - 1,
                  0, len(imu_t) - 2)
    gap = imu_t[idx + 1] - imu_t[idx]
    bad = (gap > max_gap_s) | (target_s < imu_t[0]) | (target_s > imu_t[-1])
    for j in range(imu_ch.shape[1]):
        out[:, j] = np.interp(target_s, imu_t, imu_ch[:, j])
    out[bad] = np.nan
    return out


# ── motion diagnostics ───────────────────────────────────────────────────────

def bandpass(x, lo, hi, fs, order: int = 4):
    """Zero-phase Butterworth band-pass of the demeaned series.

    Parameters
    ----------
    x : numpy.ndarray
        Evenly sampled series without NaN.
    lo, hi : float
        Band edges [Hz].
    fs : float
        Sampling rate [Hz].
    order : int, optional
        Order of the Butterworth prototype.

    Returns
    -------
    numpy.ndarray
        ``x - mean(x)`` filtered forward and backward (``filtfilt`` with
        transfer-function coefficients).
    """
    from scipy import signal
    b, a = signal.butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return signal.filtfilt(b, a, x - np.mean(x))


def band_variance(x, band, fs):
    """Variance of a series inside a frequency band.

    NaN gaps are interpolated with :func:`fill_gaps` before a fourth-order
    zero-phase Butterworth band-pass (second-order sections).

    Parameters
    ----------
    x : numpy.ndarray
        Evenly sampled series, may contain NaN.
    band : tuple of float
        ``(low, high)`` band edges [Hz].
    fs : float
        Sampling rate [Hz].

    Returns
    -------
    float
        Band-passed variance, or NaN when fewer than 200 values are finite.
    """
    from scipy import signal
    ok = np.isfinite(x)
    if ok.sum() < 200:
        return np.nan
    filled, _ = fill_gaps(x)
    sos = signal.butter(4, [band[0] / (fs / 2), band[1] / (fs / 2)],
                        btype="band", output="sos")
    return float(np.var(signal.sosfiltfilt(sos, filled - filled.mean())))


def heave_velocity(az, fs, lo, hi):
    """Band-passed vertical velocity from the vertical accelerometer.

    The band-pass removes gravity and the slow tilt-driven part of *az*
    before and after integration, so the integrator does not wander.

    Parameters
    ----------
    az : numpy.ndarray
        Vertical specific force [g], evenly sampled, without NaN.
    fs : float
        Sampling rate [Hz].
    lo, hi : float
        Band edges [Hz] of both band-passes.

    Returns
    -------
    numpy.ndarray
        Vertical velocity [m/s] in the band.
    """
    return bandpass(np.cumsum(bandpass((az - 1.0) * G, lo, hi, fs)) / fs, lo, hi, fs)


def xcorr(x, y, max_lag_s, fs):
    """Normalized cross-correlation ``corr(x[n], y[n + k])`` over bounded lags.

    Parameters
    ----------
    x, y : numpy.ndarray
        Evenly sampled series of equal length, without NaN.
    max_lag_s : float
        Largest absolute lag returned [s].
    fs : float
        Sampling rate [Hz].

    Returns
    -------
    lags : numpy.ndarray
        Lags [s]; positive when *x* leads *y*.
    r : numpy.ndarray
        Correlation coefficient at each lag, normalized by ``len(x)``.
    """
    from scipy import signal
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    r = signal.correlate(y, x, mode="full") / len(x)
    lags = signal.correlation_lags(len(y), len(x), mode="full") / fs
    m = np.abs(lags) <= max_lag_s
    return lags[m], r[m]


def bounded_peak_lag(x, y, max_lag_s, fs, bound_s):
    """Largest positive correlation of :func:`xcorr` within a lag bound.

    Narrow-band signals correlate again at every period, so the search is
    restricted to ``|lag| <= bound_s``.

    Parameters
    ----------
    x, y : numpy.ndarray
        Evenly sampled series of equal length, without NaN.
    max_lag_s : float
        Largest absolute lag [s] passed to :func:`xcorr`.
    fs : float
        Sampling rate [Hz].
    bound_s : float
        Largest absolute lag [s] searched for the peak.

    Returns
    -------
    r : float
        Peak correlation coefficient.
    lag : float
        Lag [s] of the peak; positive when *x* leads *y*.
    """
    lags, r = xcorr(x, y, max_lag_s, fs)
    core = np.abs(lags) <= bound_s
    j = np.where(core)[0][np.argmax(r[core])]
    return r[j], lags[j]


def pair_on_grid(imu_df, son, t0, t1, fs, *, pad_s, min_records, trim_s,
                 min_overlap_s):
    """IMU vertical acceleration and sonic wind on one uniform grid.

    IMU records within *pad_s* of ``[t0, t1)`` are screened with
    :func:`utespac.raw_processing.imu.flag_spikes` at its default limits,
    duplicate timestamps are averaged, and both records are linearly
    interpolated onto a grid spanning their overlap, trimmed by *trim_s*
    at each end. Requires pandas.

    Parameters
    ----------
    imu_df : pandas.DataFrame
        IMU records with the logger channel names of
        :mod:`utespac.raw_processing.imu` (``time``, ``ax`` ... ``yaw``).
    son : pandas.DataFrame
        Sonic records with columns ``TIMESTAMP``, ``Ux``, ``Uy``, ``Uz``.
    t0, t1 : datetime-like
        Window start (inclusive) and end (exclusive).
    fs : float
        Grid rate [Hz].
    pad_s : float
        IMU records taken beyond each end of the window [s].
    min_records : int
        Fewest IMU or sonic records accepted in the window.
    trim_s : float
        Overlap trimmed from each end before gridding [s].
    min_overlap_s : float
        Shortest trimmed overlap accepted [s].

    Returns
    -------
    dict or None
        ``grid`` (epoch seconds), ``rate`` (screened IMU records per
        second), ``az`` [g], ``u``, ``v``, ``w`` [m/s] on the grid and
        ``spd`` (mean horizontal speed [m/s]); None when a record or the
        overlap falls short of the limits.
    """
    import pandas as pd

    from utespac.averaging import epoch_s
    from utespac.raw_processing.imu import flag_spikes

    d = imu_df[(imu_df["time"] >= pd.Timestamp(t0) - pd.Timedelta(seconds=pad_s))
               & (imu_df["time"] < pd.Timestamp(t1) + pd.Timedelta(seconds=pad_s))]
    if len(d) < min_records:
        return None
    d = d[~flag_spikes(d)].groupby("time", as_index=False).mean()
    s = son[(son["TIMESTAMP"] >= t0) & (son["TIMESTAMP"] < t1)]
    if len(s) < min_records:
        return None
    ti, ts = epoch_s(d["time"]), epoch_s(s["TIMESTAMP"])
    lo, hi = max(ti[0], ts[0]) + trim_s, min(ti[-1], ts[-1]) - trim_s
    if hi - lo < min_overlap_s:
        return None
    grid = np.arange(lo, hi, 1 / fs)
    out = {"grid": grid, "rate": len(d) / (ti[-1] - ti[0])}
    out["az"] = np.interp(grid, ti, d["az"].to_numpy())
    for src, name in (("Ux", "u"), ("Uy", "v"), ("Uz", "w")):
        out[name] = np.interp(grid, ts, s[src].to_numpy().astype(float))
    out["spd"] = float(np.hypot(out["u"], out["v"]).mean())
    return out
