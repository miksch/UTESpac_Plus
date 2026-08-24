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
  conventions happens in :func:`utespac.stages.motion`, not here.
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
           "highpass", "lowpass", "integrate", "fill_gaps", "G"]


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
