"""Wind direction, speed, the tower-shadow flag and friction velocity.

:func:`wind_direction_speed` and :func:`shadow_flag` are the primitives
``utespac.stages.wind``, ``find_global_pf`` and the flux stage share.
:func:`in_direction_window` and :func:`direction_window_fraction` test
directions against a symmetric window, :func:`compass_label` names a
bearing, and :func:`ustar_from_rotated` / :func:`ustar_double_rotation`
give the friction velocity of one period of winds.
"""

from typing import Tuple

import numpy as np

MANUFACTURER_RMYOUNG = 0
MANUFACTURER_CAMPBELL = 1
MANUFACTURER_GILL = 2


def sonic_axes(u, v, manufacturer: int):
    """``(u, v)`` in the Campbell convention (u toward the transducer array
    reference, v to the left): RMYoung swaps and negates, Gill negates both."""
    if int(manufacturer) == MANUFACTURER_RMYOUNG:
        return v, -np.asarray(u, dtype=float)
    if int(manufacturer) == MANUFACTURER_GILL:
        return -np.asarray(u, dtype=float), -np.asarray(v, dtype=float)
    return u, v


def wind_direction_speed(u, v, bearing: float, manufacturer: int = MANUFACTURER_CAMPBELL
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Meteorological wind direction [deg, 0-360) and speed from sonic u, v.

    *bearing* is the sonic head orientation (``SonicLevel.orientation``);
    the manufacturer axis convention is applied first.
    """
    u_d, v_d = sonic_axes(u, v, manufacturer)
    direction = (np.degrees(np.arctan2(-np.asarray(v_d), np.asarray(u_d))) + bearing) % 360.0
    return direction, np.hypot(u_d, v_d)


def shadow_sector(tower_bearing: float, sonic_bearing: float, envelope: float
                  ) -> Tuple[float, float]:
    """``(min_a, max_a)`` of the tower-shadow sector centred on
    ``tower_bearing + sonic_bearing``; ``min_a > max_a`` when it wraps 360."""
    centre = (float(tower_bearing) + float(sonic_bearing)) % 360.0
    if centre + envelope > 360.0:
        return centre - envelope, (centre + envelope) - 360.0
    if centre - envelope < 0.0:
        return 360.0 + centre - envelope, centre + envelope
    return centre - envelope, centre + envelope


def shadow_flag(direction, tower_bearing: float, sonic_bearing: float, envelope: float
                ) -> Tuple[np.ndarray, float, float]:
    """Flag (1.0 inside the tower-shadow sector) and the sector bounds."""
    min_a, max_a = shadow_sector(tower_bearing, sonic_bearing, envelope)
    d = np.asarray(direction, dtype=float)
    if min_a > max_a:       # wraps through north
        inside = (d > min_a) | (d < max_a)
    else:
        inside = (d > min_a) & (d < max_a)
    return inside.astype(float), min_a, max_a


# ── direction windows and labels ─────────────────────────────────────────────

COMPASS_16 = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
              "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")


def compass_label(bearing_deg: float) -> str:
    """Sixteen-point compass label of a bearing [deg]; empty when not finite."""
    if not np.isfinite(bearing_deg):
        return ""
    return COMPASS_16[int(np.round(np.mod(bearing_deg, 360.0) / 22.5)) % 16]


def in_direction_window(direction, centre: float, half_width: float) -> np.ndarray:
    """Directions within *half_width* of *centre* [deg], both edges included.

    The comparison is on the shortest angular separation, so the window is
    safe across the 0/360 seam. Missing directions return False.

    Parameters
    ----------
    direction : array_like
        Directions [deg], any real value.
    centre, half_width : float
        Window centre and half-width [deg].

    Returns
    -------
    numpy.ndarray of bool
    """
    d = np.asarray(direction, dtype=float)
    delta = np.abs((d - float(centre) + 180.0) % 360.0 - 180.0)
    with np.errstate(invalid="ignore"):
        return np.isfinite(d) & (delta <= float(half_width) + 1e-9)


def direction_window_fraction(u, v, centre: float, half_width: float) -> float:
    """Fraction of samples whose instantaneous direction lies in a window.

    The direction is the sonic-frame azimuth of :func:`wind_direction_speed`
    with zero bearing and the Campbell axis convention: the direction the
    wind comes from, clockwise from the sonic +x axis. Samples with a
    missing component are left out of the fraction.

    Parameters
    ----------
    u, v : array_like
        Horizontal wind components.
    centre, half_width : float
        Window centre and half-width [deg], as in :func:`in_direction_window`.

    Returns
    -------
    float
        Fraction in [0, 1]; NaN when no sample is finite.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    ok = np.isfinite(u) & np.isfinite(v)
    if not ok.any():
        return np.nan
    d, _ = wind_direction_speed(u[ok], v[ok], 0.0, MANUFACTURER_CAMPBELL)
    return float(in_direction_window(d, centre, half_width).mean())


# ── friction velocity ────────────────────────────────────────────────────────

def ustar_from_rotated(uvw: np.ndarray) -> float:
    """Friction velocity [m/s] from one period of already rotated winds.

    ``u* = (<u'w'>^2 + <v'w'>^2)^(1/4)`` over the rows of *uvw* with all
    three components finite; no rotation is applied.

    Parameters
    ----------
    uvw : numpy.ndarray
        ``(n, 3)`` winds in the rotated frame, for example planar fit
        followed by yaw into the mean wind.

    Returns
    -------
    float
        NaN for fewer than 100 complete rows.
    """
    u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]
    ok = np.isfinite(u) & np.isfinite(v) & np.isfinite(w)
    if ok.sum() < 100:
        return np.nan
    u, v, w = u[ok] - u[ok].mean(), v[ok] - v[ok].mean(), w[ok] - w[ok].mean()
    return float((np.mean(u * w) ** 2 + np.mean(v * w) ** 2) ** 0.25)


def ustar_double_rotation(u, v, w) -> float:
    """Friction velocity [m/s] from one period of sonic winds.

    Streamwise-aligned double rotation (Kaimal and Finnigan 1994): the mean
    wind is rotated into +u and the mean vertical velocity to zero, then
    ``u* = (<u'w'>^2 + <v'w'>^2)^(1/4)``. No planar fit, no spectral or
    density correction.

    Parameters
    ----------
    u, v, w : numpy.ndarray
        Wind components of one period in the instrument frame.

    Returns
    -------
    float
        NaN for fewer than 100 samples with all three components finite.
    """
    ok = np.isfinite(u) & np.isfinite(v) & np.isfinite(w)
    if ok.sum() < 100:
        return np.nan
    u, v, w = u[ok], v[ok], w[ok]
    theta = np.arctan2(v.mean(), u.mean())
    u1 = u * np.cos(theta) + v * np.sin(theta)
    v1 = -u * np.sin(theta) + v * np.cos(theta)
    phi = np.arctan2(w.mean(), u1.mean())
    u2 = u1 * np.cos(phi) + w * np.sin(phi)
    w2 = -u1 * np.sin(phi) + w * np.cos(phi)
    uw = np.mean((u2 - u2.mean()) * (w2 - w2.mean()))
    vw = np.mean((v1 - v1.mean()) * (w2 - w2.mean()))
    return float((uw ** 2 + vw ** 2) ** 0.25)

