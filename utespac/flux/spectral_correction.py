"""Analytical spectral correction of eddy-covariance fluxes (Massman 2000).

Implements the closed-form approximation of Massman (2000), "A simple
method for estimating frequency response corrections for eddy covariance
systems", Agric. For. Meteorol. 104, 185-198,
doi:10.1016/S0168-1923(00)00164-7. The paper addresses tower
eddy-covariance systems, a sonic anemometer with an open-path or a
closed-path scalar sensor, and was tested for
"0<=z/L<=2 and 0.1 ms-1<=u<=20 ms-1" (Sect. 4, p. 195) against numerical
integration of the Kaimal et al. (1972) flat-terrain cospectra, with a
30 min block, 10 Hz sampling, 5 m measurement height and sonic paths of
0.15 m (Sect. 3.1, p. 191).

Each attenuation process is replaced by a first-order filter with the
equivalent time constant of Table 1; the low-pass constants combine by
Eq. (9); the cospectrum is the Horst (1997) form, Eq. (8), with peak
frequency ``f_x = n_x u / z``; Eq. (11) and Table 2 give the ratio of
measured to true flux. :func:`massman2000_factor` returns its inverse, the
multiplicative correction. Only the scalar-flux branch is implemented
(``n_x`` of Horst 1997, sonic line averaging for scalar flux). Tube
attenuation, volume averaging and the numerical integral of Eq. (3) are
not implemented.
"""

import numpy as np

ALPHA_UNSTABLE = 0.925      # Table 2 note: alpha for z/L <= 0 [CITED]
ALPHA_STABLE = 1.0          # Table 2 note: alpha for stable cospectra [CITED]
NX_UNSTABLE = 0.085         # Sect. 2.4 below Eq. (7): n_x for z/L <= 0 [CITED]
# Lower bound of the scalar-sensor response time above which the stable
# branch carries the extra term (1 + 0.9p)/(1 + p): "a first-order scalar
# instrument (which herein is assumed to have an intrinsic response time
# tau_beta>=0.1 s)" (p. 191); Table 2 labels the row "0.1-0.3 s" [CITED].
SLOW_SENSOR_TAU = 0.1       # [s]


# ---------------------------------------------------------------------------
# Equivalent first-order time constants, Table 1 (u is the horizontal wind)
# ---------------------------------------------------------------------------

def tau_sonic_line_scalar(path_len, U):
    """Sonic line averaging for a scalar flux, ``l_w / (8.4 u)``.

    Table 1, Kristensen and Fitzjarrald (1984) [CITED].

    Parameters
    ----------
    path_len : array_like
        Sonic vertical path length ``l_w`` [m].
    U : array_like
        Mean horizontal wind speed [m/s].

    Returns
    -------
    ndarray
        Equivalent time constant [s].
    """
    return np.asarray(path_len, float) / (8.4 * np.asarray(U, float))


def tau_lateral_separation(sep, U):
    """Lateral sensor separation, ``l_lat / (1.1 u)``.

    Table 1, Kristensen and Jensen (1979) [CITED].

    Parameters
    ----------
    sep : array_like
        Lateral separation ``l_lat`` [m].
    U : array_like
        Mean horizontal wind speed [m/s].

    Returns
    -------
    ndarray
        Equivalent time constant [s].
    """
    return np.asarray(sep, float) / (1.1 * np.asarray(U, float))


def tau_longitudinal_separation(sep, U, tau_sensor=0.0):
    """Longitudinal sensor separation, with or without a first-order sensor.

    Without a first-order instrument, ``l_lon / (1.05 u)`` (Table 1, Hicks
    1972; Kristensen and Jensen 1979) [CITED]. With one of time constant
    ``tau_1 > 0``, ``(4/3) [(l_lon/u) ((1/2)(l_lon/u) + tau_1)]^(1/2)``
    (Table 1, Appendix A) [CITED], with the tube lag ``L_t/U_t`` taken as
    removed by time shifting, as the paper assumes (p. 189), so that
    ``l*_lon = l_lon``. Table 1 (p. 188) prints
    ``(4/3)[l*_lon/u(1/2(l*_lon/u)+tau_1)]^(1/2)``; its ``tau_1 -> 0``
    limit is ``0.943 l_lon/u`` [DERIVED].

    Parameters
    ----------
    sep : array_like
        Longitudinal separation ``l_lon`` [m].
    U : array_like
        Mean horizontal wind speed [m/s].
    tau_sensor : array_like, optional
        Intrinsic time constant of the scalar sensor ``tau_1`` [s].

    Returns
    -------
    ndarray
        Equivalent time constant [s].
    """
    t_sep = np.asarray(sep, float) / np.asarray(U, float)
    tau_1 = np.asarray(tau_sensor, float)
    with np.errstate(invalid="ignore"):
        first_order = (4.0 / 3.0) * np.sqrt(t_sep * (0.5 * t_sep + tau_1))
    return np.where(tau_1 > 0.0, first_order, t_sep / 1.05)


def tau_scalar_line(path_len, U):
    """Line averaging by the scalar sensor, ``l_scalar / (4.0 u)``.

    Table 1, Gurvich (1962); Silverman (1968) [CITED].

    Parameters
    ----------
    path_len : array_like
        Scalar sensor path length [m].
    U : array_like
        Mean horizontal wind speed [m/s].

    Returns
    -------
    ndarray
        Equivalent time constant [s].
    """
    return np.asarray(path_len, float) / (4.0 * np.asarray(U, float))


def tau_anti_noise(fs):
    """Butterworth anti-noise filter at the Nyquist frequency, ``1/(pi f_s)``.

    Table 1, Moore (1986) [CITED]; equals ``1/(2 pi f_1/2)`` with the
    half-power point at ``f_s/2`` (Sect. 2.4) [DERIVED].

    Parameters
    ----------
    fs : array_like
        Sampling frequency [Hz].

    Returns
    -------
    ndarray
        Equivalent time constant [s].
    """
    return 1.0 / (np.pi * np.asarray(fs, float))


def tau_block_average(avg_period_s):
    """High-pass filtering by block averaging, ``T_b / 2.8``.

    Table 1, Kaimal et al. (1989) [CITED].

    Parameters
    ----------
    avg_period_s : array_like
        Block averaging period ``T_b`` [s].

    Returns
    -------
    ndarray
        Equivalent time constant ``tau_b`` [s].
    """
    return np.asarray(avg_period_s, float) / 2.8


def tau_linear_detrend(avg_period_s):
    """High-pass filtering by linear detrending, ``T_b / 5.3``.

    Table 1, Kristensen (1998); Rannik and Vesala (1999) [CITED].

    Parameters
    ----------
    avg_period_s : array_like
        Block averaging period ``T_b`` [s].

    Returns
    -------
    ndarray
        Equivalent time constant ``tau_h`` [s].
    """
    return np.asarray(avg_period_s, float) / 5.3


def combined_time_constant(*taus):
    """Combined low-pass time constant, ``tau_e = (sum tau_i^2)^(1/2)``.

    Eq. (9) [CITED].

    Parameters
    ----------
    *taus : array_like
        Equivalent first-order time constants of the low-pass filters [s].

    Returns
    -------
    ndarray
        ``tau_e`` [s], broadcast over the inputs.
    """
    arrs = np.broadcast_arrays(*[np.asarray(t, float) for t in taus])
    return np.sqrt(sum(t ** 2 for t in arrs))


def peak_frequency_nx(zeta):
    """Dimensionless scalar cospectral peak frequency, ``n_x = f_x z/u``.

    ``n_x = 0.085`` for ``z/L <= 0`` and ``n_x = 2.0 - 1.915/(1 + 0.5 z/L)``
    for ``z/L > 0`` (Sect. 2.4 below Eq. (7), after Horst 1997) [CITED].

    Parameters
    ----------
    zeta : array_like
        Stability ``z/L``.

    Returns
    -------
    ndarray
        ``n_x``; NaN where ``zeta`` is NaN.
    """
    zeta = np.asarray(zeta, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        stable = 2.0 - 1.915 / (1.0 + 0.5 * zeta)
    nx = np.where(zeta > 0.0, stable, NX_UNSTABLE)
    return np.where(np.isnan(zeta), np.nan, nx)


def _eq11(a, b, p):
    """Eq. (11), products divided through, finite as ``a, b -> inf``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        high = 1.0 / ((1.0 + 1.0 / a) * (1.0 + 1.0 / b))
        mixed = 1.0 / ((1.0 + p / a) * (1.0 + p / b))
        low = 1.0 / (1.0 + p)
        trend = 1.0 - p / ((a + 1.0) * (a + p))
    return high * mixed * low * trend


def attenuation_unstable(a, b, p, alpha=ALPHA_UNSTABLE):
    """Measured-to-true flux ratio for ``z/L <= 0``.

    Table 2, unstable row: Eq. (11) with ``a, b, p`` raised to ``alpha``
    (p. 190-191, Table 2) [CITED].

    Parameters
    ----------
    a, b, p : array_like
        ``2 pi f_x tau_h``, ``2 pi f_x tau_b`` and ``2 pi f_x tau_e``;
        ``a = inf`` when no trend removal is applied.
    alpha : float, optional
        Exponent; 0.925 for the flat-terrain unstable cospectrum, 1 for a
        more sharply peaked one (Table 2 note; Sect. 3.1, p. 194) [CITED].

    Returns
    -------
    ndarray
        ``(Flux)_m / Flux``.
    """
    a, b, p = (np.asarray(x, float) ** alpha for x in (a, b, p))
    return _eq11(a, b, p)


def attenuation_stable(a, b, p, slow_sensor=False, alpha=ALPHA_STABLE):
    """Measured-to-true flux ratio for ``z/L > 0``.

    Table 2, stable rows: Eq. (11), times ``(1 + 0.9 p)/(1 + p)`` for a
    scalar instrument with 0.1-0.3 s response time (p. 191, Table 2)
    [CITED].

    Parameters
    ----------
    a, b, p : array_like
        As in :func:`attenuation_unstable`.
    slow_sensor : array_like of bool, optional
        Apply the first-order-sensor term.
    alpha : float, optional
        Exponent on ``a, b, p``; 1 for "the stable flat terrain cospectra
        of Kaimal et al. (1972)" (Table 2 note) [CITED].

    Returns
    -------
    ndarray
        ``(Flux)_m / Flux``.
    """
    a, b, p = (np.asarray(x, float) ** alpha for x in (a, b, p))
    ratio = _eq11(a, b, p)
    return np.where(slow_sensor, ratio * (1.0 + 0.9 * p) / (1.0 + p), ratio)


def massman2000_factor(U, z_eff, zeta, fs, avg_period_s, tau_sensor=0.0,
                       sep_lateral=0.0, sep_longitudinal=0.0,
                       path_len_sonic=0.0, path_len_scalar=0.0,
                       detrend="linear", anti_noise_filter=False):
    """Spectral correction factor for a scalar flux, true over measured.

    ``f_x = n_x u / z`` with ``n_x`` from :func:`peak_frequency_nx`;
    ``a = 2 pi f_x tau_h``, ``b = 2 pi f_x tau_b``, ``p = 2 pi f_x tau_e``
    (Eq. (10)); the ratio follows Table 2 and the factor is its inverse
    ("the correction factor is the inverse of Eq. (3)", p. 192) [CITED].

    Five extrapolations beyond the tested range are applied without a
    warning or a label on the result [ASSUMED]: the stable branch is used
    for any ``z/L > 0``, where Table 2 reads "Stable atmospheric conditions
    (0<z/L<=2)"; the slow-sensor term for any ``tau_sensor >= 0.1`` s, where
    Table 2 reads "0.1-0.3 s response time"; the stable row without that
    term for ``0 < tau_sensor < 0.1`` s, where Table 2 labels it
    "Fast-response open path systems" (p. 191); any positive wind, where
    the method was tested for "0.1 ms-1<=u<=20 ms-1" (Sect. 4, p. 195);
    and any separation, where the paper "assumes that the maximum
    horizontal separation distance is 0.30 m" (p. 189).

    Parameters
    ----------
    U : array_like
        Mean horizontal wind speed [m/s].
    z_eff : array_like
        Measurement height above the zero-plane displacement, ``z - d`` [m].
    zeta : array_like
        Stability ``z/L``.
    fs : array_like
        Sampling frequency [Hz]; enters only through the anti-noise filter.
    avg_period_s : array_like
        Block averaging period ``T_b`` [s]; ``inf`` removes the block term,
        and the linear-detrend term, the ``b -> inf`` and ``a -> inf``
        limits of Eq. (11) [DERIVED].
    tau_sensor : array_like, optional
        First-order time constant of the scalar sensor ``tau_beta`` [s];
        enters Eq. (9) directly, as one of the low-pass filters of "finite
        response times" (Sect. 2.4, p. 190), with transfer function
        ``H_beta = 1/(1 + omega^2 tau_beta^2)`` in Eq. (A.3) [CITED], and
        through the longitudinal separation.
    sep_lateral, sep_longitudinal : array_like, optional
        Sonic-to-scalar separation across and along the wind [m].
    path_len_sonic : array_like, optional
        Sonic vertical path length [m].
    path_len_scalar : array_like, optional
        Scalar sensor path length [m].
    detrend : {'linear', 'constant'} or float, optional
        Trend removal: ``'linear'`` (``tau_h = T_b/5.3``, Table 1) [CITED],
        the default [HOUSE]; the paper's scenarios use a recursive filter
        with ``tau_r = 450`` s (p. 191) and linear detrending is "not
        directly employed in the present study" (p. 187);
        ``'constant'`` (block mean only, ``a -> inf`` in Eq. (11))
        [DERIVED]; or the time constant ``tau_r`` [s] of a high-pass
        recursive filter (Table 1) [CITED].
    anti_noise_filter : bool, optional
        Include a Butterworth anti-noise filter with its half-power point at
        ``fs/2`` (Table 1); the paper's scenarios exclude it (p. 189).

    Returns
    -------
    ndarray or float
        Factor ``>= 1`` to multiply the measured flux by, broadcast over the
        array inputs; NaN where ``U``, ``z_eff`` or ``avg_period_s`` is not
        positive, ``tau_sensor``, ``sep_lateral``, ``sep_longitudinal``,
        ``path_len_sonic`` or ``path_len_scalar`` is negative, ``fs`` is
        not positive with ``anti_noise_filter`` set, or ``zeta`` is NaN;
        also NaN where ``U``, ``z_eff``, ``tau_sensor``, ``sep_lateral``
        or ``sep_longitudinal`` is infinite.

    Raises
    ------
    ValueError
        If ``detrend`` is a ``bool``, or neither a known name nor a positive
        number.
    """
    U = np.asarray(U, float)
    z_eff = np.asarray(z_eff, float)
    zeta = np.asarray(zeta, float)
    T_b = np.asarray(avg_period_s, float)
    tau_sensor = np.asarray(tau_sensor, float)
    sep_lateral = np.asarray(sep_lateral, float)
    sep_longitudinal = np.asarray(sep_longitudinal, float)
    path_len_sonic = np.asarray(path_len_sonic, float)
    path_len_scalar = np.asarray(path_len_scalar, float)
    fs = np.asarray(fs, float)

    if isinstance(detrend, (bool, np.bool_)):
        raise ValueError(f"detrend must be a name or a time constant, "
                         f"got bool {detrend!r}")
    if isinstance(detrend, str):
        if detrend == "linear":
            tau_h = tau_linear_detrend(T_b)
        elif detrend == "constant":
            tau_h = np.inf
        else:
            raise ValueError(f"unknown detrend {detrend!r}; "
                             "use 'linear', 'constant' or a time constant")
    elif np.isscalar(detrend) and float(detrend) > 0.0:
        tau_h = float(detrend)
    else:
        raise ValueError("detrend time constant must be positive, "
                         f"got {detrend!r}")

    valid = ((U > 0.0) & (z_eff > 0.0) & (T_b > 0.0) & ~np.isnan(zeta)
             & (tau_sensor >= 0.0) & (sep_lateral >= 0.0)
             & (sep_longitudinal >= 0.0) & (path_len_sonic >= 0.0)
             & (path_len_scalar >= 0.0))
    if anti_noise_filter:
        valid = valid & (fs > 0.0)
    U_ok = np.where(valid, U, 1.0)

    lows = [tau_sensor,
            tau_lateral_separation(sep_lateral, U_ok),
            tau_longitudinal_separation(sep_longitudinal, U_ok, tau_sensor),
            tau_sonic_line_scalar(path_len_sonic, U_ok),
            tau_scalar_line(path_len_scalar, U_ok)]
    if anti_noise_filter:
        lows.append(tau_anti_noise(np.where(fs > 0.0, fs, 1.0)))
    tau_e = combined_time_constant(*lows)

    f_x = peak_frequency_nx(zeta) * U_ok / np.where(valid, z_eff, 1.0)
    omega_x = 2.0 * np.pi * f_x
    with np.errstate(invalid="ignore", divide="ignore"):
        a = omega_x * tau_h
        b = omega_x * tau_block_average(T_b)
        p = omega_x * tau_e
        a, b, p = np.broadcast_arrays(a, b, p)

        slow = tau_sensor >= SLOW_SENSOR_TAU
        ratio = np.where(zeta > 0.0,
                         attenuation_stable(a, b, p, slow_sensor=slow),
                         attenuation_unstable(a, b, p))
        factor = np.where(valid, 1.0 / ratio, np.nan)
    return factor[()] if factor.ndim == 0 else factor
