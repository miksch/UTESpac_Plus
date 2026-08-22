"""calc_dissipation_rate – TKE dissipation rate via structure function."""

import numpy as np


def calc_structure_function(u: np.ndarray, max_lag: int = None) -> np.ndarray:
    """Second-order longitudinal structure function D_LL at lags 1..max_lag.

    Parameters
    ----------
    u : 1-D ndarray
        Velocity time series; NaN samples are ignored pairwise.
    max_lag : int, optional
        Largest lag in samples (default ``len(u) // 2``).

    Returns
    -------
    sf : ndarray, length max_lag
        ``sf[k-1] = nanmean((u[k:] - u[:-k])**2)``.
    """
    u = np.asarray(u, dtype=float)
    n = len(u)
    r_max = n // 2 if max_lag is None else min(int(max_lag), n - 1)
    sf = np.full(max(r_max, 0), np.nan)
    for ri in range(1, r_max + 1):
        diff = (u[ri:] - u[:-ri]) ** 2
        if np.any(np.isfinite(diff)):
            sf[ri - 1] = np.nanmean(diff)
    return sf


def calc_dissipation_rate(u_prime: np.ndarray, u_mean: float, dt: float,
                          lag_window=(0.1, 2.0), C2: float = 2.0) -> float:
    """Estimate TKE dissipation rate ε from the compensated structure function.

    In the inertial subrange D_LL(r) = C2 (ε r)^(2/3), so
    D_LL(r) / (C2 r^(2/3)) is flat and equal to ε^(2/3). That compensated
    function is averaged over the lags whose time separation τ = lag·dt
    falls inside ``lag_window`` and raised to 3/2. Spatial lags follow
    Taylor's hypothesis, r = ū τ.

    Parameters
    ----------
    u_prime : 1-D ndarray
        Streamwise velocity perturbations [m/s].
    u_mean : float
        Mean streamwise velocity [m/s]; must be positive.
    dt : float
        Sample interval [s].
    lag_window : (float, float)
        Time-lag window [s] assumed to lie in the inertial subrange.
        Default 0.1–2 s: above the sonic path-averaging scale at typical
        wind speeds, below the production scale ~z for z of a few metres
        and up.
    C2 : float
        Kolmogorov structure-function constant (Pope 2000, Sec. 6.5: ≈ 2.0).

    Returns
    -------
    epsilon : float
        Dissipation rate [m²/s³]; NaN when fewer than three usable lags fall
        in the window or ``u_mean`` is not positive.

    References
    ----------
    Kolmogorov, A.N. (1941) Dokl. Akad. Nauk SSSR 30, 301-305 — the 2/3 law.
    Pope, S.B. (2000) Turbulent Flows, Cambridge Univ. Press, Sec. 6.5
      (D_LL = C2 (εr)^(2/3), C2 ≈ 2.0). DOI:10.1017/CBO9780511840531
    Sreenivasan, K.R. (1995) Phys. Fluids 7(11), 2778-2784 — universality
      of the Kolmogorov constant. DOI:10.1063/1.868656
    Chamecki, M. & Dias, N.L. (2004) Q. J. R. Meteorol. Soc. 130, 2733-2752
      — surface-layer application. DOI:10.1256/qj.03.155
    """
    u_prime = np.asarray(u_prime, dtype=float)
    n = len(u_prime)
    if n < 4 or not np.isfinite(u_mean) or u_mean <= 0 or not dt > 0:
        return np.nan

    lags = np.arange(1, n // 2 + 1)
    tau = lags * dt
    sel = (tau >= lag_window[0]) & (tau <= lag_window[1])
    if sel.sum() < 3:
        return np.nan
    max_lag = int(lags[sel].max())

    sf = calc_structure_function(u_prime, max_lag)
    r = u_mean * tau[:max_lag]
    comp = sf[sel[:max_lag]] / (C2 * r[sel[:max_lag]] ** (2.0 / 3.0))
    comp = comp[np.isfinite(comp) & (comp > 0)]
    if len(comp) < 3:
        return np.nan
    return float(np.mean(comp) ** 1.5)
