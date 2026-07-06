"""calc_dissipation_rate – TKE dissipation rate via structure function."""

import numpy as np


def calc_structure_function(u: np.ndarray) -> np.ndarray:
    """Second-order longitudinal structure function D_LL(r).

    Parameters
    ----------
    u : 1-D ndarray  velocity time series

    Returns
    -------
    sf : ndarray, length floor(N/2)
    """
    n = len(u)
    r_max = n // 2
    sf = np.zeros(r_max)
    for ri in range(1, r_max + 1):
        diff = (u[ri:] - u[:-ri]) ** 2
        sf[ri - 1] = np.nanmean(diff)
    return sf


def calc_dissipation_rate(u_prime: np.ndarray, u_mean: float, dt: float) -> float:
    """Estimate TKE dissipation rate ε from the structure function.

    Parameters
    ----------
    u_prime : 1-D ndarray   Velocity perturbation time series [m/s].
    u_mean  : float          Mean velocity to convert time lags to spatial lags [m/s].
    dt      : float          Time step [s].

    Returns
    -------
    epsilon : float   Dissipation rate [m²/s³].

    References
    ----------
    Inertial-subrange 2/3 law, D_LL(r) = C2 * eps^(2/3) * r^(2/3):
      Kolmogorov, A.N. (1941) "The Local Structure of Turbulence in
      Incompressible Viscous Fluid for Very Large Reynolds Numbers."
      Dokl. Akad. Nauk SSSR 30, 301-305.
    Kolmogorov constant C2 ~ 2.0:
      Pope, S.B. (2000) "Turbulent Flows." Cambridge Univ. Press,
      Sec. 6.2, Eq. 6.30. DOI:10.1017/CBO9780511840531
      Sreenivasan, K.R. (1995) "On the universality of the Kolmogorov
      constant." Phys. Fluids 7(11), 2778-2784. DOI:10.1063/1.868656
    Atmospheric surface-layer application:
      Chamecki, M. & Dias, N.L. (2004) "The local isotropy hypothesis and
      the turbulent kinetic energy dissipation rate in the atmospheric
      surface layer." Q. J. R. Meteorol. Soc. 130, 2733-2752.
      DOI:10.1256/qj.03.155
    """
    u_prime = np.asarray(u_prime, dtype=float)
    n = len(u_prime)
    sf = calc_structure_function(u_prime)

    r_values = np.linspace(dt * u_mean, (n // 2) * dt * u_mean, n // 2)
    y2 = r_values ** (2.0 / 3.0)

    # Find the index where the 2/3 power law best matches the structure function
    n_fit = min(20, len(y2))
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = np.abs(np.log(y2[:n_fit]) - np.log(np.where(sf[:n_fit] > 0, sf[:n_fit], np.nan)))
    I = int(np.nanargmin(diff))

    if not sf[I] > 0:
        return np.nan
    # eps = (D_LL / (C2 * r^(2/3)))^(3/2) with C2 = 2 (0.35 ~ 2^(-3/2))
    epsilon = (sf[I] / y2[I]) ** (3.0 / 2.0) * 0.35
    return float(epsilon) if not np.isnan(epsilon) else np.nan
