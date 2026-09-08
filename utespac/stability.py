"""stability – integrated Monin-Obukhov stability functions psi_m, psi_h.

Sign convention (Foken 2008 eq. 2.83): the diabatic profile is
u(z) = (u*/kappa) [ln(z/z0) - psi_m(z/L)], so psi > 0 unstable and
psi < 0 stable.

Sources are the Businger-Dyer-type rows of Foken (2008) Appendix A4
("Universal Functions", pp. 249-252): unstable phi_m = (1 - g_m z)^(-1/4)
and phi_h = a_h (1 - g_h z)^(-1/2), stable phi linear in z. Only these
integrate to the Paulson (1970) closed forms Foken gives in eqs.
2.85-2.89; the appendix's other entries (Swinbank, Zilitinkevich/
Tschalikov, Skeib, Gavrilov/Petrov, Beljaars/Holtslag, King, Handorf)
are not power-law/linear and Foken states no integrals for them, so
they are not offered here.
"""

import numpy as np

# [CITED] Foken (2008) "Micrometeorology" Appendix A4 and Table 2.8.
# Coefficients (gamma_m, beta_m, a_h, gamma_h, beta_h) in
#   unstable:  phi_m = (1 - gamma_m z)^(-1/4),  phi_h = a_h (1 - gamma_h z)^(-1/2)
#   stable:    phi_m = 1 + beta_m z,            phi_h = a_h + beta_h z
# "hoegstroem1988" is Businger et al. (1971) recalculated by Högström
# (1988) for kappa = 0.40 (Foken's recommended set, eqs. 2.85-2.89);
# "businger1971" the original kappa = 0.35 fit; "dyer1974" kappa = 0.41.
# Foken notes the functions are generally defined for -1 < z/L < 1
# (A4 ranges differ per source); values are returned uncapped.
PSI_SOURCES = {
    "hoegstroem1988": (19.3, 6.0, 0.95, 11.6, 7.8),
    "businger1971":   (15.0, 4.7, 0.74, 9.0, 4.7),
    "dyer1974":       (16.0, 5.0, 1.00, 16.0, 5.0),
}
DEFAULT_SOURCE = "hoegstroem1988"


def _coeffs(source):
    try:
        return PSI_SOURCES[source]
    except KeyError:
        raise ValueError(f"unknown stability source {source!r}; "
                         f"one of {sorted(PSI_SOURCES)}") from None


def psi_m(zeta, source=DEFAULT_SOURCE):
    """Integrated stability function for momentum, psi_m(z/L).

    Unstable (zeta < 0): ln[((1+x^2)/2) ((1+x)/2)^2] - 2 atan(x) + pi/2
    with x = (1 - gamma_m zeta)^(1/4) (Paulson 1970; Foken 2008 eqs.
    2.85, 2.87); stable (zeta >= 0): -beta_m zeta (eq. 2.88). The
    coefficients follow *source* (see PSI_SOURCES). NaN in, NaN out.
    """
    gamma_m, beta_m, _, _, _ = _coeffs(source)
    zeta = np.asarray(zeta, dtype=float)
    out = np.full(zeta.shape, np.nan)
    stab = zeta >= 0
    out[stab] = -beta_m * zeta[stab]
    uns = zeta < 0
    x = (1.0 - gamma_m * zeta[uns]) ** 0.25
    out[uns] = (np.log((1.0 + x ** 2) / 2.0 * ((1.0 + x) / 2.0) ** 2)
                - 2.0 * np.arctan(x) + np.pi / 2.0)
    return out if out.shape else float(out)


def psi_h(zeta, source=DEFAULT_SOURCE):
    """Integrated stability function for heat, psi_h(z/L).

    Unstable (zeta < 0): 2 ln((1+y)/2) with y = a_h (1 - gamma_h zeta)^(1/2)
    (Foken 2008 eqs. 2.86, 2.87); stable (zeta >= 0): -beta_h zeta
    (eq. 2.89, the a_h offset dropped as Foken drops the 0.95). The
    coefficients follow *source* (see PSI_SOURCES). NaN in, NaN out.
    """
    _, _, a_h, gamma_h, beta_h = _coeffs(source)
    zeta = np.asarray(zeta, dtype=float)
    out = np.full(zeta.shape, np.nan)
    stab = zeta >= 0
    out[stab] = -beta_h * zeta[stab]
    uns = zeta < 0
    y = a_h * (1.0 - gamma_h * zeta[uns]) ** 0.5
    out[uns] = 2.0 * np.log((1.0 + y) / 2.0)
    return out if out.shape else float(out)
