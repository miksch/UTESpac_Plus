"""sonic_temperature – humidity relations between sonic and air temperature.

Equations and loci in library/writeups/sonic_temperature_flux.md.
"""

import numpy as np

# [CITED] Schotanus et al. (1983) eqs. 3, 5: c² = γR T (1 + 0.51 q), so the
# sonic temperature is T_s = T (1 + 0.51 q). Kaimal & Gaynor (1991) eq. 3
# write the same as T_s = T (1 + 0.32 e/p); with q = 0.622 e/p that is
# 0.51 q. The virtual-temperature coefficient 0.61 is a different quantity.
SONIC_HUMIDITY_COEFF = 0.51


def air_temperature_from_sonic(Ts_K, q):
    """Mean air temperature [K] from sonic temperature T_s [K] and specific
    humidity q [kg/kg]: T = T_s / (1 + 0.51 q) (Schotanus 1983 eq. 5)."""
    return np.asarray(Ts_K, dtype=float) / (1.0 + SONIC_HUMIDITY_COEFF * np.asarray(q, dtype=float))


def air_temperature_perturbation(TsP, qP, T_mean_K):
    """Air-temperature perturbation T′ from sonic-temperature perturbation
    T_s′ and specific-humidity perturbation q′ (Schotanus 1983 eq. 6 without
    the crosswind term, which CSAT3/IRGASON heads remove internally;
    Kaimal & Gaynor 1991 eq. 4, Liu et al. 2001 eq. 10):

        T′ = T_s′ − 0.51 · T̄ · q′

    so that w′T′ = w′T_s′ − 0.51 T̄ w′q′ (Schotanus eq. 8, Liu eq. 12).
    """
    return np.asarray(TsP, dtype=float) - SONIC_HUMIDITY_COEFF * float(T_mean_K) * np.asarray(qP, dtype=float)
