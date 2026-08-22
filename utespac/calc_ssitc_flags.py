"""calc_ssitc_flags – Forest/complex-terrain SSITC + SS-only quality flags.

Outputs 8 flags per sonic per averaging period:
  TAU_SSITC_TEST, TAU_SS_ONLY_TEST
  H_SSITC_TEST,   H_SS_ONLY_TEST
  LE_SSITC_TEST,  LE_SS_ONLY_TEST
  FC_SSITC_TEST,  FC_SS_ONLY_TEST

SSITC flag:   combined steady-state test + canopy-aware w-based ITC test
SS_ONLY flag: steady-state test only (for slope-flow / complex terrain where
              ITC assumptions are knowingly violated)

Flag values:  0 = high quality, 1 = moderate / usable, 2 = poor / exclude

Port of calc_SSITC_flags_ForestComplexTerrain.m (Diane Wang).
"""

import numpy as np


def calc_ssitc_flags(
    wPF_P, uPF_P, vPF_P, ThvP,
    H2Op, rhov_ext,
    rho_CO2p, rhoc_ext,
    ustar, L_val,
    rot_flag, Ts_flag, h2o_flag, co2_flag,
    n_sub, height, d=0.0,
    canopy_height=np.nan, use_canopy_itc=True,
    latitude=None,
):
    """Return 8 flag values per period.

    Returns
    -------
    tau_ssitc, tau_ss_only, H_ssitc, H_ss_only,
    LE_ssitc,  LE_ss_only,  FC_ssitc, FC_ss_only
    Each is int 0/1/2 or NaN.

    Parameters
    ----------
    wPF_P, uPF_P, vPF_P : ndarray
        Full-period detrended PF-rotated wind fluctuations.
    ThvP : ndarray
        Full-period detrended virtual potential temperature fluctuations.
    H2Op : ndarray or None
        Detrended H2O fluctuations [g/m³].
    rhov_ext : ndarray or None
        WPL H2O external fluctuation [kg/m³].
    rho_CO2p : ndarray or None
        Detrended CO2 fluctuations [kg/m³].
    rhoc_ext : ndarray or None
        WPL CO2 external fluctuation [kg/m³].
    ustar : float
        Friction velocity [m/s].
    L_val : float
        Obukhov length [m].
    rot_flag, Ts_flag, h2o_flag, co2_flag : bool
        Period-level QC flags.
    n_sub : int
        Number of sub-periods (avgPer / SSITC_subAvgMin).
    height : float
        Sonic measurement height [m].
    d : float
        Zero-plane displacement height [m] (default 0).
    canopy_height : float
        Canopy height [m]; NaN disables canopy ITC branch.
    use_canopy_itc : bool
        If True and z ≤ canopy_height, use Rannik et al. canopy model.
    latitude : float or None
        Site latitude [deg N] for the Thomas & Foken (2002) stable-side
        σ_w/u* form (needs the Coriolis parameter); None → fallback.
    """
    # ── ITC_w deviation ──────────────────────────────────────────────────────
    n_w    = np.sum(~np.isnan(wPF_P))
    sig_w  = np.nanstd(wPF_P, ddof=1) if n_w > 1 else np.nan

    if np.isnan(ustar) or ustar <= 0 or np.isnan(sig_w):
        itc_dev = np.nan
    else:
        measured = sig_w / ustar
        if np.isnan(measured) or measured <= 0:
            itc_dev = np.nan
        else:
            z_eff = max(height - d, 0.1)
            if use_canopy_itc and not np.isnan(canopy_height) and height <= canopy_height:
                model = _canopy_sigmaw(height, canopy_height)
            else:
                zeta = z_eff / L_val if (np.isfinite(L_val) and L_val != 0) else np.nan
                model = _above_canopy_sigmaw(zeta, ustar, latitude)
            if np.isnan(model) or model <= 0:
                itc_dev = np.nan
            else:
                itc_dev = abs((model - measured) / model) * 100.0

    # ── TAU ──────────────────────────────────────────────────────────────────
    ss_uw = _ss_dev(uPF_P, wPF_P, n_sub)
    ss_vw = _ss_dev(vPF_P, wPF_P, n_sub)
    devs  = [v for v in [ss_uw, ss_vw] if not np.isnan(v)]
    tau_ss = max(devs) if devs else np.nan

    tau_ssitc   = _ssitc_flag(tau_ss, itc_dev)
    tau_ss_only = _ss_only_flag(tau_ss)
    if rot_flag:
        tau_ssitc = tau_ss_only = 2

    # ── H ────────────────────────────────────────────────────────────────────
    H_ss     = _ss_dev(wPF_P, ThvP, n_sub)
    H_ssitc   = _ssitc_flag(H_ss, itc_dev)
    H_ss_only = _ss_only_flag(H_ss)
    if rot_flag or Ts_flag:
        H_ssitc = H_ss_only = 2

    # ── LE ───────────────────────────────────────────────────────────────────
    if H2Op is not None and rhov_ext is not None:
        H2O_WPL   = H2Op + rhov_ext * 1e3
        LE_ss     = _ss_dev(wPF_P, H2O_WPL, n_sub)
        LE_ssitc   = _ssitc_flag(LE_ss, itc_dev)
        LE_ss_only = _ss_only_flag(LE_ss)
        if rot_flag or h2o_flag:
            LE_ssitc = LE_ss_only = 2
    else:
        LE_ssitc = LE_ss_only = np.nan

    # ── FC ───────────────────────────────────────────────────────────────────
    if rho_CO2p is not None and rhoc_ext is not None:
        CO2_WPL   = rho_CO2p + rhoc_ext
        FC_ss     = _ss_dev(wPF_P, CO2_WPL, n_sub)
        FC_ssitc   = _ssitc_flag(FC_ss, itc_dev)
        FC_ss_only = _ss_only_flag(FC_ss)
        if rot_flag or co2_flag:
            FC_ssitc = FC_ss_only = 2
    else:
        FC_ssitc = FC_ss_only = np.nan

    return (tau_ssitc, tau_ss_only,
            H_ssitc,   H_ss_only,
            LE_ssitc,  LE_ss_only,
            FC_ssitc,  FC_ss_only)


# ── SS deviation (sub-period means removed) ───────────────────────────────────

def _ss_dev(x, w, n_sub):
    """Foken-style SS deviation [%] with sub-period mean removal."""
    if x is None or w is None:
        return np.nan
    valid = ~(np.isnan(x) | np.isnan(w))
    if valid.sum() < 10:
        return np.nan

    # Full-period covariance (inputs already detrended, mean ≈ 0)
    x_fp  = x - np.nanmean(x)
    w_fp  = w - np.nanmean(w)
    cov_full = np.nanmean(x_fp * w_fp)
    if np.isnan(cov_full) or abs(cov_full) < np.finfo(float).eps:
        return np.nan

    # Sub-period covariances with local mean removal
    n   = len(x)
    bps = np.round(np.linspace(0, n, n_sub + 1)).astype(int)
    cov_sub = np.full(n_sub, np.nan)
    for k in range(n_sub):
        xs = x[bps[k]:bps[k + 1]]
        ws = w[bps[k]:bps[k + 1]]
        ok = ~(np.isnan(xs) | np.isnan(ws))
        if ok.sum() < 3:
            continue
        xs = xs - np.nanmean(xs)
        ws = ws - np.nanmean(ws)
        cov_sub[k] = np.nanmean(xs * ws)

    return abs((np.nanmean(cov_sub) - cov_full) / cov_full) * 100.0


# ── ITC reference models ──────────────────────────────────────────────────────

OMEGA_EARTH = 7.2921e-5   # rad/s


def coriolis_parameter(latitude_deg):
    """f = 2 Ω sin(φ) [s⁻¹]."""
    return 2.0 * OMEGA_EARTH * np.sin(np.radians(float(latitude_deg)))


def _above_canopy_sigmaw(zeta, ustar=np.nan, latitude=None):
    """Above-canopy σ_w/u* reference (library/writeups/itc_sigmaw.md).

    [CITED] Foken (2008) Table 2.11 = Foken et al. (2004) Table 9.1 = Foken
    et al. (2012) Table 4.2: σ_w/u* = 1.3 for −0.032 < z/L < 0 and
    2.0 (−z/L)^(1/8) for z/L < −0.032 (the two meet at |z/L| = 0.0319).
    [CITED] Thomas & Foken (2002) via Foken (2008) Table 2.12: for
    −0.2 < z/L < 0.4, σ_w/u* = 0.21 ln(z+ f/u*) + 3.1 with z+ = 1 m and f
    the Coriolis parameter; used here on the stable side 0 ≤ z/L ≤ 0.4
    when ``latitude`` is given (the unstable side keeps Table 2.11, which
    the same tables list for it).
    [CITED] Pahlow et al. (2001) eq. 14, a = 1.1, b = 0.9, c = 0.6:
    σ_w/u* = 1.1 + 0.9 (z/L)^0.6, fitted from near neutral to z/L = 32.9;
    used for z/L > 0.4, and for 0 < z/L ≤ 0.4 when no latitude is set.
    The two stable forms do not meet at z/L = 0.4 (≈1.4 vs 1.62); that
    step is a deviation recorded in the note and the ledger.
    """
    if np.isnan(zeta):
        return np.nan
    if zeta < 0:
        return 2.0 * (-zeta) ** (1.0 / 8.0) if zeta < -0.032 else 1.3
    if zeta <= 0.4 and latitude is not None and np.isfinite(ustar) and ustar > 0:
        f = abs(coriolis_parameter(latitude))
        if f > 0:
            return 0.21 * np.log(1.0 * f / ustar) + 3.1
    return 1.1 + 0.9 * zeta ** 0.6


def _canopy_sigmaw(z, hc):
    """Rannik et al. neutral canopy σ_w/u* (a=1.25, α=0.9, β=1.2, γ=-0.63)."""
    if np.isnan(z) or np.isnan(hc) or hc <= 0:
        return np.nan
    zh    = min(max(z / hc, 0.0), 1.0)
    ai, alpha_i, beta_i, gamma_i = 1.25, 0.9, 1.2, -0.63
    return ai * (np.exp(-alpha_i * (1.0 - zh) ** beta_i) * (1.0 - gamma_i) + gamma_i)


# ── flag combiners ────────────────────────────────────────────────────────────

def _ssitc_flag(ss_dev, itc_dev):
    """Combined SS + ITC → 0/1/2."""
    if np.isnan(ss_dev) or np.isnan(itc_dev):
        return 2
    if ss_dev < 30 and itc_dev < 30:
        return 0
    if ss_dev < 100 and itc_dev < 100:
        return 1
    return 2


def _ss_only_flag(ss_dev):
    """SS-only → 0/1/2 (for slope-flow / complex terrain)."""
    if np.isnan(ss_dev):
        return 2
    if ss_dev < 30:
        return 0
    if ss_dev < 100:
        return 1
    return 2
