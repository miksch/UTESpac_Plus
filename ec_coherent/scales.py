"""LSM/VLSM scale separation: per-band variance and flux fractions.

Science and sources: library/writeups/ec_scales.md. Bands small | lsm | vlsm
by streamwise wavelength (Taylor); fractions from exact periodogram band sums,
so they sum to 1 -- the module's invariant test.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from . import ampmod
from . import preprocess as pp
from . import spectra as sp
from .io import HFFile, iter_windows, token

GROUP = "scale_separation"

BANDS = ("small", "lsm", "vlsm")
PAIRS = {"uw": ("u_pf", "w_pf"), "wTs": ("w_pf", "ts"), "wrhov": ("w_pf", "rho_h2o")}
# Balakumar & Adrian 2007 pp. 666, 671 [CITED]: LSM band 0.1*pi*delta_0 .. pi*delta_0
# (k_x delta = 20 .. 2), one decade wide; VLSM above it.
LSM_DECADE = 10.0
VLSM_CUT_DELTA = np.pi


def band_edges(cfg, x_u: np.ndarray, fs: float, U: float, z: float
               ) -> Tuple[float, float, int]:
    """(lambda_small_lsm, lambda_lsm_vlsm [m], source index) for one window.

    ``delta``: cuts at 0.1*pi*delta and pi*delta [CITED ratios] Balakumar & Adrian
    2007; ``spectral_gap`` (default): LSM|VLSM cut at the pre-multiplied u-spectrum
    gap (ampmod.spectral_gap), small|LSM one decade below, delta fallback;
    ``scaled``: multiples of z. Deviation register: ec_ampmod.md (shared).
    """
    mode = cfg.cutoff_mode
    if mode == "spectral_gap":
        fg = ampmod.spectral_gap(x_u, fs)
        if np.isfinite(fg) and fg > 0:
            lam2 = U / fg
            return lam2 / LSM_DECADE, lam2, 0
        lam2 = VLSM_CUT_DELTA * float(cfg.delta_m)
        return lam2 / LSM_DECADE, lam2, 1
    if mode == "delta":
        lam2 = VLSM_CUT_DELTA * float(cfg.delta_m)
        return lam2 / LSM_DECADE, lam2, 2
    if mode == "scaled":
        return float(cfg.z_mult_small) * z, float(cfg.z_mult_vlsm) * z, 3
    raise ValueError(f"scales cutoff_mode {mode!r}; expected spectral_gap | delta | scaled")


def band_fractions(f: np.ndarray, S: np.ndarray, U: float, lam1: float, lam2: float
                   ) -> Tuple[np.ndarray, float]:
    """Band sums of the (co-)spectral density over small/lsm/vlsm, as fractions of the total.

    *f*, *S* are the positive lines of a full-record boxcar periodogram (exact
    Parseval closure, ec_spectra.md), so the three fractions sum to 1 exactly.
    Wavelength per line lambda = U/f [CITED] Taylor 1938 (an underestimate at the
    largest scales, Kim & Adrian 1999 p. 419). Returns (fractions[3], total).
    """
    S = np.real(S)
    lam = U / f
    total = float(np.sum(S))
    if total == 0 or not np.isfinite(total):
        return np.full(3, np.nan), np.nan
    small = float(np.sum(S[lam < lam1]))
    lsm = float(np.sum(S[(lam >= lam1) & (lam < lam2)]))
    vlsm = float(np.sum(S[lam >= lam2]))
    return np.array([small, lsm, vlsm]) / total, total


def run(hf: HFFile, cfg, records: Optional[Sequence[int]] = None) -> xr.Dataset:
    """Scale-separation group for every window and height; the ``/scale_separation`` Dataset."""
    sc, pc = cfg.scales, cfg.preprocess
    sigs = list(sc.signals)
    flxs = list(sc.fluxes)
    flux_members = [m for k in flxs for m in PAIRS[k]]
    names = ["u_pf", "v_pf", "w_pf"] + [s for s in dict.fromkeys([*sigs, *flux_members])
                                        if s not in ("u_pf", "v_pf", "w_pf") and s in hf.ds]
    nr, nh, nb = len(hf.records), len(hf.heights), len(BANDS)

    var_frac = {a: np.full((nr, nh, nb), np.nan) for a in sigs}
    flux_frac = {k: np.full((nr, nh, nb), np.nan) for k in flxs}
    var_tot = {a: np.full((nr, nh), np.nan) for a in sigs}
    cov_tot = {k: np.full((nr, nh), np.nan) for k in flxs}
    lam_1, lam_2, U_mean, zeta = (np.full((nr, nh), np.nan) for _ in range(4))
    source = np.full((nr, nh), -1, dtype=np.int8)

    for win in iter_windows(hf, variables=names, records=records):
        i = win.index
        for ih in range(nh):
            if not win.has("u_pf", ih) or not win.has("w_pf", ih):
                continue
            prep = pp.prepare(win, ih, names, method=pc.detrend, tau_s=pc.filter_tau_s,
                              nan_max_frac=pc.nan_max_frac, taylor_max_ratio=pc.taylor_max_ratio)
            prime = {k: v for k, v in prep.prime.items()
                     if prep.accepted.get(k, False) and np.isfinite(v).all()}
            if "u_pf" not in prime:
                continue
            U = prep.U_mean
            z = float(hf.heights[ih])
            l1, l2, src = band_edges(sc, prime["u_pf"], win.fs, U, z)
            lam_1[i, ih], lam_2[i, ih], source[i, ih] = l1, l2, src
            U_mean[i, ih] = U
            L = win.ancillary.get("L", np.full(nh, np.nan))[ih]
            zeta[i, ih] = z / L if np.isfinite(L) and L != 0 else np.nan
            for a in sigs:
                if a not in prime:
                    continue
                spec = sp.spectrum(prime[a], win.fs)          # boxcar periodogram: exact closure
                pos = spec.f > 0
                var_frac[a][i, ih], tot = band_fractions(spec.f[pos], spec.S[pos], U, l1, l2)
                var_tot[a][i, ih] = tot * (spec.f[1] - spec.f[0])
            for k in flxs:
                a, b = PAIRS[k]
                if a not in prime or b not in prime:
                    continue
                spec = sp.spectrum(prime[a], win.fs, y=prime[b])
                pos = spec.f > 0
                flux_frac[k][i, ih], tot = band_fractions(spec.f[pos], spec.S[pos], U, l1, l2)
                cov_tot[k][i, ih] = tot * (spec.f[1] - spec.f[0])

    ds = xr.Dataset(coords={"record": ("record", hf.records),
                            "height": ("height", hf.heights),
                            "scale_band": ("scale_band", list(BANDS))})
    dims3, dims = ("record", "height", "scale_band"), ("record", "height")
    for a in sigs:
        t = token(a)
        ds[f"var_frac_{t}"] = (dims3, var_frac[a], {
            "long_name": f"fraction of the {t}' variance in each wavelength band"})
        ds[f"var_{t}"] = (dims, var_tot[a], {"long_name": f"total {t}' variance (band-sum denominator)"})
    for k in flxs:
        a, b = (token(s) for s in PAIRS[k])
        ds[f"flux_frac_{k}"] = (dims3, flux_frac[k], {
            "long_name": f"fraction of the {a}'{b}' covariance in each wavelength band"})
        ds[f"cov_{k}"] = (dims, cov_tot[k], {"long_name": f"total {a}'{b}' covariance (band-sum denominator)"})
    ds["lambda_small_lsm"] = (dims, lam_1, {"units": "m", "long_name": "small|LSM band edge wavelength"})
    ds["lambda_lsm_vlsm"] = (dims, lam_2, {"units": "m", "long_name": "LSM|VLSM band edge wavelength"})
    ds["cutoff_source"] = (dims, source, {
        "flag_values": np.arange(len(ampmod.CUTOFF_SOURCES), dtype=np.int8),
        "flag_meanings": " ".join(ampmod.CUTOFF_SOURCES),
        "long_name": "how the band edges were set (-1: window not processed)"})
    ds["U_mean"] = (dims, U_mean, {"units": "m s-1", "long_name": "window mean horizontal wind (Taylor advection speed)"})
    ds["zeta"] = (dims, zeta, {"long_name": "z/L from the EC ancillary Obukhov length"})
    ds.attrs.update(cutoff_mode=sc.cutoff_mode, delta_m=sc.delta_m,
                    z_mult_small=sc.z_mult_small, z_mult_vlsm=sc.z_mult_vlsm,
                    detrend_method=pc.detrend,
                    bands="small: lambda < edge1; lsm: edge1 <= lambda < edge2; vlsm: lambda >= edge2",
                    closure="var_frac and flux_frac sum to 1 over scale_band (exact periodogram band sums)",
                    library_note="library/writeups/ec_scales.md")
    return ds
