"""Multiresolution (Haar) flux decomposition and the cospectral gap.

Science and sources: library/writeups/ec_mrd.md. Orthogonal dyadic
decomposition on 2^M samples (Howell & Mahrt 1997 eqs. 1-7, FHT appendix);
the sum of MR (co)spectra over scales equals the window (co)variance -- the
module's invariant test. Gap detection follows Vickers & Mahrt 2003 §4.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from . import preprocess as pp
from .io import HFFile, iter_windows

GROUP = "mrd"

PAIRS = {"uw": ("u", "w"), "vw": ("v", "w"), "wTs": ("w", "Ts"), "wrhov": ("w", "rhov")}


def to_grid(x: np.ndarray, mode: str = "trim") -> Tuple[np.ndarray, float]:
    """Map a window of R samples onto 2^M; returns (series, dt multiplier).

    ``trim``: leading 2^M samples, 2^M <= R (gameplan choice [ASSUMED]).
    ``interp``: linear interpolation onto the coarser 2^M grid with 2^M < R,
    spacing (R-1)/(2^M-1) original intervals -- Vickers & Mahrt 2003 eq. 8
    [CITED]; all points used. A record already of length 2^M passes through.
    """
    n = len(x)
    M = int(np.floor(np.log2(n)))
    if 2 ** M == n:
        return x.copy(), 1.0
    if mode == "trim":
        return x[: 2 ** M].copy(), 1.0
    if mode == "interp":
        new = np.interp(np.linspace(0.0, n - 1.0, 2 ** M), np.arange(n, dtype=float), x)
        return new, (n - 1.0) / (2 ** M - 1.0)
    raise ValueError(f"mrd grid {mode!r}; expected trim | interp")


def fht(x: np.ndarray) -> np.ndarray:
    """In-place forward Fast Haar Transform of a 2^M series (Howell & Mahrt 1997 appendix).

    After the transform, position 0 holds the record mean and the scale-m
    detail (DEL) coefficients sit at positions (n - 1/2) 2^m, n = 1..2^(M-m).
    """
    n = len(x)
    M = int(np.log2(n))
    if 2 ** M != n:
        raise ValueError(f"fht needs 2^M samples, got {n}")
    for m in range(1, M + 1):
        half = 2 ** (m - 1)
        idx = np.arange(0, n, 2 ** m)
        a, b = x[idx], x[idx + half]
        x[idx], x[idx + half] = (a + b) / 2.0, (a - b) / 2.0
    return x


def mr_spectrum(x: np.ndarray, y: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MR (co)spectrum D(m), per-scale sample std, and window counts, m = 1..M.

    D(m) is the mean product of the same-scale FHT detail coefficients
    (Howell & Mahrt 1997 eq. 15); sum(D) equals the (co)variance about the
    record mean (eq. 6). The std is their eq. 16 (NaN at m = M, one sample);
    the relative sampling error is (std/D)/sqrt(count) (eq. 12).
    """
    hx = fht(np.asarray(x, dtype=float).copy())
    hy = hx if y is None else fht(np.asarray(y, dtype=float).copy())
    n = len(hx)
    M = int(np.log2(n))
    D = np.empty(M)
    s = np.full(M, np.nan)
    counts = np.empty(M, dtype=np.int64)
    for m in range(1, M + 1):
        pos = 2 ** (m - 1) + np.arange(2 ** (M - m)) * 2 ** m
        prods = hx[pos] * hy[pos]
        D[m - 1] = prods.mean()
        counts[m - 1] = prods.size
        if prods.size > 1:
            s[m - 1] = prods.std(ddof=1)
    return D, s, counts


def smooth121(D: np.ndarray) -> np.ndarray:
    """1-2-1 filter (interior points; ends kept) -- Vickers & Mahrt 2003 §4."""
    out = D.astype(float).copy()
    if len(D) > 2:
        out[1:-1] = 0.25 * D[:-2] + 0.5 * D[1:-1] + 0.25 * D[2:]
    return out


def gap_scale(tau: np.ndarray, D: np.ndarray, level_frac: float = 0.01) -> float:
    """Cospectral gap timescale of one MR cospectrum (Vickers & Mahrt 2003 §4).

    Scan the 1-2-1-smoothed |D| from the shortest scale for the first peak
    (a decrease in magnitude with increasing scale); the gap is the first
    later scale where |D| increases or the accumulative flux levels off
    (changes by *level_frac* or less). NaN when no peak or no gap is found
    (their weak-turbulence exclusion). *D* may be a magnitude series (their
    momentum practice) or a signed cospectrum.
    """
    Ds = np.abs(smooth121(np.nan_to_num(D, nan=0.0)))
    C = np.cumsum(np.nan_to_num(D, nan=0.0))
    M = len(D)
    peak = None
    for m in range(M - 1):
        if Ds[m + 1] < Ds[m] and Ds[m] > 0:
            peak = m
            break
    if peak is None:
        return np.nan
    for g in range(peak + 1, M):
        level = abs(C[g] - C[g - 1]) <= level_frac * abs(C[g - 1]) if C[g - 1] != 0 else False
        if Ds[g] > Ds[g - 1] or level:
            return float(tau[g])
    return np.nan


def run(hf: HFFile, cfg, records: Optional[Sequence[int]] = None) -> xr.Dataset:
    """MR spectra/cospectra, sampling errors, gap scales; the ``/mrd`` Dataset."""
    mc, pc = cfg.mrd, cfg.preprocess
    sigs = [s for s in mc.signals if s in ("u", "v", "w") or s in hf.ds]
    flxs = [k for k in mc.fluxes if all(m in ("u", "v", "w") or m in hf.ds for m in PAIRS[k])]
    names = list(dict.fromkeys(["u", "v", "w"] + sigs + [m for k in flxs for m in PAIRS[k]]))

    n_win = hf.n_per_window
    M = int(np.floor(np.log2(n_win)))
    dt = 1.0 / hf.fs
    dt_mult = 1.0 if (mc.grid == "trim" or 2 ** M == n_win) else (n_win - 1.0) / (2 ** M - 1.0)
    tau = 2.0 ** np.arange(1, M + 1) * dt * dt_mult

    nr, nh = len(hf.records), len(hf.heights)
    D_sig = {a: np.full((nr, nh, M), np.nan) for a in sigs}
    D_flx = {k: np.full((nr, nh, M), np.nan) for k in flxs}
    err_flx = {k: np.full((nr, nh, M), np.nan) for k in flxs}
    cov = {k: np.full((nr, nh), np.nan) for k in flxs}
    gap_wTs = np.full((nr, nh), np.nan)
    gap_uw = np.full((nr, nh), np.nan)

    counts = None
    for win in iter_windows(hf, variables=names, records=records):
        i = win.index
        for ih in range(nh):
            if not win.has("w", ih):
                continue
            prep = pp.prepare(win, ih, names, method=pc.detrend, tau_s=pc.filter_tau_s,
                              nan_max_frac=pc.nan_max_frac, taylor_max_ratio=pc.taylor_max_ratio)
            prime = {k: v for k, v in prep.prime.items()
                     if prep.accepted.get(k, False) and np.isfinite(v).all()}
            grid = {k: to_grid(v, mc.grid)[0] for k, v in prime.items()}
            for a in sigs:
                if a in grid:
                    D_sig[a][i, ih], _, counts = mr_spectrum(grid[a])
            for k in flxs:
                a, b = PAIRS[k]
                if a in grid and b in grid:
                    D, s, counts = mr_spectrum(grid[a], y=grid[b])
                    D_flx[k][i, ih] = D
                    with np.errstate(divide="ignore", invalid="ignore"):
                        err_flx[k][i, ih] = np.abs(s / D) / np.sqrt(counts)
                    cov[k][i, ih] = D.sum()
            if "wTs" in flxs and np.isfinite(D_flx["wTs"][i, ih]).all():
                gap_wTs[i, ih] = gap_scale(tau, D_flx["wTs"][i, ih], mc.gap_level_frac)
            if {"uw", "vw"} <= set(flxs) and np.isfinite(D_flx["uw"][i, ih]).all() \
                    and np.isfinite(D_flx["vw"][i, ih]).all():
                mag = np.hypot(D_flx["uw"][i, ih], D_flx["vw"][i, ih])
                gap_uw[i, ih] = gap_scale(tau, mag, mc.gap_level_frac)

    ds = xr.Dataset(coords={"record": ("record", hf.records),
                            "height": ("height", hf.heights),
                            "mr_scale": ("mr_scale", tau)})
    ds["mr_scale"].attrs.update(units="s", long_name="MR averaging timescale 2^m dt")
    dims3, dims = ("record", "height", "mr_scale"), ("record", "height")
    for a in sigs:
        ds[f"D_{a}{a}"] = (dims3, D_sig[a], {
            "long_name": f"MR spectrum of {a}' (variance per dyadic averaging scale)"})
    for k in flxs:
        a, b = PAIRS[k]
        ds[f"D_{k}"] = (dims3, D_flx[k], {
            "long_name": f"MR cospectrum of {a}'{b}' (covariance per dyadic averaging scale)"})
        ds[f"err_{k}"] = (dims3, err_flx[k], {
            "long_name": f"relative random sampling error of D_{k} (Howell & Mahrt 1997 eq. 12)"})
        ds[f"cov_{k}"] = (dims, cov[k], {
            "long_name": f"sum of D_{k} over scales = window {a}'{b}' covariance (closure)"})
    ds["gap_wTs"] = (dims, gap_wTs, {"units": "s", "long_name":
                     "cospectral gap timescale of w'Ts' (Vickers & Mahrt 2003 algorithm)"})
    ds["gap_uw"] = (dims, gap_uw, {"units": "s", "long_name":
                    "cospectral gap timescale of the (uw, vw) stress magnitude"})
    n_used = 2 ** M
    ds.attrs.update(grid=mc.grid, n_window=n_win, n_used=n_used, M=M,
                    dt_multiplier=dt_mult, gap_level_frac=mc.gap_level_frac,
                    trim_fraction=n_used / n_win if mc.grid == "trim" else 1.0,
                    detrend_method=pc.detrend,
                    closure="sum over mr_scale of D equals the window (co)variance about the grid-series mean",
                    gap_caveat="per-record gap estimates carry significant sampling error; "
                               "Vickers & Mahrt 2003 average them over an experiment",
                    library_note="library/writeups/ec_mrd.md")
    return ds
