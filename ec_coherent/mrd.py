"""Multiresolution (Haar) flux decomposition and the cospectral gap.

Science and sources: library/writeups/ec_mrd.md. Orthogonal dyadic
decomposition on 2^M samples (Howell & Mahrt 1997 eqs. 1-7, FHT appendix);
the sum of MR (co)spectra over scales equals the window (co)variance -- the
module's invariant test. Gap detection follows Vickers & Mahrt 2003 sec. 4.
"""

import warnings
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from . import preprocess as pp
from .io import HFFile, iter_windows, token

GROUP = "mrd"

PAIRS = {"uw": ("u_pf", "w_pf"), "vw": ("v_pf", "w_pf"), "wTs": ("w_pf", "ts"), "wrhov": ("w_pf", "rho_h2o")}


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
    """1-2-1 filter (interior points; ends kept), Vickers & Mahrt 2003 sec. 4.
    """
    out = D.astype(float).copy()
    if len(D) > 2:
        out[1:-1] = 0.25 * D[:-2] + 0.5 * D[1:-1] + 0.25 * D[2:]
    return out


def gap_scale(tau: np.ndarray, D: np.ndarray, level_frac: float = 0.01,
              tau_min: Optional[float] = None) -> float:
    """Cospectral gap timescale of one MR cospectrum (Vickers & Mahrt 2003).

    Scan the 1-2-1-smoothed |D| from the shortest scale for the first peak
    (a decrease in magnitude with increasing scale); the gap is the first
    later scale where |D| increases or the accumulative flux levels off
    (changes by *level_frac* or less). NaN when no peak or no gap is found
    (their weak-turbulence exclusion). *D* may be a magnitude series (their
    momentum practice) or a signed cospectrum. Non-finite entries of *D*
    (NaN, +-inf) count as zero in both the smoothed |D| and the
    accumulative flux.

    Vickers & Mahrt 2003 sec. 4, p. 664: "The algorithm scans the MR cospectra
    [Eq. (3)] beginning with the shortest averaging timescale and
    progressing to longer timescales. The first peak in the cospectra is
    identified by a decrease in magnitude with increasing scale and is
    associated with turbulence. The gap between turbulence and mesoscale
    motions is identified when the cospectra either increase or level off
    at an averaging time scale longer than the scale associated with the
    turbulence peak. A leveling off is identified when the accumulative
    flux changes by 1% or less with an increase in timescale." [CITED]
    (written for tower sonic records, 1-h periods).

    Parameters
    ----------
    tau : ndarray
        1-D MR averaging timescales 2^m dt [s], ascending, one per entry
        of *D*.
    D : ndarray
        1-D MR cospectrum (or stress-magnitude series) per scale.
    level_frac : float
        Leveling-off fraction of the accumulative flux [CITED] 1 %, in
        ``[0, 1)``.
    tau_min : float, optional
        Scales with ``tau < tau_min`` are excluded from the peak scan
        [HOUSE]; the 1-2-1 smoothing carries a noisy scale into its
        neighbour, so ``tau_min`` must lie at least two scales above it.
        The accumulative flux still sums every scale. None keeps the
        published scan from the shortest scale.

    Returns
    -------
    float
        Gap timescale [s], or NaN.

    Raises
    ------
    ValueError
        If *D* is not 1-D, *tau* does not match the shape of *D*, *tau* is
        not strictly ascending, or *level_frac* lies outside ``[0, 1)``.
    """
    tau = np.asarray(tau, dtype=float)
    D = np.asarray(D, dtype=float)
    if D.ndim != 1:
        raise ValueError(f"gap_scale needs a 1-D D, got shape {D.shape}")
    if tau.shape != D.shape:
        raise ValueError(f"tau shape {tau.shape} does not match D shape "
                         f"{D.shape}")
    if not np.all(np.diff(tau) > 0):
        raise ValueError("tau is not strictly ascending")
    if not 0.0 <= level_frac < 1.0:
        raise ValueError(f"level_frac {level_frac!r} outside [0, 1)")
    Ds = np.abs(smooth121(np.where(np.isfinite(D), D, 0.0)))
    C = np.cumsum(np.where(np.isfinite(D), D, 0.0))
    M = len(D)
    start = (0 if tau_min is None
             else int(np.searchsorted(tau, tau_min, side="left")))
    peak = None
    for m in range(start, M - 1):
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


def gap_scale_median(gaps) -> Union[float, np.ndarray]:
    """NaN-aware median of per-record gap timescales over records (axis 0).

    Vickers & Mahrt 2003 p. 665 average the per-hour gap scale "over the
    entire experiment to reduce random errors"; the median is used here
    instead [ASSUMED], and fewer than three finite records give NaN
    [ASSUMED] (library/writeups/ec_mrd.md, deviation register).

    Parameters
    ----------
    gaps : array_like
        Gap timescales [s], at least 1-D, with records on the first axis,
        e.g. ``gap_wTs`` of shape (record,) or (record, height); NaN and
        infinite entries count as missing.

    Returns
    -------
    float or ndarray
        Median gap timescale [s]: a float for 1-D input, otherwise an
        ndarray of shape ``gaps.shape[1:]``, one value per trailing index.
    """
    g = np.asarray(gaps, dtype=float)
    g = np.where(np.isfinite(g), g, np.nan)
    n_finite = np.isfinite(g).sum(axis=0)
    with warnings.catch_warnings():
        # all-NaN or empty slices
        warnings.simplefilter("ignore", RuntimeWarning)
        med = (np.nanmedian(g, axis=0) if g.shape[0]
               else np.full(g.shape[1:], np.nan))
    out = np.where(n_finite >= 3, med, np.nan)
    return float(out) if out.ndim == 0 else out


def run(hf: HFFile, cfg, records: Optional[Sequence[int]] = None) -> xr.Dataset:
    """MR spectra/cospectra, sampling errors, gap scales; the ``/mrd`` Dataset."""
    mc, pc = cfg.mrd, cfg.preprocess
    sigs = [s for s in mc.signals if s in ("u_pf", "v_pf", "w_pf") or s in hf.ds]
    flxs = [k for k in mc.fluxes if all(m in ("u_pf", "v_pf", "w_pf") or m in hf.ds for m in PAIRS[k])]
    names = list(dict.fromkeys(["u_pf", "v_pf", "w_pf"] + sigs + [m for k in flxs for m in PAIRS[k]]))

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
            if not win.has("w_pf", ih):
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
                gap_wTs[i, ih] = gap_scale(tau, D_flx["wTs"][i, ih],
                                           mc.gap_level_frac,
                                           mc.gap_tau_min)
            if {"uw", "vw"} <= set(flxs) and np.isfinite(D_flx["uw"][i, ih]).all() \
                    and np.isfinite(D_flx["vw"][i, ih]).all():
                mag = np.hypot(D_flx["uw"][i, ih], D_flx["vw"][i, ih])
                gap_uw[i, ih] = gap_scale(tau, mag, mc.gap_level_frac,
                                          mc.gap_tau_min)

    ds = xr.Dataset(coords={"record": ("record", hf.records),
                            "height": ("height", hf.heights),
                            "mr_scale": ("mr_scale", tau)})
    ds["mr_scale"].attrs.update(units="s", long_name="MR averaging timescale 2^m dt")
    dims3, dims = ("record", "height", "mr_scale"), ("record", "height")
    for a in sigs:
        t = token(a)
        ds[f"D_{t}_{t}"] = (dims3, D_sig[a], {
            "long_name": f"MR spectrum of {t}' (variance per dyadic averaging scale)"})
    for k in flxs:
        a, b = (token(s) for s in PAIRS[k])
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
