"""Shared preprocessing: gap policy, detrend, Taylor helpers, rotation diagnostics.

Science and sources: library/writeups/ec_preprocess.md.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy import signal

DETREND_METHODS = ("block", "linear", "recursive", "butterworth")


@dataclass
class GapResult:
    """Outcome of :func:`fill_gaps`."""
    x: np.ndarray
    nan_frac: float
    accepted: bool


def fill_gaps(x: np.ndarray, max_frac: float = 0.1) -> GapResult:
    """Linearly interpolate NaNs in *x*; reject when the NaN fraction exceeds *max_frac*.

    Leading/trailing NaNs take the nearest valid value. Policy is [ASSUMED]
    (ec_preprocess.md, "Missing samples").
    """
    x = np.asarray(x, dtype=float)
    bad = ~np.isfinite(x)
    frac = float(bad.mean()) if x.size else 1.0
    if frac > max_frac or frac == 1.0:
        return GapResult(x.copy(), frac, False)
    if not bad.any():
        return GapResult(x.copy(), 0.0, True)
    idx = np.arange(x.size)
    y = x.copy()
    y[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return GapResult(y, frac, True)


def recursive_filter(x: np.ndarray, fs: float, tau_s: float, init: Optional[float] = None) -> np.ndarray:
    """Low-pass part of *x* by the recursive RC filter  [CITED] Moncrieff et al. 2004 eq. 2.23:
    ``x~_k = a x~_{k-1} + (1 - a) x_k``, ``a = exp(-dt/tau)``. Started at *init* (default: the
    series mean -- [ASSUMED], the window has no preceding samples)."""
    x = np.asarray(x, dtype=float)
    a = float(np.exp(-1.0 / (fs * tau_s)))
    out = np.empty_like(x)
    prev = float(np.nanmean(x)) if init is None else float(init)
    for k in range(x.size):
        prev = a * prev + (1.0 - a) * x[k]
        out[k] = prev
    return out


def detrend(x: np.ndarray, method: str = "block", fs: Optional[float] = None,
            tau_s: float = 300.0) -> np.ndarray:
    """Perturbation series of *x* (NaN preserved; >90 % NaN -> all NaN, as ``utespac.nandetrend``).

    Parameters
    ----------
    x : 1-D array
    method : {"block", "linear", "recursive", "butterworth"}
        Window mean (Moncrieff et al. 2004 eq. 2.8-2.9), least-squares line (eq. 2.14,
        2.17-2.18), recursive RC filter of time constant *tau_s* (eq. 2.23; needs *fs*), or a
        zero-phase 2nd-order Butterworth low-pass at 1/(2 pi tau_s) [ASSUMED]; see
        library/writeups/ec_preprocess.md.
    """
    x = np.asarray(x, dtype=float).ravel()
    if method not in DETREND_METHODS:
        raise ValueError(f"detrend method {method!r}; expected one of {DETREND_METHODS}")
    ok = np.isfinite(x)
    n_ok = int(ok.sum())
    if x.size == 0 or n_ok / x.size < 0.1:
        return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    if method == "block" or n_ok < 2:
        out[ok] = x[ok] - x[ok].mean()
        return out
    if method == "linear":
        idx = np.flatnonzero(ok).astype(float)
        c = np.polyfit(idx, x[ok], 1)
        out[ok] = x[ok] - np.polyval(c, idx)
        return out
    if fs is None:
        raise ValueError(f"detrend(method={method!r}) needs fs")
    filled = fill_gaps(x, max_frac=1.0).x          # gaps bridged for the filter only
    if method == "recursive":
        trend = recursive_filter(filled, fs, tau_s)
    else:
        fc = 1.0 / (2.0 * np.pi * tau_s)
        wn = min(fc / (fs / 2.0), 0.999)
        sos = signal.butter(2, wn, btype="low", output="sos")
        trend = signal.sosfiltfilt(sos, filled - filled.mean()) + filled.mean()
    out[ok] = x[ok] - trend[ok]
    return out


def mean_wind(u: np.ndarray, v: Optional[np.ndarray] = None) -> float:
    """Per-window mean horizontal wind: hypot of the component means (v=0 after yaw)."""
    ub = float(np.nanmean(u))
    vb = float(np.nanmean(v)) if v is not None else 0.0
    return float(np.hypot(ub, vb))


def taylor_wavelength(f: np.ndarray, U: float) -> np.ndarray:
    """lambda = U/f  [CITED] Taylor 1938 eq. 7, 11."""
    f = np.asarray(f, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(f > 0, U / f, np.inf)


def taylor_wavenumber(f: np.ndarray, U: float) -> np.ndarray:
    """k = 2 pi f / U  [CITED] Taylor 1938 eq. 9-11; Stull 1988 eq. 1.4c."""
    return 2.0 * np.pi * np.asarray(f, dtype=float) / U


def taylor_check(u: np.ndarray, v: Optional[np.ndarray] = None, max_ratio: float = 0.5) -> Dict[str, float]:
    """sigma_M / U_mean against the Willis & Deardorff limit  [CITED] Stull 1988 eq. 1.4d."""
    U = mean_wind(u, v)
    M = np.hypot(u, v) if v is not None else np.abs(u)
    sig = float(np.nanstd(M))
    ratio = sig / U if U > 0 else np.inf
    return {"U_mean": U, "sigma_M": sig, "taylor_ratio": ratio, "taylor_ok": float(ratio < max_ratio)}


def rotation_check(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    """Residual per-window means after the upstream planar fit + yaw (Wilczak 2001 pp. 142-143)."""
    wb = float(np.nanmean(w))
    sw = float(np.nanstd(w))
    return {"w_mean": wb, "v_mean": float(np.nanmean(v)), "u_mean": float(np.nanmean(u)),
            "w_mean_over_sigma_w": wb / sw if sw > 0 else np.nan}


@dataclass
class Prepared:
    """Perturbation series of one window at one height, ready for a module."""
    prime: Dict[str, np.ndarray]
    nan_frac: Dict[str, float]
    accepted: Dict[str, bool]
    U_mean: float
    diagnostics: Dict[str, float]


def prepare(window, ih: int, names, method: str = "block", tau_s: float = 300.0,
            nan_max_frac: float = 0.1, taylor_max_ratio: float = 0.5) -> Prepared:
    """Gap-fill, detrend and diagnose the series *names* of *window* at height *ih*."""
    prime, fracs, acc = {}, {}, {}
    for name in names:
        x = window.series(name, ih)
        g = fill_gaps(x, nan_max_frac)
        fracs[name] = g.nan_frac
        acc[name] = g.accepted
        prime[name] = detrend(g.x, method, fs=window.fs, tau_s=tau_s) if g.accepted \
            else np.full_like(x, np.nan)
    u, v, w = (window.series(k, ih) for k in ("u", "v", "w"))
    diag = {}
    diag.update(taylor_check(u, v, taylor_max_ratio))
    diag.update(rotation_check(u, v, w))
    return Prepared(prime=prime, nan_frac=fracs, accepted=acc, U_mean=diag["U_mean"], diagnostics=diag)
