"""Spectra, cospectra, quadrature spectra and ogives per window and height.

Science and sources: library/writeups/ec_spectra.md. Closure identities
(integral of S recovers the variance, of Co the covariance) are the tests.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from scipy import signal

from . import preprocess as pp
from .io import HFFile, Window, iter_windows, token

GROUP = "spectra"
# analysis-file pair label -> the two HF series of the cross spectrum
PAIRS = {"uw": ("u_pf", "w_pf"), "wTs": ("w_pf", "ts"), "wrhov": ("w_pf", "rho_h2o"),
         "wrhoCO2": ("w_pf", "rho_co2"), "wtheta_v": ("w_pf", "theta_v")}


@dataclass
class Spectrum:
    """One-sided spectral estimate on the raw frequency axis."""
    f: np.ndarray
    S: np.ndarray          # auto: real density; cross: complex density (Co = real, Qu = imag)


def spectrum(x: np.ndarray, fs: float, y: Optional[np.ndarray] = None, taper: str = "boxcar",
             nperseg: Optional[int] = None) -> Spectrum:
    """One-sided (cross-)spectral density with ``scaling="density"`` [CITED] Kaimal 1972 eq. 1.

    *x* (and *y*) are perturbation series; no detrending is applied here. With
    ``nperseg=None`` the full record is one segment (periodogram); otherwise
    Welch segments of *nperseg* samples with 50 % overlap.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    win = taper if taper else "boxcar"
    nseg = n if nperseg is None else int(min(nperseg, n))
    if y is None:
        f, S = signal.welch(x, fs=fs, window=win, nperseg=nseg, noverlap=None if nperseg else 0,
                            detrend=False, return_onesided=True, scaling="density")
    else:
        y = np.asarray(y, dtype=float)
        f, S = signal.csd(x, y, fs=fs, window=win, nperseg=nseg, noverlap=None if nperseg else 0,
                          detrend=False, return_onesided=True, scaling="density")
    return Spectrum(f, S)


def integrate(f: np.ndarray, S: np.ndarray) -> float:
    """Integral over the uniform frequency grid, sum(S) * df (Stull 1988 eq. 8.6.1b-8.6.2b)."""
    df = float(f[1] - f[0])
    return float(np.sum(np.real(S)) * df)


def ogive(f: np.ndarray, Co: np.ndarray) -> np.ndarray:
    """Og(f_k) = sum over f_j >= f_k of Co_j df: the cospectrum integrated from the Nyquist
    end down to f_k  [CITED] Foken & Wichura 1996 eq. 10; Og(0) is the covariance."""
    df = float(f[1] - f[0])
    return np.cumsum(np.real(Co)[::-1] * df)[::-1]


def log_bins(f: np.ndarray, n_per_decade: int = 10) -> np.ndarray:
    """Log-spaced bin edges over the positive lines of the uniform grid *f*, snapped to the
    half-line points (k + 1/2) df so every bin holds >= 1 line and has width n_b df; the
    band sum of the binned density then equals the variance exactly."""
    df = float(f[1] - f[0])
    n_lines = int(np.sum(f > 0))
    lo, hi = np.log10(df), np.log10(n_lines * df)
    n = max(int(np.ceil((hi - lo) * n_per_decade)), 1)
    cand = np.logspace(lo, hi, n + 1)
    m = np.clip(np.round(cand / df - 0.5), 0, n_lines).astype(int)
    m = np.unique(np.concatenate([[0], m, [n_lines]]))
    return (m + 0.5) * df


def log_bin(f: np.ndarray, S: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average *S* within *edges*; returns (mean line frequency per bin, binned S, counts)."""
    idx = np.digitize(f, edges) - 1
    nb = len(edges) - 1
    out = np.full(nb, np.nan, dtype=S.dtype if np.iscomplexobj(S) else float)
    cnt = np.zeros(nb, dtype=int)
    fc = np.sqrt(edges[:-1] * edges[1:])
    for b in range(nb):
        m = idx == b
        if m.any():
            out[b] = S[m].mean()
            cnt[b] = int(m.sum())
            fc[b] = f[m].mean()
    return fc, out, cnt


def sample_at(f: np.ndarray, y: np.ndarray, fq: np.ndarray) -> np.ndarray:
    """Linear interpolation of *y(f)* at *fq* (ogive on the bin axis)."""
    return np.interp(fq, f, y)


# ---------------------------------------------------------------- Kaimal neutral curves

def kaimal_neutral(f: np.ndarray, which: str) -> np.ndarray:
    """Kaimal et al. (1972) eq. 21a-g at z/L = 0, f = nz/U  [CITED] p. 579.

    *which*: "u", "v", "w", "T" (normalised by u*^2 or T*^2) or "uw", "wT", "uT"
    (cospectra, -nC_uw/u*^2, -nC_wT/(u* T*), +nC_uT/(u* T*)). Valid 0.01 < f < 4.
    """
    f = np.asarray(f, dtype=float)
    if which == "u":
        return 105 * f / (1 + 33 * f) ** (5 / 3)
    if which == "v":
        return 17 * f / (1 + 9.5 * f) ** (5 / 3)
    if which == "w":
        return 2 * f / (1 + 5.3 * f ** (5 / 3))
    if which == "T":
        return np.where(f <= 0.15, 53.4 * f / (1 + 24 * f) ** (5 / 3),
                        24.4 * f / (1 + 12.5 * f) ** (5 / 3))
    if which == "uw":
        return 14 * f / (1 + 9.6 * f) ** 2.4
    if which == "wT":
        return np.where(f <= 1.0, 11 * f / (1 + 13.3 * f) ** 1.75,
                        4.4 * f / (1 + 3.8 * f) ** 2.4)
    if which == "uT":
        return 40 * f / (1 + 14 * f) ** 2.6
    raise ValueError(f"kaimal_neutral: unknown curve {which!r}")


# ---------------------------------------------------------------- module driver

def _pairs_for(names: Sequence[str]) -> List[str]:
    """Labels of :data:`PAIRS` whose two series are both in *names*."""
    return [k for k, (a, b) in PAIRS.items() if a in names and b in names]


def run(hf: HFFile, cfg, records: Optional[Sequence[int]] = None) -> xr.Dataset:
    """Spectra group for every window and height of *hf*; returns the ``/spectra`` Dataset."""
    sc, pc = cfg.spectra, cfg.preprocess
    names = ["u_pf", "v_pf", "w_pf"] + [s for s in sc.scalars if s in hf.ds]
    if "theta_v" in hf.ds and "theta_v" not in names:
        names.append("theta_v")
    pairs = _pairs_for(names)
    n_win = hf.n_per_window
    f_raw = np.fft.rfftfreq(n_win if sc.nperseg is None else int(sc.nperseg), d=1.0 / hf.fs)
    edges = log_bins(f_raw, sc.n_bins_per_decade)
    fb = log_bin(f_raw, np.zeros_like(f_raw), edges)[0]
    nb, nr, nh = len(fb), len(hf.records), len(hf.heights)

    def _arr():
        return np.full((nr, nh, nb), np.nan)
    S = {k: _arr() for k in names}
    Co = {k: _arr() for k in pairs}
    Qu = {k: _arr() for k in pairs}
    Og = {k: _arr() for k in pairs}
    var = {k: np.full((nr, nh), np.nan) for k in names}
    cov = {k: np.full((nr, nh), np.nan) for k in pairs}
    diag_names = ("U_mean", "sigma_M", "taylor_ratio", "w_mean", "v_mean", "w_mean_over_sigma_w")
    diag = {k: np.full((nr, nh), np.nan) for k in diag_names}
    nan_frac = np.full((nr, nh), np.nan)
    n_valid = np.zeros((nr, nh), dtype=int)

    for win in iter_windows(hf, variables=names, records=records):
        i = win.index
        for ih in range(nh):
            if not win.has("u_pf", ih) or not win.has("w_pf", ih):
                continue
            prep = pp.prepare(win, ih, names, method=pc.detrend, tau_s=pc.filter_tau_s,
                              nan_max_frac=pc.nan_max_frac, taylor_max_ratio=pc.taylor_max_ratio)
            for k in diag_names:
                diag[k][i, ih] = prep.diagnostics.get(k, np.nan)
            nan_frac[i, ih] = max(prep.nan_frac.get(k, 0.0) for k in ("u_pf", "v_pf", "w_pf"))
            n_valid[i, ih] = int(np.isfinite(prep.prime["w_pf"]).sum())
            for k in names:
                if not prep.accepted.get(k, False):
                    continue
                x = prep.prime[k]
                if not np.isfinite(x).all():
                    continue
                sp = spectrum(x, hf.fs, taper=sc.taper, nperseg=sc.nperseg)
                S[k][i, ih] = log_bin(sp.f, sp.S, edges)[1]
                var[k][i, ih] = float(np.var(x))
            for key in pairs:
                a, b = PAIRS[key]
                if not (prep.accepted.get(a) and prep.accepted.get(b)):
                    continue
                x, y = prep.prime[a], prep.prime[b]
                if not (np.isfinite(x).all() and np.isfinite(y).all()):
                    continue
                sp = spectrum(x, hf.fs, y=y, taper=sc.taper, nperseg=sc.nperseg)
                binned = log_bin(sp.f, sp.S, edges)[1]
                Co[key][i, ih] = np.real(binned)
                Qu[key][i, ih] = np.imag(binned)
                Og[key][i, ih] = sample_at(sp.f, ogive(sp.f, sp.S), fb)
                cov[key][i, ih] = float(np.mean(x * y))

    ds = xr.Dataset(coords={"record": ("record", hf.records), "height": ("height", hf.heights),
                            "frequency": ("frequency", fb)})
    ds["frequency"].attrs.update(units="Hz", long_name="mean line frequency of each log bin")
    ds["frequency_edges"] = ("frequency_edge", edges)
    units = {"u_pf": "m2 s-2 Hz-1", "v_pf": "m2 s-2 Hz-1", "w_pf": "m2 s-2 Hz-1", "ts": "K2 Hz-1",
             "theta_v": "K2 Hz-1", "rho_h2o": "g2 m-6 Hz-1", "rho_co2": "mg2 m-6 Hz-1"}
    dims3 = ("record", "height", "frequency")
    for k in names:
        t = token(k)
        ds[f"S_{t}"] = (dims3, S[k], {"units": units.get(k, "1"), "long_name": f"one-sided power spectral density of {t}'"})
        ds[f"var_{t}"] = (("record", "height"), var[k], {"long_name": f"variance of {t}' (closure target)"})
    for key in pairs:
        a, b = (token(s) for s in PAIRS[key])
        ds[f"Co_{key}"] = (dims3, Co[key], {"long_name": f"cospectrum of {a}'{b}' (real part of the one-sided CSD)"})
        ds[f"Qu_{key}"] = (dims3, Qu[key], {"long_name": f"quadrature spectrum of {a}'{b}'"})
        ds[f"ogive_{key}"] = (dims3, Og[key], {"long_name": f"ogive of {a}'{b}': integral of Co from f_Nyquist down to f (Foken & Wichura 1996 eq. 10)"})
        ds[f"cov_{key}"] = (("record", "height"), cov[key], {"long_name": f"covariance {a}'{b}' (closure target)"})
    for k in diag_names:
        ds[k] = (("record", "height"), diag[k])
    ds["U_mean"].attrs.update(units="m s-1", long_name="window mean horizontal wind (Taylor advection speed)")
    ds["taylor_ratio"].attrs.update(long_name="sigma_M / U_mean; Stull 1988 eq. 1.4d wants < 0.5")
    ds["nan_filled_frac"] = (("record", "height"), nan_frac, {"long_name": "largest NaN fraction filled among u, v, w"})
    ds["n_valid"] = (("record", "height"), n_valid)
    ds.attrs.update(taper=sc.taper, nperseg=0 if sc.nperseg is None else int(sc.nperseg),
                    n_bins_per_decade=sc.n_bins_per_decade, detrend_method=pc.detrend,
                    estimator="full-record periodogram, log-binned" if sc.nperseg is None else "Welch, log-binned",
                    closure="integral of S_x over f = var_x; integral of Co_xy = cov_xy; ogive(f->0) = cov_xy",
                    library_note="library/writeups/ec_spectra.md")
    return ds
