"""Amplitude modulation of small scales by large scales (Mathis et al. 2009 decoupling).

Science and sources: library/writeups/ec_ampmod.md. Single-point form; the
cutoff-scale substitution for the missing z_i is the registered deviation.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from scipy import signal as sig

from . import preprocess as pp
from . import spectra as sp
from .io import HFFile, iter_windows

GROUP = "amplitude_mod"

CUTOFF_SOURCES = ("spectral_gap", "delta_fallback", "delta", "scaled")


def lowpass_sharp(x: np.ndarray, fs: float, fc: float) -> np.ndarray:
    """Large-scale part of *x*: Fourier lines with f > *fc* zeroed  [CITED] Salesky &
    Anderson 2018 p. 141 ("sharp spectral filter"). x = lowpass + (x - lowpass) exactly."""
    x = np.asarray(x, dtype=float)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, d=1.0 / fs)
    X[f > fc] = 0.0
    return np.fft.irfft(X, n=x.size)


def envelope(x: np.ndarray) -> np.ndarray:
    """Modulus of the analytic signal, the envelope of *x*  [CITED] Mathis 2009 eq. 4.5."""
    return np.abs(sig.hilbert(np.asarray(x, dtype=float)))


def am_coefficient(b_l: np.ndarray, env_l: np.ndarray) -> float:
    """R = correlation of the large-scale modulator with the filtered envelope
    [CITED] Mathis 2009 eq. 5.1; Salesky & Anderson 2018 eq. 1.6 (fluctuating parts)."""
    b = b_l - np.mean(b_l)
    e = env_l - np.mean(env_l)
    denom = np.sqrt(np.mean(b ** 2)) * np.sqrt(np.mean(e ** 2))
    if denom == 0 or not np.isfinite(denom):
        return np.nan
    return float(np.mean(b * e) / denom)


def spectral_gap(x: np.ndarray, fs: float, n_per_decade: int = 10) -> float:
    """Gap frequency of the pre-multiplied spectrum of *x*, or NaN.

    Interior minimum of log-binned f*S(f) that separates an outer (lower-f)
    local maximum from the main turbulent maximum -- where the sources put the
    large/small cutoff (Salesky & Anderson 2018 p. 141: "the spectral plateau
    separating the inner and outer peaks"). NaN when the binned curve has no
    interior minimum below the global maximum (no scale separation resolved).
    """
    spec = sp.spectrum(x, fs)
    pos = spec.f > 0
    f, S = spec.f[pos], np.real(spec.S[pos])
    edges = sp.log_bins(spec.f, n_per_decade)
    fb, Sb, cnt = sp.log_bin(f, S, edges)
    ok = cnt > 0
    fb, Sb = fb[ok], Sb[ok]
    if fb.size < 5:
        return np.nan
    p = fb * Sb                                   # pre-multiplied, log-f axis
    p = np.convolve(p, np.ones(3) / 3.0, mode="same")   # light smoothing against bin noise [ASSUMED]
    maxima = [i for i in range(1, p.size - 1)
              if p[i] >= p[i - 1] and p[i] >= p[i + 1] and (p[i] > p[i - 1] or p[i] > p[i + 1])]
    if len(maxima) < 2:
        return np.nan
    order = sorted(maxima, key=lambda i: p[i], reverse=True)
    i_a = order[0]                                # largest maximum; partner >= half a decade away
    i_b = next((i for i in order[1:] if abs(i - i_a) >= n_per_decade // 2), None)
    if i_b is None:
        return np.nan
    i_lo, i_hi = sorted((i_a, i_b))
    i_min = i_lo + int(np.argmin(p[i_lo: i_hi + 1]))
    if i_min in (i_lo, i_hi):
        return np.nan
    if p[i_min] > 0.9 * min(p[i_lo], p[i_hi]):    # dip depth >= 10 % of the smaller peak [ASSUMED]
        return np.nan
    return float(fb[i_min])


def cutoff_frequency(cfg, x_mod: np.ndarray, fs: float, U: float, z: float
                     ) -> Tuple[float, float, int]:
    """(lambda_c [m], f_c [Hz], source index into CUTOFF_SOURCES) for one window.

    Modes (locked decision 3; deviation register in ec_ampmod.md):
    ``spectral_gap`` from *x_mod* with the ``delta`` value as fallback;
    ``delta``: lambda_c = cfg.delta_m; ``scaled``: lambda_c = cfg.z_mult * z.
    """
    mode = cfg.cutoff_mode
    if mode == "spectral_gap":
        fg = spectral_gap(x_mod, fs)
        if np.isfinite(fg) and fg > 0:
            return U / fg, fg, 0
        lam = float(cfg.delta_m)
        return lam, U / lam, 1
    if mode == "delta":
        lam = float(cfg.delta_m)
        return lam, U / lam, 2
    if mode == "scaled":
        lam = float(cfg.z_mult) * z
        return lam, U / lam, 3
    raise ValueError(f"ampmod cutoff_mode {mode!r}; expected spectral_gap | delta | scaled")


_FLUX_PAIRS = {"uw": ("u", "w"), "wTs": ("w", "Ts"), "wrhov": ("w", "rhov")}


def _flux_series(prime: Dict[str, np.ndarray], key: str) -> Optional[np.ndarray]:
    """Instantaneous-flux series ('uw' -> u'w') from the perturbation dict, or None."""
    if key not in _FLUX_PAIRS:
        raise ValueError(f"ampmod flux {key!r}; expected one of {sorted(_FLUX_PAIRS)}")
    a, b = _FLUX_PAIRS[key]
    if a not in prime or b not in prime:
        return None
    p = prime[a] * prime[b]
    return p if np.isfinite(p).all() else None


def run(hf: HFFile, cfg, records: Optional[Sequence[int]] = None) -> xr.Dataset:
    """Amplitude-modulation group for every window and height; the ``/amplitude_mod`` Dataset."""
    ac, pc = cfg.ampmod, cfg.preprocess
    mods = list(ac.modulators)
    sigs = list(ac.signals)
    flxs = list(ac.fluxes)
    flux_members = [m for k in flxs for m in _FLUX_PAIRS[k]]
    names = ["u", "v", "w"] + [s for s in dict.fromkeys([*sigs, *mods, *flux_members])
                               if s not in ("u", "v", "w") and s in hf.ds]
    nr, nh = len(hf.records), len(hf.heights)

    def _arr():
        return np.full((nr, nh), np.nan)
    R = {(m, a): _arr() for m in mods for a in sigs + flxs}
    lam_c, f_c, U_mean, zeta = _arr(), _arr(), _arr(), _arr()
    source = np.full((nr, nh), -1, dtype=np.int8)

    for win in iter_windows(hf, variables=names, records=records):
        i = win.index
        for ih in range(nh):
            if not win.has("u", ih) or not win.has("w", ih):
                continue
            prep = pp.prepare(win, ih, names, method=pc.detrend, tau_s=pc.filter_tau_s,
                              nan_max_frac=pc.nan_max_frac, taylor_max_ratio=pc.taylor_max_ratio)
            prime = {k: v for k, v in prep.prime.items()
                     if prep.accepted.get(k, False) and np.isfinite(v).all()}
            if "u" not in prime:
                continue
            U = prep.U_mean
            z = float(hf.heights[ih])
            lam, fc, src = cutoff_frequency(ac, prime["u"], win.fs, U, z)
            lam_c[i, ih], f_c[i, ih], source[i, ih] = lam, fc, src
            U_mean[i, ih] = U
            L = win.ancillary.get("L", np.full(nh, np.nan))[ih]
            zeta[i, ih] = z / L if np.isfinite(L) and L != 0 else np.nan
            f1 = win.fs / len(prime["u"])
            if fc <= f1:                       # no resolved line below the cutoff
                continue
            large = {m: lowpass_sharp(prime[m], win.fs, fc) for m in mods if m in prime}
            series: Dict[str, np.ndarray] = {a: prime[a] for a in sigs if a in prime}
            for k in flxs:
                p = _flux_series(prime, k)
                if p is not None:
                    series[k] = p
            for a, x in series.items():
                x_s = x - lowpass_sharp(x, win.fs, fc)
                env_l = lowpass_sharp(envelope(x_s), win.fs, fc)
                for m, b_l in large.items():
                    R[(m, a)][i, ih] = am_coefficient(b_l, env_l)

    ds = xr.Dataset(coords={"record": ("record", hf.records),
                            "height": ("height", hf.heights)})
    dims = ("record", "height")
    for (m, a), arr in R.items():
        kind = "instantaneous flux" if a in flxs else "signal"
        ds[f"R_{m}L_{a}S"] = (dims, arr, {
            "long_name": f"amplitude-modulation coefficient of small-scale {a} ({kind}) by large-scale {m}",
            "reference": "Mathis et al. 2009 eq. 5.1; Salesky & Anderson 2018 eq. 1.6 (single-point)"})
    ds["cutoff_lambda"] = (dims, lam_c, {"units": "m", "long_name": "large/small cutoff wavelength lambda_c"})
    ds["cutoff_f"] = (dims, f_c, {"units": "Hz", "long_name": "large/small cutoff frequency U_mean/lambda_c"})
    ds["cutoff_source"] = (dims, source, {
        "flag_values": np.arange(len(CUTOFF_SOURCES), dtype=np.int8),
        "flag_meanings": " ".join(CUTOFF_SOURCES),
        "long_name": "how the cutoff was set (-1: window not processed)"})
    ds["U_mean"] = (dims, U_mean, {"units": "m s-1", "long_name": "window mean horizontal wind (Taylor advection speed)"})
    ds["zeta"] = (dims, zeta, {"long_name": "z/L from the EC ancillary Obukhov length"})
    ds.attrs.update(cutoff_mode=ac.cutoff_mode, delta_m=ac.delta_m, z_mult=ac.z_mult,
                    detrend_method=pc.detrend,
                    method="sharp spectral split at lambda_c; Hilbert envelope of the small scales, "
                           "low-passed at the same cutoff; R = corr(large-scale modulator, filtered envelope)",
                    library_note="library/writeups/ec_ampmod.md")
    return ds
