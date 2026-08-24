"""Coherent-structure flux fractions.

Two estimators, labelled by ``method``: conditional averages about the
ramp-module event times with the triple decomposition (Collineau & Brunet
1993 II eqs. 3-7; Thomas & Foken 2007 eqs. 3-7), and the ejection-plus-
sweep quadrant contribution above a hyperbolic hole (Thomas & Foken 2007
section 3.3). Science and sources: library/writeups/ec_coherent_flux.md.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from . import preprocess as pp
from .io import HFFile, iter_windows
from .quadrant import PAIRS, quadrant_stats
from .ramps import _slope_for, detect, log_scales

GROUP = "coherent_flux"


def conditional_average(x: np.ndarray, fs: float, times_s: np.ndarray,
                        half_window_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """(lag_s, <x>(lag)): mean of x(t + t_i) over events  [CITED] CB 1993 II eq. 3.

    The window spans ``+-half_window_s`` about each detection; lags that fall
    outside the record for an event are excluded from that lag's mean
    (NaN-padded sampling), so edge events still contribute where they can.
    """
    x = np.asarray(x, dtype=float)
    h = int(round(half_window_s * fs))
    lag = np.arange(-h, h + 1)
    idx = np.round(np.asarray(times_s, dtype=float) * fs).astype(int)[:, None] + lag[None, :]
    inside = (idx >= 0) & (idx < x.size)
    samp = np.where(inside, x[np.clip(idx, 0, x.size - 1)], 0.0)
    cnt = inside.sum(axis=0)
    mean = np.divide(samp.sum(axis=0), cnt, out=np.full(lag.size, np.nan),
                     where=cnt > 0)
    return lag / fs, mean


def flux_contribution(xp: np.ndarray, yp: np.ndarray, fs: float, times_s: np.ndarray,
                      half_window_s: float) -> Dict[str, float]:
    """Triple-decomposition flux terms for one pair and one event set.

    F_tot = tilde<x'y'>, F_cs = tilde(<x'><y'>), F_t = F_tot - F_cs
    [CITED] Thomas & Foken 2007 eqs. 5-7; F_ej / F_sw from the [-D_e, 0] /
    [0, +D_e] halves of the same operator (their p. 322), so
    F_ej + F_sw = F_cs exactly. ``ratio`` is F_tot / cov with the 0.8-1.2
    validity gate (their §4.2); ``flux_error`` is their eq. 12b with
    Delta x = -tilde<x'>  [DERIVED, see the library note].
    """
    nan = dict(F_tot=np.nan, F_cs=np.nan, F_t=np.nan, F_ej=np.nan, F_sw=np.nan,
               ratio=np.nan, valid=False, flux_error=np.nan, n=int(np.size(times_s)))
    if np.size(times_s) == 0 or not np.isfinite(half_window_s) or half_window_s <= 0:
        return nan
    lag, cx = conditional_average(xp, fs, times_s, half_window_s)
    _, cy = conditional_average(yp, fs, times_s, half_window_s)
    _, cxy = conditional_average(xp * yp, fs, times_s, half_window_s)
    prod = cx * cy
    if not (np.isfinite(prod).all() and np.isfinite(cxy).all()):
        return nan
    nl = lag.size
    F_tot = float(np.mean(cxy))
    F_cs = float(np.mean(prod))
    at0 = float(prod[nl // 2])
    F_ej = float((prod[lag < 0].sum() + 0.5 * at0) / nl)
    F_sw = float((prod[lag > 0].sum() + 0.5 * at0) / nl)
    cov = float(np.nanmean(xp * yp))
    ratio = F_tot / cov if cov != 0 else np.nan
    dx, dy = -float(np.mean(cx)), -float(np.mean(cy))
    n = xp.size
    err = float(np.sum(xp * dy + yp * dx - dy * dx) / (n - 1))
    return dict(F_tot=F_tot, F_cs=F_cs, F_t=F_tot - F_cs, F_ej=F_ej, F_sw=F_sw,
                ratio=ratio, valid=bool(np.isfinite(ratio) and 0.8 <= ratio <= 1.2),
                flux_error=err, n=int(np.size(times_s)))


def haar_coefficients(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(W, valid): redundant dyadic Haar coefficients W(m, b), m = 1..floor(log2 n).

    W(m, x) = sum f(x') Psi^(m)(x' - x) with Psi^(m) = 2^(-m/2) Psi^(0)(x/2^m)
    [CITED] Turner & Leclerc 1994 eqs. 1-3; support 2^m samples, positions where
    the support runs past the record are zero and flagged invalid.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    M = int(np.floor(np.log2(n)))
    S = np.concatenate([[0.0], np.cumsum(x)])
    W = np.zeros((M, n))
    valid = np.zeros((M, n), dtype=bool)
    for j, m in enumerate(range(1, M + 1)):
        h = 1 << (m - 1)
        b = np.arange(0, n - 2 * h + 1)
        W[j, b] = ((S[b + h] - S[b]) - (S[b + 2 * h] - S[b + h])) * 2.0 ** (-m / 2)
        valid[j, b] = True
    return W, valid


def haar_reconstruct(W: np.ndarray) -> np.ndarray:
    """Inverse transform f(x) = sum_m 2^(-m) int W(m, x') Psi^(m)(x - x') dx'
    [CITED] Turner & Leclerc 1994 eq. 4 (normalization factor 1 for the Haar basis)."""
    M, n = W.shape
    out = np.zeros(n)
    t = np.arange(n)
    for j, m in enumerate(range(1, M + 1)):
        h = 1 << (m - 1)
        C = np.concatenate([[0.0], np.cumsum(W[j])])
        lo1, hi1 = np.clip(t - h + 1, 0, n), np.clip(t + 1, 0, n)
        lo2, hi2 = np.clip(t - 2 * h + 1, 0, n), np.clip(t - h + 1, 0, n)
        out += (C[hi1] - C[lo1] - (C[hi2] - C[lo2])) * 2.0 ** (-1.5 * m)
    return out


def turner_split(x: np.ndarray, K: float = 4.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(strong, weak, reconstruction): coefficient-threshold conditional sampling.

    Coefficients with |W(m, x)| > K [1/N sum W^2(m, x)]^(1/2) at each scale, rms
    over the entire record, are the strong set [CITED] Turner & Leclerc 1994
    eq. 5 (K = 4 their default); the two sets are inverse-transformed separately
    and sum to the reconstruction exactly. The reconstruction drops the
    long-term trend and the edge-truncated large-scale content (their §3).
    """
    W, valid = haar_coefficients(x)
    ms = np.sqrt(np.sum(W * W * valid, axis=1) / np.maximum(valid.sum(axis=1), 1))
    strong = np.where(np.abs(W) > K * ms[:, None], W, 0.0)
    xs = haar_reconstruct(strong)
    xr = haar_reconstruct(W)
    return xs, xr - xs, xr


def turner_fraction(xp: np.ndarray, yp: np.ndarray, K: float = 4.0) -> Dict[str, float]:
    """Strong-signal covariance fraction of one pair (estimator 3).

    Each series is split by its own coefficient threshold (as the paper does
    for temperature); the coherent flux is cov(x_s, y_s) and the fraction is
    against the reconstructed covariance. The covariance construction is
    [NOVEL] (no flux is computed in the paper; user sign-off 2026-08-24) --
    ``ratio`` = cov(x_r, y_r)/cov(x, y) measures reconstruction fidelity.
    """
    xs, _, xr = turner_split(xp, K)
    ys, _, yr = turner_split(yp, K)
    cov = float(np.mean(xp * yp))
    cov_r = float(np.mean(xr * yr))
    F = float(np.mean(xs * ys))
    return dict(F_strong=F, frac=F / cov_r if cov_r != 0 else np.nan,
                ratio=cov_r / cov if cov != 0 else np.nan)


def quadrant_fraction(x1p: np.ndarray, x2p: np.ndarray, hole_sizes: Sequence[float],
                      w_is: int) -> Dict[str, np.ndarray]:
    """Ejection / sweep / coherent flux fractions above each hole (estimator 2).

    The two downgradient quadrants of ``quadrant_stats`` (rms hole,
    L = x'y'/(sigma_x sigma_y) [CITED] Thomas & Foken 2007 §3.3); ejection is
    the downgradient quadrant with w' > 0 -- equivalent to their
    sign-of-r_xy quadrant selection. F_coh = F_ej + F_sw per hole.
    """
    nH = len(hole_sizes)
    nan = dict(F_ej=np.full(nH, np.nan), F_sw=np.full(nH, np.nan),
               F_coh=np.full(nH, np.nan), total=np.nan)
    st = quadrant_stats(x1p, x2p, hole_sizes, norm="rms")
    total = st["total"]
    if not np.isfinite(total) or total == 0:
        return nan
    # quadrant sign layout (x1, x2): Q1 (+,+), Q2 (-,+), Q3 (-,-), Q4 (+,-)
    signs = ((1, 1), (-1, 1), (-1, -1), (1, -1))
    down = [i for i, (s1, s2) in enumerate(signs) if s1 * s2 == (1 if total > 0 else -1)]
    ej = next(i for i in down if signs[i][w_is] > 0)
    sw = next(i for i in down if signs[i][w_is] < 0)
    return dict(F_ej=st["S"][ej], F_sw=st["S"][sw],
                F_coh=st["S"][ej] + st["S"][sw], total=total)


def run(hf: HFFile, cfg, records: Optional[Sequence[int]] = None) -> xr.Dataset:
    """``/coherent_flux`` group: both estimators per record, height, pair."""
    cf, rc, pc = cfg.coherent_flux, cfg.ramps, cfg.preprocess
    unsupported = [s for s in cf.event_signals if s == "e"]
    if unsupported:
        raise ValueError("coherent_flux conditions on wavelet ramp events; "
                         "the TKE event set ('e') is not supported")
    sigs = [s for s in cf.event_signals if s in hf.ds]
    pairs = [k for k in cf.pairs if all(m in ("u", "v", "w") or m in hf.ds for m in PAIRS[k])]
    names = list(dict.fromkeys(["u", "v", "w", "Ts"] + sigs + [m for k in pairs for m in PAIRS[k]]))
    holes = np.asarray(cf.hole_sizes, dtype=float)
    scales = log_scales(rc.a_min_s, rc.a_max_s, rc.n_scales_per_decade)
    nr, nh, nH = len(hf.records), len(hf.heights), len(holes)

    fields = ("F_tot", "F_cs", "F_t", "F_ej", "F_sw", "ratio", "flux_error")
    wl = {s: {k: {f: np.full((nr, nh), np.nan) for f in fields} for k in pairs} for s in sigs}
    valid = {s: {k: np.zeros((nr, nh), dtype=bool) for k in pairs} for s in sigs}
    n_ev = {s: np.zeros((nr, nh), dtype=int) for s in sigs}
    halfw = {s: np.full((nr, nh), np.nan) for s in sigs}
    cov = {k: np.full((nr, nh), np.nan) for k in pairs}
    qd = {k: {f: np.full((nr, nh, nH), np.nan) for f in ("F_ej", "F_sw", "F_coh")} for k in pairs}
    tu = {k: {f: np.full((nr, nh), np.nan) for f in ("F_strong", "frac", "ratio")} for k in pairs}

    for win in iter_windows(hf, variables=names, records=records):
        i = win.index
        for ih in range(nh):
            if not win.has("w", ih):
                continue
            prep = pp.prepare(win, ih, names, method=pc.detrend, tau_s=pc.filter_tau_s,
                              nan_max_frac=pc.nan_max_frac, taylor_max_ratio=pc.taylor_max_ratio)
            prime = {k: v for k, v in prep.prime.items() if prep.accepted.get(k, False)}
            wT = np.nan
            if "w" in prime and "Ts" in prime:
                wT = float(np.nanmean(prime["w"] * prime["Ts"]))
            events = {}
            for s in sigs:
                if s not in prime or not np.isfinite(prime[s]).all():
                    continue
                slope = _slope_for(s, getattr(rc, f"slope_{s}", "auto"), wT)
                ev = detect(prime[s], hf.fs, scales, slope=slope, wavelet=rc.wavelet,
                            peak=rc.peak, edge_scales=rc.edge_scales, refine=rc.refine,
                            D_min_s=rc.D_min_s)
                events[s] = ev
                n_ev[s][i, ih] = ev.n
                halfw[s][i, ih] = ev.D if cf.window == "duration" else 0.5 * cf.window_s
            for k in pairs:
                a, b = PAIRS[k]
                if a not in prime or b not in prime:
                    continue
                xp, yp = prime[a], prime[b]
                cov[k][i, ih] = float(np.nanmean(xp * yp))
                q = quadrant_fraction(xp, yp, holes, w_is=(0 if a == "w" else 1))
                for f in ("F_ej", "F_sw", "F_coh"):
                    qd[k][f][i, ih] = q[f]
                if not (np.isfinite(xp).all() and np.isfinite(yp).all()):
                    continue
                if cf.turner:
                    tf = turner_fraction(xp, yp, cf.turner_K)
                    for f in ("F_strong", "frac", "ratio"):
                        tu[k][f][i, ih] = tf[f]
                for s, ev in events.items():
                    fc = flux_contribution(xp, yp, hf.fs, ev.times, halfw[s][i, ih])
                    for f in fields:
                        wl[s][k][f][i, ih] = fc[f]
                    valid[s][k][i, ih] = fc["valid"]

    ds = xr.Dataset(coords={"record": ("record", hf.records), "height": ("height", hf.heights),
                            "hole": ("hole", holes)})
    ds["hole"].attrs.update(long_name="hyperbolic hole size L (rms normalization, TF 2007 §3.3)")
    dims = ("record", "height")
    long = {"F_tot": "window-average total flux tilde<x'y'> (TF 2007 eq. 6)",
            "F_cs": "coherent-structure flux tilde(<x'><y'>) (TF 2007 eq. 7)",
            "F_t": "high-frequency stochastic flux F_tot - F_cs",
            "F_ej": "ejection-phase flux, [-D_e, 0] half of the operator",
            "F_sw": "sweep-phase flux, [0, +D_e] half of the operator",
            "ratio": "F_tot / cov representativeness (TF 2007 §4.2 gate 0.8-1.2)",
            "flux_error": "EC flux error Delta(x'y') from coherent structures (TF 2007 eq. 12b)"}
    for s in sigs:
        ds[f"n_structures_{s}"] = (dims, n_ev[s], {"long_name": f"wavelet events of {s}' conditioned on"})
        ds[f"half_window_{s}"] = (dims, halfw[s], {"units": "s",
                                  "long_name": f"conditional-window half width about {s}' events ({cf.window})"})
        for k in pairs:
            a, b = PAIRS[k]
            for f in fields:
                ds[f"{f}_{k}_{s}"] = (dims, wl[s][k][f], {
                    "long_name": f"{long[f]} of {a}'{b}' on {s}' events", "method": "wavelet"})
            ds[f"F_frac_{k}_{s}"] = (dims, wl[s][k]["F_cs"] / wl[s][k]["F_tot"], {
                "long_name": f"coherent flux fraction F_cs/F_tot of {a}'{b}' on {s}' events",
                "method": "wavelet"})
            ds[f"valid_{k}_{s}"] = (dims, valid[s][k], {
                "long_name": "0.8 <= F_tot/cov <= 1.2 (TF 2007 §4.2)", "method": "wavelet"})
    for k in pairs:
        a, b = PAIRS[k]
        ds[f"cov_{k}"] = (dims, cov[k], {"long_name": f"total {a}'{b}' covariance"})
        for f in ("F_ej", "F_sw", "F_coh"):
            what = {"F_ej": "ejection", "F_sw": "sweep", "F_coh": "ejection + sweep"}[f]
            ds[f"{f}_quad_{k}"] = (("record", "height", "hole"), qd[k][f], {
                "long_name": f"{what} flux fraction of {a}'{b}' above the hole (TF 2007 eqs. 8-9)",
                "method": "quadrant"})
        if cf.turner:
            ds[f"F_turner_{k}"] = (dims, tu[k]["F_strong"], {
                "long_name": f"strong-signal covariance of {a}'{b}' (Turner & Leclerc 1994 K-rms split)",
                "method": "turner"})
            ds[f"F_frac_turner_{k}"] = (dims, tu[k]["frac"], {
                "long_name": f"strong-signal covariance fraction of {a}'{b}' (of the reconstructed covariance)",
                "method": "turner"})
            ds[f"ratio_turner_{k}"] = (dims, tu[k]["ratio"], {
                "long_name": "reconstructed / total covariance (Haar-frame fidelity)",
                "method": "turner"})
    ds.attrs.update(event_signals=",".join(sigs), window=cf.window, window_s=cf.window_s,
                    turner=int(cf.turner), turner_K=cf.turner_K,
                    wavelet=rc.wavelet, peak=rc.peak, D_min_s=rc.D_min_s, refine=rc.refine,
                    detrend_method=pc.detrend,
                    methods="wavelet (conditional averages, triple decomposition) | "
                            "quadrant (ejection + sweep above the hole)"
                            + (" | turner (K-rms coefficient-threshold split)" if cf.turner else ""),
                    library_note="library/writeups/ec_coherent_flux.md")
    return ds
