"""Quadrant and octant analysis of the flux-carrying events.

Science and sources: library/writeups/ec_quadrant.md. Quadrant stress and
time fractions with a hyperbolic hole (Lu & Willmarth 1973; Raupach 1981
eqs. 1-4), sign-aware ejection/sweep derived values reproducing UTESpac's
find_eta/find_delta_flux/find_delta_time exactly at H = 0, and the
Li & Bo 2019 eq. 11 octant partition. Invariant: fractions sum to 1 at
H = 0 (quadrants) and always (octants).
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from . import preprocess as pp
from .io import HFFile, iter_windows, token

GROUP = "quadrant"
OCTANT_GROUP = "octant"

# plane (x1, x2): quadrant by signs, Q1 (+,+), Q2 (-,+), Q3 (-,-), Q4 (+,-).
# uw on (u', w') -- Wallace et al. 1972 layout (Q2 ejection, Q4 sweep);
# scalars on (w', c') -- Li & Bo 2019 eq. 10 (labels sign-aware, see the note).
PAIRS = {"uw": ("u_pf", "w_pf"), "wTs": ("w_pf", "ts"), "wrhov": ("w_pf", "rho_h2o"), "wrhoCO2": ("w_pf", "rho_co2")}
QUADRANTS = ("Q1", "Q2", "Q3", "Q4")
# Li & Bo 2019 eq. 11 sign triples (u', w', c') for octants O1..O8
OCTANT_SIGNS = ((1, 1, 1), (-1, 1, 1), (-1, 1, -1), (1, 1, -1),
                (1, -1, 1), (-1, -1, 1), (-1, -1, -1), (1, -1, -1))
OCTANTS = ("O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8")


def quadrant_stats(x1: np.ndarray, x2: np.ndarray, hole_sizes: Sequence[float],
                   norm: str = "rms") -> Dict[str, np.ndarray]:
    """Quadrant flux fractions, time fractions, counts over the hole grid.

    Hole: exclude samples with |x1'x2'| < H * sigma_1 * sigma_2 (``rms``,
    Lu & Willmarth 1973 §4.3) or < H * |mean flux| (``flux``, Willmarth &
    Lu 1972; Raupach 1981 eq. 2); H_flux = |rho| H_rms. Fractions are of
    the total covariance / valid-sample count; at H = 0 the four flux
    fractions sum to 1 exactly (Raupach 1981 eq. 4).
    """
    valid = np.isfinite(x1) & np.isfinite(x2)
    n = int(valid.sum())
    nq, nH = len(QUADRANTS), len(hole_sizes)
    out = {"S": np.full((nq, nH), np.nan), "T": np.full((nq, nH), np.nan),
           "count": np.zeros((nq, nH), dtype=np.int64), "total": np.nan}
    if n == 0:
        return out
    a, b = x1[valid], x2[valid]
    p = a * b
    total = p.mean()
    out["total"] = total
    if norm == "rms":
        thresh0 = a.std() * b.std()
    elif norm == "flux":
        thresh0 = abs(total)
    else:
        raise ValueError(f"quadrant hole_norm {norm!r}; expected rms | flux")
    quad = [(a > 0) & (b > 0), (a < 0) & (b > 0), (a < 0) & (b < 0), (a > 0) & (b < 0)]
    for j, H in enumerate(hole_sizes):
        outside = np.abs(p) >= H * thresh0
        for i in range(nq):
            m = quad[i] & outside
            out["count"][i, j] = int(m.sum())
            out["T"][i, j] = m.sum() / n
            out["S"][i, j] = p[m].sum() / (n * total) if total != 0 else np.nan
    return out


def derived_h0(x1: np.ndarray, x2: np.ndarray, w_is: int) -> Dict[str, float]:
    """Sign-aware H = 0 values: delta_S, delta_D, eta, exuberance.

    Downgradient samples share the sign of the total flux; ejection is the
    downgradient half with w' > 0 (UTESpac convention -- reproduces
    find_delta_flux/find_delta_time/find_eta exactly; Li & Bou-Zeid 2011
    eqs. 8-9, 16-17). *w_is*: index (0 or 1) of w' in the (x1, x2) plane.
    """
    valid = np.isfinite(x1) & np.isfinite(x2)
    nan = dict(delta_S=np.nan, delta_D=np.nan, eta=np.nan, exuberance=np.nan)
    if not valid.any():
        return nan
    a, b = x1[valid], x2[valid]
    w = a if w_is == 0 else b
    p = a * b
    total = p.sum()
    n = p.size
    if total == 0:
        return nan
    down = (p * total) > 0
    ej, sw = down & (w > 0), down & (w < 0)
    dg = p[down].sum()
    return {
        "delta_S": p[ej].sum() / total - p[sw].sum() / total,
        "delta_D": (int(ej.sum()) - int(sw.sum())) / n,
        "eta": total / dg if dg != 0 else np.nan,
        "exuberance": total / dg - 1.0 if dg != 0 else np.nan,   # = counter/down [DERIVED] Li2011 eq. 8
    }


def octant_stats(u: np.ndarray, w: np.ndarray, c: np.ndarray) -> Dict[str, np.ndarray]:
    """Octant flux fractions of the three 2-D components, time fractions, counts.

    Octants by the signs of (u', w', c') per Li & Bo 2019 eq. 11; the flux
    fraction of component x'y' in octant i is sum(x'y' | octant i) / sum(x'y')
    (their O_i). Fractions over the 8 octants sum to 1 per component.
    """
    valid = np.isfinite(u) & np.isfinite(w) & np.isfinite(c)
    n = int(valid.sum())
    comps = ("uw", "uc", "wc")
    out = {f"F_{k}": np.full(8, np.nan) for k in comps}
    out.update({"T": np.full(8, np.nan), "count": np.zeros(8, dtype=np.int64)})
    out.update({f"total_{k}": np.nan for k in comps})
    if n == 0:
        return out
    uu, ww, cc = u[valid], w[valid], c[valid]
    prods = {"uw": uu * ww, "uc": uu * cc, "wc": ww * cc}
    for k in comps:
        out[f"total_{k}"] = prods[k].mean()
    for i, (su, sw_, sc) in enumerate(OCTANT_SIGNS):
        m = (su * uu > 0) & (sw_ * ww > 0) & (sc * cc > 0)
        out["count"][i] = int(m.sum())
        out["T"][i] = m.sum() / n
        for k in comps:
            tot = prods[k].sum()
            out[f"F_{k}"][i] = prods[k][m].sum() / tot if tot != 0 else np.nan
    return out


def _prime(win, ih, names, pc):
    prep = pp.prepare(win, ih, names, method=pc.detrend, tau_s=pc.filter_tau_s,
                      nan_max_frac=pc.nan_max_frac, taylor_max_ratio=pc.taylor_max_ratio)
    return {k: v for k, v in prep.prime.items() if prep.accepted.get(k, False)}


def run(hf: HFFile, cfg, records: Optional[Sequence[int]] = None) -> xr.Dataset:
    """Quadrant statistics for every window and pair; the ``/quadrant`` Dataset."""
    qc, pc = cfg.quadrant, cfg.preprocess
    pairs = [k for k in qc.pairs if all(m in ("u_pf", "v_pf", "w_pf") or m in hf.ds for m in PAIRS[k])]
    names = list(dict.fromkeys(["u_pf", "v_pf", "w_pf"] + [m for k in pairs for m in PAIRS[k]]))
    holes = np.asarray(qc.hole_sizes, dtype=float)
    nr, nh, nq, nH = len(hf.records), len(hf.heights), len(QUADRANTS), len(holes)

    S = {k: np.full((nr, nh, nq, nH), np.nan) for k in pairs}
    T = {k: np.full((nr, nh, nq, nH), np.nan) for k in pairs}
    count = {k: np.zeros((nr, nh, nq, nH), dtype=np.int64) for k in pairs}
    cov = {k: np.full((nr, nh), np.nan) for k in pairs}
    der = {k: {d: np.full((nr, nh), np.nan) for d in ("delta_S", "delta_D", "eta", "exuberance")}
           for k in pairs}

    for win in iter_windows(hf, variables=names, records=records):
        i = win.index
        for ih in range(nh):
            if not win.has("w_pf", ih):
                continue
            prime = _prime(win, ih, names, pc)
            for k in pairs:
                a, b = PAIRS[k]
                if a not in prime or b not in prime:
                    continue
                st = quadrant_stats(prime[a], prime[b], holes, qc.hole_norm)
                S[k][i, ih], T[k][i, ih], count[k][i, ih] = st["S"], st["T"], st["count"]
                cov[k][i, ih] = st["total"]
                d = derived_h0(prime[a], prime[b], w_is=(0 if a == "w_pf" else 1))
                for name, v in d.items():
                    der[k][name][i, ih] = v

    ds = xr.Dataset(coords={"record": ("record", hf.records),
                            "height": ("height", hf.heights),
                            "quadrant": ("quadrant", list(QUADRANTS)),
                            "hole": ("hole", holes)})
    ds["hole"].attrs.update(long_name=f"hyperbolic hole size H ({qc.hole_norm} normalization)")
    dims4, dims = ("record", "height", "quadrant", "hole"), ("record", "height")
    for k in pairs:
        a, b = (token(s) for s in PAIRS[k])
        plane = f"({a}', {b}')"
        ds[f"S_frac_{k}"] = (dims4, S[k], {
            "long_name": f"{a}'{b}' flux fraction per quadrant of the {plane} plane outside the hole"})
        ds[f"dur_frac_{k}"] = (dims4, T[k], {
            "long_name": f"time fraction per quadrant of the {plane} plane outside the hole"})
        ds[f"count_{k}"] = (dims4, count[k], {"long_name": "samples per quadrant outside the hole"})
        ds[f"cov_{k}"] = (dims, cov[k], {"long_name": f"total {a}'{b}' covariance (fraction denominator)"})
        ds[f"delta_S_{k}"] = (dims, der[k]["delta_S"], {
            "long_name": "ejection minus sweep flux fraction, sign-aware (= UTESpac delta_flux_ctrb)"})
        ds[f"delta_D_{k}"] = (dims, der[k]["delta_D"], {
            "long_name": "ejection minus sweep time fraction, sign-aware (= UTESpac delta_time_ctrb)"})
        ds[f"eta_{k}"] = (dims, der[k]["eta"], {
            "long_name": "transport efficiency total/downgradient (Li & Bou-Zeid 2011 eq. 8; = UTESpac eta)"})
        ds[f"exuberance_{k}"] = (dims, der[k]["exuberance"], {
            "long_name": "countergradient over downgradient flux (= eta - 1)"})
    ds.attrs.update(hole_norm=qc.hole_norm, detrend_method=pc.detrend,
                    quadrant_layout="Q1 (+,+), Q2 (-,+), Q3 (-,-), Q4 (+,-) on the plane in each "
                                    "variable's long_name; uw: Wallace et al. 1972; scalars: Li & Bo 2019 eq. 10",
                    closure="S_frac sums to 1 over quadrant at H = 0",
                    library_note="library/writeups/ec_quadrant.md")
    return ds


def run_octant(hf: HFFile, cfg, records: Optional[Sequence[int]] = None) -> xr.Dataset:
    """Octant partition of every configured triplet; the ``/octant`` Dataset.

    Each triplet (a, b, c) contributes variables tagged ``_<abc>``: the flux
    fractions of its three 2-D components a'b', a'c', b'c', the per-octant
    time fraction and count, and the component covariances. Triplets with a
    signal absent from the file are skipped.
    """
    qc, pc = cfg.quadrant, cfg.preprocess
    trips = []
    for t in qc.octant_triplets:
        t = tuple(t)
        if len(t) != 3:
            raise ValueError(f"octant triplet needs 3 signals, got {t}")
        if all(m in ("u_pf", "v_pf", "w_pf") or m in hf.ds for m in t):
            trips.append(t)
    names = list(dict.fromkeys(["u_pf", "v_pf", "w_pf"] + [m for t in trips for m in t]))
    nr, nh = len(hf.records), len(hf.heights)

    comps = {t: ((t[0], t[1]), (t[0], t[2]), (t[1], t[2])) for t in trips}
    F = {t: {k: np.full((nr, nh, 8), np.nan) for k in ("uw", "uc", "wc")} for t in trips}
    T = {t: np.full((nr, nh, 8), np.nan) for t in trips}
    count = {t: np.zeros((nr, nh, 8), dtype=np.int64) for t in trips}
    tot = {t: {k: np.full((nr, nh), np.nan) for k in ("uw", "uc", "wc")} for t in trips}

    for win in iter_windows(hf, variables=names, records=records):
        i = win.index
        for ih in range(nh):
            if not win.has("w_pf", ih):
                continue
            prime = _prime(win, ih, names, pc)
            for t in trips:
                if not all(m in prime for m in t):
                    continue
                st = octant_stats(prime[t[0]], prime[t[1]], prime[t[2]])
                T[t][i, ih], count[t][i, ih] = st["T"], st["count"]
                for k in ("uw", "uc", "wc"):
                    F[t][k][i, ih] = st[f"F_{k}"]
                    tot[t][k][i, ih] = st[f"total_{k}"]

    ds = xr.Dataset(coords={"record": ("record", hf.records),
                            "height": ("height", hf.heights),
                            "octant": ("octant", list(OCTANTS))})
    dims3, dims = ("record", "height", "octant"), ("record", "height")
    layouts = []
    for t in trips:
        tt = tuple(token(s) for s in t)
        tag = "_".join(tt)
        layout = ("O1..O8 by the signs of ({0}', {1}', {2}') per Li & Bo 2019 eq. 11: "
                  "(+++), (-++), (-+-), (++-), (+-+), (--+), (---), (+--)").format(*tt)
        layouts.append(f"{tag}: {layout}")
        for k, (a, b) in zip(("uw", "uc", "wc"), comps[t]):
            a, b = token(a), token(b)
            ds[f"flux_frac_{tag}_{a}{b}"] = (dims3, F[t][k], {
                "long_name": f"fraction of the {a}'{b}' covariance per ({tt[0]}', {tt[1]}', {tt[2]}') octant"})
            ds[f"cov_{tag}_{a}{b}"] = (dims, tot[t][k], {"long_name": f"total {a}'{b}' covariance"})
        ds[f"dur_frac_{tag}"] = (dims3, T[t], {
            "long_name": f"time fraction per ({tt[0]}', {tt[1]}', {tt[2]}') octant"})
        ds[f"count_{tag}"] = (dims3, count[t], {"long_name": "samples per octant"})
    ds.attrs.update(octant_triplets="; ".join(",".join(token(s) for s in t) for t in trips),
                    detrend_method=pc.detrend,
                    octant_layout=" | ".join(layouts),
                    closure="flux_frac sums to 1 over octant per component; no hole (none of the read "
                            "octant sources applies one)",
                    library_note="library/writeups/ec_quadrant.md")
    return ds
