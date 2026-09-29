"""Planar-fit and yaw rotation of sonic wind vectors (migration step 4).

The numerics of ``sonic_rotation``/``pf_coefficients`` as small, testable
pieces: :class:`PlanarFit` (the Wilczak et al. 2001 regression, its
rotation matrix and its application), :func:`sector_mask` (the
wind-direction sectors of a global fit), the sector bookkeeping of a
sectorwise fit (:func:`sector_keys`, :func:`sector_of`,
:func:`sector_edges_from_histogram`, :func:`fit_planar`,
:func:`bootstrap_tilt`), :func:`yaw_rotate` (per-period
yaw into the mean wind) and :func:`rotate_sonics`, which rotates every
sonic of a run and returns a :class:`RotationResult`; ``utespac.stages.rotate``
feeds it from the run model.

Wilczak, J. M., S. P. Oncley and S. A. Stage (2001), Sonic anemometer tilt
correction algorithms, Boundary-Layer Meteorol. 99, 127-150: eqs. 35-39
(offset c3 = b0 removed before rotation), 42-44 (P = D'C').
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .averaging import block_mean, n_periods, period_bounds
from .pf_info import PFRecord, PFTable

log = logging.getLogger("utespac")


# ── planar fit ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PlanarFit:
    """``w = b0 + b1 u + b2 v`` fitted to period-mean winds."""
    b0: float
    b1: float
    b2: float
    n: int = 0          # points in the fit (0 when taken from a stored table)

    @property
    def pitch_deg(self) -> float:
        return float(np.degrees(np.arcsin(-self.b1 / np.sqrt(1 + self.b1 ** 2))))

    @property
    def roll_deg(self) -> float:
        return float(np.degrees(np.arcsin(self.b2 / np.sqrt(1 + self.b2 ** 2))))

    @classmethod
    def fit(cls, u_bar, v_bar, w_bar, min_points: int = 4) -> Optional["PlanarFit"]:
        """Least-squares plane through the period means (NaN rows dropped);
        None with fewer than *min_points* rows or a singular system."""
        u = np.asarray(u_bar, dtype=float)
        v = np.asarray(v_bar, dtype=float)
        w = np.asarray(w_bar, dtype=float)
        ok = ~(np.isnan(u) | np.isnan(v) | np.isnan(w))
        u, v, w = u[ok], v[ok], w[ok]
        n = len(u)
        if n < min_points:
            return None
        su, sv, sw = np.sum(u), np.sum(v), np.sum(w)
        suv, suw, svw = np.sum(u * v), np.sum(u * w), np.sum(v * w)
        su2, sv2 = np.sum(u ** 2), np.sum(v ** 2)
        H = np.array([[n, su, sv], [su, su2, suv], [sv, suv, sv2]], dtype=float)
        g = np.array([sw, suw, svw], dtype=float)
        try:
            b = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return None
        return cls(float(b[0]), float(b[1]), float(b[2]), n)

    @classmethod
    def from_record(cls, rec: PFRecord) -> "PlanarFit":
        return cls(rec.b0, rec.b1, rec.b2)

    def matrix(self) -> np.ndarray:
        return pf_matrix(self.b1, self.b2)

    def apply(self, u, v, w, remove_offset: bool = True) -> np.ndarray:
        """Winds in the fitted plane, ``(n, 3)``; ``remove_offset`` subtracts b0
        from w first (Wilczak eq. 35; the MATLAB port did not)."""
        return apply_planar_fit(self.matrix(), self.b0 if remove_offset else 0.0, u, v, w)

    def describe(self) -> str:
        return (f"b0={self.b0:.3g} b1={self.b1:.3g} b2={self.b2:.3g} "
                f"pitch={self.pitch_deg:.3g} roll={self.roll_deg:.3g} deg")


def pf_matrix(b1: float, b2: float) -> np.ndarray:
    """Wilczak et al. (2001) planar-fit matrix P = D'C' (eqs. 42-44, p. 141)."""
    denom = np.sqrt(b1 ** 2 + b2 ** 2 + 1.0)
    p31, p32, p33 = -b1 / denom, -b2 / denom, 1.0 / denom
    sin_alpha = p31
    cos_alpha = np.sqrt(p32 ** 2 + p33 ** 2)
    sin_beta = -p32 / np.sqrt(p32 ** 2 + p33 ** 2)
    cos_beta = p33 / np.sqrt(p32 ** 2 + p33 ** 2)
    C = np.array([[1, 0, 0],
                  [0, cos_beta, -sin_beta],
                  [0, sin_beta, cos_beta]])
    D = np.array([[cos_alpha, 0, sin_alpha],
                  [0, 1, 0],
                  [-sin_alpha, 0, cos_alpha]])
    return D.T @ C.T


def apply_planar_fit(P: np.ndarray, b0: float, u, v, w) -> np.ndarray:
    """u_p = P (u_m - c), c = (0, 0, b0) (Wilczak et al. 2001 eqs. 35-39): the
    regression intercept is the w offset; horizontal offsets are not
    recoverable by the planar fit and are taken as zero."""
    return (P @ np.column_stack([u, v, w - b0]).T).T


def sector_mask(direction, lo: float, hi: float) -> np.ndarray:
    """Samples whose direction falls in the sector ``(lo, hi)``; ``lo == hi``
    means all directions, ``hi < lo`` wraps through north (open bounds, as
    MATLAB sonicRotation)."""
    d = np.asarray(direction, dtype=float)
    if lo == hi:
        return np.ones(d.shape, dtype=bool)
    if hi > lo:
        return (d > lo) & (d < hi)
    return (d > lo) | (d < hi)


def fit_sectors(u_bar, v_bar, w_bar, direction, bins: Sequence[float]
                ) -> Dict[Tuple[float, float], Optional[PlanarFit]]:
    """One :class:`PlanarFit` per direction sector of *bins* (closed on the
    upper edge, as ``pf_coefficients``); no bins = one all-direction sector
    keyed ``(0, 0)``."""
    u = np.asarray(u_bar, dtype=float)
    v = np.asarray(v_bar, dtype=float)
    w = np.asarray(w_bar, dtype=float)
    d = np.asarray(direction, dtype=float)
    out: Dict[Tuple[float, float], Optional[PlanarFit]] = {}
    if not bins:
        out[(0.0, 0.0)] = PlanarFit.fit(u, v, w)
        return out
    for i in range(len(bins)):
        if i < len(bins) - 1:
            lo, hi = float(bins[i]), float(bins[i + 1])
            m = (d > lo) & (d <= hi)
        else:
            lo, hi = float(bins[i]), float(bins[0])
            m = (d > lo) | (d <= hi)
        out[(lo, hi)] = PlanarFit.fit(u[m], v[m], w[m])
    return out


# ── sector bookkeeping ───────────────────────────────────────────────────────

N_BOOTSTRAP = 1000      # default resample count of :func:`bootstrap_tilt`


def sector_keys(edges: Sequence[float]) -> List[Tuple[float, float]]:
    """Sector ``(lo, hi)`` keys for *edges*, in :func:`fit_sectors` order.

    :func:`fit_sectors` pairs each edge with the next, closing the list
    through the first edge, and takes each sector closed on its upper bound.
    Edges ``(60, 250)`` therefore give ``(60, 250]`` and ``(250, 60]``, the
    second wrapping through north.

    Parameters
    ----------
    edges : sequence of float
        Sector edges [deg], as passed to :func:`fit_sectors`.

    Returns
    -------
    list of tuple of float
        One ``(lo, hi)`` key per sector; empty for no edges.
    """
    e = [float(x) for x in edges]
    return [(e[i], e[i + 1] if i < len(e) - 1 else e[0]) for i in range(len(e))]


def sector_of(direction, edges: Sequence[float]) -> list:
    """Sector key of each direction [deg], matching :func:`sector_keys`.

    Membership follows :func:`fit_sectors`: open on the lower edge, closed
    on the upper edge, wrapping through north when ``hi < lo``.

    Parameters
    ----------
    direction : array_like
        Wind directions [deg], any real value; wrapped into [0, 360).
    edges : sequence of float
        Sector edges, as passed to :func:`fit_sectors`.

    Returns
    -------
    list
        One ``(lo, hi)`` key per direction, ``None`` where the direction is
        not finite.
    """
    d = np.asarray(direction, dtype=float)
    keys = sector_keys(edges)
    out = [None] * d.size
    flat = np.mod(d.ravel(), 360.0)
    for lo, hi in keys:
        m = (flat > lo) & (flat <= hi) if hi > lo else (flat > lo) | (flat <= hi)
        for i in np.flatnonzero(m & np.isfinite(flat)):
            out[i] = (lo, hi)
    return out


def sector_label(key: Optional[Tuple[float, float]]) -> str:
    """Degree range of a sector key, ``all`` for the single-fit key ``(x, x)``
    and an empty string for ``None``."""
    if key is None:
        return ""
    lo, hi = key
    return "all" if lo == hi else f"{lo:.0f}-{hi:.0f}"


def sector_name(key: Optional[Tuple[float, float]],
                names: Dict[Tuple[float, float], str]) -> str:
    """Name of a sector key under the mapping *names*.

    Parameters
    ----------
    key : tuple of float or None
        Sector ``(lo, hi)`` key.
    names : dict
        Sector key to name, for example ``{(250.0, 60.0): "A"}``.

    Returns
    -------
    str
        The mapped name; ``all`` for the single-fit key ``(x, x)``; the
        :func:`sector_label` degree range for a key absent from *names*; an
        empty string for ``None``.
    """
    if key is None:
        return ""
    lo, hi = key
    if lo == hi:
        return "all"
    return names.get((float(lo), float(hi)), sector_label(key))


def sector_edges_from_histogram(direction, admit, bin_deg: float
                                ) -> Tuple[List[float], np.ndarray]:
    """Sector edges at the centre of every empty run of a direction histogram.

    The admitted directions are binned at *bin_deg*; a contiguous run of
    empty bins is a gap between lobes, and its centre becomes an edge. The
    runs are found on the circle, so a gap straddling north is one run.

    Parameters
    ----------
    direction : array_like
        Directions [deg], any real value; wrapped into [0, 360).
    admit : array_like of bool or index
        Selection of *direction* entering the histogram.
    bin_deg : float
        Histogram bin width [deg].

    Returns
    -------
    edges : list of float
        Edge angles [deg], ascending, in the form :func:`fit_sectors`
        expects; empty when no bin, or every bin, is empty.
    hist : numpy.ndarray
        The bin counts.
    """
    d = np.mod(np.asarray(direction, dtype=float)[admit], 360.0)
    d = d[np.isfinite(d)]
    n_bins = int(round(360.0 / bin_deg))
    hist, _ = np.histogram(d, bins=np.arange(n_bins + 1) * bin_deg)
    empty = hist == 0
    if not empty.any() or empty.all():
        return [], hist
    idx = np.flatnonzero(empty)
    runs, cur = [], [idx[0]]
    for a, b in zip(idx[:-1], idx[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == n_bins - 1:
        runs[0] = runs[-1] + runs[0]      # a run straddling north
        runs.pop()
    edges = []
    for run in runs:
        span = len(run) * bin_deg
        edges.append(float((run[0] * bin_deg + span / 2.0) % 360.0))
    return sorted(edges), hist


def fit_planar(u, v, w, direction, edges: Sequence[float], admit
               ) -> Tuple[Dict[Tuple[float, float], Optional[PlanarFit]], Optional[PlanarFit]]:
    """Sectorwise and single-sector planar fits from the admitted period means.

    Parameters
    ----------
    u, v, w : numpy.ndarray
        Period-mean wind components.
    direction : numpy.ndarray
        Period-mean wind direction [deg].
    edges : sequence of float
        Sector edges, as passed to :func:`fit_sectors`.
    admit : numpy.ndarray of bool
        Periods entering the fits.

    Returns
    -------
    sectors : dict
        :func:`fit_sectors` result over *edges*.
    single : PlanarFit or None
        The all-direction fit.
    """
    sec = fit_sectors(u[admit], v[admit], w[admit], direction[admit], list(edges))
    single = fit_sectors(u[admit], v[admit], w[admit], direction[admit], [])[(0.0, 0.0)]
    return sec, single


def sector_fit(fits: Dict[Tuple[float, float], Optional[PlanarFit]],
               fallback: Optional[PlanarFit], key: Optional[Tuple[float, float]],
               use_sector: bool = True) -> Optional[PlanarFit]:
    """The planar fit applying to one period: its sector's, or *fallback*.

    *fallback* is returned when *use_sector* is false, when *key* is
    ``None``, or when the sector has no fit in *fits*.
    """
    if not use_sector:
        return fallback
    f = fits.get(key) if key is not None else None
    return f if f is not None else fallback


def bootstrap_tilt(u, v, w, n_boot: int = N_BOOTSTRAP, seed: int = 0
                   ) -> Tuple[float, float]:
    """Bootstrap standard deviation of a planar fit's pitch and roll [deg].

    Parameters
    ----------
    u, v, w : numpy.ndarray
        Period-mean wind components of the fit.
    n_boot : int
        Number of resamples with replacement.
    seed : int
        Seed of the ``numpy.random.default_rng`` generator.

    Returns
    -------
    sd_pitch, sd_roll : float
        Population standard deviations over the resamples that gave a fit;
        NaN for fewer than 4 points or fewer than 10 successful resamples.
    """
    n = len(u)
    if n < 4:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    pitch, roll = [], []
    for _ in range(n_boot):
        k = rng.integers(0, n, n)
        f = PlanarFit.fit(u[k], v[k], w[k])
        if f is not None:
            pitch.append(f.pitch_deg)
            roll.append(f.roll_deg)
    if len(pitch) < 10:
        return np.nan, np.nan
    return float(np.std(pitch)), float(np.std(roll))


# ── global-fit application ───────────────────────────────────────────────────

def _day_bounds(rec: PFRecord) -> Tuple[float, float]:
    from .pf_info import _date_to_datenum
    from datetime import date
    d0 = _date_to_datenum(date.fromisoformat(rec.date_start))
    d1 = _date_to_datenum(date.fromisoformat(rec.date_end))
    return float(d0), float(d1) + 1.0          # [start midnight, end day's next midnight)


def apply_global_fit(u, v, w, t, direction_hf, records: Sequence[PFRecord],
                     remove_offset: bool = True) -> np.ndarray:
    """Planar-fit winds from the stored records of one height: each record
    covers its date window and direction sector; samples outside every
    record stay NaN."""
    n = len(t)
    out = np.full((n, 3), np.nan)
    t = np.asarray(t, dtype=float)
    for rec in records:
        d0, d1 = _day_bounds(rec)
        mask = (t >= d0) & (t < d1) & sector_mask(direction_hf, rec.sector_lo, rec.sector_hi)
        if not mask.any():
            continue
        out[mask] = PlanarFit.from_record(rec).apply(u[mask], v[mask], w[mask], remove_offset)
    return out


# ── yaw rotation ─────────────────────────────────────────────────────────────

def yaw_rotate(wind_pf: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Rotate each period of *wind_pf* (``(n, 3)``, planar-fit frame) about
    the vertical so the period-mean v vanishes; periods with no data stay NaN."""
    means = block_mean(wind_pf[:, :2], bounds)
    out = np.full(wind_pf.shape, np.nan)
    for j in range(len(bounds) - 1):
        s0, s1 = int(bounds[j]), int(bounds[j + 1])
        if s0 >= s1:
            continue
        u_bar, v_bar = means[j]
        denom = np.sqrt(u_bar ** 2 + v_bar ** 2)
        if denom > 0:
            cos_g, sin_g = u_bar / denom, v_bar / denom
        else:
            cos_g, sin_g = 1.0, 0.0
        M = np.array([[cos_g, sin_g, 0.0],
                      [-sin_g, cos_g, 0.0],
                      [0.0, 0.0, 1.0]])
        out[s0:s1] = (M @ wind_pf[s0:s1].T).T
    return out


# ── a run's sonics ───────────────────────────────────────────────────────────

@dataclass
class SonicSeries:
    """One sonic's high-frequency winds and the per-period information the
    rotation needs."""
    height: float
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    direction_avg: np.ndarray              # per period [deg]
    good_periods: np.ndarray               # per period, True where the LPF may use the mean wind
    u_bar: np.ndarray                      # period means of u, v, w (the LPF regression data)
    v_bar: np.ndarray
    w_bar: np.ndarray


@dataclass
class RotationResult:
    heights: List[float]
    rotated: np.ndarray                    # (n, 3 * k): planar fit + yaw, per sonic u, v, w
    pf_only: np.ndarray                    # (n, 3 * k): planar fit only
    bounds: np.ndarray                     # period edges used for the yaw and the averages
    fits: Dict[float, Optional[PlanarFit]] = field(default_factory=dict)   # LPF fits by height
    records: Dict[float, List[PFRecord]] = field(default_factory=dict)      # GPF records by height
    skipped: Dict[float, str] = field(default_factory=dict)                 # height -> reason

    @property
    def headers(self) -> List[str]:
        return [f"{h}m:{c}" for h in self.heights for c in ("u", "v", "w")]

    def averaged(self, which: str = "rotated") -> np.ndarray:
        arr = self.rotated if which == "rotated" else self.pf_only
        return block_mean(arr, self.bounds)


def rotate_sonics(sonics: Sequence[SonicSeries], t: np.ndarray, avg_per: float,
                  pf_table: Optional[PFTable] = None) -> RotationResult:
    """Planar-fit (local from each sonic's good period means, or global from
    *pf_table*) and yaw-rotate every sonic. A height the global table does
    not cover, or a local fit without enough periods, is skipped with its
    reason in ``skipped`` and NaN columns."""
    n = len(t)
    N = n_periods(t, avg_per)
    bounds = period_bounds(n, N)
    k = len(sonics)
    res = RotationResult([s.height for s in sonics], np.full((n, 3 * k), np.nan),
                         np.full((n, 3 * k), np.nan), bounds)
    for i, s in enumerate(sonics):
        if pf_table is not None:
            recs = [r for r in pf_table.records if abs(r.height - s.height) <= 0.01]
            if not recs:
                res.skipped[s.height] = (f"PF info at {s.height} m does not exist. "
                                         "Check/re-run find_global_pf.")
                continue
            res.records[s.height] = recs
            dir_hf = np.full(n, np.nan)
            for j in range(N):
                dir_hf[bounds[j]:bounds[j + 1]] = s.direction_avg[j] if j < len(s.direction_avg) else np.nan
            wind_pf = apply_global_fit(s.u, s.v, s.w, t, dir_hf, recs)
        else:
            g = s.good_periods
            fit = PlanarFit.fit(s.u_bar[g], s.v_bar[g], s.w_bar[g])
            res.fits[s.height] = fit
            if fit is None:
                res.skipped[s.height] = "insufficient data for planar fit"
                log.info(f"  Sonic @ {s.height}m: insufficient data for planar fit, skipping.")
                continue
            log.info(f"  Sonic @ {s.height}m  pitch={fit.pitch_deg:.3g}°  roll={fit.roll_deg:.3g}°  "
                     f"b0={fit.b0:.3g} m/s")
            wind_pf = fit.apply(s.u, s.v, s.w)
        c0 = 3 * i
        res.pf_only[:, c0:c0 + 3] = wind_pf
        res.rotated[:, c0:c0 + 3] = yaw_rotate(wind_pf, bounds)
    return res
