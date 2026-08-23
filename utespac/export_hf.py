"""High-frequency products of a run -> CF netCDF (integration notes A.2).

One ``<Site>_hf_<PF>_<Det>_<date>.nc`` per processed date, written by
``save_data`` when ``saveRawConditionedData`` is on: dims ``time``
(datetime64) and ``height`` (sonic heights, ascending); the rotated winds,
sonic temperatures, scalars and pressure of ``Run.raw`` as ``(time,
height)`` variables; the per-window ancillaries of the averaged products
(u*, L, wind direction, QC flags, SSITC flags) on a ``record`` dimension;
the planar-fit coefficients in a ``planar_fit`` group; site and run
provenance in the global attributes.

``ec_coherent`` consumes these files; ``utespac`` never reads them back.
"""

import json
import logging
import os
import re
import warnings
from typing import Dict, List, Optional

import numpy as np

from .labeled import _require_netcdf4, _time_to_ms, parse_label, run_attrs, tables
from .model import TIME_HF, Run, raw_to_legacy, to_legacy_output

log = logging.getLogger("utespac")

FORMAT = "utespac-hf-1"
_TIME_UNITS = "milliseconds since 1970-01-01 00:00:00"

# netCDF name, raw key, units, long_name  -- (time, height) variables
_HF_VARS = [
    ("u", "uPF", "m s-1", "streamwise wind, planar fit + per-period yaw rotation"),
    ("v", "vPF", "m s-1", "crosswind, planar fit + per-period yaw rotation"),
    ("w", "wPF", "m s-1", "vertical wind, planar fit + per-period yaw rotation"),
    ("u_tilt", "u_tilt", "m s-1", "u after planar fit only (no yaw)"),
    ("v_tilt", "v_tilt", "m s-1", "v after planar fit only (no yaw)"),
    ("w_tilt", "w_tilt", "m s-1", "w after planar fit only (no yaw)"),
    ("Ts", "sonTs", "degC", "sonic temperature"),
    ("theta_v", "Theta_v_son", "degC", "sonic virtual potential temperature (dry-adiabatic offset to the reference height)"),
    ("WD", "WD", "degree", "wind direction from the unrotated sonic components and boom azimuth"),
    ("spd", "spd", "m s-1", "horizontal wind speed from the unrotated sonic components"),
    ("T_fw", "fwT", "degC", "fine-wire thermocouple temperature"),
    ("theta_fw", "fwTh", "degC", "fine-wire potential temperature"),
]
_SCALAR_VARS = [
    ("rhov", "rhov", "z_h2o", "g m-3", "water vapour density (not specific humidity)"),
    ("rhoCO2", "rhoCO2", "z_co2", "mg m-3", "CO2 density"),
]
_LPF_RE = re.compile(r"([\d.]+)m b0=([-+\d.eE]+|nan) b1=([-+\d.eE]+|nan) b2=([-+\d.eE]+|nan)")


def _sanitize(label: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_")
    return s or "x"


def _col_for_height(tab, z: float, pred) -> Optional[int]:
    """Index of the first column at height *z* whose height-stripped label
    (``parse_label(label)[1]``) satisfies *pred*."""
    for j, (h, lab) in enumerate(zip(tab.heights, tab.labels)):
        if h is not None and abs(h - z) < 0.01 and pred(parse_label(lab)[1]):
            return j
    return None


def _lpf_records(notes, date: str) -> List[Dict]:
    """Local planar-fit coefficients from the dataInfo strings the rotation stage writes."""
    recs = []
    for col in notes:
        for s in col:
            m = _LPF_RE.search(str(s))
            if m:
                recs.append({"height": float(m.group(1)), "date_start": date, "date_end": date,
                             "sector_lo": 0.0, "sector_hi": 0.0,
                             "b0": float(m.group(2)), "b1": float(m.group(3)), "b2": float(m.group(4))})
    return recs


def write_hf(run: Run, out_path, output: Optional[Dict] = None, attrs: Optional[Dict] = None,
             dtype: str = "f4", complevel: int = 4, include_primes: bool = False) -> str:
    """Write the HF netCDF of *run* (``run.raw`` must be set); returns *out_path*.

    Parameters
    ----------
    run : Run
        A run after the flux stage with the raw products kept.
    output : dict, optional
        ``to_legacy_output(run)`` with ``dataInfo`` (built here when None).
    attrs : dict, optional
        Extra or overriding global attributes.
    dtype : {"f4", "f8"}
        Storage type of the HF variables.
    include_primes : bool
        Also write the stored ``*Prime`` fluctuation series of the scalars.
    """
    if run.raw is None:
        raise ValueError("write_hf: the run carries no raw products (saveRawConditionedData off?)")
    nc = _require_netcdf4()
    info = run.site
    raw = raw_to_legacy(run.raw)
    if output is None:
        output = to_legacy_output(run)
        output["dataInfo"] = run.notes
    pf_mode = "GPF" if info.get("PF", {}).get("globalCalculation") == "global" else "LPF"
    date = str(info.get("date", "")).replace("_", "-")

    t_dn = np.asarray(raw["t"], dtype=float)
    t64 = run.raw[TIME_HF].values
    n = len(t64)
    z = np.asarray(raw["z"], dtype=float)
    order = np.argsort(z)
    z_sorted = z[order]
    dt_s = float(np.nanmedian(np.diff(t_dn[: min(n, 20000)]))) * 86400.0 if n > 1 else np.nan
    fs = round(1.0 / dt_s, 3) if np.isfinite(dt_s) and dt_s > 0 else np.nan

    out_path = str(out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)

    chunk_t = min(n, 2 ** 16)
    with nc.Dataset(out_path, "w", format="NETCDF4") as ds:
        # -- global attributes ------------------------------------------------------
        g = run_attrs(info, output)
        g.update({
            "Conventions": "CF-1.10",
            "utespac_format": FORMAT,
            "title": "UTESpac high-frequency series (rotated, despiked, gap-interpolated)",
            "sampling_frequency_hz_measured": fs,
            "detrend_upstream": g.get("detrend", ""),
            "rotation": "planar_fit+yaw (applied upstream by UTESpac sonic_rotation)",
            "despiking": "Vickers & Mahrt spikes interpolated in place upstream (condition_data)",
            "humidity_note": "rhov is a vapour density [g m-3], not a specific humidity",
            "dataInfo": json.dumps(run.notes, default=str),
        })
        g.setdefault("sampling_frequency_hz", fs)
        g.update(attrs or {})
        for k, v in g.items():
            ds.setncattr(k, v)

        # -- dims and coordinates -----------------------------------------------------
        ds.createDimension("time", n)
        ds.createDimension("height", len(z_sorted))
        tv = ds.createVariable("time", "i8", ("time",), zlib=True, complevel=complevel,
                               chunksizes=(chunk_t,))
        tv.units = _TIME_UNITS
        tv.calendar = "proleptic_gregorian"
        tv.standard_name = "time"
        tv.long_name = "sample time (Campbell convention: first sample one interval after midnight)"
        tv[:] = _time_to_ms(t64)
        hv = ds.createVariable("height", "f8", ("height",))
        hv.units = "m"
        hv.standard_name = "height"
        hv.long_name = "sonic anemometer height above ground"
        hv.positive = "up"
        hv[:] = z_sorted

        def _hf_var(name, arr, dims, units, long_name, chunks):
            v = ds.createVariable(name, dtype, dims, fill_value=np.nan, zlib=True,
                                  complevel=complevel, chunksizes=chunks)
            v.units = units
            v.long_name = long_name
            v[:] = arr
            return v

        for name, key, units, long_name in _HF_VARS:
            if key not in raw:
                continue
            arr = np.asarray(raw[key], dtype=float)
            if arr.ndim == 1:
                arr = arr[:, None]
            if arr.shape[1] != len(z):
                warnings.warn(f"{key}: {arr.shape[1]} columns for {len(z)} heights; skipped")
                continue
            _hf_var(name, arr[:, order], ("time", "height"), units, long_name,
                    (chunk_t, len(z_sorted)))

        for name, key, zkey, units, long_name in _SCALAR_VARS:
            if key not in raw:
                continue
            arr = np.asarray(raw[key], dtype=float)
            if arr.ndim == 1:
                arr = arr[:, None]
            if zkey in raw:
                zs = np.asarray(raw[zkey], dtype=float)
            elif arr.shape[1] == len(z):
                zs = z
            else:
                warnings.warn(f"{key}: heights unknown (no {zkey} on the raw products); written as NaN")
                zs = np.full(arr.shape[1], np.nan)
            dim = f"height_{name}"
            o = np.argsort(zs) if np.all(np.isfinite(zs)) else np.arange(len(zs))
            ds.createDimension(dim, len(zs))
            zv = ds.createVariable(dim, "f8", (dim,), fill_value=np.nan)
            zv.units = "m"
            zv.long_name = f"{name} sensor height above ground"
            zv[:] = zs[o]
            _hf_var(name, arr[:, o], ("time", dim), units, long_name, (chunk_t, len(zs)))
            if include_primes:
                for suffix, desc in (("Prime", "detrended fluctuation (pipeline detrend)"),
                                     ("extenalPrime", "fluctuation about the external reference")):
                    k2 = key + suffix
                    if k2 in raw:
                        a2 = np.asarray(raw[k2], dtype=float)
                        a2 = a2[:, None] if a2.ndim == 1 else a2
                        _hf_var(f"{name}_{suffix.lower()}", a2[:, o], ("time", dim), units,
                                f"{long_name}: {desc}", (chunk_t, len(zs)))

        if "P" in raw:
            P = np.asarray(raw["P"], dtype=float)
            if P.ndim == 2 and P.shape[1] >= 2:
                pt, pv = P[:, 0], P[:, 1]
                if len(pt) == n and np.allclose(pt, t_dn, rtol=0, atol=1e-9):
                    p_on_t = pv
                else:
                    ok = np.isfinite(pt) & np.isfinite(pv)
                    p_on_t = np.interp(t_dn, pt[ok], pv[ok], left=np.nan, right=np.nan) \
                        if ok.any() else np.full(n, np.nan)
                pvar = ds.createVariable("P", dtype, ("time",), fill_value=np.nan, zlib=True,
                                         complevel=complevel, chunksizes=(chunk_t,))
                pvar.units = "kPa"
                pvar.long_name = "air pressure (barometer nearest the reference height, interpolated to time)"
                pvar[:] = p_on_t

        # -- per-window ancillaries from the averaged products -----------------------------
        tabs = tables(output)
        ref = next((tb.time for tb in tabs.values() if tb.time_in_col0), None)
        if ref is not None:
            nr = len(ref)
            ds.createDimension("record", nr)
            rv = ds.createVariable("record", "i8", ("record",))
            rv.units = _TIME_UNITS
            rv.calendar = "proleptic_gregorian"
            rv.long_name = "end of averaging period"
            rv[:] = _time_to_ms(ref)

            def _rec_var(name, units, long_name, fill_fn, dtype_="f8"):
                data = np.full((nr, len(z_sorted)), np.nan)
                found = False
                for k, zz in enumerate(z_sorted):
                    col = fill_fn(zz)
                    if col is not None:
                        data[:, k] = col
                        found = True
                if not found:
                    return
                v = ds.createVariable(name, dtype_, ("record", "height"), fill_value=np.nan,
                                      zlib=True, complevel=complevel)
                v.units = units
                v.long_name = long_name
                v[:] = data

            tau = tabs.get("tau")
            if tau is not None:
                def _ustar(zz):
                    j = _col_for_height(tau, zz, lambda s: s.startswith("sqrt(uPF'wPF'"))
                    return np.sqrt(tau.values[:, j]) if j is not None else None
                _rec_var("ustar", "m s-1", "friction velocity from the rotated covariances", _ustar)
            Lt = tabs.get("L")
            if Lt is not None:
                _rec_var("L", "m", "Obukhov length", lambda zz: (
                    Lt.values[:, _col_for_height(Lt, zz, lambda s: s.startswith("L"))]
                    if _col_for_height(Lt, zz, lambda s: s.startswith("L")) is not None else None))
            sd = tabs.get("spdAndDir")
            if sd is not None:
                _rec_var("wdir", "degree", "mean wind direction", lambda zz: (
                    sd.values[:, _col_for_height(sd, zz, lambda s: s == "direction")]
                    if _col_for_height(sd, zz, lambda s: s == "direction") is not None else None))
                _rec_var("wind_flag", "1", "wind-direction (tower shadow) flag", lambda zz: (
                    sd.values[:, _col_for_height(sd, zz, lambda s: s.startswith("flag"))]
                    if _col_for_height(sd, zz, lambda s: s.startswith("flag")) is not None else None))
            for flag_name, suffix, desc in (("spike_flag", "SpikeFlag", "spike test failed"),
                                            ("nan_flag", "NanFlag", "too many missing samples")):
                flag_tabs = [tb for k, tb in tabs.items() if k.endswith(suffix)]

                def _flag(zz, flag_tabs=flag_tabs):
                    cols = []
                    for tb in flag_tabs:
                        idx = [j for j, h in enumerate(tb.heights)
                               if h is not None and abs(h - zz) < 0.01]
                        if idx:
                            cols.append(tb.values[:, idx].any(axis=1))
                    return np.any(cols, axis=0).astype(float) if cols else None
                _rec_var(flag_name, "1", f"any column at this height: {desc}", _flag)
            qc = tabs.get("fluxQC")
            if qc is not None:
                for kind in sorted({parse_label(s)[1] for s in qc.labels}):   # "TAU_SSITC_TEST", ...
                    _rec_var("ssitc_" + _sanitize(kind).lower(), "1",
                             f"SSITC quality flag {kind} (Foken 0/1/2 scale)", lambda zz, kind=kind: (
                                 qc.values[:, _col_for_height(qc, zz, lambda s: s == kind)]
                                 if _col_for_height(qc, zz, lambda s: s == kind) is not None else None))

        # -- planar-fit coefficients ----------------------------------------------------
        recs: List[Dict] = []
        if pf_mode == "GPF" and run.pf_table is not None:
            recs = [r.__dict__ for r in run.pf_table.records]
        elif pf_mode == "LPF":
            recs = _lpf_records(run.notes, date)
        if recs:
            grp = ds.createGroup("planar_fit")
            grp.description = ("coefficients of the planar fit applied upstream: w = b0 + b1 u + b2 v; "
                               "sector_lo == sector_hi means all wind directions")
            grp.createDimension("pf_record", len(recs))
            for key, kind in (("height", "f8"), ("sector_lo", "f8"), ("sector_hi", "f8"),
                              ("b0", "f8"), ("b1", "f8"), ("b2", "f8"),
                              ("date_start", str), ("date_end", str)):
                v = grp.createVariable(key, kind, ("pf_record",))
                vals = [r[key] for r in recs]
                v[:] = np.array(vals, dtype=object) if kind is str else np.array(vals, dtype=float)
            grp["height"].units = "m"
            grp["sector_lo"].units = grp["sector_hi"].units = "degree"
            grp["b0"].units = "m s-1"
    log.info("  Saved HF netCDF: %s", out_path)
    return out_path
