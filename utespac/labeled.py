"""Labeled view of the averaged UTESpac output, and its netCDF form.

An averaged output dict holds 2-D matrices addressed by position with the
column names in a parallel ``<field>Header`` list. :func:`tables` turns
every such field into a :class:`LabeledTable` (``datetime64`` time,
column labels, parsed heights); :func:`to_frames` gives the same as pandas
DataFrames. :func:`write_netcdf` / :func:`read_netcdf` persist the dict as
a self-describing netCDF -- one ``time`` dimension, one labeled column
dimension per field, provenance in the global attributes -- and read it
back into the legacy shape, so the pickle and the netCDF carry the same
content (migration step 3, labeled I/O boundary).
"""

import json
import logging
import os
import re
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from .campbell_date import datetime64_to_matlab_datenum, matlab_datenum_to_datetime64

log = logging.getLogger("utespac")

FORMAT = "utespac-averaged-1"
_TIME_LABELS = {"time", "timestamp"}
_LABEL_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*m\s*:?\s*(.*?)\s*$")
_FLAG_SUFFIXES = ("SpikeFlag", "NanFlag")
_NOT_TABLES = {"dataInfo", "tableNames", "warnings", "infoString", "z"}
_TIME_UNITS = "milliseconds since 1970-01-01 00:00:00"


def parse_label(label: str) -> Tuple[Optional[float], str]:
    """``"10.85m son:Ts'w'"`` -> ``(10.85, "son:Ts'w'")``; no height -> ``(None, label)``."""
    m = _LABEL_RE.match(str(label))
    if m and m.group(2):
        return float(m.group(1)), m.group(2)
    return None, str(label).strip()


def is_time_label(label) -> bool:
    return str(label).strip().lower() in _TIME_LABELS


def header_key(output: Dict, field: str) -> Optional[str]:
    """Key of the header list paired with *field*, if any."""
    for cand in (f"{field}Header", f"{field}header"):
        if cand in output:
            return cand
    return None


def header_rows(hdr) -> Tuple[Optional[List[str]], Optional[list]]:
    """``(labels, heights)`` from a flat ``[...]`` or nested ``[names, heights]`` header."""
    if not isinstance(hdr, list) or not hdr:
        return None, None
    if isinstance(hdr[0], list):
        labels = [str(h) for h in hdr[0]]
        heights = list(hdr[1]) if len(hdr) > 1 and isinstance(hdr[1], list) else None
        return labels, heights
    if all(isinstance(h, str) for h in hdr):
        return list(hdr), None
    return None, None


@dataclass
class LabeledTable:
    """One averaged field with its time axis and column labels."""
    name: str
    time: np.ndarray                 # datetime64[ms], shape (n,)
    labels: List[str]                # one per data column (the time column removed)
    values: np.ndarray               # (n, len(labels)); float or bool
    heights: List[Optional[float]]   # per column, parsed from the label or the header
    time_in_col0: bool               # the legacy matrix carried the timestamp in column 0
    time_label: Optional[str] = None
    header_key: Optional[str] = None   # "Hheader", "MySite_20HzHeader", ... (None: unlabeled)
    header_nested: bool = False        # header stored as [names, heights]
    header_of: Optional[str] = None    # flags: the table whose header names the columns

    @property
    def n(self) -> int:
        return int(self.values.shape[0])

    def column(self, label: str) -> np.ndarray:
        return self.values[:, self.labels.index(label)]


def reference_time(output: Dict) -> Optional[np.ndarray]:
    """Shared time axis (datetime64) from the first averaged table, else from
    the first labeled field whose leading column is a timestamp."""
    for tname in output.get("tableNames", []) or []:
        arr = output.get(tname)
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1]:
            return matlab_datenum_to_datetime64(arr[:, 0])
    for key, val in output.items():
        if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[1]:
            hk = header_key(output, key)
            labels, _ = header_rows(output.get(hk)) if hk else (None, None)
            if labels and is_time_label(labels[0]) and len(labels) == val.shape[1]:
                return matlab_datenum_to_datetime64(val[:, 0])
    return None


def tables(output: Dict, time: Optional[np.ndarray] = None) -> Dict[str, LabeledTable]:
    """Every 2-D numeric field of *output* as a :class:`LabeledTable`."""
    ref = time if time is not None else reference_time(output)
    out: Dict[str, LabeledTable] = {}
    for key, val in output.items():
        if key in _NOT_TABLES or key.endswith(("Header", "header")):
            continue
        if not isinstance(val, np.ndarray) or val.ndim != 2:
            continue
        ncol = val.shape[1]
        header_of = None
        hk = header_key(output, key)
        if hk is None:
            for suf in _FLAG_SUFFIXES:
                if key.endswith(suf):
                    header_of = key[: -len(suf)]
                    hk = header_key(output, header_of)
                    break
        labels, hdr_heights = header_rows(output.get(hk)) if hk else (None, None)
        if labels is not None and len(labels) != ncol:
            labels, hdr_heights = None, None       # width mismatch: treat as unlabeled
        if labels is None:
            hk, hdr_heights, header_of = None, None, None
            labels = [f"col{i}" for i in range(ncol)]
        nested = hk is not None and isinstance(output[hk][0], list)
        time_in_col0 = (header_of is None and hk is not None
                        and bool(labels) and is_time_label(labels[0]))
        if time_in_col0:
            t = matlab_datenum_to_datetime64(val[:, 0])
            data, labels = val[:, 1:], labels[1:]
            hdr_heights = hdr_heights[1:] if hdr_heights is not None else None
            time_label = str(output[hk][0][0] if nested else output[hk][0])
        else:
            if ref is None or len(ref) != val.shape[0]:
                warnings.warn(f"labeled.tables: no time axis for field {key!r}; skipped")
                continue
            t, data, time_label = ref, val, None
        if hdr_heights is not None and len(hdr_heights) == len(labels):
            heights = [None if h is None or (isinstance(h, float) and np.isnan(h)) else float(h)
                       for h in hdr_heights]
        else:
            heights = [parse_label(lab)[0] for lab in labels]
        out[key] = LabeledTable(key, t, list(labels), data, heights, time_in_col0,
                                time_label, hk, nested, header_of)
    return out


def to_frames(output: Dict, time: Optional[np.ndarray] = None) -> Dict:
    """Every table as a pandas DataFrame with a ``DatetimeIndex`` named ``time``."""
    import pandas as pd
    frames = {}
    for name, tab in tables(output, time).items():
        frames[name] = pd.DataFrame(tab.values, index=pd.DatetimeIndex(tab.time, name="time"),
                                    columns=tab.labels)
    return frames


# -- provenance ---------------------------------------------------------------

def git_commit(path=None) -> Optional[str]:
    """Short hash of the checked-out commit, or None outside a git tree."""
    here = path or os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here,
                              capture_output=True, text=True, timeout=10,
                              check=True).stdout.strip() or None
    except Exception:
        return None


def run_attrs(info: Dict, output: Optional[Dict] = None) -> Dict:
    """Global attributes for a product written from *info* (site facts applied)."""
    pf_mode = info.get("PF", {}).get("globalCalculation", "local")
    scan = info.get("tableScanFrequency") or []
    lat, lon = info.get("latitude"), info.get("longitude")
    if lat is None or lon is None:
        warnings.warn(f"{info.get('siteFolder')}: latitude/longitude missing from siteInfo; "
                      "written as NaN")
    attrs = {
        "site_id": info.get("siteFolder"),
        "latitude": float("nan") if lat is None else float(lat),
        "longitude": float("nan") if lon is None else float(lon),
        "elevation": info.get("siteElevation"),
        "canopy_height": info.get("canopyHeight"),
        "displacement_height": info.get("displacementHeight"),
        "sampling_frequency_hz": float(max(scan)) if scan else None,
        "flux_averaging_s": float(info.get("avgPer", 30)) * 60.0,
        "pf_type": "GPF" if pf_mode == "global" else "LPF",
        "rotation": "planar_fit+yaw (applied by UTESpac sonic_rotation)",
        "detrend": info.get("detrendingFormat", "linear"),
        "utespac_version": info.get("UTESpacVersion"),
        "git_commit": git_commit(),
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if output is not None:
        files = [row[0][len("file: "):] for row in output.get("dataInfo", [])
                 if row and isinstance(row[0], str) and row[0].startswith("file: ")]
        attrs["source_files"] = ", ".join(files)
    return {k: v for k, v in attrs.items() if v is not None}


# -- netCDF -------------------------------------------------------------------

def _require_netcdf4():
    try:
        import netCDF4
    except ImportError as exc:
        raise ImportError("netCDF output needs the netCDF4 package "
                          "(conda install netcdf4)") from exc
    return netCDF4


def _time_to_ms(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t).astype("datetime64[ms]")
    ms = t.astype("int64")
    ms[np.isnat(t)] = np.iinfo("int64").min
    return ms


def write_netcdf(output: Dict, path, attrs: Optional[Dict] = None,
                 time: Optional[np.ndarray] = None) -> str:
    """Write an averaged output dict as a labeled netCDF; returns *path*.

    Layout: dimension ``time`` with a CF coordinate; per field ``<f>`` a
    dimension ``<f>_column`` with the string coordinate ``<f>_column``
    (labels), ``<f>_height`` [m] and the data variable ``<f>(time,
    <f>_column)``; metadata lists as JSON global attributes.
    """
    nc = _require_netcdf4()
    tabs = tables(output, time)
    ref = time if time is not None else reference_time(output)
    if ref is None:
        if not tabs:
            raise ValueError("write_netcdf: output has no labeled tables")
        ref = next(iter(tabs.values())).time
    n = len(ref)
    path = str(path)
    if os.path.exists(path):
        os.remove(path)
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.setncattr("Conventions", "CF-1.10")
        ds.setncattr("utespac_format", FORMAT)
        for k, v in (attrs or {}).items():
            ds.setncattr(k, v)
        for key in ("tableNames", "dataInfo", "warnings", "infoString"):
            if key in output:
                ds.setncattr(key, json.dumps(output[key], default=str))
        ds.createDimension("time", n)
        tv = ds.createVariable("time", "i8", ("time",))
        tv.units = _TIME_UNITS
        tv.calendar = "proleptic_gregorian"
        tv.standard_name = "time"
        tv.long_name = "end of averaging period"
        tv[:] = _time_to_ms(ref)
        for name, tab in tabs.items():
            if tab.n != n:
                warnings.warn(f"write_netcdf: {name!r} has {tab.n} rows, time has {n}; skipped")
                continue
            if tab.time_in_col0 and not np.array_equal(tab.time, ref):
                warnings.warn(f"write_netcdf: {name!r} carries its own timestamps, which differ "
                              "from the file time axis; the file axis is what is written")
            cdim = f"{name}_column"
            ds.createDimension(cdim, len(tab.labels))
            cv = ds.createVariable(cdim, str, (cdim,))
            cv[:] = np.array(tab.labels, dtype=object)
            cv.long_name = f"column labels of {name}"
            hv = ds.createVariable(f"{name}_height", "f8", (cdim,), fill_value=np.nan)
            hv.units = "m"
            hv.long_name = f"measurement height of each {name} column"
            hv[:] = np.array([np.nan if h is None else h for h in tab.heights], dtype=float)
            if tab.values.dtype == bool:
                dv = ds.createVariable(name, "i1", ("time", cdim), zlib=True, complevel=4)
                dv[:] = tab.values.astype("i1")
                dv.flag_values = np.array([0, 1], dtype="i1")
                dv.flag_meanings = "clear set"
                dv.legacy_dtype = "bool"
            else:
                dv = ds.createVariable(name, "f8", ("time", cdim), fill_value=np.nan,
                                       zlib=True, complevel=4)
                dv[:] = tab.values.astype(float)
                dv.legacy_dtype = str(tab.values.dtype)
            dv.coordinates = f"time {cdim} {name}_height"
            dv.time_in_col0 = int(tab.time_in_col0)
            if tab.time_label is not None:
                dv.time_label = tab.time_label
            if tab.header_key is not None:
                dv.header_key = tab.header_key
                dv.header_nested = int(tab.header_nested)
            if tab.header_of is not None:
                dv.header_of = tab.header_of
            if tab.time_in_col0 and tab.header_nested:
                # the nested header's height row has a leading entry for the time column
                dv.time_height = json.dumps(output[tab.header_key][1][0])
    log.info("  Saved netCDF: %s", path)
    return path


def read_netcdf(path, frames: bool = False) -> Dict:
    """Read a netCDF written by :func:`write_netcdf`.

    Returns the legacy-shaped dict (matrices with datenum time columns and
    ``<field>Header`` lists), or with ``frames=True`` a dict of DataFrames.
    """
    nc = _require_netcdf4()
    output: Dict = {}
    with nc.Dataset(str(path), "r") as ds:
        fmt = ds.getncattr("utespac_format") if "utespac_format" in ds.ncattrs() else None
        if fmt != FORMAT:
            raise ValueError(f"{path}: not a {FORMAT} file (utespac_format={fmt!r})")
        for key in ("tableNames", "dataInfo", "warnings", "infoString"):
            if key in ds.ncattrs():
                output[key] = json.loads(ds.getncattr(key))
        ms = np.array(ds["time"][:]).astype("int64")
        t = ms.astype("datetime64[ms]")
        t[ms == np.iinfo("int64").min] = np.datetime64("NaT", "ms")
        t_datenum = datetime64_to_matlab_datenum(t)
        for name, var in ds.variables.items():
            if len(var.dimensions) != 2 or var.dimensions[0] != "time":
                continue
            cdim = var.dimensions[1]
            labels = [str(s) for s in np.array(ds[cdim][:], dtype=object)]
            heights = np.array(ds[f"{name}_height"][:], dtype=float)
            data = np.array(var[:])
            if getattr(var, "legacy_dtype", "") == "bool":
                data = data.astype(bool)
            else:
                data = np.ma.filled(np.ma.masked_invalid(data), np.nan).astype(float)
            time_in_col0 = bool(int(getattr(var, "time_in_col0", 0)))
            hk = getattr(var, "header_key", None)
            nested = bool(int(getattr(var, "header_nested", 0)))
            h_list = [None if np.isnan(h) else float(h) for h in heights]
            if time_in_col0:
                data = np.column_stack([t_datenum, data])
                labels = [getattr(var, "time_label", "time")] + labels
                h_list = [json.loads(getattr(var, "time_height", "null"))] + h_list
            output[name] = data
            if hk is not None and hk not in output:
                output[hk] = [labels, h_list] if nested else labels
    if frames:
        return to_frames(output)
    return output
