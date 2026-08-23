"""netCDF I/O: the ``utespac-hf-1`` reader, the window iterator, the analysis writer.

Windows follow the upstream averaging period: ``record`` is the END of each
period, so record *r* covers the samples ``(r - T, r]`` (``T`` =
``flux_averaging_s``). See library/writeups/ec_preprocess.md.
"""

import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import xarray as xr

log = logging.getLogger("ec_coherent")

HF_FORMAT = "utespac-hf-1"
OUT_FORMAT = "ec-coherent-1"

# (time, height) variables carried into every window when present
HF_VARIABLES = ("u", "v", "w", "Ts", "theta_v", "T_fw", "theta_fw", "rhov", "rhoCO2")
# per-record ancillaries (record, height) carried into every window when present
ANCILLARIES = ("ustar", "L", "wdir", "wind_flag", "spike_flag", "nan_flag")


@dataclass
class HFFile:
    """An opened ``utespac-hf-1`` file."""
    path: str
    ds: xr.Dataset
    fs: float
    window_s: float
    heights: np.ndarray
    site_id: str
    attrs: Dict[str, object]
    time: np.ndarray = field(repr=False)
    records: np.ndarray = field(repr=False)

    @property
    def n_per_window(self) -> int:
        return int(round(self.window_s * self.fs))

    def close(self):
        self.ds.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@dataclass
class Window:
    """One averaging period of one HF file: series on ``(time, height)`` plus ancillaries."""
    index: int
    record: np.datetime64
    start: np.datetime64
    fs: float
    heights: np.ndarray
    data: Dict[str, np.ndarray]            # name -> (n, n_heights), NaN where absent
    ancillary: Dict[str, np.ndarray]       # name -> (n_heights,)
    scalar_heights: Dict[str, np.ndarray]  # rhov/rhoCO2 sensor height mapped to each sonic height

    @property
    def n(self) -> int:
        return next(iter(self.data.values())).shape[0] if self.data else 0

    def series(self, name: str, ih: int = 0) -> np.ndarray:
        """1-D series of *name* at height index *ih* (NaN-filled if absent)."""
        if name not in self.data:
            return np.full(self.n, np.nan)
        return self.data[name][:, ih]

    def has(self, name: str, ih: int = 0) -> bool:
        return name in self.data and np.isfinite(self.data[name][:, ih]).any()


def open_hf(path) -> HFFile:
    """Open a ``utespac-hf-1`` file lazily (xarray, netCDF4 engine)."""
    ds = xr.open_dataset(path, engine="netcdf4", decode_times=True, chunks=None)
    fmt = ds.attrs.get("utespac_format", "")
    if fmt != HF_FORMAT:
        raise ValueError(f"{path}: utespac_format {fmt!r}, expected {HF_FORMAT!r}")
    fs = float(ds.attrs["sampling_frequency_hz"])
    window_s = float(ds.attrs["flux_averaging_s"])
    heights = np.asarray(ds["height"].values, dtype=float)
    time = ds["time"].values.astype("datetime64[ns]")
    records = ds["record"].values.astype("datetime64[ns]") if "record" in ds else \
        _records_from_time(time, window_s)
    return HFFile(path=str(path), ds=ds, fs=fs, window_s=window_s, heights=heights,
                  site_id=str(ds.attrs.get("site_id", "")), attrs=dict(ds.attrs),
                  time=time, records=records)


def _records_from_time(time: np.ndarray, window_s: float) -> np.ndarray:
    """Window ends on the ``window_s`` grid spanning *time* (fallback when no ``record``)."""
    step = np.timedelta64(int(round(window_s * 1e9)), "ns")
    t0 = time[0].astype("datetime64[D]").astype("datetime64[ns]")
    first = t0 + step
    out = []
    r = first
    while r <= time[-1] + np.timedelta64(1, "ns"):
        out.append(r)
        r = r + step
    return np.array(out, dtype="datetime64[ns]")


def _nearest_height_map(z_sonic: np.ndarray, z_other: np.ndarray, tol: float = 0.5):
    """Column index of *z_other* nearest to each sonic height (None beyond *tol* m)."""
    out = []
    for z in z_sonic:
        if z_other.size == 0 or not np.isfinite(z_other).any():
            out.append(None)
            continue
        j = int(np.nanargmin(np.abs(z_other - z)))
        out.append(j if abs(z_other[j] - z) <= tol or z_other.size == 1 else None)
    return out


def iter_windows(hf: HFFile, variables: Sequence[str] = HF_VARIABLES,
                 records: Optional[Sequence[int]] = None) -> Iterator[Window]:
    """Yield the averaging windows of *hf* in record order.

    Parameters
    ----------
    hf : HFFile
    variables : sequence of str
        HF variables to load (those absent from the file are skipped).
    records : sequence of int, optional
        Record indices to yield; default all.
    """
    ds = hf.ds
    n_win = hf.n_per_window
    z = hf.heights
    nh = len(z)
    present = [v for v in variables if v in ds]
    scalar_map: Dict[str, list] = {}
    scalar_z: Dict[str, np.ndarray] = {}
    for v in present:
        dims = ds[v].dims
        if len(dims) == 2 and dims[1] != "height":
            zz = np.asarray(ds[dims[1]].values, dtype=float)
            scalar_z[v] = zz
            scalar_map[v] = _nearest_height_map(z, zz)
    anc_present = [a for a in ANCILLARIES if a in ds] + \
        [a for a in ds.data_vars if a.startswith("ssitc_")]
    rec_index = None
    if "record" in ds:
        rec_index = {np.datetime64(r, "ns"): i for i, r in enumerate(ds["record"].values)}

    idx_all = range(len(hf.records)) if records is None else records
    for i in idx_all:
        r = hf.records[i]
        end = int(np.searchsorted(hf.time, r, side="right"))
        start = end - n_win
        if start < 0 or end > len(hf.time):
            log.debug("record %s: window outside the file (start %d, end %d); skipped", r, start, end)
            continue
        sl = slice(start, end)
        data: Dict[str, np.ndarray] = {}
        sh: Dict[str, np.ndarray] = {}
        for v in present:
            arr = np.asarray(ds[v].isel(time=sl).values, dtype=float)
            if arr.ndim == 1:
                arr = arr[:, None]
            if v in scalar_map:
                out = np.full((arr.shape[0], nh), np.nan)
                zs = np.full(nh, np.nan)
                for k, j in enumerate(scalar_map[v]):
                    if j is not None:
                        out[:, k] = arr[:, j]
                        zs[k] = scalar_z[v][j]
                data[v] = out
                sh[v] = zs
            else:
                data[v] = arr
        anc: Dict[str, np.ndarray] = {}
        if rec_index is not None and np.datetime64(r, "ns") in rec_index:
            ri = rec_index[np.datetime64(r, "ns")]
            for a in anc_present:
                vals = np.asarray(ds[a].isel(record=ri).values, dtype=float)
                anc[a] = vals if vals.ndim == 1 else np.full(nh, float(vals))
        yield Window(index=i, record=r, start=hf.time[start], fs=hf.fs, heights=z,
                     data=data, ancillary=anc, scalar_heights=sh)


# ----------------------------------------------------------------------------- output

def output_path(hf_path: str, suffix: str = "coherent") -> str:
    """``<Site>_hf_<...>.nc`` -> ``<Site>_<suffix>_<...>.nc`` in the same folder."""
    d, base = os.path.split(str(hf_path))
    return os.path.join(d, base.replace("_hf_", f"_{suffix}_", 1))


def _git_commit() -> str:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=here,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return ""


def init_output(path: str, hf: HFFile, cfg, extra: Optional[Dict] = None) -> str:
    """Create the analysis file with the shared coordinates and provenance globals."""
    attrs = {
        "Conventions": "CF-1.10",
        "ec_coherent_format": OUT_FORMAT,
        "title": "ec_coherent coherent-structure analysis of UTESpac high-frequency data",
        "source_file": os.path.basename(hf.path),
        "source_format": hf.attrs.get("utespac_format", ""),
        "source_git_commit": hf.attrs.get("git_commit", ""),
        "git_commit": _git_commit(),
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "site_id": hf.site_id,
        "sampling_frequency_hz": hf.fs,
        "flux_averaging_s": hf.window_s,
        "taylor_hypothesis": "lambda = U_mean/f, k = 2 pi f/U_mean (Taylor 1938 eq. 7; Stull 1988 eq. 1.4c)",
        "detrend_method": cfg.preprocess.detrend,
        "nan_max_frac": cfg.preprocess.nan_max_frac,
        "modules": ",".join(cfg.modules),
    }
    for k in ("latitude", "longitude", "elevation", "canopy_height", "displacement_height",
              "pf_type", "rotation", "detrend_upstream", "despiking", "humidity_note",
              "utespac_version"):
        if k in hf.attrs:
            attrs[k] = hf.attrs[k]
    attrs.update(extra or {})
    root = xr.Dataset(coords={"record": ("record", hf.records),
                              "height": ("height", hf.heights)})
    root["record"].attrs.update(long_name="end of averaging period")
    root["height"].attrs.update(units="m", long_name="sonic anemometer height above ground",
                                positive="up")
    root.attrs.update(attrs)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    root.to_netcdf(path, mode="w", engine="netcdf4",
                   encoding={"record": {"units": "milliseconds since 1970-01-01 00:00:00"}})
    return path


def write_group(path: str, name: str, ds: xr.Dataset, complevel: int = 4) -> None:
    """Append *ds* as group *name* of the analysis file at *path*."""
    enc = {}
    for v in ds.data_vars:
        if np.issubdtype(ds[v].dtype, np.floating):
            enc[v] = {"zlib": True, "complevel": complevel, "dtype": "f4"} \
                if ds[v].ndim >= 3 else {"zlib": True, "complevel": complevel}
    if "record" in ds.coords:
        enc["record"] = {"units": "milliseconds since 1970-01-01 00:00:00"}
    ds.to_netcdf(path, mode="a", group=name, engine="netcdf4", encoding=enc)
    log.info("  wrote /%s to %s", name, os.path.basename(path))


def read_group(path: str, name: str) -> xr.Dataset:
    """Load group *name* of an analysis file."""
    with xr.open_dataset(path, group=name, engine="netcdf4") as ds:
        return ds.load()
