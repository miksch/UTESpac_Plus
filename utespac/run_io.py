"""Persist and read the labeled run model as netCDF (``utespac-run-2``).

One file per processing date holds the averaged products of a
:class:`~utespac.model.Run` as netCDF groups — ``periods/<table>``,
``flags/<table>``, ``wind``, ``rotation`` (period means and fit records;
the high-frequency rotated winds live in the raw product), ``products/
<name>`` — plus a ``sensors`` group and the run facts, notes and warnings
as JSON global attributes. :func:`read_run` gives the Run back (without the
high-frequency input tables); :func:`load_products` concatenates the
products of a site's run files along ``time``.
"""

import glob
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import xarray as xr

from .model import (HEIGHT, TIME, TIME_HF, Run, Sensor, Sensors)

FORMAT = "utespac-run-2"
_ENCODING_TIME = {"units": "milliseconds since 1970-01-01 00:00:00", "calendar": "proleptic_gregorian"}


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict
        return _json_safe(asdict(obj))
    return str(obj)


def _sensors_dataset(sensors: Sensors) -> xr.Dataset:
    n = len(sensors)
    return xr.Dataset({
        "field": ("sensor", np.array([s.field for s in sensors], dtype=object)),
        "table": ("sensor", np.array([s.table for s in sensors], dtype=object)),
        "column": ("sensor", np.array([s.column for s in sensors], dtype=object)),
        "height": ("sensor", np.array([s.height for s in sensors], dtype=float)),
        "orientation": ("sensor", np.array([np.nan if s.orientation is None else s.orientation
                                            for s in sensors], dtype=float)),
        "manufacturer": ("sensor", np.array([-1 if s.manufacturer is None else s.manufacturer
                                             for s in sensors], dtype=int)),
    }, coords={"sensor": np.arange(n)})


def _sensors_from_dataset(ds: xr.Dataset) -> Sensors:
    out = Sensors()
    for i in range(ds.sizes["sensor"]):
        orient = float(ds["orientation"].values[i])
        manuf = int(ds["manufacturer"].values[i])
        out.append(Sensor(str(ds["field"].values[i]), str(ds["table"].values[i]),
                          str(ds["column"].values[i]), float(ds["height"].values[i]),
                          None if np.isnan(orient) else orient, None if manuf < 0 else manuf))
    return out


def _encoding(ds: xr.Dataset) -> Dict:
    enc = {}
    for name in ds.coords:
        if np.issubdtype(ds[name].dtype, np.datetime64):
            enc[name] = dict(_ENCODING_TIME)
    for name, var in ds.data_vars.items():
        if var.dtype.kind == "f" and var.ndim >= 1:
            enc[name] = {"zlib": True, "complevel": 4}
        elif var.dtype.kind == "b":
            enc[name] = {"dtype": "i1", "zlib": True, "complevel": 4}
    return enc


def _write_group(ds: xr.Dataset, path: str, group: str, first: bool) -> None:
    ds = ds.copy()
    for name in list(ds.coords) + list(ds.data_vars):
        if ds[name].dtype == object:
            ds[name] = ds[name].astype(str)
    ds.to_netcdf(path, group=group, mode="w" if first else "a", engine="netcdf4",
                 format="NETCDF4", encoding=_encoding(ds))


def write_run(run: Run, path, attrs: Optional[Dict] = None) -> str:
    """Write the averaged part of *run* as a ``utespac-run-2`` netCDF; returns *path*."""
    path = str(path)
    if os.path.exists(path):
        os.remove(path)
    first = True
    root = xr.Dataset(attrs={
        "Conventions": "CF-1.10",
        "utespac_format": FORMAT,
        **{k: v for k, v in (attrs or {}).items() if v is not None},
        "run_facts": json.dumps(_json_safe(run.site)),
        "table_names": json.dumps(list(run.table_names)),
        "headers": json.dumps(_json_safe(run.headers)),
        "notes": json.dumps(_json_safe(run.notes)),
        "warnings": json.dumps(list(run.warnings)),
    })
    root.to_netcdf(path, mode="w", engine="netcdf4", format="NETCDF4")
    first = False
    _write_group(_sensors_dataset(run.sensors), path, "sensors", first)
    for name, ds in run.periods.items():
        _write_group(ds, path, f"periods/{name}", first)
    for name, ds in run.flags.items():
        _write_group(ds, path, f"flags/{name}", first)
    if run.wind is not None:
        _write_group(run.wind, path, "wind", first)
    if run.rotation is not None:
        rot = run.rotation.drop_vars([v for v in ("rotated", "pf") if v in run.rotation])
        rot = rot.drop_dims([TIME_HF], errors="ignore")
        _write_group(rot, path, "rotation", first)
    for name, ds in run.products.items():
        _write_group(ds, path, f"products/{name}", first)
    if run.pf_table is not None:
        root_pf = xr.Dataset(attrs={"pf_table": json.dumps(run.pf_table.to_dict())})
        root_pf.to_netcdf(path, group="planar_fit", mode="a", engine="netcdf4", format="NETCDF4")
    return path


def _groups(path: str) -> List[str]:
    import netCDF4
    out = []

    def walk(g, prefix):
        for name, sub in g.groups.items():
            full = f"{prefix}/{name}" if prefix else name
            out.append(full)
            walk(sub, full)
    with netCDF4.Dataset(path, "r") as ds:
        walk(ds, "")
    return out


def _open(path: str, group: str) -> xr.Dataset:
    with xr.open_dataset(path, group=group, engine="netcdf4", decode_timedelta=False) as ds:
        return ds.load()


def read_run(path) -> Run:
    """A :class:`Run` from a ``utespac-run-2`` file (no high-frequency tables)."""
    path = str(path)
    with xr.open_dataset(path, engine="netcdf4") as root:
        attrs = dict(root.attrs)
    if attrs.get("utespac_format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file (utespac_format={attrs.get('utespac_format')!r})")
    groups = _groups(path)
    headers = {k: (list(v[0]), [None if h is None else h for h in v[1]])
               for k, v in json.loads(attrs.get("headers", "{}")).items()}
    run = Run(site=json.loads(attrs.get("run_facts", "{}")),
              sensors=_sensors_from_dataset(_open(path, "sensors")) if "sensors" in groups else Sensors(),
              table_names=json.loads(attrs.get("table_names", "[]")), headers=headers,
              notes=json.loads(attrs.get("notes", "[]")), warnings=json.loads(attrs.get("warnings", "[]")))
    for g in groups:
        if g.startswith("periods/"):
            run.periods[g.split("/", 1)[1]] = _open(path, g)
        elif g.startswith("flags/"):
            ds = _open(path, g)
            for v in ("spike", "nan"):
                if v in ds:
                    ds[v] = ds[v].astype(bool)
            run.flags[g.split("/", 1)[1]] = ds
        elif g == "wind":
            run.wind = _open(path, g)
        elif g == "rotation":
            run.rotation = _open(path, g)
        elif g.startswith("products/"):
            run.products[g.split("/", 1)[1]] = _open(path, g)
        elif g == "planar_fit":
            from .pf_info import PFTable
            with xr.open_dataset(path, group=g, engine="netcdf4") as ds:
                run.pf_table = PFTable.from_dict(json.loads(ds.attrs["pf_table"]))
    run.attrs = {k: v for k, v in attrs.items()
                 if k not in ("run_facts", "table_names", "headers", "notes", "warnings")}
    return run


def run_files(root, site: str, avg_per: Optional[int] = None, qualifier: Optional[str] = None
              ) -> List[str]:
    """The site's ``utespac-run-2`` files, sorted (same name pattern as the pickles)."""
    from .site_config import resolve_site_dir
    out_dir = os.path.join(root, resolve_site_dir(root, site), "output")
    parts = ["*"]
    if avg_per is not None:
        parts.append(f"_{avg_per}")
    parts.append("*")
    if qualifier is not None:
        parts.append(f"{qualifier}*")
    parts.append(".nc")
    files = []
    for f in sorted(glob.glob(os.path.join(out_dir, "".join(parts)))):
        try:
            with xr.open_dataset(f, engine="netcdf4") as ds:
                if ds.attrs.get("utespac_format") == FORMAT:
                    files.append(f)
        except Exception:
            continue
    return files


def load_products(root, site: str, avg_per: Optional[int] = None, qualifier: Optional[str] = None,
                  names: Optional[Sequence[str]] = None) -> Dict[str, xr.Dataset]:
    """The products of every run file of *site* concatenated along ``time``
    (``join="outer"``: a date missing a variable or a height contributes NaN)."""
    files = run_files(root, site, avg_per, qualifier)
    if not files:
        raise FileNotFoundError(f"no {FORMAT} files for {site!r} under {root}")
    per_name: Dict[str, List[xr.Dataset]] = {}
    for f in files:
        for g in _groups(f):
            if g.startswith("products/"):
                name = g.split("/", 1)[1]
                if names is not None and name not in names:
                    continue
                per_name.setdefault(name, []).append(_open(f, g))
    return {name: xr.concat(parts, dim=TIME, join="outer", combine_attrs="override")
            for name, parts in per_name.items()}
