"""Persist and read the labeled run model as netCDF (``utespac-run-3``).

One file per processing date holds the averaged products of a
:class:`~utespac.model.Run` as netCDF groups — ``periods/<table>``,
``flags/<table>``, ``wind``, ``rotation`` (period means and fit records;
the high-frequency rotated winds live in the raw product), ``products/
<group>`` — plus a ``sensors`` group and the run facts, notes and warnings
as JSON global attributes. :func:`read_run` gives the Run back (without the
high-frequency input tables); :func:`load_products` concatenates the
products of a site's run files along ``time``.

``utespac-run-2`` files (products under the legacy table and column names)
are read as well: :func:`products_from_v2` renames their groups and
variables through :mod:`utespac.names`. Only ``utespac-run-3`` is written.
"""

import glob
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import xarray as xr

from .model import (HEIGHT, TIME, TIME_HF, Run, Sensor, Sensors, set_product_attrs)
from .names import GROUPS, legacy_to_new

FORMAT = "utespac-run-3"
FORMAT_V2 = "utespac-run-2"                  # read-only, through products_from_v2
READABLE_FORMATS = (FORMAT, FORMAT_V2)
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
        "units": ("sensor", np.array(["" if s.units is None else s.units
                                      for s in sensors], dtype=object)),
    }, coords={"sensor": np.arange(n)})


def _sensors_from_dataset(ds: xr.Dataset) -> Sensors:
    out = Sensors()
    for i in range(ds.sizes["sensor"]):
        orient = float(ds["orientation"].values[i])
        manuf = int(ds["manufacturer"].values[i])
        units = str(ds["units"].values[i]) if "units" in ds else ""
        out.append(Sensor(str(ds["field"].values[i]), str(ds["table"].values[i]),
                          str(ds["column"].values[i]), float(ds["height"].values[i]),
                          None if np.isnan(orient) else orient, None if manuf < 0 else manuf,
                          units or None))
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
        elif np.issubdtype(ds[name].dtype, np.datetime64):
            # coarser-than-ns stamps encode to NaT under the millisecond units
            ds[name] = ds[name].astype("datetime64[ns]")
    ds.to_netcdf(path, group=group, mode="w" if first else "a", engine="netcdf4",
                 format="NETCDF4", encoding=_encoding(ds))


def write_run(run: Run, path, attrs: Optional[Dict] = None) -> str:
    """Write the averaged part of *run* as a ``utespac-run-3`` netCDF; returns *path*."""
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
        "header_units": json.dumps(_json_safe(run.header_units)),
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


def _walk_groups(root) -> List[str]:
    out = []

    def walk(g, prefix):
        for name, sub in g.groups.items():
            full = f"{prefix}/{name}" if prefix else name
            out.append(full)
            walk(sub, full)
    walk(root, "")
    return out


def _open_file(path: str):
    import netCDF4
    return netCDF4.Dataset(path, "r")


def _load_group(nc, group: str) -> xr.Dataset:
    """One group of an open ``netCDF4.Dataset`` *nc* as a loaded Dataset
    (a store on the shared handle -- the file is opened once per read)."""
    store = xr.backends.NetCDF4DataStore(nc, group=group)
    return xr.open_dataset(store, decode_timedelta=False).load()


def _format_of(path: str) -> Optional[str]:
    """``utespac_format`` of a netCDF file, None if absent or unreadable."""
    import netCDF4
    try:
        with netCDF4.Dataset(path, "r") as nc:
            return nc.getncattr("utespac_format") if "utespac_format" in nc.ncattrs() else None
    except Exception:
        return None


def products_from_v2(products: Dict[str, xr.Dataset]) -> Dict[str, xr.Dataset]:
    """``utespac-run-2`` product groups under the current names.

    Group names come from :data:`utespac.names.GROUPS` and variable names
    from :func:`utespac.names.legacy_to_new`, so the third moment of the
    sonic temperature lands in ``transport``; variables with no alias are
    kept under their own name in the renamed group.
    """
    out: Dict[str, xr.Dataset] = {}

    def target(group: str, like: xr.Dataset) -> xr.Dataset:
        if group not in out:
            out[group] = xr.Dataset(coords=like.coords,
                                    attrs={**like.attrs, "legacy_header_key": f"{group}Header"})
        return out[group]

    for table, ds in products.items():
        target(GROUPS.get(table, table), ds)
    for table, ds in products.items():
        for key, values in ds.data_vars.items():
            try:
                group, name = legacy_to_new(table, str(key))
            except KeyError:
                group, name = GROUPS.get(table, table), str(key)
            grp = target(group, ds)
            grp[name] = values.rename(name)
            grp[name].attrs.clear()
            try:
                set_product_attrs(grp[name], group, name)
            except KeyError:
                pass
    return out


def read_run(path) -> Run:
    """A :class:`Run` from a ``utespac-run-3`` (or ``utespac-run-2``) file, without
    the high-frequency tables."""
    path = str(path)
    with _open_file(path) as nc:
        attrs = {k: nc.getncattr(k) for k in nc.ncattrs()}
        fmt = attrs.get("utespac_format")
        if fmt not in READABLE_FORMATS:
            raise ValueError(f"{path}: not a {FORMAT} file (utespac_format={fmt!r})")
        groups = _walk_groups(nc)
        headers = {k: (list(v[0]), [None if h is None else h for h in v[1]])
                   for k, v in json.loads(attrs.get("headers", "{}")).items()}
        run = Run(site=json.loads(attrs.get("run_facts", "{}")),
                  sensors=(_sensors_from_dataset(_load_group(nc, "sensors"))
                           if "sensors" in groups else Sensors()),
                  table_names=json.loads(attrs.get("table_names", "[]")), headers=headers,
                  header_units={k: list(v) for k, v in
                                json.loads(attrs.get("header_units", "{}")).items()},
                  notes=json.loads(attrs.get("notes", "[]")), warnings=json.loads(attrs.get("warnings", "[]")))
        for g in groups:
            if g.startswith("periods/"):
                run.periods[g.split("/", 1)[1]] = _load_group(nc, g)
            elif g.startswith("flags/"):
                ds = _load_group(nc, g)
                for v in ("spike", "nan"):
                    if v in ds:
                        ds[v] = ds[v].astype(bool)
                run.flags[g.split("/", 1)[1]] = ds
            elif g == "wind":
                run.wind = _load_group(nc, g)
            elif g == "rotation":
                run.rotation = _load_group(nc, g)
            elif g.startswith("products/"):
                run.products[g.split("/", 1)[1]] = _load_group(nc, g)
            elif g == "planar_fit":
                from .pf_info import PFTable
                run.pf_table = PFTable.from_dict(json.loads(nc[g].getncattr("pf_table")))
        if fmt == FORMAT_V2:
            run.products = products_from_v2(run.products)
    run.attrs = {k: v for k, v in attrs.items()
                 if k not in ("run_facts", "table_names", "headers", "header_units",
                              "notes", "warnings")}
    return run


def read_run_legacy(path) -> Dict:
    """A run file as the legacy output dict (``to_legacy_output`` plus ``dataInfo``
    and ``tableNames``) -- what the pickle held; for readers not yet on the Run."""
    from .model import to_legacy_output
    run = read_run(path)
    out = to_legacy_output(run)
    out["dataInfo"] = run.notes
    out["tableNames"] = list(run.table_names)
    return out


def run_files(root, site: str, avg_per: Optional[int] = None, qualifier: Optional[str] = None
              ) -> List[str]:
    """The site's run files (``utespac-run-3`` or ``-2``), sorted (the pickles' name
    pattern)."""
    from .site_config import resolve_site_dir
    out_dir = os.path.join(root, resolve_site_dir(root, site), "output")
    parts = ["*"]
    if avg_per is not None:
        parts.append(f"_{avg_per}")
    parts.append("*")
    if qualifier is not None:
        parts.append(f"{qualifier}*")
    parts.append(".nc")
    return [f for f in sorted(glob.glob(os.path.join(out_dir, "".join(parts))))
            if _format_of(f) in READABLE_FORMATS]


def load_products(root, site: str, avg_per: Optional[int] = None, qualifier: Optional[str] = None,
                  names: Optional[Sequence[str]] = None) -> Dict[str, xr.Dataset]:
    """The products of every run file of *site* concatenated along ``time``
    (``join="outer"``: a date missing a variable or a height contributes NaN)."""
    files = run_files(root, site, avg_per, qualifier)
    if not files:
        raise FileNotFoundError(f"no {FORMAT} files for {site!r} under {root}")
    per_name: Dict[str, List[xr.Dataset]] = {}
    for f in files:
        with _open_file(f) as nc:
            fmt = nc.getncattr("utespac_format") if "utespac_format" in nc.ncattrs() else None
            found = {g.split("/", 1)[1]: _load_group(nc, g)
                     for g in _walk_groups(nc) if g.startswith("products/")}
        if fmt == FORMAT_V2:
            found = products_from_v2(found)
        for name, ds in found.items():
            if names is not None and name not in names:
                continue
            per_name.setdefault(name, []).append(ds)
    return {name: xr.concat(parts, dim=TIME, join="outer", combine_attrs="override")
            for name, parts in per_name.items()}
