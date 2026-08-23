"""The labeled run model: sensors as records, data as xarray Datasets.

Design: tasks/active/2026-08-22_labeled-run-model.md. A :class:`Run`
carries one processing date of one site — the high-frequency logger
tables, the per-period flags and means, the wind statistics, the rotated
winds, the flux products and the raw high-frequency products — as
``xarray.Dataset`` objects with ``datetime64`` time, plus the
:class:`Sensors` (what each header column is) and the run facts. The
converters at the bottom translate each piece from and to the legacy
structures (``data`` arrays with a datenum column, ``sensor_info`` arrays,
the ``output`` dict of matrices and ``<name>Header`` lists, the ``raw``
dict), so the stages can move onto the model one at a time while the
others still speak the legacy form.

Kernels keep taking numpy; the model is the currency between stages and
the shape of what is persisted.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from .campbell_date import datetime64_to_matlab_datenum, matlab_datenum_to_datetime64
from .flux.tables import TABLE_SPECS, height_label

TIME = "time"          # period-end stamps (averaged products)
TIME_HF = "time_hf"    # high-frequency stamps
HEIGHT = "height"
COMPONENT = "component"
COLUMN = "column"


# ── sensors ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Sensor:
    """One header column that matched a sensor template."""
    field: str                      # template key: "u", "Tson", "irgaH2O", ...
    table: str                      # logger table name
    column: str                     # header label, e.g. "Ux_10.85"
    height: float                   # [m]
    orientation: Optional[float] = None     # sonic head bearing (field "u")
    manufacturer: Optional[int] = None      # 0 RMYoung, 1 Campbell, 2 Gill (field "u")


class Sensors(list):
    """The sensors of a run, with lookups by field and height."""

    def by_field(self, field_name: str) -> List[Sensor]:
        return [s for s in self if s.field == field_name]

    def at(self, field_name: str, height: float, tol: float = 0.01) -> Optional[Sensor]:
        for s in self:
            if s.field == field_name and abs(s.height - height) <= tol:
                return s
        return None

    def heights(self, field_name: str) -> List[float]:
        return [s.height for s in self.by_field(field_name)]

    def has(self, field_name: str) -> bool:
        return any(s.field == field_name for s in self)

    # -- legacy --
    @classmethod
    def from_legacy(cls, sensor_info: Dict[str, np.ndarray], headers: Sequence,
                    table_names: Sequence[str]) -> "Sensors":
        """From the ``find_instruments`` arrays ``[table, col, height(, bearing, manufacturer)]``."""
        out = cls()
        for field_name, arr in sensor_info.items():
            for row in np.atleast_2d(arr):
                tbl_i, col_i = int(row[0]), int(row[1])
                label = str(headers[tbl_i][0][col_i])
                orient = float(row[3]) if len(row) > 3 else None
                manuf = int(row[4]) if len(row) > 4 else None
                out.append(Sensor(field_name, table_names[tbl_i], label, float(row[2]), orient, manuf))
        return out

    def to_legacy(self, headers: Sequence, table_names: Sequence[str]) -> Dict[str, np.ndarray]:
        """The ``sensor_info`` dict the legacy stages read (discovery order kept)."""
        out: Dict[str, List[List[float]]] = {}
        for s in self:
            tbl_i = list(table_names).index(s.table)
            col_i = list(headers[tbl_i][0]).index(s.column)
            row = [tbl_i, col_i, s.height]
            if s.field == "u":
                row += [0.0 if s.orientation is None else float(s.orientation),
                        1.0 if s.manufacturer is None else float(s.manufacturer)]
            out.setdefault(s.field, []).append(row)
        return {k: np.array(v, dtype=float) for k, v in out.items()}


# ── the run ──────────────────────────────────────────────────────────────────

@dataclass
class Run:
    site: Dict[str, Any]                       # resolved run/site facts (the legacy info dict for now)
    sensors: Sensors
    table_names: List[str]
    headers: Dict[str, Tuple[List[str], List[Optional[float]]]]   # name -> (labels, heights)
    tables: Dict[str, xr.Dataset] = field(default_factory=dict)     # high-frequency, dim time_hf
    flags: Dict[str, xr.Dataset] = field(default_factory=dict)      # per period, dims (time, column)
    periods: Dict[str, xr.Dataset] = field(default_factory=dict)    # per period, dim time
    wind: Optional[xr.Dataset] = None                                # (time, height)
    rotation: Optional[xr.Dataset] = None                            # (time_hf, height, component) + means
    products: Dict[str, xr.Dataset] = field(default_factory=dict)   # (time, height) per flux table
    raw: Optional[xr.Dataset] = None                                 # (time_hf, height) products
    notes: List[List[str]] = field(default_factory=list)             # legacy dataInfo
    pf_table: Any = None                                             # PFTable or None
    warnings: List[str] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)              # provenance of a read run file

    @property
    def time_hf(self) -> np.ndarray:
        """High-frequency time axis of the first sonic table."""
        u = self.sensors.by_field("u")
        name = u[0].table if u else self.table_names[0]
        return self.tables[name][TIME_HF].values

    def sonic_heights(self) -> List[float]:
        return self.sensors.heights("u")

    # -- kernel-side accessors (numpy views; time as datenum for the kernels) --
    def hf(self, sensor: "Sensor") -> np.ndarray:
        """High-frequency series of a sensor."""
        return self.tables[sensor.table][sensor.column].values

    def hf_time(self, table: str) -> np.ndarray:
        """Datenum time axis of a table (cached per table)."""
        cache = self.__dict__.setdefault("_datenum_cache", {})
        if table not in cache:
            cache[table] = to_datenum(self.tables[table][TIME_HF].values)
        return cache[table]

    @property
    def time_hf_datenum(self) -> np.ndarray:
        u = self.sensors.by_field("u")
        return self.hf_time(u[0].table if u else self.table_names[0])

    def period_mean(self, sensor: "Sensor") -> Optional[np.ndarray]:
        """Per-period mean of a sensor's column, None when the table was not averaged."""
        per = self.periods.get(sensor.table)
        if per is None or sensor.column not in per:
            return None
        return per[sensor.column].values

    def period_time(self) -> np.ndarray:
        """Period-end stamps (datetime64) of the first averaged table."""
        for name in self.table_names:
            if name in self.periods:
                return self.periods[name][TIME].values
        raise ValueError("no averaged table on the run")

    def flag(self, sensor: "Sensor", which: str, n: int) -> np.ndarray:
        """Per-period ``spike``/``nan`` flag of a sensor's column; zeros when absent."""
        ds = self.flags.get(sensor.table)
        if ds is None or which not in ds or sensor.column not in list(ds[COLUMN].values):
            return np.zeros(n, dtype=bool)
        j = list(ds[COLUMN].values).index(sensor.column)
        return ds[which].values[:, j].astype(bool)


# ── time helpers ─────────────────────────────────────────────────────────────

def to_datetime64(datenum) -> np.ndarray:
    return matlab_datenum_to_datetime64(datenum).astype("datetime64[ns]")


def to_datenum(t64) -> np.ndarray:
    return datetime64_to_matlab_datenum(np.asarray(t64).astype("datetime64[ns]"))


# ── high-frequency tables ────────────────────────────────────────────────────

def tables_from_legacy(data: Sequence[Optional[np.ndarray]], headers: Sequence,
                       table_names: Sequence[str], scan_hz: Optional[Sequence[float]] = None
                       ) -> Dict[str, xr.Dataset]:
    """``data[i]`` (datenum column 0 + one column per header label) -> Dataset per table."""
    out = {}
    for i, tbl in enumerate(data):
        if tbl is None or tbl.size == 0:
            continue
        labels = list(headers[i][0])
        t = to_datetime64(tbl[:, 0])
        ds = xr.Dataset({lab: (TIME_HF, tbl[:, j]) for j, lab in enumerate(labels) if j > 0},
                        coords={TIME_HF: t})
        ds.attrs["time_label"] = labels[0]
        if scan_hz is not None and i < len(scan_hz):
            ds.attrs["scan_hz"] = float(scan_hz[i])
        out[table_names[i]] = ds
    return out


def tables_to_legacy(tables: Dict[str, xr.Dataset], headers: Dict, table_names: Sequence[str]
                     ) -> List[Optional[np.ndarray]]:
    """Back to the list of ``[datenum, columns...]`` arrays (None where absent)."""
    out: List[Optional[np.ndarray]] = []
    for name in table_names:
        ds = tables.get(name)
        if ds is None:
            out.append(None)
            continue
        labels = headers[name][0]
        cols = [to_datenum(ds[TIME_HF].values)] + [ds[lab].values for lab in labels[1:]]
        out.append(np.column_stack(cols))
    return out


# ── flags and period means ───────────────────────────────────────────────────

def flags_from_legacy(output: Dict, headers: Dict, table_names: Sequence[str],
                      time: np.ndarray) -> Dict[str, xr.Dataset]:
    """``<table>SpikeFlag``/``<table>NanFlag`` (periods × table columns) -> Dataset per table."""
    out = {}
    for name in table_names:
        sf, nf = output.get(f"{name}SpikeFlag"), output.get(f"{name}NanFlag")
        if sf is None and nf is None:
            continue
        labels = list(headers[name][0])
        ref = sf if sf is not None else nf
        ds = xr.Dataset(coords={TIME: time, COLUMN: labels[: ref.shape[1]]})
        if sf is not None:
            ds["spike"] = ((TIME, COLUMN), np.asarray(sf, dtype=bool))
        if nf is not None:
            ds["nan"] = ((TIME, COLUMN), np.asarray(nf, dtype=bool))
        out[name] = ds
    return out


def flags_to_legacy(flags: Dict[str, xr.Dataset]) -> Dict[str, np.ndarray]:
    out = {}
    for name, ds in flags.items():
        if "spike" in ds:
            out[f"{name}SpikeFlag"] = ds["spike"].values.astype(bool)
        if "nan" in ds:
            out[f"{name}NanFlag"] = ds["nan"].values.astype(bool)
    return out


def periods_from_legacy(output: Dict, headers: Dict, table_names: Sequence[str]
                        ) -> Dict[str, xr.Dataset]:
    """``output[<table>]`` (period-end datenum col 0 + columns) -> Dataset per table."""
    out = {}
    for name in table_names:
        mat = output.get(name)
        if not isinstance(mat, np.ndarray) or mat.ndim != 2:
            continue
        labels = list(headers[name][0])
        t = to_datetime64(mat[:, 0])
        ds = xr.Dataset({lab: (TIME, mat[:, j]) for j, lab in enumerate(labels) if j > 0 and j < mat.shape[1]},
                        coords={TIME: t})
        ds.attrs["time_label"] = labels[0]
        out[name] = ds
    return out


def periods_to_legacy(periods: Dict[str, xr.Dataset], headers: Dict) -> Dict[str, Any]:
    out = {}
    for name, ds in periods.items():
        labels = headers[name][0]
        cols = [to_datenum(ds[TIME].values)] + [ds[lab].values for lab in labels[1:] if lab in ds]
        out[name] = np.column_stack(cols)
        out[f"{name}Header"] = [list(labels), list(headers[name][1])]
    return out


def reference_time(output: Dict, table_names: Sequence[str]) -> np.ndarray:
    """Period-end stamps of the first averaged table, as datetime64."""
    for name in table_names:
        mat = output.get(name)
        if isinstance(mat, np.ndarray) and mat.ndim == 2:
            return to_datetime64(mat[:, 0])
    raise ValueError("no averaged table in output")


# ── wind ─────────────────────────────────────────────────────────────────────

_FLAG_RE = re.compile(r"^(?P<h>[\d.]+)m flag (?P<lo>[-\d.eE+]+)<dir<(?P<hi>[-\d.eE+]+)$")


def wind_from_legacy(output: Dict, heights: Sequence[float]) -> Optional[xr.Dataset]:
    """``spdAndDir`` + header -> Dataset (time, height): direction, speed, shadow_flag,
    sector_min, sector_max."""
    mat, hdr = output.get("spdAndDir"), output.get("spdAndDirHeader")
    if mat is None or hdr is None:
        return None
    t = to_datetime64(mat[:, 0])
    n, k = mat.shape[0], len(heights)
    d = np.full((n, k), np.nan)
    s = np.full((n, k), np.nan)
    f = np.full((n, k), np.nan)
    lo = np.full(k, np.nan)
    hi = np.full(k, np.nan)
    for i, h in enumerate(heights):
        for j, lab in enumerate(hdr):
            if lab == f"{h}m direction":
                d[:, i] = mat[:, j]
            elif lab == f"{h}m speed":
                s[:, i] = mat[:, j]
            elif str(lab).startswith(f"{h}m flag"):
                f[:, i] = mat[:, j]
                m = _FLAG_RE.match(str(lab))
                if m:
                    lo[i], hi[i] = float(m["lo"]), float(m["hi"])
    return xr.Dataset({"direction": ((TIME, HEIGHT), d), "speed": ((TIME, HEIGHT), s),
                       "shadow_flag": ((TIME, HEIGHT), f),
                       "sector_min": (HEIGHT, lo), "sector_max": (HEIGHT, hi)},
                      coords={TIME: t, HEIGHT: list(heights)})


def wind_to_legacy(wind: xr.Dataset) -> Dict[str, Any]:
    heights = [float(h) for h in wind[HEIGHT].values]
    n, k = wind.sizes[TIME], len(heights)
    mat = np.full((n, 1 + 3 * k), np.nan)
    mat[:, 0] = to_datenum(wind[TIME].values)
    hdr = ["timeStamp"] + [""] * (3 * k)
    for i, h in enumerate(heights):
        c0 = 1 + 3 * i
        mat[:, c0] = wind["direction"].values[:, i]
        mat[:, c0 + 1] = wind["speed"].values[:, i]
        mat[:, c0 + 2] = wind["shadow_flag"].values[:, i]
        lo, hi = float(wind["sector_min"].values[i]), float(wind["sector_max"].values[i])
        hdr[c0] = f"{h}m direction"
        hdr[c0 + 1] = f"{h}m speed"
        hdr[c0 + 2] = f"{h}m flag {lo:.3g}<dir<{hi:.3g}"
    return {"spdAndDir": mat, "spdAndDirHeader": hdr}


# ── rotation ─────────────────────────────────────────────────────────────────

def rotation_from_legacy(rotated: np.ndarray, pf_only: np.ndarray, output: Dict,
                         heights: Sequence[float], time_hf: np.ndarray, time: np.ndarray
                         ) -> Optional[xr.Dataset]:
    """``rotated_sonic_data``/``pf_sonic_data`` (n, 3k) + ``rotatedSonic``/``PFSonic`` means."""
    if rotated is None or np.size(rotated) == 0:
        return None
    k = len(heights)
    comps = ["u", "v", "w"]
    ds = xr.Dataset(coords={TIME_HF: time_hf, HEIGHT: list(heights), COMPONENT: comps, TIME: time})
    ds["rotated"] = ((TIME_HF, HEIGHT, COMPONENT), np.asarray(rotated).reshape(len(time_hf), k, 3))
    ds["pf"] = ((TIME_HF, HEIGHT, COMPONENT), np.asarray(pf_only).reshape(len(time_hf), k, 3))
    for key, name in (("rotatedSonic", "rotated_mean"), ("PFSonic", "pf_mean")):
        if key in output:
            ds[name] = ((TIME, HEIGHT, COMPONENT), np.asarray(output[key]).reshape(len(time), k, 3))
    return ds


def rotation_to_legacy(rot: xr.Dataset) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """``(rotated, pf_only, output entries)``; the high-frequency arrays are
    empty when the Dataset carries only the period means (a read run file)."""
    k = rot.sizes[HEIGHT]
    if "rotated" in rot:
        n = rot.sizes[TIME_HF]
        rotated = rot["rotated"].values.reshape(n, 3 * k)
        pf_only = rot["pf"].values.reshape(n, 3 * k)
    else:
        rotated = pf_only = np.empty((0, 3 * k))
    hdr = [f"{float(h)}m:{c}" for h in rot[HEIGHT].values for c in ("u", "v", "w")]
    out = {"rotatedSonicHeader": hdr, "PFSonicHeader": list(hdr)}
    if "rotated_mean" in rot:
        out["rotatedSonic"] = rot["rotated_mean"].values.reshape(rot.sizes[TIME], 3 * k)
    if "pf_mean" in rot:
        out["PFSonic"] = rot["pf_mean"].values.reshape(rot.sizes[TIME], 3 * k)
    return rotated, pf_only, out


# ── flux products ────────────────────────────────────────────────────────────

_DERIVED_RE = re.compile(r"^(?P<h>[\d.]+) m: (?P<key>.+)$")
_SPECIFIC_KEYS = {"q(g/kg)": ("q", "g/kg"), "q(g/g)": ("q", "g/g"),
                  "virtualThetaAvg(K)": ("virtualThetaAvg", "K"), "rAvg(g/kg)": ("rAvg", "g/kg"),
                  "rho_airmoistAvg(kg/m^3)": ("rho_airmoistAvg", "kg/m^3"),
                  "rho_airdryAvg(kg/m^3)": ("rho_airdryAvg", "kg/m^3"),
                  "rho_H2OAvg(kg/m^3)": ("rho_H2OAvg", "kg/m^3")}
_SPECIFIC_LABELS = {v[0]: k for k, v in _SPECIFIC_KEYS.items() if v[0] != "q"}
PRODUCT_NAMES = list(TABLE_SPECS) + ["derivedT", "specificHum"]


def _header_key(output: Dict, name: str) -> Optional[str]:
    for k in (f"{name}Header", f"{name}header"):
        if k in output:
            return k
    return None


def products_from_legacy(output: Dict, heights: Sequence[float]) -> Dict[str, xr.Dataset]:
    """Every flux table of *output* -> Dataset (time, height) with variables by column key."""
    out = {}
    hn_of = {height_label(h): h for h in heights}
    for name in PRODUCT_NAMES:
        hk = _header_key(output, name)
        mat = output.get(name)
        if hk is None or not isinstance(mat, np.ndarray) or mat.ndim != 2:
            continue
        labels = list(output[hk])
        t = to_datetime64(mat[:, 0])
        ds = xr.Dataset(coords={TIME: t, HEIGHT: list(heights)})
        ds.attrs["legacy_header_key"] = hk
        if name in TABLE_SPECS:
            spec = TABLE_SPECS[name]
            lab_to_key = {}
            for hn, h in hn_of.items():
                for key, tmpl in spec.columns:
                    lab_to_key[tmpl.format(hn=hn)] = (key, h, tmpl)
            for j, lab in enumerate(labels):
                if j == 0:
                    continue
                if lab in spec.fixed:
                    ds[lab] = (TIME, mat[:, j])
                    continue
                if lab not in lab_to_key:
                    continue
                key, h, tmpl = lab_to_key[lab]
                if key not in ds:
                    ds[key] = ((TIME, HEIGHT), np.full((len(t), len(heights)), np.nan))
                    ds[key].attrs["label"] = tmpl
                ds[key].values[:, list(heights).index(h)] = mat[:, j]
        else:
            units = {}
            for j, lab in enumerate(labels):
                if j == 0:
                    continue
                m = _DERIVED_RE.match(str(lab))
                if not m:
                    continue
                h = float(m["h"])
                raw_key = m["key"]
                key, unit = _SPECIFIC_KEYS.get(raw_key, (raw_key, None))
                if key not in ds:
                    ds[key] = ((TIME, HEIGHT), np.full((len(t), len(heights)), np.nan))
                    if unit:
                        ds[key].attrs["units"] = unit
                    units[key] = unit
                ds[key].values[:, list(heights).index(h)] = mat[:, j]
        out[name] = ds
    return out


def products_to_legacy(products: Dict[str, xr.Dataset]) -> Dict[str, Any]:
    """Back to ``<name>`` matrices and headers, all-NaN columns dropped (legacy trim)."""
    out: Dict[str, Any] = {}
    for name, ds in products.items():
        heights = [float(h) for h in ds[HEIGHT].values]
        hk = ds.attrs.get("legacy_header_key", f"{name}Header")
        cols = [to_datenum(ds[TIME].values)]
        hdr = ["time"]
        if name in TABLE_SPECS:
            spec = TABLE_SPECS[name]
            for lab in spec.fixed[1:]:
                if lab in ds:
                    cols.append(ds[lab].values)
                    hdr.append(lab)
            for h in heights:
                hn = height_label(h)
                for key, tmpl in spec.columns:
                    if key in ds:
                        cols.append(ds[key].values[:, heights.index(h)])
                        hdr.append(tmpl.format(hn=hn))
        else:
            order = (["theta_fw", "theta_v_son", "theta_v_fw", "T_son_air"] if name == "derivedT"
                     else ["q", "virtualThetaAvg", "rAvg", "rho_airmoistAvg", "rho_airdryAvg", "rho_H2OAvg"])
            for h in heights:
                for key in order:
                    if key not in ds:
                        continue
                    col = ds[key].values[:, heights.index(h)]
                    if key == "q":
                        lab = f"{h} m: q({ds[key].attrs.get('units', 'g/kg')})"
                    elif name == "specificHum":
                        lab = f"{h} m: {_SPECIFIC_LABELS[key]}"
                    else:
                        lab = f"{h} m: {key}"
                    cols.append(col)
                    hdr.append(lab)
        mat = np.column_stack(cols)
        keep = np.any(~np.isnan(mat), axis=0)
        out[name] = mat[:, keep]
        out[hk] = [lab for lab, k in zip(hdr, keep) if k]
    return out


# ── raw high-frequency products ──────────────────────────────────────────────

_RAW_SONIC = ("uPF", "vPF", "wPF", "u_tilt", "v_tilt", "w_tilt", "WD", "spd", "sonTs", "Theta_v_son",
              "fwThPrime", "fwTh", "fwT")
_RAW_H2O = ("rhov", "rhovPrime", "rhovextenalPrime")
_RAW_CO2 = ("rhoCO2", "rhoCO2Prime", "rhoCO2extenalPrime")


def raw_from_legacy(raw: Optional[Dict], time_hf: np.ndarray) -> Optional[xr.Dataset]:
    """The raw dict -> Dataset on (time_hf, height) / (time_hf, height_h2o) / (time_hf, height_co2)."""
    if raw is None:
        return None
    ds = xr.Dataset(coords={TIME_HF: time_hf})
    if "z" in raw:
        ds.coords[HEIGHT] = np.asarray(raw["z"], dtype=float)
    if "z_h2o" in raw:
        ds.coords["height_h2o"] = np.asarray(raw["z_h2o"], dtype=float)
    if "z_co2" in raw:
        ds.coords["height_co2"] = np.asarray(raw["z_co2"], dtype=float)
    for key in _RAW_SONIC:
        if key in raw:
            ds[key] = ((TIME_HF, HEIGHT), np.asarray(raw[key]))
    for key in _RAW_H2O:
        if key in raw:
            ds[key] = ((TIME_HF, "height_h2o"), np.asarray(raw[key]))
    for key in _RAW_CO2:
        if key in raw:
            ds[key] = ((TIME_HF, "height_co2"), np.asarray(raw[key]))
    if "P" in raw:
        P = np.asarray(raw["P"])
        ds["P"] = ("time_P", P[:, 1])
        ds.coords["time_P"] = to_datetime64(P[:, 0])
    return ds


def raw_to_legacy(ds: Optional[xr.Dataset]) -> Optional[Dict]:
    if ds is None:
        return None
    out: Dict[str, Any] = {"t": to_datenum(ds[TIME_HF].values)}
    if HEIGHT in ds.coords:
        out["z"] = ds[HEIGHT].values.astype(float)
    if "height_h2o" in ds.coords:
        out["z_h2o"] = ds["height_h2o"].values.astype(float)
    if "height_co2" in ds.coords:
        out["z_co2"] = ds["height_co2"].values.astype(float)
    for key in _RAW_SONIC + _RAW_H2O + _RAW_CO2:
        if key in ds:
            out[key] = ds[key].values
    if "P" in ds:
        out["P"] = np.column_stack([to_datenum(ds["time_P"].values), ds["P"].values])
    return out


# ── whole-output adapters ────────────────────────────────────────────────────

def run_from_legacy(info: Dict, data, headers, table_names, sensor_info, output: Optional[Dict] = None,
                    rotated=None, pf_only=None, raw: Optional[Dict] = None, data_info=None,
                    pf_table=None) -> Run:
    """Build a :class:`Run` from whatever legacy pieces exist at a point in the pipeline."""
    hdrs = {name: (list(headers[i][0]), list(headers[i][1])) for i, name in enumerate(table_names)}
    sensors = Sensors.from_legacy(sensor_info, headers, table_names)
    run = Run(site=info, sensors=sensors, table_names=list(table_names), headers=hdrs,
              tables=tables_from_legacy(data, headers, table_names, info.get("tableScanFrequency")),
              notes=list(data_info or []), pf_table=pf_table)
    if output:
        heights = run.sonic_heights()
        t = reference_time(output, table_names)
        run.periods = periods_from_legacy(output, hdrs, table_names)
        run.flags = flags_from_legacy(output, hdrs, table_names, t)
        run.wind = wind_from_legacy(output, heights)
        if rotated is not None and np.size(rotated):
            run.rotation = rotation_from_legacy(rotated, pf_only, output, heights, run.time_hf, t)
        run.products = products_from_legacy(output, heights)
        run.warnings = list(output.get("warnings", []))
    if raw is not None:
        run.raw = raw_from_legacy(raw, run.time_hf)
    return run


def to_legacy_output(run: Run) -> Dict[str, Any]:
    """The ``output`` dict the legacy stages and ``save_data`` read."""
    out: Dict[str, Any] = {}
    out.update(periods_to_legacy(run.periods, run.headers))
    out.update(flags_to_legacy(run.flags))
    if run.wind is not None:
        out.update(wind_to_legacy(run.wind))
    if run.rotation is not None:
        _, _, rot_out = rotation_to_legacy(run.rotation)
        out.update(rot_out)
    out.update(products_to_legacy(run.products))
    out["warnings"] = list(run.warnings)
    return out
