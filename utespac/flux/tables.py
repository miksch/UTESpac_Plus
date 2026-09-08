"""Named per-period output tables of the flux stage.

Each output group (``sensible_heat``, ``momentum``, ``sigma``, ...) is a
:class:`FluxTable`: a ``time`` column, optional fixed columns, then the
same block of named columns for every sonic level. Values are written by
column name (``tab.set(row, height, "w_ts_cov_raw", value)``), the CSV
label is ``<name>_<height>`` (:func:`utespac.names.csv_header`), and
:meth:`FluxTable.trimmed` drops all-NaN columns exactly as the legacy
``_trim_both`` did, so the stored matrices and ``<group>Header`` lists
keep their form. :data:`TABLE_SPECS` is built from
:data:`utespac.names.VARIABLES`, the single source of truth for the
names, in its column order.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..names import GROUP_ORDER, VARIABLES, csv_header

# groups written only with info["storeExtraStats"]
_EXTRA_GROUPS = frozenset({"correlation", "obukhov", "eta", "delta_flux", "delta_time",
                           "transport", "dissipation", "skewness", "sensible_heat_snsp"})
# columns of a group that carry no height, in header order after "time"
_FIXED_COLUMNS: Dict[str, Tuple[str, ...]] = {"sensible_heat": ("rho_air_ref", "cp_ref")}
# groups assembled in the flux stage from the level inputs, not from a FluxTable
_ASSEMBLED = ("temperature", "humidity")


def height_label(height: float) -> str:
    """``10.85`` -> ``"10.85"``, ``2.0`` -> ``"2"`` (the legacy ``_h``)."""
    return f"{height:g}"


@dataclass
class TableSpec:
    name: str                       # group name, e.g. "sensible_heat"
    header_key: str                 # "sensible_heatHeader", "momentumHeader", ...
    columns: Sequence[Tuple[str, str]]   # per-sonic (name, label template with {hn})
    fixed: Sequence[str] = ("time",)     # leading columns (names == labels)
    extra: bool = False             # only stored with info["storeExtraStats"]


# storage order of the groups in the output dict
STORE_ORDER = [g for g in GROUP_ORDER if g not in _ASSEMBLED]


def _spec(group: str) -> TableSpec:
    fixed = _FIXED_COLUMNS.get(group, ())
    columns = [(v.name, csv_header(v.name, "{hn}")) for v in VARIABLES[group]
               if v.name not in fixed]
    return TableSpec(group, f"{group}Header", columns, ("time",) + fixed, group in _EXTRA_GROUPS)


TABLE_SPECS: Dict[str, TableSpec] = {g: _spec(g) for g in STORE_ORDER}


class FluxTable:
    """One named table: ``(n_periods, n_columns)`` values with a name and a
    label per column."""

    def __init__(self, spec: TableSpec, n_periods: int, heights: Sequence[float],
                 drop_keys: Iterable[str] = ()):
        self.spec = spec
        self.name = spec.name
        self.keys: List[str] = list(spec.fixed)
        self.labels: List[str] = list(spec.fixed)
        drop = set(drop_keys)
        for h in heights:
            hn = height_label(h)
            for key, tmpl in spec.columns:
                if key in drop:
                    continue
                self.keys.append(f"{hn}|{key}")
                self.labels.append(tmpl.format(hn=hn))
        self._index = {k: i for i, k in enumerate(self.keys)}
        self.values = np.full((n_periods, len(self.keys)), np.nan)

    # -- writing --
    def set(self, row: int, height: Optional[float], key: str, value) -> None:
        self.values[row, self._col(height, key)] = value

    def set_many(self, row: int, height: Optional[float], values: Dict[str, float]) -> None:
        for key, val in values.items():
            self.values[row, self._col(height, key)] = val

    def blank(self, row: int, height: float, keys: Iterable[str]) -> None:
        for key in keys:
            self.values[row, self._col(height, key)] = np.nan

    def get(self, row: int, height: Optional[float], key: str) -> float:
        return float(self.values[row, self._col(height, key)])

    def _col(self, height: Optional[float], key: str) -> int:
        k = key if height is None else f"{height_label(height)}|{key}"
        return self._index[k]

    def has(self, height: Optional[float], key: str) -> bool:
        k = key if height is None else f"{height_label(height)}|{key}"
        return k in self._index

    # -- reading out --
    def trimmed(self) -> Tuple[np.ndarray, List[str]]:
        """Values and labels with all-NaN columns dropped (legacy ``_trim_both``)."""
        keep = np.any(~np.isnan(self.values), axis=0)
        return self.values[:, keep], [lab for lab, k in zip(self.labels, keep) if k]

    @property
    def any_data(self) -> bool:
        """True when some non-time column holds a value."""
        return bool(np.any(~np.isnan(self.values[:, 1:])))


@dataclass
class FluxTables:
    """The output table set for a run."""
    n_periods: int
    heights: List[float]
    has_fw: bool
    tables: Dict[str, FluxTable] = field(init=False)

    def __post_init__(self):
        self.tables = {}
        for name in STORE_ORDER:
            spec = TABLE_SPECS[name]
            drop = () if (self.has_fw or name != "sigma") else ("t_fw_sigma",)
            self.tables[name] = FluxTable(spec, self.n_periods, self.heights, drop)

    def __getitem__(self, name: str) -> FluxTable:
        return self.tables[name]

    def set_time(self, row: int, ts: float) -> None:
        for tab in self.tables.values():
            tab.values[row, 0] = ts

    def store(self, output: Dict, store_extra: bool = True) -> Dict:
        """Write every table (trimmed) and its header into *output*; ``co2_flux``
        only when it carries data, the extras only with *store_extra*."""
        for name in STORE_ORDER:
            tab = self.tables[name]
            if tab.spec.extra and not store_extra:
                continue
            if name == "co2_flux" and not tab.any_data:
                continue
            output[name], output[tab.spec.header_key] = tab.trimmed()
        return output
