"""Named per-period output tables of the flux stage.

Each legacy output matrix (``H``, ``tau``, ``sigma``, ...) is a
:class:`FluxTable`: a ``time`` column, optional fixed columns, then the
same block of named columns for every sonic level. Values are written by
column *key* (``tab.set(row, height, "Ts_w", value)``), the display label
is rendered from the key's template, and :meth:`FluxTable.trimmed` drops
all-NaN columns exactly as the legacy ``_trim_both`` did, so the stored
matrices and ``<name>Header`` lists keep their form. :data:`TABLE_SPECS`
is the legacy column order, without the MATLAB artifacts retired on
2026-08-22 (the duplicate ``R_wPF_CO2`` column, the ``skew_Theata_v``
label).
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def height_label(height: float) -> str:
    """``10.85`` -> ``"10.85"``, ``2.0`` -> ``"2"`` (the legacy ``_h``)."""
    return f"{height:g}"


@dataclass
class TableSpec:
    name: str                       # output key, e.g. "H"
    header_key: str                 # "Hheader", "tauHeader", ...
    columns: Sequence[Tuple[str, str]]   # per-sonic (key, label template with {hn})
    fixed: Sequence[str] = ("time",)     # leading columns (keys == labels)
    extra: bool = False             # only stored with info["storeExtraStats"]


# Column key -> label template, in the legacy header order. {hn} is the
# height label; derivedT/specificHum are assembled separately.
TABLE_SPECS: Dict[str, TableSpec] = {s.name: s for s in [
    TableSpec("H", "Hheader", [
        ("Ts_w", "{hn}m son:Ts'w'"), ("Thv_wPF", "{hn}m son:Theta_v'wPF'"),
        ("Tair_w", "{hn}m son:T_air'w'"), ("Tair_wPF", "{hn}m son:T_air'wPF'"),
        ("fwT_w", "{hn}m fw:T'w'"), ("fwT_wPF", "{hn}m fw:T'wPF'"),
        ("Ths_w", "{hn}m son:Th_s'w'"), ("Ths_wPF", "{hn}m son:Th_s'wPF'"),
        ("fwTh_w", "{hn}m fw:Th'w'"), ("fwTh_wPF", "{hn}m fw:Th'wPF'"),
        ("fwVTh_w", "{hn}m fw:VTh'w'"), ("fwVTh_wPF", "{hn}m fw:VTh'wPF'"),
    ], fixed=("time", "rho", "cp")),
    TableSpec("Hlat", "HlatHeader", [
        ("Ts_u", "{hn}m son:Ts'u'"), ("Ts_uPF", "{hn}m son:Ts'uPF'"),
        ("Ts_v", "{hn}m son:Ts'v'"), ("Ts_vPF", "{hn}m son:Ts'vPF'"),
        ("Ths_u", "{hn}m son:Th_s'u'"), ("Ths_uPF", "{hn}m son:Th_s'uPF'"),
        ("Ths_v", "{hn}m son:Th_s'v'"), ("Ths_vPF", "{hn}m son:Th_s'vPF'"),
        ("fwTh_u", "{hn}m fw:Th'u'"), ("fwTh_uPF", "{hn}m fw:Th'uPF'"),
        ("fwTh_v", "{hn}m fw:Th'v'"), ("fwTh_vPF", "{hn}m fw:Th'vPF'"),
    ]),
    TableSpec("tau", "tauHeader", [
        ("tau_raw", "{hn}m :sqrt(u'w'^2+v'w'^2)"),
        ("tau_PF", "{hn}m :sqrt(uPF'wPF'^2+vPF'wPF'^2)"),
        ("uPF_uPF", "{hn}m :uPF'uPF'"), ("vPF_vPF", "{hn}m :vPF'vPF'"), ("wPF_wPF", "{hn}m :wPF'wPF'"),
        ("uPF_vPF", "{hn}m :uPF'vPF'"), ("uPF_wPF", "{hn}m :uPF'wPF'"), ("vPF_wPF", "{hn}m :vPF'wPF'"),
        ("uT_uT", "{hn}m :uTilt'uTilt'"), ("vT_vT", "{hn}m :vTilt'vTilt'"), ("wT_wT", "{hn}m :wTilt'wTilt'"),
        ("uT_vT", "{hn}m :uTilt'vTilt'"), ("uT_wT", "{hn}m :uTilt'wTilt'"), ("vT_wT", "{hn}m :vTilt'wTilt'"),
    ]),
    TableSpec("tke", "tkeHeader", [("tke", "{hn}m :0.5(u'^2+v'^2+w'^2)")]),
    TableSpec("sigma", "sigmaHeader", [
        ("sigma_u", "{hn}m :sigma_u"), ("sigma_v", "{hn}m :sigma_v"), ("sigma_w", "{hn}m :sigma_w"),
        ("sigma_uPF", "{hn}m :sigma_uPF"), ("sigma_vPF", "{hn}m :sigma_vPF"), ("sigma_wPF", "{hn}m :sigma_wPF"),
        ("sigma_Tson", "{hn}m :sigma_Tson"),
        ("wPFP_TsonP_TsonP", "{hn}m :wPFP_TsonP_TsonP"),
        ("sigma_Theta_v", "{hn}m :sigma_Theta_v"),
        ("sigma_H2O", "{hn}m :sigma_H2O"), ("sigma_CO2", "{hn}m :sigma_CO2"),
        ("sigma_CO2_WPL", "{hn}m :sigma_CO2_WPL"),
        ("sigma_TFW", "{hn}m :sigma_TFW"),          # only with a fine-wire (see FluxTables)
    ]),
    TableSpec("R", "RHeader", [
        ("R_uPFwPF_wPFThetav", "{hn}m :R_uPFwPF_wPFThetav"),
        ("R_uPFwPF_wPFH2O", "{hn}m :R_uPFwPF_wPFH2O"),
        ("R_wPFH2O_wPFThetav", "{hn}m :R_wPFH2O_wPFThetav"),
        ("R_uPFwPF_wPFCO2_WPL", "{hn}m :R_uPFwPF_wPFCO2_WPL"),
        ("R_wPFCO2_WPL_wPFThetav", "{hn}m :R_wPFCO2_WPL_wPFThetav"),
        ("R_wPF_uPF", "{hn}m :R_wPF_uPF"),
        ("R_wPF_Theta_v", "{hn}m :R_wPF_Theta_v"),
        ("R_wPF_H2O", "{hn}m :R_wPF_H2O"),
        ("R_wPF_CO2", "{hn}m :R_wPF_CO2"),
        ("R_wPF_CO2_WPL", "{hn}m :R_wPF_CO2_WPL"),
        ("R_uPFwPF_wPFCO2", "{hn}m :R_uPFwPF_wPFCO2"),
        ("R_wPFCO2_wPFThetav", "{hn}m :R_wPFCO2_wPFThetav"),
        ("R_uPFwPF_wPFH2O_WPL", "{hn}m :R_uPFwPF_wPFH2O_WPL"),
        ("R_wPFH2O_WPL_wPFThetav", "{hn}m :R_wPFH2O_WPL_wPFThetav"),
        ("R_wPF_H2O_WPL", "{hn}m :R_wPF_H2O_WPL"),
    ], extra=True),
    TableSpec("L", "Lheader", [("L", "{hn}m L:sqrt(uPF'*wPF'+vPF'*wPF')^3/2*Th_v/(k*g*wThv_vert)")],
              extra=True),
    TableSpec("scaling", "scalingHeader", [
        ("theta_star_SL", "{hn}m :theta_star_SL(K)"),
        ("q_star_SL", "{hn}m :q_star_SL(g/kg)"),
        ("psi_m", "{hn}m :psi_m(z/L)"),
        ("psi_h", "{hn}m :psi_h(z/L)"),
    ]),
    TableSpec("eta", "etaHeader", [
        ("eta_wPFuPF", "{hn}m :eta_wPFuPF"), ("eta_wPFThetav", "{hn}m :eta_wPFThetav"),
        ("eta_wPFH2O", "{hn}m :eta_wPFH2O"), ("eta_wPFH2O_WPL", "{hn}m :eta_wPFH2O_WPL"),
        ("eta_wPFCO2_WPL", "{hn}m :eta_wPFCO2_WPL"), ("eta_wPFCO2", "{hn}m :eta_wPFCO2"),
    ], extra=True),
    TableSpec("delta_flux_ctrb", "delta_flux_ctrbHeader", [
        ("S_wPFuPF", "{hn}m :S_wPFuPF"), ("S_wPFThetav", "{hn}m :S_wPFThetav"),
        ("S_wPFH2O", "{hn}m :S_wPFH2O"), ("S_wPFH2O_WPL", "{hn}m :S_wPFH2O_WPL"),
        ("S_wPFCO2_WPL", "{hn}m :S_wPFCO2_WPL"), ("S_wPFCO2", "{hn}m :S_wPFCO2"),
    ], extra=True),
    TableSpec("delta_time_ctrb", "delta_time_ctrbHeader", [
        ("D_wPFuPF", "{hn}m :D_wPFuPF"), ("D_wPFThetav", "{hn}m :D_wPFThetav"),
        ("D_wPFH2O", "{hn}m :D_wPFH2O"), ("D_wPFH2O_WPL", "{hn}m :D_wPFH2O_WPL"),
        ("D_wPFCO2_WPL", "{hn}m :D_wPFCO2_WPL"), ("D_wPFCO2", "{hn}m :D_wPFCO2"),
    ], extra=True),
    TableSpec("turbtr", "turbtrHeader", [
        ("w_e", "{hn}m :mean(w'e')"), ("w_uw", "{hn}m :mean(w'u'w')"),
        ("w_thvw", "{hn}m :mean(w'theta_v'w')"),
        ("w_H2Ow", "{hn}m :mean(w'H2O'w')"), ("w_H2OWPLw", "{hn}m :mean(w'H2O_WPL'w')"),
        ("w_CO2w", "{hn}m :mean(w'CO2'w')"), ("w_CO2WPLw", "{hn}m :mean(w'CO2_WPL'w')"),
    ], extra=True),
    TableSpec("epsilon", "epsilonHeader", [("epsilon", "{hn}m :epsilon")], extra=True),
    TableSpec("skew", "skewHeader", [
        ("skew_uPF", "{hn}m :skew_uPF"), ("skew_vPF", "{hn}m :skew_vPF"), ("skew_wPF", "{hn}m :skew_wPF"),
        ("skew_Theta_v", "{hn}m :skew_Theta_v"),
        ("skew_H2O", "{hn}m :skew_H2O"), ("skew_H2O_WPL", "{hn}m :skew_H2O_WPL"),
        ("skew_CO2", "{hn}m :skew_CO2"), ("skew_CO2_WPL", "{hn}m :skew_CO2_WPL"),
    ], extra=True),
    TableSpec("H_SNSP", "H_SNSPHeader", [
        ("uTHv", "{hn}m :uTHv"), ("vTHv", "{hn}m :vTHv"),
        ("wTHv", "{hn}m :wTHv"), ("wTHv_vert", "{hn}m :wTHv_vert"),
    ], extra=True),
    TableSpec("Flux_lat", "Flux_latHeader", [
        ("uPF_thv", "{hn}m :uPF'theta_v'"),
        ("uPF_H2O", "{hn}m :uPF'H2O'"), ("uPF_H2O_WPL", "{hn}m :uPF'H2O_WPL'"),
        ("uPF_CO2", "{hn}m :uPF'CO2'"), ("uPF_CO2_WPL", "{hn}m :uPF'CO2_WPL'"),
    ]),
    TableSpec("LHflux", "LHfluxHeader", [
        ("Lv", "{hn}m Lv(J/g)"),
        ("E_w", "{hn}m w':E(g/m^2s)"), ("E_wPF", "{hn}m wPF':E(g/m^2s)"),
        ("wPF_qWPL", "{hn}m wPF'q_WPL'(m/s kg/m3)"),
        ("LE_WPL_w", "{hn}m WPL, w' (W/m^2)"), ("LE_WPL_wPF", "{hn}m WPL, wPF' (W/m^2)"),
        ("LE_O2_w", "{hn}m O2 no WPL,w' (W/m^2)"), ("LE_O2_wPF", "{hn}m O2 no WPL,wPF' (W/m^2)"),
    ]),
    TableSpec("fluxQC", "fluxQCHeader", [
        ("TAU_SSITC", "{hn}m:TAU_SSITC_TEST"), ("TAU_SS", "{hn}m:TAU_SS_ONLY_TEST"),
        ("H_SSITC", "{hn}m:H_SSITC_TEST"), ("H_SS", "{hn}m:H_SS_ONLY_TEST"),
        ("LE_SSITC", "{hn}m:LE_SSITC_TEST"), ("LE_SS", "{hn}m:LE_SS_ONLY_TEST"),
        ("FC_SSITC", "{hn}m:FC_SSITC_TEST"), ("FC_SS", "{hn}m:FC_SS_ONLY_TEST"),
    ]),
    TableSpec("CO2flux", "CO2fluxHeader", [
        ("Fc_w", "{hn}m w':CO2(kg/m^2s)"),
        ("Fc_wPF", "{hn}m wPF':CO2(kg/m^2s)"),
        ("Fc_WPL", "{hn}m WPL,wPF':CO2(mol/m^2s)"),
        ("wPF_CO2WPL", "{hn}m: wPF'CO2_WPL'(m/s kg/m^3)"),
        ("ppm", "{hn}m: CO2 (ppm, moist-air molar ratio)"),
    ]),
]}

# storage order of the legacy output dict
STORE_ORDER = ["H", "Hlat", "tau", "tke", "sigma", "R", "L", "scaling", "eta", "delta_flux_ctrb",
               "delta_time_ctrb", "turbtr", "epsilon", "skew", "H_SNSP", "Flux_lat", "LHflux",
               "fluxQC", "CO2flux"]


class FluxTable:
    """One named table: ``(n_periods, n_columns)`` values with a key and a
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
    """The legacy table set for a run."""
    n_periods: int
    heights: List[float]
    has_fw: bool
    tables: Dict[str, FluxTable] = field(init=False)

    def __post_init__(self):
        self.tables = {}
        for name in STORE_ORDER:
            spec = TABLE_SPECS[name]
            drop = () if (self.has_fw or name != "sigma") else ("sigma_TFW",)
            self.tables[name] = FluxTable(spec, self.n_periods, self.heights, drop)

    def __getitem__(self, name: str) -> FluxTable:
        return self.tables[name]

    def set_time(self, row: int, ts: float) -> None:
        for tab in self.tables.values():
            tab.values[row, 0] = ts

    def store(self, output: Dict, store_extra: bool = True) -> Dict:
        """Write every table (trimmed) and its header into *output*; ``CO2flux``
        only when it carries data, the extras only with *store_extra*."""
        for name in STORE_ORDER:
            tab = self.tables[name]
            if tab.spec.extra and not store_extra:
                continue
            if name == "CO2flux" and not tab.any_data:
                continue
            output[name], output[tab.spec.header_key] = tab.trimmed()
        return output
