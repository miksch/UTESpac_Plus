"""Site configuration: typed SiteInfo dataclass and loaders.

Replaces the exec()-based loading of per-site ``siteInfo.py`` files. The
dataclass fields define the full set of recognized site keys; unrecognized
names in a site file raise a warning so typos are not silently dropped.

TODO: simplify these inputs by modeling the site as a tower profile —
per-level instrument entries (height, orientation, manufacturer, paired
HMP height) instead of the parallel lists sonicOrientation /
sonicManufact / shiftsSonHeight / shiftsHMPHeight.
"""

import runpy
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Union

_UNSET = None


@dataclass
class SiteInfo:
    """Per-site configuration. Fields left as None fall back to the
    pipeline defaults already present in the info dict."""

    sonicOrientation: Optional[list] = _UNSET     # sonic azimuths [deg]
    sonicManufact: Optional[list] = _UNSET        # 0 RMYoung, 1 CSAT/IRGASON, 2 Gill
    tower: Union[float, str, None] = _UNSET       # tower bearing [deg]
    siteElevation: Optional[float] = _UNSET       # [m]
    angle: Optional[float] = _UNSET               # slope angle [deg]
    tableNames: Optional[list] = _UNSET
    tableScanFrequency: Optional[list] = _UNSET   # [Hz]
    tableNumberOfColumns: Optional[list] = _UNSET
    useTrefHMP: Optional[bool] = _UNSET
    avgSlowFreq: Optional[float] = _UNSET         # [min]
    shiftsSonHeight: Optional[list] = _UNSET      # sonic heights with paired HMPs [m]
    shiftsHMPHeight: Optional[list] = _UNSET      # corresponding HMP heights [m]
    shiftzRef: Optional[bool] = _UNSET
    zRefLowestSon: Optional[float] = _UNSET       # [m]
    ascending: Optional[bool] = _UNSET
    SSITC_subAvgMin: Optional[float] = _UNSET     # sub-period length [min]
    displacementHeight: Optional[float] = _UNSET  # [m]
    canopyHeight: Optional[float] = _UNSET        # [m]
    useCanopyITC: Optional[bool] = _UNSET

    @classmethod
    def field_names(cls):
        return [f.name for f in fields(cls)]

    @classmethod
    def from_mapping(cls, ns, source="siteInfo"):
        """Build a SiteInfo from a dict, warning on unrecognized keys."""
        known = set(cls.field_names())
        values = {k: v for k, v in ns.items()
                  if k in known and not k.startswith("_")}
        unknown = [k for k in ns
                   if k not in known and not k.startswith("_")
                   and not callable(ns[k]) and not hasattr(ns[k], "__file__")]
        if unknown:
            warnings.warn(
                f"{source}: unrecognized key(s) ignored: {', '.join(sorted(unknown))}"
            )
        return cls(**values)

    def apply_to(self, info):
        """Copy all set (non-None) fields into the pipeline info dict."""
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                info[f.name] = value
        return info


def load_site_info(site_path) -> SiteInfo:
    """Load site configuration from ``siteInfo.toml`` or ``siteInfo.py``.

    TOML takes precedence when both exist. Raises FileNotFoundError when
    neither is present.
    """
    site_path = Path(site_path)

    toml_file = site_path / "siteInfo.toml"
    if toml_file.is_file():
        import tomllib
        with open(toml_file, "rb") as fh:
            ns = tomllib.load(fh)
        return SiteInfo.from_mapping(ns, source=str(toml_file))

    py_file = site_path / "siteInfo.py"
    if py_file.is_file():
        ns = runpy.run_path(str(py_file))
        return SiteInfo.from_mapping(ns, source=str(py_file))

    raise FileNotFoundError(f"No siteInfo.toml or siteInfo.py in {site_path}")
