"""Site configuration: typed SiteInfo dataclass and loaders.

Replaces the exec()-based loading of per-site ``siteInfo.py`` files. The
dataclass fields define the full set of recognized site keys; unrecognized
names in a site file raise a warning so typos are not silently dropped.

Sonic instruments may be given either as a tower profile (``sonics`` — a
list of per-level entries with height, orientation, manufacturer, and an
optional paired HMP height) or, for backward compatibility, as the parallel
lists sonicOrientation / sonicManufact / shiftsSonHeight / shiftsHMPHeight.
The profile is preferred; consumers look sonics up by height rather than by
discovery order.
"""

import runpy
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Union

_UNSET = None


@dataclass
class SonicLevel:
    """One sonic anemometer on the tower profile."""

    height: float                          # sonic height [m], matches header height
    orientation: float                     # boom azimuth [deg]
    manufacturer: int                      # 0 RMYoung, 1 CSAT/IRGASON, 2 Gill
    hmp_height: Optional[float] = None      # paired physical HMP height [m], if any


def sonic_for(sonics, height, tol=0.01):
    """Return the SonicLevel whose height matches *height*, or None."""
    for s in sonics or []:
        if abs(float(s.height) - float(height)) < tol:
            return s
    return None


@dataclass
class SiteInfo:
    """Per-site configuration. Fields left as None fall back to the
    pipeline defaults already present in the info dict."""

    sonics: Optional[list] = _UNSET               # tower profile: list of SonicLevel
    sonicOrientation: Optional[list] = _UNSET     # legacy: sonic azimuths [deg]
    sonicManufact: Optional[list] = _UNSET        # legacy: 0 RMYoung, 1 CSAT/IRGASON, 2 Gill
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
        if isinstance(values.get("sonics"), list):
            values["sonics"] = [s if isinstance(s, SonicLevel) else SonicLevel(**s)
                                 for s in values["sonics"]]
        return cls(**values)

    def apply_to(self, info):
        """Copy all set (non-None) fields into the pipeline info dict."""
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                info[f.name] = value
        return info


def has_site_info(site_path) -> bool:
    """True when *site_path* holds a ``siteInfo.toml`` or ``siteInfo.py``."""
    site_path = Path(site_path)
    return (site_path / "siteInfo.toml").is_file() or (site_path / "siteInfo.py").is_file()


def list_sites(root) -> list:
    """Site folder names under *root*, sorted.

    A site is any directory carrying a ``siteInfo.*`` file (the ``data/``
    layout) or, for legacy trees, any directory whose name starts with
    ``site``.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (has_site_info(d) or d.name.startswith("site"))
    )


def resolve_site_dir(root, site) -> str:
    """Map a user-given site name to a folder name under *root*.

    Accepts the folder name itself (``"VAC001"``, ``"siteGill..."``) or the
    legacy bare id for a ``site``-prefixed folder (``"Gill..."``).
    """
    available = list_sites(root)
    if site in available:
        return site
    if f"site{site}" in available:
        return f"site{site}"
    raise FileNotFoundError(f"No site folder {site!r} under {root}; "
                            f"available: {', '.join(available) or 'none'}")


def site_input_dir(site_path) -> Path:
    """Folder holding the UTESpac header and 48-h data files for a site.

    ``<site>/utespac/`` in the ``data/`` layout; the site folder itself for
    legacy trees.
    """
    site_path = Path(site_path)
    sub = site_path / "utespac"
    return sub if sub.is_dir() else site_path


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
