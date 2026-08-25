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
    leverArm: Optional[list] = None         # [m] sonic head rel. to the IMU (x, y, z),
    #                                         platform frame; overrides IMUInfo.leverArm


@dataclass
class IMUInfo:
    """The colocated IMU of a floating-platform site (``[imu]`` in
    siteInfo.toml). Consumed by :func:`utespac.stages.motion`; the sign
    lists map the IMU hardware axes into the sonic's right-handed z-up
    platform frame, and the mount angles rotate a residual IMU-to-platform
    mounting offset out (applied exactly to accel/gyro vectors and to the
    vendor attitude via its rotation matrix)."""

    leverArm: list                          # [m] sonic head rel. to IMU (x, y, z), platform frame
    accelUnits: str = "m/s2"                # "m/s2" | "g"
    gyroUnits: str = "deg/s"                # "deg/s" | "rad/s"
    attitudeUnits: str = "deg"              # "deg" | "rad"
    accelSigns: list = None                 # per-axis +-1 into the platform frame (default [1,1,1])
    gyroSigns: list = None
    attitudeSigns: list = None
    mountRoll: float = 0.0                  # [deg] IMU-to-platform mounting rotation
    mountPitch: float = 0.0
    mountYaw: float = 0.0
    Tcf: float = 20.0                       # [s] complementary-filter cutoff period
    Ta: float = 20.0                        # [s] accel-integration high-pass cutoff period
    yawHandling: str = "demean"             # "demean" | "full" | "zero"
    useVendorAttitude: bool = True          # prefer fused roll/pitch/yaw channels when logged

    def __post_init__(self):
        if len(self.leverArm) != 3:
            raise ValueError("imu.leverArm must have three components (x, y, z)")
        for name, val, allowed in (("accelUnits", self.accelUnits, ("m/s2", "g")),
                                   ("gyroUnits", self.gyroUnits, ("deg/s", "rad/s")),
                                   ("attitudeUnits", self.attitudeUnits, ("deg", "rad")),
                                   ("yawHandling", self.yawHandling, ("demean", "full", "zero"))):
            if val not in allowed:
                raise ValueError(f"imu.{name} must be one of {allowed}, got {val!r}")
        for name in ("accelSigns", "gyroSigns", "attitudeSigns"):
            val = getattr(self, name)
            if val is None:
                object.__setattr__(self, name, [1.0, 1.0, 1.0])
            elif len(val) != 3 or any(abs(v) != 1 for v in val):
                raise ValueError(f"imu.{name} must be three values of +-1")


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
    imu: Optional[IMUInfo] = _UNSET               # floating platform: colocated IMU ([imu] table)
    sonicOrientation: Optional[list] = _UNSET     # legacy: sonic azimuths [deg]
    sonicManufact: Optional[list] = _UNSET        # legacy: 0 RMYoung, 1 CSAT/IRGASON, 2 Gill
    tower: Union[float, str, None] = _UNSET       # tower bearing [deg]
    siteElevation: Optional[float] = _UNSET       # [m]
    latitude: Optional[float] = _UNSET            # [deg N], Coriolis parameter for the ITC stable side
    longitude: Optional[float] = _UNSET           # [deg E], provenance in the netCDF products
    angle: Optional[float] = _UNSET               # slope angle [deg]
    downslopeAspect: Optional[float] = _UNSET     # fall-line direction from north [deg]
    slopeAxis: Optional[str] = _UNSET             # planar-fit horizontal axis along the fall line: "u" or "v"
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
        if values.get("slopeAxis") is not None and values["slopeAxis"] not in ("u", "v"):
            raise ValueError(f"{source}: slopeAxis must be 'u' or 'v', got {values['slopeAxis']!r}")
        if isinstance(values.get("sonics"), list):
            values["sonics"] = [s if isinstance(s, SonicLevel) else SonicLevel(**s)
                                 for s in values["sonics"]]
        if isinstance(values.get("imu"), dict):
            known_imu = {f.name for f in fields(IMUInfo)}
            unknown_imu = sorted(set(values["imu"]) - known_imu)
            if unknown_imu:
                warnings.warn(f"{source}: unrecognized imu key(s) ignored: "
                              f"{', '.join(unknown_imu)}")
            values["imu"] = IMUInfo(**{k: v for k, v in values["imu"].items()
                                       if k in known_imu})
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

    Accepts the folder name itself (``"MySite"``, ``"siteGill..."``) or the
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
