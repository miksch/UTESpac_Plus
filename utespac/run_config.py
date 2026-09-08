"""Typed, frozen run configuration for UTESpac (packaged TOMLs + overrides).

Four stage configs mirror the TOMLs in :mod:`utespac.config`:
:class:`QCConfig` (``qc.toml``), :class:`PFConfig` (``pf.toml``),
:class:`FluxConfig` (``flux.toml``) and the run-level :class:`RunConfig`
(``run.toml``, which also carries the output flags and the sensor-name
templates). Each dataclass is the schema and holds the defaults; a TOML
overrides them; per-call keywords override the TOML.

``RunConfig.to_info()`` renders the legacy ``info`` dict the processing
stages still read, so the pipeline runs unchanged while the stages are
migrated one by one (migration gameplan step 4).
"""

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Dict, List, Optional

from . import config as _config


def _check_unknown(cls, keys, where):
    names = {f.name for f in fields(cls)}
    unknown = set(keys) - names
    if unknown:
        raise ValueError(f"unknown {where} config parameter(s): {sorted(unknown)}. "
                         f"Valid keys: {sorted(names)}")


@dataclass(frozen=True)
class SpikeTestConfig:
    """Vickers & Mahrt despiking (``[spikeTest]``)."""
    maxRuns: int = 20
    windowSizeFraction: float = 1
    maxConsecutiveOutliers: int = 10
    maxPercent: float = 2
    spikeDef: Dict[str, float] = field(default_factory=lambda: {
        "u": 3.5, "v": 3.5, "w": 5.0, "Tson": 3.5, "fw": 3.5, "irgaCO2": 3.5,
        "irgaH2O": 3.5, "KH2O": 3.5, "cup": 3.5, "birdSpd": 3.5,
        "imuAx": 5.0, "imuAy": 5.0, "imuAz": 5.0, "imuGx": 5.0, "imuGy": 5.0,
        "imuGz": 5.0, "imuRoll": 5.0, "imuPitch": 5.0, "imuYaw": 5.0,
        "otherInstrument": 5.0})


@dataclass(frozen=True)
class DiagnosticTestConfig:
    """Sensor diagnostic limits (``[diagnosticTest]``)."""
    H2OminSignal: float = 0.7
    CO2minSignal: float = 0.7
    meanGasDiagnosticLimit: float = 0.1
    meanSonicDiagnosticLimit: float = 50
    meanLiGasDiagnosticLimit: float = 220


@dataclass(frozen=True)
class QCConfig:
    """Data-conditioning tests (``qc.toml``)."""
    spikeTest: SpikeTestConfig = field(default_factory=SpikeTestConfig)
    absoluteLimitsTest: Dict[str, List[float]] = field(default_factory=lambda: {
        "u": [-50, 50], "v": [-50, 50], "w": [-10, 10], "Tson": [-20, 80],
        "fw": [-20, 80], "irgaCO2": [0, 1500], "irgaH2O": [0, 50],
        "KH2O": [0, 50], "cup": [0, 50], "birdSpd": [0, 50],
        "imuAx": [-50, 50], "imuAy": [-50, 50], "imuAz": [-50, 50],
        "imuGx": [-500, 500], "imuGy": [-500, 500], "imuGz": [-500, 500],
        "imuRoll": [-180, 180], "imuPitch": [-180, 180], "imuYaw": [-360, 360]})
    windDirectionEnvelope: float = 20      # [deg] (legacy windDirectionTest.envelopeSize)
    nanMaxPercent: float = 55              # [%]   (legacy nanTest.maxPercent)
    diagnosticTest: DiagnosticTestConfig = field(default_factory=DiagnosticTestConfig)

    @classmethod
    def from_config(cls, config=None, **overrides):
        """Build from ``qc.toml`` (see :func:`utespac.config.resolve`); the
        sections of the file are the legacy ``info`` blocks. ``overrides``
        are field names of this class."""
        raw = _config.resolve("qc", config, flatten=False)
        known = {"spikeTest", "absoluteLimitsTest", "windDirectionTest", "nanTest",
                 "diagnosticTest"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown qc config section(s): {sorted(unknown)}")
        vals: Dict[str, Any] = {}
        if "spikeTest" in raw:
            st = dict(raw["spikeTest"])
            spike_def = st.pop("spikeDef", None)
            _check_unknown(SpikeTestConfig, st, "qc.spikeTest")
            base = SpikeTestConfig(**st)
            if spike_def is not None:
                base = replace(base, spikeDef={**base.spikeDef, **spike_def})
            vals["spikeTest"] = base
        if "absoluteLimitsTest" in raw:
            vals["absoluteLimitsTest"] = {**cls().absoluteLimitsTest,
                                          **{k: list(v) for k, v in raw["absoluteLimitsTest"].items()}}
        if "windDirectionTest" in raw:
            vals["windDirectionEnvelope"] = raw["windDirectionTest"]["envelopeSize"]
        if "nanTest" in raw:
            vals["nanMaxPercent"] = raw["nanTest"]["maxPercent"]
        if "diagnosticTest" in raw:
            _check_unknown(DiagnosticTestConfig, raw["diagnosticTest"], "qc.diagnosticTest")
            vals["diagnosticTest"] = DiagnosticTestConfig(**raw["diagnosticTest"])
        _check_unknown(cls, overrides, "qc")
        return cls(**{**vals, **overrides})

    def to_info(self) -> Dict[str, Any]:
        """Legacy ``info`` blocks."""
        return {
            "spikeTest": {**asdict(self.spikeTest)},
            "absoluteLimitsTest": {k: list(v) for k, v in self.absoluteLimitsTest.items()},
            "windDirectionTest": {"envelopeSize": self.windDirectionEnvelope},
            "nanTest": {"maxPercent": self.nanMaxPercent},
            "diagnosticTest": asdict(self.diagnosticTest),
        }


@dataclass(frozen=True)
class PFConfig:
    """Planar fit (``pf.toml``)."""
    globalCalculation: str = "global"
    recalculateGlobalCoefficients: bool = True
    avgPer: int = 30
    globalCalcMaxWind: float = 20
    globalCalcMinWind: float = 0.5

    @classmethod
    def from_config(cls, config=None, **overrides):
        vals = _config.resolve("pf", config)
        _check_unknown(cls, set(vals) | set(overrides), "pf")
        obj = cls(**{**vals, **overrides})
        if obj.globalCalculation not in ("global", "local"):
            raise ValueError("pf.globalCalculation must be 'global' or 'local', "
                             f"got {obj.globalCalculation!r}")
        return obj

    def to_info(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FluxConfig:
    """Flux stage (``flux.toml``)."""
    detrendingFormat: str = "linear"
    calcDissipation: bool = False
    storeExtraStats: bool = True
    qRef: float = 12
    stabilitySource: str = "hoegstroem1988"

    @classmethod
    def from_config(cls, config=None, **overrides):
        from .stability import PSI_SOURCES
        vals = _config.resolve("flux", config)
        _check_unknown(cls, set(vals) | set(overrides), "flux")
        obj = cls(**{**vals, **overrides})
        if obj.detrendingFormat not in ("linear", "constant"):
            raise ValueError("flux.detrendingFormat must be 'linear' or 'constant', "
                             f"got {obj.detrendingFormat!r}")
        if obj.stabilitySource not in PSI_SOURCES:
            raise ValueError(f"flux.stabilitySource must be one of {sorted(PSI_SOURCES)}, "
                             f"got {obj.stabilitySource!r}")
        return obj

    def to_info(self) -> Dict[str, Any]:
        return asdict(self)


_DEFAULT_TEMPLATE = {
    "u": "Ux_*", "v": "Uy_*", "w": "Uz_*", "Tson": "T_Sonic_*",
    "sonDiagnostic": "diagnostic_*", "fw": "FW_*", "RH": "RH_*", "T": "Temp_*",
    "P": "Pressure_*", "irgaH2O": "H2O_*", "irgaH2OsigStrength": "H2Osig_*",
    "irgaCO2": "CO2_*", "irgaCO2sigStrength": "CO2sig_*", "irgaGasDiag": "gas_diag_*",
    "LiH2O": "LiH2O_*", "LiCO2": "LiCO2_*", "LiGasDiag": "Li_gas_diag_*",
    "KH2O": "KH2O_H2O_*", "cup": "cup_*", "birdSpd": "wbSpd_*", "birdDir": "wbDir_*",
    "imuAx": "IMU_Ax*", "imuAy": "IMU_Ay*", "imuAz": "IMU_Az*",
    "imuGx": "IMU_Gx*", "imuGy": "IMU_Gy*", "imuGz": "IMU_Gz*",
    "imuRoll": "IMU_Roll*", "imuPitch": "IMU_Pitch*", "imuYaw": "IMU_Yaw*",
}


@dataclass(frozen=True)
class RunConfig:
    """Run-level configuration (``run.toml``) plus the three stage configs.

    ``rootFolder`` is the directory holding one folder per site (the
    ``data/`` tree); it is machine-specific and never read from the
    packaged TOML.
    """
    rootFolder: Optional[str] = None
    UTESpacVersion: str = "5.0-Python"
    avgPer: int = 30
    saveRawConditionedData: bool = True
    saveCSV: bool = True
    template: Dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_TEMPLATE))
    qc: QCConfig = field(default_factory=QCConfig)
    pf: PFConfig = field(default_factory=PFConfig)
    flux: FluxConfig = field(default_factory=FluxConfig)

    @classmethod
    def from_config(cls, config=None, *, qc=None, pf=None, flux=None, **overrides):
        """Resolve ``run.toml`` (and the stage files) with overrides.

        ``config`` follows :func:`utespac.config.resolve` for ``run.toml``;
        ``qc``/``pf``/``flux`` are a stage dataclass, a dict of overrides, or
        ``None`` (auto-load that stage's TOML). Keyword ``overrides`` are
        ``RunConfig`` fields (``rootFolder`` included).
        """
        raw = _config.resolve("run", config, flatten=False)
        known = {"run", "output", "template"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown run config section(s): {sorted(unknown)}")
        vals: Dict[str, Any] = {}
        vals.update(raw.get("run", {}))
        vals.update(raw.get("output", {}))
        if "template" in raw:
            vals["template"] = {**_DEFAULT_TEMPLATE, **raw["template"]}
        _check_unknown(cls, set(vals) | set(overrides), "run")

        def _stage(obj, klass):
            if obj is None:
                return klass.from_config()
            if isinstance(obj, klass):
                return obj
            if isinstance(obj, dict):
                return klass.from_config(None, **obj)
            raise TypeError(f"{klass.__name__} expected, a dict of overrides, or None")

        stages = {"qc": _stage(qc, QCConfig), "pf": _stage(pf, PFConfig),
                  "flux": _stage(flux, FluxConfig)}
        return cls(**{**vals, **stages, **overrides})

    def with_(self, **changes) -> "RunConfig":
        """Copy with top-level fields changed (``dataclasses.replace``)."""
        _check_unknown(type(self), changes, "run")
        return replace(self, **changes)

    def with_pf(self, **changes) -> "RunConfig":
        _check_unknown(PFConfig, changes, "pf")
        return replace(self, pf=replace(self.pf, **changes))

    def with_flux(self, **changes) -> "RunConfig":
        _check_unknown(FluxConfig, changes, "flux")
        return replace(self, flux=replace(self.flux, **changes))

    def to_info(self) -> Dict[str, Any]:
        """The legacy ``info`` dict the processing stages read.

        Site facts are added later by ``find_files`` (``SiteInfo.apply_to``),
        as before.
        """
        if self.rootFolder is None:
            raise ValueError("RunConfig.rootFolder is not set")
        info: Dict[str, Any] = {
            "rootFolder": str(self.rootFolder),
            "UTESpacVersion": self.UTESpacVersion,
            "avgPer": self.avgPer,
            "saveRawConditionedData": self.saveRawConditionedData,
            "saveCSV": self.saveCSV,
            "PF": self.pf.to_info(),
        }
        info.update(self.flux.to_info())
        info.update(self.qc.to_info())
        return info
