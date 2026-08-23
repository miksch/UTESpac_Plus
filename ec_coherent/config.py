"""Typed, frozen configuration for ec_coherent (packaged TOML + overrides).

:class:`ECConfig` is the schema and holds the defaults; ``config/ec_coherent.toml``
(packaged) mirrors it with comments and provenance labels; a
``config/ec_coherent.toml`` in the working directory overrides the packaged
file; keywords to :meth:`ECConfig.from_config` override both. Same resolution
order as :mod:`utespac.config`.
"""

import tomllib
from dataclasses import dataclass, field, fields, replace
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_NAME = "ec_coherent"


@dataclass(frozen=True)
class PreprocessConfig:
    """Shared preprocessing (``[preprocess]``; library/writeups/ec_preprocess.md)."""
    detrend: str = "block"          # block | linear | recursive | butterworth   [ASSUMED] default
    filter_tau_s: float = 300.0     # RC time constant of 'recursive' / 'butterworth' [s]  [ASSUMED]
    nan_max_frac: float = 0.1       # reject a window above this NaN fraction [ASSUMED]
    taylor_max_ratio: float = 0.5   # sigma_M / U_mean validity limit [CITED] Stull 1988 eq. 1.4d


@dataclass(frozen=True)
class SpectraConfig:
    """Spectra module (``[spectra]``; library/writeups/ec_spectra.md)."""
    taper: str = "boxcar"           # boxcar closes Parseval exactly; hann available  [ASSUMED]
    nperseg: Optional[int] = None   # None: full-record periodogram; int: Welch segments
    n_bins_per_decade: int = 10     # log-binning density [ASSUMED]
    scalars: Tuple[str, ...] = ("Ts", "rhov", "rhoCO2")


@dataclass(frozen=True)
class RampsConfig:
    """Ramp detection, wavelet path (``[ramps]``; library/writeups/ec_ramps.md)."""
    signals: Tuple[str, ...] = ("Ts", "u")   # "e" (TKE) waits on Mangan et al. 2022
    wavelet: str = "mhat"                    # [CITED] CB 1993 zero-crossing method
    peak: str = "smallest_scale"             # smallest_scale (Thomas & Foken 2007) | global (CB 1993)
    D_min_s: float = 6.2                     # variance-peak search from this event duration up  [CITED] Thomas & Foken 2007 §3.1 (their site)
    a_min_s: float = 1.0                     # scale grid [s]  [ASSUMED]
    a_max_s: float = 300.0
    n_scales_per_decade: int = 16
    slope_Ts: str = "auto"                   # negative | positive | both | auto (sign of w'Ts')
    slope_u: str = "positive"                # CB 1993 I p. 375
    edge_scales: float = 3.0                 # drop detections within edge_scales*a0 of the ends  [ASSUMED]
    refine: str = "none"                     # none | ramp | haar: re-time at the first-derivative wavelet extremum (DECIDE, task doc)
    max_events: int = 300                    # padding of the event axis


@dataclass(frozen=True)
class ECConfig:
    """Run-level configuration for ec_coherent."""
    modules: Tuple[str, ...] = ("spectra",)
    output_suffix: str = "coherent"       # <Site>_coherent_<PF>_<Det>_<date>.nc
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    spectra: SpectraConfig = field(default_factory=SpectraConfig)
    ramps: RampsConfig = field(default_factory=RampsConfig)

    @classmethod
    def from_config(cls, config=None, **overrides) -> "ECConfig":
        """Resolve the TOML (packaged, cwd override, path or dict) with keyword overrides.

        Top-level overrides are field names of this class; ``preprocess`` and
        ``spectra`` accept a dict of their own fields.
        """
        raw = _resolve(config)
        vals: Dict[str, Any] = {}
        for sect, klass in (("preprocess", PreprocessConfig), ("spectra", SpectraConfig),
                            ("ramps", RampsConfig)):
            d = dict(raw.pop(sect, {}) or {})
            d.update(overrides.pop(sect, {}) or {})
            _check_unknown(klass, d, sect)
            if "nperseg" in d and d["nperseg"] in (0, "none", "None"):
                d["nperseg"] = None
            for key in ("scalars", "signals"):
                if key in d:
                    d[key] = tuple(d[key])
            vals[sect] = klass(**d)
        top = dict(raw.get("run", {}) or {})
        top.update({k: v for k, v in raw.items() if k != "run"})
        top.update(overrides)
        _check_unknown(cls, top, "run")
        if "modules" in top:
            top["modules"] = tuple(top["modules"])
        return cls(**top, **vals)

    def with_(self, **changes) -> "ECConfig":
        return replace(self, **changes)


def _check_unknown(klass, keys, where):
    names = {f.name for f in fields(klass)}
    unknown = set(keys) - names
    if unknown:
        raise ValueError(f"unknown ec_coherent [{where}] parameter(s): {sorted(unknown)}. "
                         f"Valid keys: {sorted(names)}")


def _resolve(config) -> Dict[str, Any]:
    if config is None:
        cwd = Path.cwd() / "config" / f"{_NAME}.toml"
        if cwd.is_file():
            with open(cwd, "rb") as fh:
                return tomllib.load(fh)
        res = files(__package__) / "config" / f"{_NAME}.toml"
        with res.open("rb") as fh:
            return tomllib.load(fh)
    if isinstance(config, (str, Path)):
        with open(config, "rb") as fh:
            return tomllib.load(fh)
    if isinstance(config, dict):
        return dict(config)
    raise TypeError("config must be None, a TOML path, or a dict")
