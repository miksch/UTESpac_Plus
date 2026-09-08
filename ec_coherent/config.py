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
    scalars: Tuple[str, ...] = ("ts", "rho_h2o", "rho_co2")


@dataclass(frozen=True)
class RampsConfig:
    """Ramp detection, wavelet path (``[ramps]``; library/writeups/ec_ramps.md)."""
    signals: Tuple[str, ...] = ("ts", "u_pf", "e")   # wavelet detector: ts, u_pf; "e" runs the TKE trigger (Mangan 2022)
    wavelet: str = "mhat"                    # [CITED] CB 1993 zero-crossing method
    peak: str = "smallest_scale"             # smallest_scale (Thomas & Foken 2007) | global (CB 1993)
    D_min_s: float = 6.2                     # variance-peak search from this event duration up  [CITED] Thomas & Foken 2007 §3.1 (their site)
    a_min_s: float = 1.0                     # scale grid [s]  [ASSUMED]
    a_max_s: float = 300.0
    n_scales_per_decade: int = 16
    slope_ts: str = "auto"                   # negative | positive | both | auto (sign of w'ts')
    slope_u: str = "positive"                # CB 1993 I p. 375
    edge_scales: float = 3.0                 # drop detections within edge_scales*a0 of the ends  [ASSUMED]
    refine: str = "none"                     # none | ramp | haar: re-time at the first-derivative wavelet extremum (DECIDE, task doc)
    max_events: int = 300                    # padding of the event axis
    # structure-function (surface renewal) detector
    sr_signals: Tuple[str, ...] = ("ts", "u_pf")  # series the Van Atta cubic runs on; flux for ts only
    sr_lags_s: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)  # time lags r [CITED] Spano 1997 p. 261
    sr_min_period_lags: float = 10.0         # drop lags with l+s < this * r  [CITED] Spano 1997 p. 261
    sr_alpha: float = 1.0                    # weighting factor, 1 well above the canopy [CITED] Spano 1997 eq. 2
    sr_alpha_mode: str = "fixed"             # fixed (sr_alpha) | castellvi (similarity, per record) | fit ([SITE-TUNED] vs w'Ts')
    sr_d: float = 0.0                        # zero-plane displacement [m] for castellvi mode  [ASSUMED]
    # TKE trigger (Mangan et al. 2022)
    tke_lp_s: float = 10.0                   # centred moving-mean window on u_TKE [CITED] Mangan 2022 eq. 5 (their sites)
    tke_a_s: float = 10.0                    # fixed MHAT scale, ~40 s period [CITED] Mangan 2022 p. 52 (their sites)
    tke_thresh: float = 1.25                 # trigger when wave amplitude > thresh * record mean [CITED] Mangan 2022 p. 54


@dataclass(frozen=True)
class MrdConfig:
    """Multiresolution decomposition (``[mrd]``; library/writeups/ec_mrd.md)."""
    signals: Tuple[str, ...] = ("u_pf", "w_pf", "ts")    # MR variance spectra D_xx
    fluxes: Tuple[str, ...] = ("uw", "vw", "wTs")  # MR cospectra D_xy; vw feeds the momentum gap scan
    grid: str = "trim"              # trim | interp -- map the window onto 2^M samples (DECIDE, task doc);
                                    # both sources interpolate (Howell 1997 eq. 8 up, Vickers 2003 eq. 8 down)
    gap_level_frac: float = 0.01    # leveling-off rule: |accumulative flux change| <= this fraction [CITED] Vickers 2003 §4


@dataclass(frozen=True)
class QuadrantConfig:
    """Quadrant/octant analysis (``[quadrant]``; library/writeups/ec_quadrant.md)."""
    pairs: Tuple[str, ...] = ("uw", "wTs", "wrhov")   # quadrant planes; uw on (u',w'), scalars on (w',c')
    hole_sizes: Tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0,
                                     5.0, 6.0, 8.0, 10.0, 15.0, 20.0)  # range [CITED] Raupach 1981 figs. 6-7; spacing [ASSUMED]
    hole_norm: str = "rms"          # rms (H*sigma_x*sigma_w, Lu & Willmarth 1973; Li & Bo 2019) |
                                    # flux (H*|mean flux|, Willmarth & Lu 1972; Raupach 1981)
    octant_triplets: Tuple[Tuple[str, ...], ...] = (("u_pf", "w_pf", "ts"), ("w_pf", "ts", "rho_h2o"))
    # (u_pf, w_pf, ts) [CITED] Li & Bo 2019 eq. 11; (w_pf, ts, rho_h2o) scalar-dissimilarity [ASSUMED]
    # (ruled in 2026-08-23); any signal names accepted (v_pf, rho_co2, ...)


@dataclass(frozen=True)
class AmpmodConfig:
    """Amplitude modulation (``[ampmod]``; library/writeups/ec_ampmod.md)."""
    modulators: Tuple[str, ...] = ("u_pf", "w_pf")       # large-scale signals b_l [CITED] Salesky & Anderson 2018 §1.3
    signals: Tuple[str, ...] = ("u_pf", "w_pf", "ts")    # small-scale signals a_s
    fluxes: Tuple[str, ...] = ("uw", "wTs")        # instantaneous-flux series decomposed like signals [CITED] SA18
    cutoff_mode: str = "spectral_gap"   # spectral_gap | delta | scaled (locked decision 3; ec_ampmod.md register)
    delta_m: float = 1000.0             # [m] assumed outer scale for 'delta' and the gap fallback [ASSUMED] (Salesky 2020 AHATS)
    z_mult: float = 100.0               # 'scaled': lambda_c = z_mult * z [ASSUMED]


@dataclass(frozen=True)
class ScalesConfig:
    """LSM/VLSM separation (``[scales]``; library/writeups/ec_scales.md)."""
    signals: Tuple[str, ...] = ("u_pf", "w_pf", "ts")    # per-band variance fractions
    fluxes: Tuple[str, ...] = ("uw", "wTs")        # per-band covariance fractions
    cutoff_mode: str = "spectral_gap"   # spectral_gap | delta | scaled (shared register entry, ec_ampmod.md)
    delta_m: float = 1000.0             # [m] 'delta': cuts at 0.1*pi*delta and pi*delta [CITED ratios] Balakumar 2007
    z_mult_small: float = 10.0          # 'scaled': small|LSM cut at z_mult_small * z [ASSUMED]
    z_mult_vlsm: float = 100.0          # 'scaled': LSM|VLSM cut at z_mult_vlsm * z [ASSUMED]


@dataclass(frozen=True)
class CoherentFluxConfig:
    """Coherent-structure flux fractions (``[coherent_flux]``; library/writeups/ec_coherent_flux.md)."""
    event_signals: Tuple[str, ...] = ("u_pf", "ts")   # ramp event sets conditioned on; u default, Ts alongside (ruling 2026-08-23)
    pairs: Tuple[str, ...] = ("uw", "wTs", "wrhov")  # flux pairs, keys of quadrant.PAIRS
    window: str = "duration"        # duration (half-width D_e, Thomas & Foken 2007 eq. 5) | fixed (window_s)
    window_s: float = 30.0          # [s] full width of the 'fixed' window [CITED] Collineau & Brunet 1993 II p. 62
    hole_sizes: Tuple[float, ...] = (0.0, 0.5, 1.0)  # quadrant-estimator holes [CITED] Thomas & Foken 2007 figs. 3-5
    turner: bool = False            # third estimator: Turner & Leclerc 1994 K-rms split (option, ruling 2026-08-24)
    turner_K: float = 4.0           # coefficient threshold K [CITED] Turner & Leclerc 1994 p. 208 (their default)


@dataclass(frozen=True)
class ECConfig:
    """Run-level configuration for ec_coherent."""
    modules: Tuple[str, ...] = ("spectra",)
    output_suffix: str = "coherent"       # <Site>_coherent_<PF>_<Det>_<date>.nc
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    spectra: SpectraConfig = field(default_factory=SpectraConfig)
    mrd: MrdConfig = field(default_factory=MrdConfig)
    quadrant: QuadrantConfig = field(default_factory=QuadrantConfig)
    ramps: RampsConfig = field(default_factory=RampsConfig)
    ampmod: AmpmodConfig = field(default_factory=AmpmodConfig)
    scales: ScalesConfig = field(default_factory=ScalesConfig)
    coherent_flux: CoherentFluxConfig = field(default_factory=CoherentFluxConfig)

    @classmethod
    def from_config(cls, config=None, **overrides) -> "ECConfig":
        """Resolve the TOML (packaged, cwd override, path or dict) with keyword overrides.

        Top-level overrides are field names of this class; ``preprocess`` and
        ``spectra`` accept a dict of their own fields.
        """
        raw = _resolve(config)
        vals: Dict[str, Any] = {}
        for sect, klass in (("preprocess", PreprocessConfig), ("spectra", SpectraConfig),
                            ("mrd", MrdConfig), ("quadrant", QuadrantConfig),
                            ("ramps", RampsConfig), ("ampmod", AmpmodConfig),
                            ("scales", ScalesConfig), ("coherent_flux", CoherentFluxConfig)):
            d = dict(raw.pop(sect, {}) or {})
            d.update(overrides.pop(sect, {}) or {})
            _check_unknown(klass, d, sect)
            if "nperseg" in d and d["nperseg"] in (0, "none", "None"):
                d["nperseg"] = None
            for key in ("scalars", "signals", "sr_signals", "sr_lags_s", "modulators", "fluxes",
                        "pairs", "hole_sizes", "event_signals"):
                if key in d:
                    d[key] = tuple(d[key])
            if "octant_triplets" in d:
                d["octant_triplets"] = tuple(tuple(t) for t in d["octant_triplets"])
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
