"""fluxes – turbulent statistics (H, τ, LE, CO2, σ, L, η, ε, skew, …) per
averaging period and sonic level.

Stage orchestrator over the :mod:`utespac.flux` package: the reference
state (:func:`~utespac.flux.reference.reference_state`), one
:class:`~utespac.flux.levels.LevelInputs` per sonic, the per-period engine
(:func:`~utespac.flux.engine.compute_period`) and the named output tables
(:class:`~utespac.flux.tables.FluxTables`), which store the legacy
matrices and ``<name>Header`` lists into ``output``.
"""

import logging
import traceback
import warnings
from dataclasses import replace
from typing import Dict, List, Optional

import numpy as np

from .averaging import block_last, n_periods, period_bounds
from .flux.engine import FluxOptions, _corr, _skew, compute_period  # noqa: F401
from .flux.levels import LevelInputs, build_level
from .flux.reference import Md, Mv, Rd, Rv, reference_state  # noqa: F401
from .flux.tables import FluxTables
from .wind_stats import wind_direction_speed

log = logging.getLogger("utespac")


def fluxes(
    data: List[Optional[np.ndarray]],
    rotated_sonic_data: np.ndarray,
    pf_sonic_data: np.ndarray,
    info: Dict,
    output: Dict,
    sensor_info: Dict,
    table_names: List[str],
) -> tuple:
    """Compute all turbulent fluxes and statistics for all sonic levels.

    Returns
    -------
    output : dict
    raw    : dict or None
    """
    log.info("Computing Fluxes")

    if "u" not in sensor_info:
        return output, None

    save_raw = bool(info.get("saveRawConditionedData", False))
    raw: Optional[Dict] = {} if save_raw else None

    # ---- reference state, periods, run options ----
    t = data[int(sensor_info["u"][0, 0])][:, 0]
    ref = reference_state(data, sensor_info, info, t)
    if save_raw and ref.raw_P is not None:
        raw["P"] = ref.raw_P

    num_sonics = sensor_info["u"].shape[0]
    heights = [float(sensor_info["u"][ii, 2]) for ii in range(num_sonics)]
    N = n_periods(t, info["avgPer"], whole_days=False)
    bp = period_bounds(len(t), N)
    has_fw = "fw" in sensor_info
    matlab_compat = bool(info.get("matlabCompat", False))

    # Slope geometry (SiteInfo): slope angle, fall-line direction, and which
    # planar-fit horizontal axis lies along the fall line. Only the angle
    # matters on flat sites; a sloped site without the other two keys gets
    # the French Meadows values with a warning.
    angle = float(info.get("angle", 0))
    downslope_aspect = info.get("downslopeAspect")
    slope_axis = info.get("slopeAxis")
    if angle != 0 and (downslope_aspect is None or slope_axis is None):
        warnings.warn(
            f"angle = {angle} deg but downslopeAspect/slopeAxis are not set in "
            "siteInfo; assuming French Meadows geometry (downslopeAspect = 30, "
            "slopeAxis = 'v') for the slope-normal heat flux and L."
        )
    downslope_aspect = 30.0 if downslope_aspect is None else float(downslope_aspect)
    slope_axis = "v" if slope_axis is None else str(slope_axis)
    if slope_axis not in ("u", "v"):
        raise ValueError(f"info['slopeAxis'] must be 'u' or 'v', got {slope_axis!r}")

    opts = FluxOptions(
        detrend=info.get("detrendingFormat", "linear"),
        n_sub=info["avgPer"] // info.get("SSITC_subAvgMin", 5),
        displacement_height=float(info.get("displacementHeight", 0.0)),
        canopy_height=float(info.get("canopyHeight", np.nan)),
        use_canopy_itc=bool(info.get("useCanopyITC", True)),
        latitude=info.get("latitude"),
        calc_dissipation=bool(info.get("calcDissipation", False)),
        matlab_compat=matlab_compat,
        angle=angle,
        downslope_aspect=downslope_aspect,
    )

    tables = FluxTables(N, heights, has_fw)
    derivedT_cols = []
    cp = 1004.67 * (1 + 0.84 * ref.q)

    # ---- per-sonic loop ----
    for ii in range(num_sonics):
        height = heights[ii]
        try:
            lev = build_level(ii, data, output, sensor_info, info, table_names, ref,
                              rotated_sonic_data, pf_sonic_data, N, t, slope_axis, matlab_compat)
            scan = info.get("tableScanFrequency") or []
            lev_opts = replace(opts, scan_freq=float(scan[lev.table_index])
                               if lev.table_index < len(scan) else opts.scan_freq)

            _store_specific_hum(output, lev, t, bp)
            derivedT_cols.extend(lev.derived_T_cols)
            if save_raw:
                if ii == 0:
                    _init_raw(raw, t, num_sonics, sensor_info)
                _fill_raw_level(raw, lev, ii)

            for jj in range(N):
                s0, s1 = int(bp[jj]), int(bp[jj + 1])
                if s0 >= s1:
                    continue
                if ii == 0:
                    tables.set_time(jj, t[s1 - 1])
                    tables["H"].set(jj, None, "rho", ref.rho[jj] if jj < len(ref.rho) else np.nan)
                    tables["H"].set(jj, None, "cp", cp[jj] if jj < len(cp) else np.nan)
                res = compute_period(lev, ref, lev_opts, jj, s0, s1, t)
                for tname, vals in res.values.items():
                    tables[tname].set_many(jj, height, vals)
                if save_raw:
                    _fill_raw_period(raw, lev, ii, s0, s1, res.samples, t)

        except Exception as exc:
            warnings.warn(f"Flux computation failed at height {height}m: {exc}")
            traceback.print_exc()

    # ---- store ----
    tables.store(output, store_extra=bool(info.get("storeExtraStats", True)))

    # derivedT: block-averaged derived temperatures (theta_v_son, T_son_air, fw temps)
    if derivedT_cols:
        timestamps = block_last(t, period_bounds(len(t), n_periods(t, info["avgPer"])))
        mat = np.column_stack([timestamps] + [c for _, c in derivedT_cols])
        hdr = ["time"] + [h for h, _ in derivedT_cols]
        keep = np.any(~np.isnan(mat), axis=0)
        output["derivedT"] = mat[:, keep]
        output["derivedTheader"] = [h for h, k in zip(hdr, keep) if k]

    return output, raw


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _store_specific_hum(output: Dict, lev: LevelInputs, t, bp) -> None:
    """Append the level's HMP-derived columns to ``output['specificHum']``
    (time column created with the first contributing level)."""
    if not lev.specific_hum_cols:
        return
    if "specificHum" not in output:
        output["specificHum"] = block_last(t, bp).reshape(-1, 1)
        output["specificHumHeader"] = ["time"]
    nrows = output["specificHum"].shape[0]
    for hdr, col in lev.specific_hum_cols:
        output["specificHum"] = np.column_stack([output["specificHum"],
                                                 np.asarray(col)[:nrows].reshape(-1, 1)])
        output["specificHumHeader"].append(hdr)


def _init_raw(raw: Dict, t, num_sonics: int, sensor_info: Dict) -> None:
    n = len(t)
    for key in ("uPF", "vPF", "wPF", "u_tilt", "v_tilt", "w_tilt", "WD", "spd", "sonTs", "Theta_v_son"):
        raw[key] = np.full((n, num_sonics), np.nan)
    raw["t"] = t
    raw["z"] = np.full(num_sonics, np.nan)
    if "irgaH2O" in sensor_info or "LiH2O" in sensor_info:
        h2o_info = sensor_info.get("irgaH2O", sensor_info.get("LiH2O"))
        nh = h2o_info.shape[0]
        raw["rhov"] = np.full((n, nh), np.nan)
        raw["rhovPrime"] = np.full((n, nh), np.nan)
        raw["rhovextenalPrime"] = np.full((n, nh), np.nan)
        raw["z_h2o"] = h2o_info[:, 2].astype(float)        # hygrometer heights [m]
    if "irgaCO2" in sensor_info or "LiCO2" in sensor_info:
        co2_info = sensor_info.get("irgaCO2", sensor_info.get("LiCO2"))
        nc = co2_info.shape[0]
        raw["rhoCO2"] = np.full((n, nc), np.nan)
        raw["rhoCO2Prime"] = np.full((n, nc), np.nan)
        raw["rhoCO2extenalPrime"] = np.full((n, nc), np.nan)
        raw["z_co2"] = co2_info[:, 2].astype(float)        # CO2 sensor heights [m]


def _fill_raw_level(raw: Dict, lev: LevelInputs, ii: int) -> None:
    raw["uPF"][:, ii] = lev.u_pf
    raw["vPF"][:, ii] = lev.v_pf
    raw["wPF"][:, ii] = lev.w_pf
    raw["u_tilt"][:, ii] = lev.pf_u
    raw["v_tilt"][:, ii] = lev.pf_v
    raw["w_tilt"][:, ii] = lev.w_tilt
    raw["sonTs"][:, ii] = lev.T_son
    raw["Theta_v_son"][:, ii] = lev.theta_son
    raw["z"][ii] = lev.height
    raw["WD"][:, ii], raw["spd"][:, ii] = wind_direction_speed(lev.u, lev.v, lev.bearing, lev.manufacturer)


def _fill_raw_period(raw: Dict, lev: LevelInputs, ii: int, s0: int, s1: int,
                     samples: Dict[str, np.ndarray], t) -> None:
    if "fwThPrime" in samples:
        if "fwThPrime" not in raw:
            for key in ("fwThPrime", "fwTh", "fwT"):
                raw[key] = np.full((len(t), raw["uPF"].shape[1]), np.nan)
        for key in ("fwThPrime", "fwTh", "fwT"):
            raw[key][s0:s1, ii] = samples[key]
    if "rhov" in raw and "rhov" in samples:
        for key in ("rhov", "rhovPrime", "rhovextenalPrime"):
            raw[key][s0:s1, lev.h2o_sensor_index] = samples[key]
    if "rhoCO2" in raw and "rhoCO2" in samples:
        for key in ("rhoCO2", "rhoCO2Prime", "rhoCO2extenalPrime"):
            raw[key][s0:s1, lev.co2_sensor_index] = samples[key]
