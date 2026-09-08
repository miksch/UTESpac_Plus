"""The processing stages on the labeled run model (:mod:`utespac.model`).

``load_run`` → ``condition`` → ``motion`` → ``average`` → ``wind`` →
``rotate`` → ``flux``; each takes a :class:`~utespac.model.Run`, reads the Datasets it
needs, calls the numpy kernels and writes its Dataset back. The pipeline
(:mod:`utespac.pipeline`) calls these in order and writes the products
through the legacy adapters of :mod:`utespac.model`.
"""

import logging
import traceback
import warnings
from dataclasses import replace
from typing import Dict, Sequence

import numpy as np
import xarray as xr

from .averaging import block_average, block_last, n_periods, period_bounds
from .condition_data import qc_table
from .model import (COLUMN, COMPONENT, HEIGHT, TIME, TIME_HF, Run, Sensors, header_units,
                    tables_from_legacy, to_datenum, to_datetime64)
from .rotation import SonicSeries, rotate_sonics
from .wind_stats import shadow_flag, wind_direction_speed

log = logging.getLogger("utespac")


def load_run(info: Dict, data, headers, table_names: Sequence[str], sensor_info: Dict,
             data_info=None, pf_table=None) -> Run:
    """The Run right after ``load_data``/``find_serial_date``: tables and sensors."""
    hdrs = {name: (list(headers[i][0]), list(headers[i][1])) for i, name in enumerate(table_names)}
    units = {name: header_units(headers[i]) for i, name in enumerate(table_names)}
    return Run(site=info, sensors=Sensors.from_legacy(sensor_info, headers, table_names),
               table_names=list(table_names), headers=hdrs, header_units=units,
               tables=tables_from_legacy(data, headers, table_names, info.get("tableScanFrequency")),
               notes=list(data_info or []), pf_table=pf_table)


def _table_matrix(ds: xr.Dataset, labels: Sequence[str]) -> np.ndarray:
    """``[datenum, columns...]`` view of a table Dataset (the kernels' form)."""
    return np.column_stack([to_datenum(ds[TIME_HF].values)] + [ds[lab].values for lab in labels[1:]])


def condition(run: Run, template: Dict[str, str]) -> Run:
    """QC every table in place; ``run.flags[<table>]`` gets ``spike``/``nan``
    on ``(time, column)``."""
    info = run.site
    for name in run.table_names:
        ds = run.tables.get(name)
        if ds is None:
            continue
        labels = run.headers[name][0]
        mat = _table_matrix(ds, labels)
        hz = float(ds.attrs.get("scan_hz", 20))
        mat, spike, nan = qc_table(mat, labels, info, template, hz)
        for j, lab in enumerate(labels):
            if j:
                ds[lab].values[:] = mat[:, j]
        t_end, _ = block_average(mat[:, 0], mat[:, :1], info["avgPer"])
        run.flags[name] = xr.Dataset({"spike": ((TIME, COLUMN), spike), "nan": ((TIME, COLUMN), nan)},
                                     coords={TIME: to_datetime64(t_end), COLUMN: list(labels)})
    return run


_IMU_ACC = ("imuAx", "imuAy", "imuAz")
_IMU_GYRO = ("imuGx", "imuGy", "imuGz")
_IMU_ATT = ("imuRoll", "imuPitch", "imuYaw")


def motion(run: Run) -> Run:
    """Motion-correct every sonic's u/v/w in place in ``run.tables`` from the
    colocated IMU (floating-platform sites: ``info["imu"]`` + IMU columns);
    a no-op without both. Attitude and platform velocity -> ``run.motion``
    on ``time_hf`` for the validation diagnostics."""
    from .motion import G, MotionParams, correct_wind, euler_T, euler_from_T
    from .site_config import IMUInfo, sonic_for

    info = run.site
    imu_cfg = info.get("imu")
    sonics = run.sensors.by_field("u")
    if imu_cfg is None or not sonics:
        return run
    if isinstance(imu_cfg, dict):
        imu_cfg = IMUInfo(**imu_cfg)

    def _triad(fields):
        found = [run.sensors.by_field(f) for f in fields]
        if all(found):
            return [f[0] for f in found]
        if any(found):
            missing = [name for name, f in zip(fields, found) if not f]
            msg = f"IMU triad incomplete (missing {', '.join(missing)}); ignored"
            warnings.warn(msg)
            run.warnings.append(msg)
        return None

    acc_s, gyro_s, att_s = _triad(_IMU_ACC), _triad(_IMU_GYRO), _triad(_IMU_ATT)
    if not imu_cfg.useVendorAttitude and acc_s and gyro_s:
        att_s = None
    if att_s is None and not (acc_s and gyro_s):
        msg = ("imu configured but the IMU columns give neither a fused attitude "
               "nor accel + gyro; motion correction skipped")
        warnings.warn(msg)
        run.warnings.append(msg)
        return run

    ref_table = sonics[0].table
    n = run.tables[ref_table].sizes[TIME_HF]
    fs = float(run.tables[ref_table].attrs.get("scan_hz", 20))

    def _stack(triple, signs, convert):
        cols = []
        for s, sign in zip(triple, signs):
            x = np.asarray(run.hf(s), dtype=float)
            if len(x) != n:
                raise ValueError(f"IMU column {s.column} has {len(x)} samples, "
                                 f"sonic table {ref_table} has {n}")
            cols.append(float(sign) * convert(x))
        return np.column_stack(cols)

    ident = lambda x: x
    try:
        acc = _stack(acc_s, imu_cfg.accelSigns,
                     (lambda x: x * G) if imu_cfg.accelUnits == "g" else ident) \
            if acc_s else None
        gyro = _stack(gyro_s, imu_cfg.gyroSigns,
                      np.deg2rad if imu_cfg.gyroUnits == "deg/s" else ident) \
            if gyro_s else None
        att = _stack(att_s, imu_cfg.attitudeSigns,
                     np.deg2rad if imu_cfg.attitudeUnits == "deg" else ident) \
            if att_s else None

        # residual IMU-to-platform mounting rotation
        if imu_cfg.mountRoll or imu_cfg.mountPitch or imu_cfg.mountYaw:
            Rm = euler_T(*np.deg2rad([imu_cfg.mountRoll, imu_cfg.mountPitch,
                                      imu_cfg.mountYaw]))
            if acc is not None:
                acc = acc @ Rm.T
            if gyro is not None:
                gyro = gyro @ Rm.T
            if att is not None:
                T_p = euler_T(att[:, 0], att[:, 1], att[:, 2]) @ Rm.T
                att = np.column_stack(euler_from_T(T_p))

        res = None
        for s in sonics:
            sv, sw = run.sensors.at("v", s.height), run.sensors.at("w", s.height)
            ds = run.tables[s.table]
            if sv is None or sw is None or ds.sizes[TIME_HF] != n:
                msg = f"motion correction skipped at {s.height}m: no aligned u/v/w"
                warnings.warn(msg)
                run.warnings.append(msg)
                continue
            level = sonic_for(info.get("sonics"), s.height)
            r = (level.leverArm if level is not None and level.leverArm is not None
                 else imu_cfg.leverArm)
            params = MotionParams(fs=fs, lever_arm=tuple(float(v) for v in r),
                                  Tcf=imu_cfg.Tcf, Ta=imu_cfg.Ta,
                                  yaw_handling=imu_cfg.yawHandling)
            uvw = np.column_stack([run.hf(s), run.hf(sv), run.hf(sw)])
            res = correct_wind(uvw, params, acc=acc, gyro=gyro, attitude=att)
            ds[s.column].values[:] = res.uvw[:, 0]
            ds[sv.column].values[:] = res.uvw[:, 1]
            ds[sw.column].values[:] = res.uvw[:, 2]
            log.info(f"  Motion-corrected sonic @ {s.height}m "
                     f"(roll std {np.degrees(np.nanstd(res.roll)):.2f} deg, "
                     f"w_plat std {np.nanstd(res.v_platform[:, 2]):.3f} m/s)")
    except ValueError as exc:
        msg = f"motion correction skipped: {exc}"
        warnings.warn(msg)
        run.warnings.append(msg)
        return run

    if res is not None:
        run.motion = xr.Dataset(
            {"roll": (TIME_HF, np.degrees(res.roll)),
             "pitch": (TIME_HF, np.degrees(res.pitch)),
             "yaw": (TIME_HF, np.degrees(res.yaw)),
             "u_platform": (TIME_HF, res.v_platform[:, 0]),
             "v_platform": (TIME_HF, res.v_platform[:, 1]),
             "w_platform": (TIME_HF, res.v_platform[:, 2])},
            coords={TIME_HF: run.tables[ref_table][TIME_HF].values},
            attrs={"Tcf_s": imu_cfg.Tcf, "Ta_s": imu_cfg.Ta,
                   "yaw_handling": imu_cfg.yawHandling,
                   "attitude_source": "vendor" if att is not None else "complementary",
                   **{f"imu_nan_frac_{k}": v for k, v in res.imu_nan_frac.items()}})
    return run


def average(run: Run) -> Run:
    """Block means of every table at ``avgPer`` -> ``run.periods[<table>]``
    (propeller directions vector-averaged)."""
    info = run.site
    for name in run.table_names:
        ds = run.tables.get(name)
        if ds is None:
            continue
        labels = run.headers[name][0]
        log.info(f"Averaging {name}")
        mat = _table_matrix(ds, labels)
        pairs = []
        for s_dir in run.sensors.by_field("birdDir"):
            s_spd = run.sensors.at("birdSpd", s_dir.height)
            if s_dir.table == name and s_spd is not None:
                pairs.append((labels.index(s_dir.column), labels.index(s_spd.column)))
        t_end, means = block_average(mat[:, 0], mat, info["avgPer"], vector_cols=pairs)
        per = xr.Dataset({lab: (TIME, means[:, j]) for j, lab in enumerate(labels) if j},
                         coords={TIME: to_datetime64(t_end)})
        per.attrs["time_label"] = labels[0]
        run.periods[name] = per
    return run


def wind(run: Run) -> Run:
    """Period-mean wind direction/speed per sonic and the tower-shadow flag
    -> ``run.wind`` on ``(time, height)``."""
    info = run.site
    sonics = run.sensors.by_field("u")
    if not sonics:
        return run
    tower = float(info.get("tower", 0))
    env = float(info["windDirectionTest"]["envelopeSize"])
    heights = [s.height for s in sonics]
    ref = run.periods[sonics[0].table]
    n = ref.sizes[TIME]
    d = np.full((n, len(heights)), np.nan)
    sp = np.full((n, len(heights)), np.nan)
    fl = np.full((n, len(heights)), np.nan)
    lo = np.full(len(heights), np.nan)
    hi = np.full(len(heights), np.nan)
    for i, s in enumerate(sonics):
        try:
            v = run.sensors.at("v", s.height)
            per = run.periods[s.table]
            direction, speed = wind_direction_speed(per[s.column].values, per[v.column].values,
                                                    s.orientation or 0.0, s.manufacturer or 1)
            flag, lo[i], hi[i] = shadow_flag(direction, tower, s.orientation or 0.0, env)
            d[:, i], sp[:, i], fl[:, i] = direction, speed, flag
        except Exception as exc:
            msg = f"Unable to find wind stats at {s.height}m: {exc}"
            warnings.warn(msg)
            run.warnings.append(msg)
    run.wind = xr.Dataset({"direction": ((TIME, HEIGHT), d), "speed": ((TIME, HEIGHT), sp),
                           "shadow_flag": ((TIME, HEIGHT), fl),
                           "sector_min": (HEIGHT, lo), "sector_max": (HEIGHT, hi)},
                          coords={TIME: ref[TIME].values, HEIGHT: heights})
    return run


def rotate(run: Run) -> Run:
    """Planar-fit (local, or global from ``run.pf_table``) and yaw rotation of
    every sonic -> ``run.rotation`` on ``(time_hf, height, component)`` with
    the period means on ``(time, height, component)``."""
    info = run.site
    sonics = run.sensors.by_field("u")
    if not sonics:
        return run
    t = run.time_hf_datenum
    n = len(t)
    global_pf = info["PF"]["globalCalculation"] == "global"
    pf_table = run.pf_table if global_pf else None
    if global_pf and pf_table is None:
        msg = "PFinfo failed to load. Check/re-run find_global_pf."
        warnings.warn(msg)
        run.warnings.append(msg)
    if pf_table is not None:
        # MATLAB sonicRotation.m lines 37-40: the PFinfo infoString columns travel
        # with the output (dataInfo) so save_data preserves them.
        for col in pf_table.info_string or []:
            run.notes.append(list(col))

    heights = [s.height for s in sonics]
    w_hts = [float(h) for h in run.wind[HEIGHT].values] if run.wind is not None else []
    n_per = run.period_time().shape[0]
    series = []
    for s in sonics:
        sv, sw = run.sensors.at("v", s.height), run.sensors.at("w", s.height)
        if s.height in w_hts:
            j = w_hts.index(s.height)
            direction = run.wind["direction"].values[:, j]
            wind_flag = np.nan_to_num(run.wind["shadow_flag"].values[:, j]).astype(bool)
        else:
            direction, wind_flag = np.zeros(n_per), np.zeros(n_per, dtype=bool)
        good = ~(run.flag(s, "spike", n_per) | run.flag(sv, "spike", n_per)
                 | run.flag(sw, "spike", n_per) | wind_flag)
        series.append(SonicSeries(s.height, run.hf(s), run.hf(sv), run.hf(sw), direction, good,
                                  run.period_mean(s), run.period_mean(sv), run.period_mean(sw)))

    res = rotate_sonics(series, t, info["avgPer"], pf_table)
    k = len(heights)
    ds = xr.Dataset(coords={TIME_HF: run.time_hf, HEIGHT: heights, COMPONENT: ["u", "v", "w"],
                            TIME: run.period_time()})
    ds["rotated"] = ((TIME_HF, HEIGHT, COMPONENT), res.rotated.reshape(n, k, 3))
    ds["pf"] = ((TIME_HF, HEIGHT, COMPONENT), res.pf_only.reshape(n, k, 3))
    ds["rotated_mean"] = ((TIME, HEIGHT, COMPONENT), res.averaged("rotated").reshape(n_per, k, 3))
    ds["pf_mean"] = ((TIME, HEIGHT, COMPONENT), res.averaged("pf_only").reshape(n_per, k, 3))
    for i, s in enumerate(sonics):
        if s.height in res.skipped:
            msg = res.skipped[s.height]
            if pf_table is not None:            # the local case is logged by rotate_sonics
                warnings.warn(f"Sonic rotation failed at {s.height}m: {msg}")
                run.warnings.append(f"Sonic rotation failed at {s.height}m: {msg}")
            continue
        fit = res.fits.get(s.height)
        if fit is not None:
            col = i + 1
            while len(run.notes) <= col:
                run.notes.append([])
            run.notes[col].append(f"{s.height}m {fit.describe()}")
            for key, val in (("b0", fit.b0), ("b1", fit.b1), ("b2", fit.b2),
                             ("pitch_deg", fit.pitch_deg), ("roll_deg", fit.roll_deg)):
                if key not in ds:
                    ds[key] = (HEIGHT, np.full(k, np.nan))
                ds[key].values[i] = val
    run.rotation = ds
    return run


def flux(run: Run) -> Run:
    """Flux products per period and sonic -> ``run.products`` (one Dataset per
    table on ``(time, height)``) and ``run.raw`` (high-frequency products)."""
    from .flux.engine import FluxOptions, compute_period
    from .flux.levels import build_level
    from .flux.reference import reference_state
    from .flux.tables import FluxTables
    from .model import products_from_legacy, raw_from_legacy

    log.info("Computing Fluxes")
    info = run.site
    sonics = run.sensors.by_field("u")
    if not sonics:
        return run
    save_raw = bool(info.get("saveRawConditionedData", False))
    raw = {} if save_raw else None

    t = run.time_hf_datenum
    ref = reference_state(run)
    if save_raw and ref.raw_P is not None:
        raw["P"] = ref.raw_P
    heights = [s.height for s in sonics]
    N = n_periods(t, info["avgPer"], whole_days=False)
    bp = period_bounds(len(t), N)
    has_fw = run.sensors.has("fw")

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
        angle=angle,
        downslope_aspect=downslope_aspect,
        stability_source=info.get("stabilitySource", "hoegstroem1988"),
    )

    tables = FluxTables(N, heights, has_fw)
    temperature_cols = []
    humidity_cols = None
    cp = ref.cp

    for ii, s in enumerate(sonics):
        height = s.height
        try:
            lev = build_level(run, ii, ref, N, slope_axis)
            lev_opts = replace(opts, scan_freq=lev.scan_hz)
            if lev.specific_hum_cols:
                if humidity_cols is None:
                    humidity_cols = ([block_last(t, bp)], ["time"])
                for hdr, col in lev.specific_hum_cols:
                    humidity_cols[0].append(np.asarray(col)[:N])
                    humidity_cols[1].append(hdr)
            temperature_cols.extend(lev.derived_T_cols)
            if save_raw:
                if ii == 0:
                    _init_raw(raw, t, len(sonics), run)
                _fill_raw_level(raw, lev, ii)
            for jj in range(N):
                s0, s1 = int(bp[jj]), int(bp[jj + 1])
                if s0 >= s1:
                    continue
                if ii == 0:
                    tables.set_time(jj, t[s1 - 1])
                    tables["sensible_heat"].set(jj, None, "rho_air_ref",
                                                ref.rho[jj] if jj < len(ref.rho) else np.nan)
                    tables["sensible_heat"].set(jj, None, "cp_ref",
                                                cp[jj] if jj < len(cp) else np.nan)
                res = compute_period(lev, ref, lev_opts, jj, s0, s1, t)
                for tname, vals in res.values.items():
                    tables[tname].set_many(jj, height, vals)
                if save_raw:
                    _fill_raw_period(raw, lev, ii, s0, s1, res.samples, t)
        except Exception as exc:
            warnings.warn(f"Flux computation failed at height {height}m: {exc}")
            traceback.print_exc()

    # legacy-shaped matrices + headers -> Datasets (the same converter the
    # boundary uses; the CSV writer and the HF ancillaries read the legacy form)
    legacy = tables.store({}, store_extra=bool(info.get("storeExtraStats", True)))
    if temperature_cols:
        timestamps = block_last(t, period_bounds(len(t), n_periods(t, info["avgPer"])))
        mat = np.column_stack([timestamps] + [c for _, c in temperature_cols])
        hdr = ["time"] + [h for h, _ in temperature_cols]
        keep = np.any(~np.isnan(mat), axis=0)
        legacy["temperature"] = mat[:, keep]
        legacy["temperatureHeader"] = [h for h, k in zip(hdr, keep) if k]
    if humidity_cols is not None:
        legacy["humidity"] = np.column_stack(humidity_cols[0])
        legacy["humidityHeader"] = list(humidity_cols[1])
    run.products = products_from_legacy(legacy, heights)
    if raw is not None:
        run.raw = raw_from_legacy(raw, run.time_hf)
    return run


def _init_raw(raw, t, num_sonics: int, run: Run) -> None:
    n = len(t)
    for key in ("uPF", "vPF", "wPF", "u_tilt", "v_tilt", "w_tilt", "WD", "spd", "sonTs", "Theta_v_son"):
        raw[key] = np.full((n, num_sonics), np.nan)
    raw["t"] = t
    raw["z"] = np.full(num_sonics, np.nan)
    for field_name in ("irgaH2O", "LiH2O"):
        if run.sensors.has(field_name):
            nh = len(run.sensors.by_field(field_name))
            raw["rhov"] = np.full((n, nh), np.nan)
            raw["rhovPrime"] = np.full((n, nh), np.nan)
            raw["rhovextenalPrime"] = np.full((n, nh), np.nan)
            raw["z_h2o"] = np.array(run.sensors.heights(field_name), dtype=float)
            break
    for field_name in ("irgaCO2", "LiCO2"):
        if run.sensors.has(field_name):
            nc = len(run.sensors.by_field(field_name))
            raw["rhoCO2"] = np.full((n, nc), np.nan)
            raw["rhoCO2Prime"] = np.full((n, nc), np.nan)
            raw["rhoCO2extenalPrime"] = np.full((n, nc), np.nan)
            raw["z_co2"] = np.array(run.sensors.heights(field_name), dtype=float)
            break


def _fill_raw_level(raw, lev, ii: int) -> None:
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


def _fill_raw_period(raw, lev, ii: int, s0: int, s1: int, samples, t) -> None:
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
