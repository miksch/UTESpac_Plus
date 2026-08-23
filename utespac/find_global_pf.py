"""findGlobalPF – compute or load multi-sector global planar-fit coefficients."""

import logging
import os
import pickle
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from .get_data import get_data
from .pf_coefficients import pf_coefficients
from .site_config import list_sites
from .stp_dn import stp_dn
from .wind_stats import wind_direction_speed

log = logging.getLogger("utespac")

_MATLAB_EPOCH = 719529  # MATLAB datenum for 1970-01-01


def _matlab_to_datetime(serial):
    """Convert a MATLAB serial date float to a Python datetime."""
    return datetime(1970, 1, 1) + timedelta(days=float(serial) - _MATLAB_EPOCH)


def _compute_direction(u, v, bearing, manufact):
    """Wind direction [0, 360) from u/v (``wind_stats.wind_direction_speed``)."""
    return wind_direction_speed(u, v, bearing, manufact)[0]


def find_global_pf(info: Dict, template: Dict, sensor_info: Dict,
                   prompter=None) -> Dict:
    """Compute (or load) global planar-fit coefficients for each sonic height.

    Reproduces MATLAB findGlobalPF behaviour:

    * Flagged rows (NaN + spike + wind-direction + sonic-diagnostic) are
      **removed** from the arrays before stpDn, matching MATLAB's
      ``data(~totalFlag, col)`` boolean indexing.
    * Date barriers split the data into windows; within each window the
      selection is either all data at once or day-by-day approval (the
      console prompter shows a cumulative pitch/roll figure per bin).
    * The result is saved to ``<siteFolder>/PFinfo.pkl``.

    Every decision (skip a height, bin boundaries, date barriers, use-all vs
    day-by-day, accepted days, final confirmation) comes from ``prompter``
    (:mod:`utespac.prompts`); ``None`` means the scripted default: one
    0–360 sector, no barriers, all data, confirmed.
    """
    if prompter is None:
        from .prompts import ScriptedPFSelection
        prompter = ScriptedPFSelection()
    from .prompts import PFHeightContext

    site_path = os.path.join(info["rootFolder"], info["siteFolder"])

    if not info["PF"]["recalculateGlobalCoefficients"]:
        pf_info = load_pf_info(site_path)
        if pf_info is not None:
            log.info("Reusing planar-fit coefficients from %s\n%s", site_path, format_pf_info(pf_info))
            prompter.confirm(pf_info)
            return pf_info

    log.info("Finding Global Planar Fit Coefficients (b0, b1, b2)")

    # ── load LPF-averaged data ────────────────────────────────────────────────
    all_sites = list_sites(info["rootFolder"])
    site_num  = all_sites.index(info["siteFolder"]) + 1
    site_name = info["siteFolder"].removeprefix("site")

    data = get_data(
        info["rootFolder"],
        site=site_num,
        avg_per=info["avgPer"],
        qualifier="LPF",
        rows=0,
    )

    # ── determine sonic heights ───────────────────────────────────────────────
    hdr    = data.get("spdAndDirHeader", [])
    z_vals = []
    for h in hdr:
        nums = re.findall(r"[\d.]+", str(h))
        if nums:
            z_vals.append(float(nums[0]))
    ascending = info.get("ascending", True)
    z_vals = sorted(set(z_vals[1:]), reverse=not ascending)

    table_names = data.get("tableNames", [])
    pf_info: Dict = {}

    for ii, z in enumerate(z_vals):
        # ── skip (MATLAB: skipFlag = input(...)) ─────────────────────────────
        if prompter.skip_height(z):
            old = load_pf_info(site_path)
            cm_key = f"cm_{round(z * 100)}"
            if old is not None and cm_key in old:
                pf_info[cm_key] = old[cm_key]
                continue
            log.warning("No existing PFinfo found — cannot skip %s m.", z)

        varname_u    = template["u"].replace("*", str(z))
        varname_v    = template["v"].replace("*", str(z))
        varname_w    = template["w"].replace("*", str(z))
        varname_diag = template.get("sonDiagnostic", "diagnostic_*").replace("*", str(z))

        # ── find table and columns ────────────────────────────────────────────
        local_table = u_col = v_col = w_col = None
        for tname in table_names:
            hdr_key = f"{tname}Header"
            if hdr_key not in data:
                continue
            names = data[hdr_key][0] if isinstance(data[hdr_key][0], list) else data[hdr_key]
            if varname_u in names:
                local_table = tname
                u_col = names.index(varname_u)
                v_col = names.index(varname_v) if varname_v in names else None
                w_col = names.index(varname_w) if varname_w in names else None
                break

        if local_table is None or u_col is None:
            log.warning("Could not find sonic data at %s m — skipping.", z)
            continue

        n_periods = data[local_table].shape[0]

        # ── assemble total flag (MATLAB lines 73–93) ─────────────────────────
        def _flag_col(flag_key, col):
            arr = data.get(flag_key)
            if arr is None:
                return np.zeros(n_periods, dtype=bool)
            a = np.asarray(arr)
            return a[:, col].astype(bool) if a.ndim == 2 and col < a.shape[1] \
                   else np.zeros(n_periods, dtype=bool)

        nan_flag   = (_flag_col(f"{local_table}NanFlag",   u_col) |
                      _flag_col(f"{local_table}NanFlag",   v_col if v_col is not None else u_col) |
                      _flag_col(f"{local_table}NanFlag",   w_col if w_col is not None else u_col))
        spike_flag = (_flag_col(f"{local_table}SpikeFlag", u_col) |
                      _flag_col(f"{local_table}SpikeFlag", v_col if v_col is not None else u_col) |
                      _flag_col(f"{local_table}SpikeFlag", w_col if w_col is not None else u_col))

        wind_flag_col = next(
            (j for j, h in enumerate(hdr)
             if str(h).startswith(f"{z}m flag")), None)
        wind_flag = data["spdAndDir"][:, wind_flag_col].astype(bool) \
                    if wind_flag_col is not None else np.zeros(n_periods, dtype=bool)

        tbl_hdr = data.get(f"{local_table}Header", [])
        if tbl_hdr and isinstance(tbl_hdr[0], list):
            tbl_hdr = tbl_hdr[0]
        diag_col = tbl_hdr.index(varname_diag) if varname_diag in tbl_hdr else None
        if diag_col is not None:
            diag_vals = data[local_table][:, diag_col].copy().astype(float)
            diag_vals = np.where(np.isnan(diag_vals), 0.0, diag_vals)
            limit = info.get("diagnosticTest", {}).get("meanSonicDiagnosticLimit", 50)
            diag_flag = (diag_vals >= limit)
        else:
            diag_flag = np.zeros(n_periods, dtype=bool)

        total_flag = nan_flag | spike_flag | wind_flag | diag_flag

        # ── REMOVE flagged rows (MATLAB: u = data(~totalFlag, uCol)) ─────────
        # MATLAB deletes rows before stpDn; Python was NaN-filling.  Removing
        # rows means stpDn blocks are formed from good data only.
        keep = ~total_flag
        u_raw = data[local_table][keep, u_col].copy()
        v_raw = data[local_table][keep, v_col].copy() if v_col is not None \
                else np.full(keep.sum(), np.nan)
        w_raw = data[local_table][keep, w_col].copy() if w_col is not None \
                else np.full(keep.sum(), np.nan)
        t_raw = data[local_table][keep, 0].copy()

        spd_col_idx = next((j for j, h in enumerate(hdr) if h == f"{z}m speed"), None)
        spd_raw = data["spdAndDir"][keep, spd_col_idx].copy() if spd_col_idx is not None \
                  else np.full(keep.sum(), np.nan)

        # ── speed filter (NaN on compact arrays, matching MATLAB lines 104-107) ─
        max_w = info["PF"]["globalCalcMaxWind"]
        min_w = info["PF"]["globalCalcMinWind"]
        spd_mask = (spd_raw > max_w) | (spd_raw < min_w)
        u_raw[spd_mask] = v_raw[spd_mask] = w_raw[spd_mask] = np.nan
        spd_raw[spd_mask] = np.nan

        # ── step-down to PF averaging period (MATLAB lines 117-121) ──────────
        factor = max(1, round(info["PF"]["avgPer"] / info["avgPer"]))
        u_dn   = stp_dn(u_raw.reshape(-1, 1),   factor)[:, 0]
        v_dn   = stp_dn(v_raw.reshape(-1, 1),   factor)[:, 0]
        w_dn   = stp_dn(w_raw.reshape(-1, 1),   factor)[:, 0]
        t_dn   = stp_dn(t_raw.reshape(-1, 1),   factor)[:, 0]
        spd_dn = stp_dn(spd_raw.reshape(-1, 1), factor)[:, 0]

        # ── direction (MATLAB lines 123-137) ─────────────────────────────────
        bearing  = float(sensor_info["u"][ii, 3]) if sensor_info["u"].shape[1] > 3 else 0.0
        manufact = int(sensor_info["u"][ii, 4])   if sensor_info["u"].shape[1] > 4 else 1
        direction = _compute_direction(u_dn, v_dn, bearing, manufact)

        # ── selection (overview figure, bins, barriers) via the prompter ─────
        prompter.begin_height(PFHeightContext(
            z=z, site_name=site_name, direction=direction, t_dn=t_dn,
            u_dn=u_dn, v_dn=v_dn, w_dn=w_dn, spd_dn=spd_dn))
        bins = sorted(set(prompter.bin_boundaries(z)))
        bins = [b for b in bins if 0 <= b <= 360]

        # ── date barriers (MATLAB lines 179-191) ─────────────────────────────
        t_valid   = t_dn[~np.isnan(t_dn)]
        all_days  = np.unique(np.floor(t_valid))
        log.info("%s m: data spans %d day(s), %s to %s", z, len(all_days),
                 _matlab_to_datetime(all_days[0]).strftime("%Y-%m-%d"),
                 _matlab_to_datetime(all_days[-1]).strftime("%Y-%m-%d"))
        # MATLAB: dateBarriers(dateBarriers<t(1)|dateBarriers>t(end)) = []
        date_barriers = sorted({db for db in prompter.date_barriers(z, t_valid)
                                if t_valid[0] <= db <= t_valid[-1]})
        prompter.end_selection(z)

        # ── build date windows (MATLAB lines 197-211) ─────────────────────────
        # MATLAB sentinel: empty → [0,0]; barriers → [allDays(1),…,allDays(end)]
        # Each window's localDays = allDays[i_start : i_end+1] (inclusive at both ends).
        if not date_barriers:
            # single window: all days
            windows = [(all_days[0], all_days[-1], all_days)]
        else:
            edges = np.array([all_days[0]] + date_barriers + [all_days[-1]])
            windows = []
            for j in range(len(edges) - 1):
                d_start, d_end = edges[j], edges[j + 1]
                local_days = all_days[(all_days >= d_start) & (all_days <= d_end)]
                windows.append((d_start, d_end, local_days))

        cm_key = f"cm_{round(z * 100)}"
        pf_info[cm_key] = {}

        # ── iterate through windows (MATLAB lines 204-345) ───────────────────
        for d_start, d_end, local_days in windows:
            d0_str = _matlab_to_datetime(local_days[0]).strftime("%Y-%m-%d")
            d1_str = _matlab_to_datetime(local_days[-1]).strftime("%Y-%m-%d")
            log.info("Finding PF for %s to %s", d0_str, d1_str)

            # data in this window (MATLAB: local_u = u(t<=localDays(end)+1), lower-bound fixed)
            w_mask = (t_dn >= local_days[0]) & (t_dn <= local_days[-1] + 1)
            local_u   = u_dn[w_mask]
            local_v   = v_dn[w_mask]
            local_w   = w_dn[w_mask]
            local_dir = direction[w_mask]
            local_t   = t_dn[w_mask]

            day_key = f"day_{int(local_days[0])}to{int(local_days[-1])}"

            # MATLAB: useAllDatesFlag = input(...)
            if prompter.use_all_dates(z, local_days[0], local_days[-1]):
                # ── use-all path ─────────────────────────────────────────────
                coef = pf_coefficients(
                    np.column_stack([local_u, local_v, local_w, local_dir]),
                    bins,
                )
                pf_info[cm_key][day_key] = coef
            else:
                # ── day-by-day path (MATLAB lines 232-344) ────────────────────
                prompter.begin_day_by_day(z, bins)
                cum_rows: List[int] = []   # indices into local_* arrays
                cum_coef = None

                for k, day in enumerate(local_days):
                    # rows belonging to this day
                    day_mask = (np.floor(local_t) == day)
                    day_idx  = np.where(day_mask)[0].tolist()
                    if not day_idx:
                        continue

                    day_u   = local_u[day_mask]
                    day_v   = local_v[day_mask]
                    day_w   = local_w[day_mask]
                    day_dir = local_dir[day_mask]

                    # daily PF coefficients
                    day_coef = pf_coefficients(
                        np.column_stack([day_u, day_v, day_w, day_dir]),
                        bins,
                    )

                    # daily pitch/roll shown by the prompter (MATLAB: 'x' pitch, 'd' roll)
                    prompter.show_day(z, k, len(local_days), day_coef, bins)
                    use_day = "1" if prompter.use_day(z, day) else "0"

                    if use_day == "1":
                        cum_rows.extend(day_idx)

                        # cumulative data (MATLAB: cumU = local_u(cumRows))
                        cum_u   = local_u[cum_rows]
                        cum_v   = local_v[cum_rows]
                        cum_w   = local_w[cum_rows]
                        cum_dir = local_dir[cum_rows]

                        cum_coef = pf_coefficients(
                            np.column_stack([cum_u, cum_v, cum_w, cum_dir]),
                            bins,
                        )

                        # cumulative pitch/roll shown by the prompter
                        prompter.show_cumulative(z, cum_coef, bins, len(local_days))

                prompter.end_day_by_day(z)

                if cum_coef is not None:
                    pf_info[cm_key][day_key] = cum_coef
                else:
                    log.warning("No days accepted for window %s — skipping.", day_key)

    # ── build infoString and save ─────────────────────────────────────────────
    pf_info["infoString"] = _build_info_string(pf_info)
    save_pf_info(pf_info, site_path, site=info["siteFolder"])
    log.info("PFinfo saved to %s\n%s", site_path, format_pf_info(pf_info))

    prompter.confirm(pf_info)
    return pf_info


def save_pf_info(pf_info: Dict, site_path, site: Optional[str] = None) -> Dict[str, str]:
    """Write ``PFinfo.json`` (labeled :class:`~utespac.pf_info.PFTable`) and the
    legacy ``PFinfo.pkl`` into *site_path*; returns both paths."""
    from .pf_info import PFTable
    pkl_path = os.path.join(site_path, "PFinfo.pkl")
    json_path = os.path.join(site_path, "PFinfo.json")
    with open(pkl_path, "wb") as fh:
        pickle.dump(pf_info, fh)
    PFTable.from_legacy(pf_info, site=site).save(json_path)
    return {"pkl": pkl_path, "json": json_path}


def load_pf_info(site_path) -> Optional[Dict]:
    """Legacy-shaped coefficient dict from ``PFinfo.json`` (preferred) or
    ``PFinfo.pkl``; None when the site has neither."""
    from .pf_info import PFTable
    json_path = os.path.join(site_path, "PFinfo.json")
    pkl_path = os.path.join(site_path, "PFinfo.pkl")
    if os.path.isfile(json_path):
        return PFTable.load(json_path).to_legacy()
    if os.path.isfile(pkl_path):
        with open(pkl_path, "rb") as fh:
            pf_info = pickle.load(fh)
        if "infoString" not in pf_info:
            pf_info["infoString"] = _build_info_string(pf_info)
        return pf_info
    return None


def _build_info_string(pf_info: Dict) -> List[List[str]]:
    """Build the infoString saved in PFinfo (mirrors MATLAB findGlobalPF.m lines 384-388).

    Returns a list of columns (one per sonic height).  Each column is a list of
    strings laid out as:
      [0]  'Global PF Info'
      [1]  'Sonic Height: cm_XXXX m'
      For each date window j (0-indexed):
        [j*3+2]  '<d0> to <d1>'            date range (MATLAB datestr format)
        [j*3+3]  '<last direction bin key>' direction bin (MATLAB overwrites per k)
        [j*3+4]  'b0=..., b1=..., b2=... -- pitch=... deg, roll=... deg'
    """
    columns: List[List[str]] = []
    for cm_key, date_dict in pf_info.items():
        if cm_key == "infoString":
            continue
        col: List[str] = ["Global PF Info", f"Sonic Height: {cm_key} m"]
        for j, (day_key, dir_bins_dict) in enumerate(date_dict.items()):
            # date range — match MATLAB datestr "dd-mmm-yyyy"
            try:
                nums = day_key.split("_")[1].split("to")
                d0_str = _matlab_to_datetime(int(nums[0])).strftime("%d-%b-%Y")
                d1_str = _matlab_to_datetime(int(nums[1])).strftime("%d-%b-%Y")
                col.append(f"{d0_str} to {d1_str}")
            except Exception:
                col.append(day_key)
            # only last direction bin survives (MATLAB overwrites same row for each k)
            last_bin_key = list(dir_bins_dict.keys())[-1]
            coef = np.asarray(dir_bins_dict[last_bin_key], dtype=float)
            b0, b1, b2 = coef[0], coef[1], coef[2]
            col.append(last_bin_key)
            if not any(np.isnan([b0, b1, b2])):
                pitch = np.degrees(np.arcsin(-b1 / np.sqrt(1.0 + b1**2)))
                roll  = np.degrees(np.arcsin( b2 / np.sqrt(1.0 + b2**2)))
                col.append(f"b0={b0:.3g}, b1={b1:.3g}, b2={b2:.3g} -- "
                           f"pitch={pitch:.3g} deg, roll={roll:.3g} deg")
            else:
                col.append("b0=nan, b1=nan, b2=nan")
        columns.append(col)
    return columns


def _bin_key(bins: List[float], m: int) -> str:
    """Return the pf_coefficients dict key for the m-th bin (matches pf_coefficients.py)."""
    if m < len(bins) - 1:
        return f"degrees_{bins[m]}_to_{bins[m+1]}"
    return f"degrees_{bins[m]}_to_{bins[0]}"


def format_pf_info(pf_info: Dict) -> str:
    """Human-readable listing of the planar-fit coefficients per height/window/bin."""
    lines: List[str] = []
    for cm_key, date_dict in pf_info.items():
        if cm_key == "infoString":
            continue
        lines.append(f"  Sonic height: {cm_key}")
        for day_key, coef_dict in date_dict.items():
            try:
                d0 = int(day_key.split("_")[1].split("to")[0])
                d1 = int(day_key.split("to")[1])
                lines.append(f"    {_matlab_to_datetime(d0).strftime('%Y-%m-%d')} → "
                             f"{_matlab_to_datetime(d1).strftime('%Y-%m-%d')}")
            except Exception:
                lines.append(f"    {day_key}")
            for bin_key, coef in coef_dict.items():
                b0, b1, b2 = coef[0], coef[1], coef[2]
                pitch = np.degrees(np.arcsin(-b1 / np.sqrt(1 + b1**2)))
                roll  = np.degrees(np.arcsin( b2 / np.sqrt(1 + b2**2)))
                lines.append(f"      {bin_key}:  b0={b0:.3g}  b1={b1:.3g}  b2={b2:.3g}"
                             f"  pitch={pitch:.3g}°  roll={roll:.3g}°")
    return "\n".join(lines)
