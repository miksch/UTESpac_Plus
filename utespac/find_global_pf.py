"""findGlobalPF – compute or load multi-sector global planar-fit coefficients."""

import os
import pickle
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from .get_data import get_data
from .pf_coefficients import pf_coefficients
from .stp_dn import stp_dn

_MATLAB_EPOCH = 719529  # MATLAB datenum for 1970-01-01


def _matlab_to_datetime(serial):
    """Convert a MATLAB serial date float to a Python datetime."""
    return datetime(1970, 1, 1) + timedelta(days=float(serial) - _MATLAB_EPOCH)


def _compute_direction(u, v, bearing, manufact):
    """Compute wind direction [0, 360) from u/v, matching MATLAB findGlobalPF.

    Manufacturer corrections:
      1 (Campbell)  : u, v as-is
      0 (RMYoung)   : swap u↔v, negate new v
      2 (Gill)      : negate both u and v
    """
    if manufact == 0:
        u_dir, v_dir = v, u * -1
    elif manufact == 2:
        u_dir, v_dir = -u, -v
    else:
        u_dir, v_dir = u, v
    return np.mod(np.arctan2(-v_dir, u_dir) * 180.0 / np.pi + bearing, 360.0)


def _show_selection_figure(z, site_name, direction, t_dn, u_dn, v_dn, w_dn, spd_dn):
    """Create the two-panel figure (histogram + time series) and return (fig, ax1, ax2)."""
    try:
        import matplotlib
        matplotlib.use("TkAgg")
    except Exception:
        pass
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f"{site_name}  {z} m", fontsize=13)

    valid = direction[~np.isnan(direction)]
    ax1.hist(valid, bins=100, range=(0, 360), color="steelblue", edgecolor="none")
    ax1.set_xlim(0, 360)
    ax1.set_xlabel("Direction (°)")
    ax1.set_ylabel("Occurrences")
    ax1.set_title("Wind direction — enter bin boundaries below")

    try:
        dates = [_matlab_to_datetime(d) for d in t_dn]
    except Exception:
        dates = list(range(len(t_dn)))

    ax2.plot(dates, u_dn,   "b.", markersize=3, label="u")
    ax2.plot(dates, v_dn,   "g.", markersize=3, label="v")
    ax2.plot(dates, w_dn,   "r.", markersize=3, label="w")
    ax2.plot(dates, spd_dn, "c.", markersize=3, label="spd")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_title(f"{site_name}  {z} m — enter date barriers after selecting bins")
    if dates and isinstance(dates[0], datetime):
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        fig.autofmt_xdate(rotation=30)
    ax2.autoscale(axis="x", tight=True)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    return fig, ax1, ax2


def _show_dayby_day_figure(num_bins: int, bins: List[float]):
    """Create the day-by-day figure (one subplot per bin) and return (fig, axes)."""
    try:
        import matplotlib
        matplotlib.use("TkAgg")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    ncols = max(1, int(np.ceil(num_bins / 2)))
    nrows = 2 if num_bins > 1 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 5 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for m in range(num_bins):
        ax = axes_flat[m]
        if m < num_bins - 1:
            ax.set_title(f"{bins[m]:.4g}° → {bins[m+1]:.4g}°", fontsize=9)
        else:
            ax.set_title(f"{bins[m]:.4g}° → {bins[0]:.4g}°", fontsize=9)
        ax.set_xlabel("Day index")
        ax.set_ylabel("Angle (°)")

    for m in range(num_bins, len(axes_flat)):
        axes_flat[m].set_visible(False)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    return fig, axes_flat[:num_bins]


def find_global_pf(info: Dict, template: Dict, sensor_info: Dict) -> Dict:
    """Compute (or load) global planar-fit coefficients for each sonic height.

    Reproduces MATLAB findGlobalPF behaviour:

    * Flagged rows (NaN + spike + wind-direction + sonic-diagnostic) are
      **removed** from the arrays before stpDn, matching MATLAB's
      ``data(~totalFlag, col)`` boolean indexing.
    * Date barriers split the data into windows; within each window the user
      chooses to use all data at once or approve days one-by-one (day-by-day
      mode shows a cumulative pitch/roll figure per direction bin).
    * The result is saved to ``<siteFolder>/PFinfo.pkl``.
    """
    pf_path = os.path.join(info["rootFolder"], info["siteFolder"], "PFinfo.pkl")

    if not info["PF"]["recalculateGlobalCoefficients"] and os.path.isfile(pf_path):
        with open(pf_path, "rb") as fh:
            pf_info = pickle.load(fh)
        if "infoString" not in pf_info:
            pf_info["infoString"] = _build_info_string(pf_info)
        _display_and_confirm_pf(pf_info)
        return pf_info

    print("Finding Global Planar Fit Coefficients (b0, b1, b2)")

    # ── load LPF-averaged data ────────────────────────────────────────────────
    all_sites = sorted(d for d in os.listdir(info["rootFolder"]) if d.startswith("site"))
    site_num  = all_sites.index(info["siteFolder"]) + 1
    site_name = info["siteFolder"][4:]

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
        # ── skip prompt (MATLAB: skipFlag = input(...)) ───────────────────────
        skip = input(f"\nSkip {z} m? (1=process, 0=skip): ").strip()
        if skip == "0":
            if os.path.isfile(pf_path):
                with open(pf_path, "rb") as fh:
                    old = pickle.load(fh)
                cm_key = f"cm_{round(z * 100)}"
                if cm_key in old:
                    pf_info[cm_key] = old[cm_key]
                    continue
            print("  No existing PFinfo.pkl found — cannot skip this level.")

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
            print(f"  Could not find sonic data at {z} m — skipping.")
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

        # ── show overview figure ──────────────────────────────────────────────
        try:
            fig, ax1, ax2 = _show_selection_figure(
                z, site_name, direction, t_dn, u_dn, v_dn, w_dn, spd_dn)
            has_figure = True
        except Exception as exc:
            print(f"  (Could not show figure: {exc})")
            has_figure = False

        # ── direction bin selection (MATLAB lines 159-173) ───────────────────
        print(f"\nSelect direction bin boundaries for {z} m.")
        print("  Enter one value per prompt; press Enter with no value when done.")
        bins: List[float] = []
        y_lim = ax1.get_ylim() if has_figure else (0, 1)
        while True:
            val = input("  Bin boundary (° 0–360, or Enter to finish): ").strip()
            if not val:
                break
            try:
                b = float(val)
            except ValueError:
                continue
            if 0 <= b <= 360:
                bins.append(b)
                if has_figure:
                    import matplotlib.pyplot as plt
                    ax1.plot([b, b], [y_lim[0], y_lim[1]], "g--", linewidth=2)
                    plt.draw(); plt.pause(0.05)
        bins = sorted(set(bins))

        # ── date barrier selection (MATLAB lines 179-191) ────────────────────
        t_valid   = t_dn[~np.isnan(t_dn)]
        all_days  = np.unique(np.floor(t_valid))
        print(f"\nData spans {len(all_days)} day(s): "
              f"{_matlab_to_datetime(all_days[0]).strftime('%Y-%m-%d')} to "
              f"{_matlab_to_datetime(all_days[-1]).strftime('%Y-%m-%d')}")
        print("  Optionally split data by date for different PF calculations.")
        print("  Enter MATLAB serial dates; press Enter to skip.")
        date_barriers: List[float] = []
        if has_figure:
            import matplotlib.pyplot as plt
            ax2_ylim = ax2.get_ylim()
        while True:
            val = input("  Date barrier (MATLAB serial date, or Enter to finish): ").strip()
            if not val:
                break
            try:
                db = float(val)
            except ValueError:
                continue
            # MATLAB: dateBarriers(dateBarriers<t(1)|dateBarriers>t(end)) = []
            if t_valid[0] <= db <= t_valid[-1]:
                date_barriers.append(db)
                if has_figure:
                    dt = _matlab_to_datetime(db)
                    ax2.plot([dt, dt], [ax2_ylim[0], ax2_ylim[1]], "g--", linewidth=2)
                    plt.draw(); plt.pause(0.05)
        date_barriers = sorted(set(date_barriers))

        if has_figure:
            import matplotlib.pyplot as plt
            plt.close(fig)

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
            print(f"\nFinding PF for {d0_str} to {d1_str}")

            # data in this window (MATLAB: local_u = u(t<=localDays(end)+1), lower-bound fixed)
            w_mask = (t_dn >= local_days[0]) & (t_dn <= local_days[-1] + 1)
            local_u   = u_dn[w_mask]
            local_v   = v_dn[w_mask]
            local_w   = w_dn[w_mask]
            local_dir = direction[w_mask]
            local_t   = t_dn[w_mask]

            day_key = f"day_{int(local_days[0])}to{int(local_days[-1])}"

            # MATLAB: useAllDatesFlag = input(...)
            use_all = input("  Use all data in range (1) or select day by day (0)?: ").strip()

            if use_all != "0":
                # ── use-all path ─────────────────────────────────────────────
                coef = pf_coefficients(
                    np.column_stack([local_u, local_v, local_w, local_dir]),
                    bins,
                )
                pf_info[cm_key][day_key] = coef
            else:
                # ── day-by-day path (MATLAB lines 232-344) ────────────────────
                num_bins = len(bins)
                try:
                    dbd_fig, dbd_axes = _show_dayby_day_figure(num_bins, bins)
                    has_dbd_fig = True
                except Exception as exc:
                    print(f"  (Could not show day-by-day figure: {exc})")
                    has_dbd_fig = False

                cum_rows: List[int] = []   # indices into local_* arrays
                cum_coef = None
                pitch_cum: List[List[float]] = [[] for _ in range(num_bins)]
                roll_cum:  List[List[float]] = [[] for _ in range(num_bins)]

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

                    # plot daily pitch/roll (MATLAB: 'x' for pitch, 'd' for roll)
                    if has_dbd_fig:
                        import matplotlib.pyplot as plt
                        for m, ax in enumerate(dbd_axes):
                            bin_key = _bin_key(bins, m)
                            if bin_key in day_coef:
                                b1d = day_coef[bin_key][1]
                                b2d = day_coef[bin_key][2]
                                p_d = np.degrees(np.arcsin(-b1d / np.sqrt(1 + b1d**2)))
                                r_d = np.degrees(np.arcsin( b2d / np.sqrt(1 + b2d**2)))
                                ax.plot(k + 1, p_d, "bx", markersize=8)
                                ax.plot(k + 1, r_d, "gd", markersize=8)
                                ax.set_xlim(0, len(local_days) + 1)
                        plt.draw(); plt.pause(0.05)

                    use_day = input(
                        f"  Use {_matlab_to_datetime(day).strftime('%Y-%m-%d')} "
                        "for cumulative calculation (1=yes, 0=no)?: "
                    ).strip()

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

                        # plot cumulative pitch/roll (MATLAB: 'b.-' pitch, 'g.-' roll)
                        if has_dbd_fig:
                            import matplotlib.pyplot as plt
                            for m, ax in enumerate(dbd_axes):
                                bin_key = _bin_key(bins, m)
                                if bin_key in cum_coef:
                                    b1c = cum_coef[bin_key][1]
                                    b2c = cum_coef[bin_key][2]
                                    p_c = np.degrees(np.arcsin(-b1c / np.sqrt(1 + b1c**2)))
                                    r_c = np.degrees(np.arcsin( b2c / np.sqrt(1 + b2c**2)))
                                    pitch_cum[m].append(p_c)
                                    roll_cum[m].append(r_c)
                                    xs = list(range(1, len(pitch_cum[m]) + 1))
                                    # re-draw cumulative lines from scratch each update
                                    for ln in [l for l in ax.lines
                                               if getattr(l, "_cum_line", False)]:
                                        ln.remove()
                                    lp, = ax.plot(xs, pitch_cum[m], "b.-",
                                                  label="Pitch" if len(pitch_cum[m]) == 1 else "")
                                    lr, = ax.plot(xs, roll_cum[m],  "g.-",
                                                  label="Roll"  if len(roll_cum[m])  == 1 else "")
                                    lp._cum_line = True
                                    lr._cum_line = True
                                    if len(pitch_cum[m]) == 1:
                                        ax.legend(fontsize=8)
                                    ax.set_xlim(0, len(local_days) + 1)
                            plt.draw(); plt.pause(0.05)

                if has_dbd_fig:
                    import matplotlib.pyplot as plt
                    plt.close(dbd_fig)

                if cum_coef is not None:
                    pf_info[cm_key][day_key] = cum_coef
                else:
                    print(f"  No days accepted for window {day_key} — skipping.")

    # ── build infoString and save ─────────────────────────────────────────────
    pf_info["infoString"] = _build_info_string(pf_info)
    with open(pf_path, "wb") as fh:
        pickle.dump(pf_info, fh)
    print(f"\nPFinfo saved to {pf_path}")

    _display_and_confirm_pf(pf_info)
    return pf_info


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


def _display_and_confirm_pf(pf_info: Dict) -> None:
    """Display PF coefficients and ask the user to confirm before analysis."""
    print("\nVerify Global Planar Fit Coefficients\n")
    for cm_key, date_dict in pf_info.items():
        if cm_key == "infoString":
            continue
        print(f"  Sonic height: {cm_key}")
        for day_key, coef_dict in date_dict.items():
            try:
                d0 = int(day_key.split("_")[1].split("to")[0])
                d1 = int(day_key.split("to")[1])
                print(f"    {_matlab_to_datetime(d0).strftime('%Y-%m-%d')} → "
                      f"{_matlab_to_datetime(d1).strftime('%Y-%m-%d')}")
            except Exception:
                print(f"    {day_key}")
            for bin_key, coef in coef_dict.items():
                b0, b1, b2 = coef[0], coef[1], coef[2]
                pitch = np.degrees(np.arcsin(-b1 / np.sqrt(1 + b1**2)))
                roll  = np.degrees(np.arcsin( b2 / np.sqrt(1 + b2**2)))
                print(f"      {bin_key}:  b0={b0:.3g}  b1={b1:.3g}  b2={b2:.3g}"
                      f"  pitch={pitch:.3g}°  roll={roll:.3g}°")
        print()

    ok = input("Is this correct? (1=yes, 0=no): ").strip()
    if ok != "1":
        raise RuntimeError("PF coefficients rejected by user. Check inputs and rerun.")

    ok2 = input("Okay to begin analysis? (1=yes, 0=no): ").strip()
    if ok2 != "1":
        raise RuntimeError("Program stopped by user.")
