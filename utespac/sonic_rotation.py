"""sonicRotation – apply planar-fit and yaw rotation to sonic wind vectors."""

import re
import warnings
from typing import Dict, List, Optional
import logging
import numpy as np
from .simple_avg import simple_avg
from .pf_coefficients import pf_coefficients

log = logging.getLogger("utespac")


def sonic_rotation(
    output: Dict,
    data: List[Optional[np.ndarray]],
    sensor_info: Dict,
    info: Dict,
    data_info: List,
    table_names: List[str],
    pf_info: Optional[Dict] = None,
) -> tuple:
    """Apply planar-fit (Wilczak et al. 2000) then yaw rotation (v̄ → 0).

    Returns
    -------
    rotated_sonic_data : ndarray, shape (N, 3*numSonics)
    pf_sonic_data      : ndarray, shape (N, 3*numSonics)  (planar-fit only, no yaw)
    output             : dict (updated with rotatedSonic, PFSonic, rotatedSonicHeader)
    data_info          : list (updated with PF coefficient strings)
    """
    if "u" not in sensor_info:
        return np.array([]), np.array([]), output, data_info

    output.setdefault("warnings", [])
    num_sonics = sensor_info["u"].shape[0]
    # Reproduce the MATLAB rotation exactly (wrong GPF coefficient indexing, no
    # b0 removal) only when parity with MATLAB outputs is being tested.
    matlab_compat = bool(info.get("matlabCompat", False))

    t = data[int(sensor_info["u"][0, 0])][:, 0]
    n_pts = len(t)

    rotated_sonic_data = np.full((n_pts, 3 * num_sonics), np.nan)
    pf_sonic_data      = np.full((n_pts, 3 * num_sonics), np.nan)

    output.setdefault("rotatedSonicHeader", [""] * (3 * num_sonics))
    output.setdefault("PFSonicHeader",      [""] * (3 * num_sonics))

    for ii in range(num_sonics):
        try:
            tbl_idx  = int(sensor_info["u"][ii, 0])
            height   = float(sensor_info["u"][ii, 2])

            mask_h = sensor_info["u"][:, 2] == height
            u_col  = int(sensor_info["u"][mask_h, 1][0])
            v_col  = int(sensor_info["v"][sensor_info["v"][:, 2] == height, 1][0])
            w_col  = int(sensor_info["w"][sensor_info["w"][:, 2] == height, 1][0])

            u = data[tbl_idx][:, u_col]
            v = data[tbl_idx][:, v_col]
            w = data[tbl_idx][:, w_col]

            # Average wind direction (from output.spdAndDir)
            dir_header = f"{height}m direction"
            dir_col = output["spdAndDirHeader"].index(dir_header) if dir_header in output["spdAndDirHeader"] else None
            dir_avg = output["spdAndDir"][:, dir_col] if dir_col is not None else np.zeros(len(output["spdAndDir"]))
            t_avg   = output["spdAndDir"][:, 0]

            # ---- PLANAR FIT ----
            wind_pf = np.full((n_pts, 3), np.nan)

            if info["PF"]["globalCalculation"] == "global":
                # MATLAB sonicRotation.m lines 123-128: error when PFinfo or the
                # height-specific field is missing, rather than silently leaving NaN.
                if pf_info is None:
                    raise RuntimeError(
                        "PFinfo failed to load. Check/re-run find_global_pf."
                    )
                cm_key = f"cm_{round(height * 100)}"
                if cm_key not in pf_info:
                    raise RuntimeError(
                        f"PF info at {height} m does not exist. "
                        "Check/re-run find_global_pf."
                    )
                # MATLAB sonicRotation.m lines 37-40: on the first sonic, append
                # all PFinfo.infoString columns to dataInfo so they are preserved
                # in the output pkl via save_data.
                if ii == 0 and "infoString" in pf_info:
                    for info_col in pf_info["infoString"]:
                        data_info.append(list(info_col))
                for date_bin_key, dir_bins_dict in pf_info[cm_key].items():
                    # Parse date range from key, e.g. 'day_730485to730515'
                    nums = re.findall(r"[\d.]+", date_bin_key)
                    if len(nums) >= 2:
                        d0, d1 = float(nums[0]), float(nums[1])
                    else:
                        d0, d1 = t[0], t[-1]

                    date_mask = np.zeros(n_pts, dtype=bool)
                    # d1 is integer midnight; use < d1+1 so the whole last day is covered
                    date_mask[(t >= d0) & (t < d1 + 1)] = True

                    # Expand dir_avg to high-frequency
                    reps = n_pts // len(dir_avg)
                    if reps > 0:
                        dir_hf = np.repeat(dir_avg, reps)[:n_pts]
                    else:
                        dir_hf = np.interp(np.arange(n_pts),
                                           np.linspace(0, n_pts - 1, len(dir_avg)),
                                           dir_avg)

                    for dir_bin_key, coef in dir_bins_dict.items():
                        nums2 = re.findall(r"[\d.]+", dir_bin_key)
                        if len(nums2) >= 2:
                            d_lo, d_hi = float(nums2[0]), float(nums2[1])
                        else:
                            d_lo, d_hi = 0.0, 360.0

                        # MATLAB sonicRotation.m lines 101-103: when d_lo == d_hi
                        # (single-sector key "degrees_0_to_0"), use all directions.
                        # Python's old else-branch gave (dir>0)|(dir<0), excluding
                        # the exact value 0.
                        if d_lo == d_hi:
                            dir_mask = np.ones(n_pts, dtype=bool)
                        elif d_hi > d_lo:
                            dir_mask = (dir_hf > d_lo) & (dir_hf < d_hi)
                        else:
                            dir_mask = (dir_hf > d_lo) | (dir_hf < d_hi)

                        row_mask = date_mask & dir_mask
                        if not row_mask.any():
                            continue

                        # PF_coefficients stores [b0, b1, b2]. MATLAB sonicRotation.m
                        # lines 68-69 read localCoef(1:2), i.e. b0 as the pitch slope
                        # and b1 as the roll slope; that indexing is reproduced only
                        # under matlabCompat for parity testing.
                        if matlab_compat:
                            b0, b1, b2 = 0.0, float(coef[0]), float(coef[1])
                        else:
                            b0, b1, b2 = float(coef[0]), float(coef[1]), float(coef[2])
                        P = _build_pf_matrix(b1, b2)
                        wind_pf[row_mask, :] = _apply_pf(P, b0, u[row_mask], v[row_mask], w[row_mask])

            else:
                # LOCAL planar fit
                spike_flag_name = f"{table_names[tbl_idx]}SpikeFlag"
                nan_flag_name   = f"{table_names[tbl_idx]}NanFlag"
                sf = output.get(spike_flag_name, np.zeros((len(dir_avg), sensor_info["u"].shape[1]), dtype=bool))
                nf = output.get(nan_flag_name,   np.zeros((len(dir_avg), sensor_info["u"].shape[1]), dtype=bool))

                u_sf = sf[:, u_col] if sf.shape[1] > u_col else np.zeros(len(dir_avg), dtype=bool)
                v_sf = sf[:, v_col] if sf.shape[1] > v_col else np.zeros(len(dir_avg), dtype=bool)
                w_sf = sf[:, w_col] if sf.shape[1] > w_col else np.zeros(len(dir_avg), dtype=bool)

                flag_name = f"{height}m flag"
                wind_flag_col = next(
                    (j for j, h in enumerate(output["spdAndDirHeader"]) if h.startswith(flag_name)), None
                )
                wind_flag = output["spdAndDir"][:, wind_flag_col].astype(bool) if wind_flag_col is not None else np.zeros(len(dir_avg), dtype=bool)

                total_flag = u_sf | v_sf | w_sf | wind_flag

                u_bar = output[table_names[tbl_idx]][~total_flag, u_col]
                v_bar = output[table_names[tbl_idx]][~total_flag, v_col]
                w_bar = output[table_names[tbl_idx]][~total_flag, w_col]

                # Strip NaN rows (all-NaN averaging periods pass the flag check but have no data)
                valid = ~(np.isnan(u_bar) | np.isnan(v_bar) | np.isnan(w_bar))
                u_bar, v_bar, w_bar = u_bar[valid], v_bar[valid], w_bar[valid]

                n_r = len(u_bar)
                if n_r < 4:
                    log.info(f"  Sonic @ {height}m: insufficient data for planar fit, skipping.")
                    continue

                su  = np.sum(u_bar); sv  = np.sum(v_bar); sw  = np.sum(w_bar)
                suv = np.sum(u_bar * v_bar); suw = np.sum(u_bar * w_bar); svw = np.sum(v_bar * w_bar)
                su2 = np.sum(u_bar ** 2); sv2 = np.sum(v_bar ** 2)

                H_mat = np.array([[n_r, su, sv], [su, su2, suv], [sv, suv, sv2]], dtype=float)
                g_vec = np.array([sw, suw, svw], dtype=float)

                try:
                    coef = np.linalg.solve(H_mat, g_vec)
                except np.linalg.LinAlgError:
                    coef = np.array([0.0, 0.0, 0.0])

                b0 = 0.0 if matlab_compat else float(coef[0])
                b1, b2 = float(coef[1]), float(coef[2])
                pitch = np.degrees(np.arcsin(-b1 / np.sqrt(1 + b1**2)))
                roll  = np.degrees(np.arcsin(b2  / np.sqrt(1 + b2**2)))
                log.info(f"  Sonic @ {height}m  pitch={pitch:.3g}°  roll={roll:.3g}°  b0={coef[0]:.3g} m/s")

                P = _build_pf_matrix(b1, b2)
                wind_pf = _apply_pf(P, b0, u, v, w)

                data_info_col = ii + 1
                while len(data_info) <= data_info_col:
                    data_info.append([])
                data_info[data_info_col].append(
                    f"{height}m b0={coef[0]:.3g} b1={b1:.3g} b2={b2:.3g} "
                    f"pitch={pitch:.3g} roll={roll:.3g} deg"
                )

            # ---- YAW ROTATION ----
            avg_dt_days = info["avgPer"] / (24.0 * 60.0)
            wind_pf_avg = simple_avg(np.column_stack([wind_pf[:, :2], t]), info["avgPer"])
            u_pf_bar = wind_pf_avg[:, 0]
            v_pf_bar = wind_pf_avg[:, 1]

            denom = np.sqrt(u_pf_bar**2 + v_pf_bar**2)
            denom_safe = np.where(denom > 0, denom, np.nan)
            cos_g = np.where(denom > 0, u_pf_bar / denom_safe, 1.0).astype(float)
            sin_g = np.where(denom > 0, v_pf_bar / denom_safe, 0.0).astype(float)

            N_avg = len(u_pf_bar)
            dt_vals = t[0] + np.arange(1, N_avg + 1) * avg_dt_days
            I0 = 0
            c0 = 3 * ii
            output["rotatedSonicHeader"][c0]     = f"{height}m:u"
            output["rotatedSonicHeader"][c0 + 1] = f"{height}m:v"
            output["rotatedSonicHeader"][c0 + 2] = f"{height}m:w"
            output["PFSonicHeader"][c0]     = f"{height}m:u"
            output["PFSonicHeader"][c0 + 1] = f"{height}m:v"
            output["PFSonicHeader"][c0 + 2] = f"{height}m:w"

            for row_idx in range(N_avg):
                I1 = int(np.searchsorted(t, dt_vals[row_idx], side="left"))
                if I1 >= n_pts:
                    I1 = n_pts
                if I0 >= I1:
                    I0 = I1
                    continue
                M = np.array([[ cos_g[row_idx], sin_g[row_idx], 0],
                               [-sin_g[row_idx], cos_g[row_idx], 0],
                               [0,               0,              1]])
                seg = wind_pf[I0:I1, :]
                rotated_sonic_data[I0:I1, c0:c0 + 3] = (M @ seg.T).T
                pf_sonic_data[I0:I1, c0:c0 + 3]      = seg
                I0 = I1

        except Exception as exc:
            msg = f"Sonic rotation failed at {height}m: {exc}"
            warnings.warn(msg)
            output["warnings"].append(msg)

    # Store averaged rotated data
    try:
        rs_avg = simple_avg(np.column_stack([rotated_sonic_data, t]), info["avgPer"])
        output["rotatedSonic"] = rs_avg[:, :-1]

        pf_avg = simple_avg(np.column_stack([pf_sonic_data, t]), info["avgPer"])
        output["PFSonic"] = pf_avg[:, :-1]
    except Exception as exc:
        warnings.warn(f"Could not average rotated sonic data: {exc}")
        rotated_sonic_data = np.array([])
        pf_sonic_data      = np.array([])

    return rotated_sonic_data, pf_sonic_data, output, data_info


def _apply_pf(P: np.ndarray, b0: float, u, v, w) -> np.ndarray:
    """Rotate measured winds into the fitted plane: u_p = P (u_m - c), c = (0, 0, b0).

    Wilczak et al. (2001) eqs. 35-39 (pp. 139-140): the regression intercept b0
    is the w offset c3 and is removed before rotation; the horizontal offsets
    are not recoverable by the planar fit and are taken as zero.
    """
    return (P @ np.column_stack([u, v, w - b0]).T).T


def _build_pf_matrix(b1: float, b2: float) -> np.ndarray:
    """Build the Wilczak et al. (2001) large-angle planar-fit matrix P = D'C'
    (eqs. 42-44, p. 141)."""
    denom = np.sqrt(b1**2 + b2**2 + 1.0)
    p31 = -b1 / denom
    p32 = -b2 / denom
    p33 =  1.0 / denom

    sin_alpha = p31
    cos_alpha = np.sqrt(p32**2 + p33**2)
    sin_beta  = -p32 / np.sqrt(p32**2 + p33**2)
    cos_beta  =  p33 / np.sqrt(p32**2 + p33**2)

    C = np.array([[1, 0,        0       ],
                  [0, cos_beta, -sin_beta],
                  [0, sin_beta,  cos_beta]])
    D = np.array([[cos_alpha, 0, sin_alpha],
                  [0,         1, 0        ],
                  [-sin_alpha, 0, cos_alpha]])
    return (D.T @ C.T)
