"""sonicRotation – apply planar-fit and yaw rotation to sonic wind vectors.

Legacy stage wrapper: gathers each sonic from ``data``/``output``/
``sensor_info``, calls :func:`utespac.rotation.rotate_sonics` and writes
``rotatedSonic``/``PFSonic`` (+ headers) back into ``output``. The global
coefficients may come as a :class:`~utespac.pf_info.PFTable` or as the
legacy ``cm_/day_/degrees_`` dict.
"""

import warnings
from typing import Dict, List, Optional
import logging
import numpy as np

from .pf_info import PFTable
from .rotation import (RotationResult, SonicSeries, apply_planar_fit, pf_matrix,
                       rotate_sonics)

log = logging.getLogger("utespac")

# names kept for callers of the former private helpers
_apply_pf = apply_planar_fit
_build_pf_matrix = pf_matrix


def sonic_rotation(
    output: Dict,
    data: List[Optional[np.ndarray]],
    sensor_info: Dict,
    info: Dict,
    data_info: List,
    table_names: List[str],
    pf_info=None,
) -> tuple:
    """Apply planar-fit (Wilczak et al. 2001) then yaw rotation (v̄ → 0).

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
    matlab_compat = bool(info.get("matlabCompat", False))
    t = data[int(sensor_info["u"][0, 0])][:, 0]
    output.setdefault("rotatedSonicHeader", [""] * (3 * num_sonics))
    output.setdefault("PFSonicHeader", [""] * (3 * num_sonics))

    pf_table = None
    if info["PF"]["globalCalculation"] == "global":
        if pf_info is None:
            msg = "PFinfo failed to load. Check/re-run find_global_pf."
            warnings.warn(msg)
            output["warnings"].append(msg)
            return np.full((len(t), 3 * num_sonics), np.nan), np.full((len(t), 3 * num_sonics), np.nan), output, data_info
        pf_table = pf_info if isinstance(pf_info, PFTable) else PFTable.from_legacy(pf_info)
        # MATLAB sonicRotation.m lines 37-40: the PFinfo infoString columns travel
        # with the output (dataInfo) so save_data preserves them.
        for col in pf_table.info_string or []:
            data_info.append(list(col))

    sonics = []
    for ii in range(num_sonics):
        sonics.append(_sonic_series(ii, data, output, sensor_info, table_names))

    res = rotate_sonics(sonics, t, info["avgPer"], pf_table, matlab_compat)

    for ii, s in enumerate(sonics):
        c0 = 3 * ii
        for k, comp in enumerate(("u", "v", "w")):
            output["rotatedSonicHeader"][c0 + k] = f"{s.height}m:{comp}"
            output["PFSonicHeader"][c0 + k] = f"{s.height}m:{comp}"
        if s.height in res.skipped:
            msg = res.skipped[s.height]
            if pf_table is not None:            # the local case is logged by rotate_sonics
                warnings.warn(f"Sonic rotation failed at {s.height}m: {msg}")
                output["warnings"].append(f"Sonic rotation failed at {s.height}m: {msg}")
            continue
        fit = res.fits.get(s.height)
        if fit is not None:
            col = ii + 1
            while len(data_info) <= col:
                data_info.append([])
            data_info[col].append(f"{s.height}m {fit.describe()}")

    output["rotatedSonic"] = res.averaged("rotated")
    output["PFSonic"] = res.averaged("pf_only")
    return res.rotated, res.pf_only, output, data_info


def _sonic_series(ii: int, data, output: Dict, sensor_info: Dict, table_names: List[str]
                  ) -> SonicSeries:
    """The ii-th sonic's winds, period-mean direction and the periods a local
    planar fit may use (spike flags on u/v/w and the tower-shadow flag clear)."""
    tbl_idx = int(sensor_info["u"][ii, 0])
    height = float(sensor_info["u"][ii, 2])
    u_col = int(sensor_info["u"][sensor_info["u"][:, 2] == height, 1][0])
    v_col = int(sensor_info["v"][sensor_info["v"][:, 2] == height, 1][0])
    w_col = int(sensor_info["w"][sensor_info["w"][:, 2] == height, 1][0])
    tname = table_names[tbl_idx]
    tbl = data[tbl_idx]
    avg_tbl = output[tname]
    n_per = avg_tbl.shape[0]

    hdr = output.get("spdAndDirHeader", [])
    dir_col = hdr.index(f"{height}m direction") if f"{height}m direction" in hdr else None
    direction_avg = output["spdAndDir"][:, dir_col] if dir_col is not None else np.zeros(n_per)
    flag_col = next((j for j, h in enumerate(hdr) if h.startswith(f"{height}m flag")), None)
    wind_flag = output["spdAndDir"][:, flag_col].astype(bool) if flag_col is not None \
        else np.zeros(n_per, dtype=bool)

    sf = output.get(f"{tname}SpikeFlag")
    def _sflag(col):
        if sf is None or sf.shape[1] <= col:
            return np.zeros(n_per, dtype=bool)
        return sf[:, col].astype(bool)
    good = ~(_sflag(u_col) | _sflag(v_col) | _sflag(w_col) | wind_flag)

    return SonicSeries(height, tbl[:, u_col], tbl[:, v_col], tbl[:, w_col], direction_avg, good,
                       avg_tbl[:, u_col], avg_tbl[:, v_col], avg_tbl[:, w_col])
