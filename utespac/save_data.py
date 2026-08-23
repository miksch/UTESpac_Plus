"""saveData – persist processed output to disk (.pkl, optional .csv, optional .nc)."""

import logging
import os
import pickle
from typing import Dict, List, Optional
import numpy as np


log = logging.getLogger("utespac")


def save_data(
    info: Dict,
    output: Dict,
    data_info: List,
    headers: List,
    table_names: List[str],
    raw_flux: Optional[Dict],
    template: Dict,
) -> Dict:
    """Save output structure to the site output directory.

    Produces:
    - A pickle file  ``<SiteName>_<avgPer>minAvg_<PFtype><detrendType><date>.pkl``
    - Optional CSV   (if ``info['saveCSV']`` is True)
    - Optional NetCDF (if ``info['saveNetCDF']`` is True)

    Returns
    -------
    paths : dict
        Written products by kind: ``"pkl"``, ``"raw"`` (when raw_flux is
        given), ``"csv"`` (directory) and ``"nc"`` when enabled.
    """
    log.info("Saving data")
    paths: Dict[str, str] = {}

    output["dataInfo"]   = data_info
    output["tableNames"] = table_names

    pf_type    = "GPF_" if info.get("PF", {}).get("globalCalculation") == "global" else "LPF_"
    det_type   = "LinDet_" if info.get("detrendingFormat", "linear") == "linear" else "ConstDet_"
    site_name  = info.get("siteFolder", "site").removeprefix("site")
    date_str   = info.get("date", "unknown")
    avg_per    = info.get("avgPer", 30)

    base_name = f"{site_name}_{avg_per}minAvg_{pf_type}{det_type}{date_str}"
    out_dir   = os.path.join(info["rootFolder"], info["siteFolder"], "output")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Pickle (averaged output) -----------------------------------------------
    pkl_path = os.path.join(out_dir, base_name + ".pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump(output, fh)
    log.info("  Saved: %s", pkl_path)
    paths["pkl"] = pkl_path

    # ---- Pickle (raw 20 Hz output) ----------------------------------------------
    if raw_flux is not None:
        raw_name = base_name.replace(f"_{avg_per}minAvg_", "_raw_", 1)
        raw_path = os.path.join(out_dir, raw_name + ".pkl")
        with open(raw_path, "wb") as fh:
            pickle.dump(raw_flux, fh)
        log.info("  Saved: %s", raw_path)
        paths["raw"] = raw_path

    # ---- CSV -------------------------------------------------------------------
    if info.get("saveCSV", False):
        paths["csv"] = _save_csv(output, out_dir, base_name) or os.path.join(out_dir, "csv")

    # ---- NetCDF ----------------------------------------------------------------
    if info.get("saveNetCDF", False):
        paths["nc"] = _save_netcdf(output, out_dir, base_name) or os.path.join(out_dir, base_name + ".nc")

    return paths


def _save_csv(output: Dict, out_dir: str, base_name: str) -> None:
    """Write numeric output arrays to CSV files."""
    csv_dir = os.path.join(out_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    skip_keys = {"dataInfo", "tableNames", "warnings", "spdAndDirHeader",
                 "rotatedSonicHeader", "PFSonicHeader"}

    for key, val in output.items():
        if key in skip_keys or key.endswith("Header") or key.endswith("Flag"):
            continue
        if not isinstance(val, np.ndarray) or val.ndim < 2:
            continue
        csv_path = os.path.join(csv_dir, f"{base_name}_{key}.csv")
        hdr_key  = key + "Header"
        header   = output.get(hdr_key, [])
        if isinstance(header, list) and len(header) == val.shape[1]:
            header_str = ",".join(str(h) for h in header)
        else:
            header_str = ",".join(f"col{i}" for i in range(val.shape[1]))
        np.savetxt(csv_path, val, delimiter=",", header=header_str, comments="")


def _save_netcdf(output: Dict, out_dir: str, base_name: str) -> None:
    try:
        import netCDF4 as nc
    except ImportError:
        import warnings
        warnings.warn("netCDF4 package not installed; skipping NetCDF output.")
        return

    nc_path = os.path.join(out_dir, base_name + ".nc")
    if os.path.exists(nc_path):
        os.remove(nc_path)

    with nc.Dataset(nc_path, "w") as ds:
        for key, val in output.items():
            if not isinstance(val, np.ndarray) or val.ndim < 1:
                continue
            clean_key = key.replace(" ", "_")[:64]
            dims = []
            for di, size in enumerate(val.shape):
                dname = f"{clean_key}_dim{di}"
                if dname not in ds.dimensions:
                    ds.createDimension(dname, size)
                dims.append(dname)
            var = ds.createVariable(clean_key, "f4", dims, fill_value=np.nan)
            var[:] = val.astype(float)
    log.info("  Saved NetCDF: %s", nc_path)
    return nc_path
