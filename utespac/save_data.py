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
    run=None,
) -> Dict:
    """Save output structure to the site output directory.

    Produces:
    - A pickle file  ``<SiteName>_<avgPer>minAvg_<PFtype><detrendType><date>.pkl``
    - Optional CSV   (if ``info['saveCSV']`` is True)
    - Optional NetCDF (if ``info['saveNetCDF']`` is True): the
      ``utespac-run-2`` run file (:mod:`utespac.run_io`) when the labeled
      *run* is given, else the older per-field ``utespac-averaged-1`` file

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

    # ---- NetCDF (labeled, CF time; utespac.labeled) ------------------------------
    if info.get("saveNetCDF", False):
        nc_path = _save_netcdf(info, output, out_dir, base_name, run)
        if nc_path:
            paths["nc"] = nc_path

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


def _save_netcdf(info: Dict, output: Dict, out_dir: str, base_name: str, run=None) -> Optional[str]:
    """The run netCDF (``utespac.run_io.write_run``) when *run* is given, else
    the ``utespac-averaged-1`` file (``utespac.labeled.write_netcdf``); None,
    with a warning, when netCDF4 is not installed."""
    from .labeled import run_attrs, write_netcdf
    nc_path = os.path.join(out_dir, base_name + ".nc")
    try:
        if run is not None:
            from .run_io import write_run
            path = write_run(run, nc_path, attrs=run_attrs(info, output))
            log.info("  Saved netCDF: %s", path)
            return path
        return write_netcdf(output, nc_path, attrs=run_attrs(info, output))
    except ImportError as exc:
        import warnings
        warnings.warn(f"{exc}; skipping NetCDF output.")
        return None
