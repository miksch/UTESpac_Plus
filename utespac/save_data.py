"""saveData – persist the products of a run: the run netCDF, the HF netCDF, CSV."""

import logging
import os
from typing import Dict
import numpy as np

from .model import Run, to_legacy_output

log = logging.getLogger("utespac")


def product_base_name(info: Dict) -> str:
    """``<Site>_<avgPer>minAvg_<PFtype>_<DetType>_<date>`` for the run facts *info*."""
    pf_type = "GPF_" if info.get("PF", {}).get("globalCalculation") == "global" else "LPF_"
    det_type = "LinDet_" if info.get("detrendingFormat", "linear") == "linear" else "ConstDet_"
    site_name = info.get("siteFolder", "site").removeprefix("site")
    return f"{site_name}_{info.get('avgPer', 30)}minAvg_{pf_type}{det_type}{info.get('date', 'unknown')}"


def save_data(run: Run) -> Dict[str, str]:
    """Write the products of *run* to ``<rootFolder>/<siteFolder>/output``.

    Produces:
    - the run netCDF ``<base>.nc`` (``utespac-run-2``, :mod:`utespac.run_io`) — always
    - the high-frequency netCDF ``<Site>_hf_<PF>_<Det>_<date>.nc``
      (``utespac-hf-1``, :mod:`utespac.export_hf`) when ``saveRawConditionedData``
      is on and the flux stage kept the raw products
    - CSV files under ``output/csv/`` when ``saveCSV`` is on

    Returns
    -------
    paths : dict
        Written products by kind: ``"nc"``, ``"hf"``, ``"csv"`` (directory).
    """
    from .labeled import run_attrs
    from .run_io import write_run

    log.info("Saving data")
    info = run.site
    paths: Dict[str, str] = {}
    base_name = product_base_name(info)
    out_dir = os.path.join(info["rootFolder"], info["siteFolder"], "output")
    os.makedirs(out_dir, exist_ok=True)

    output = to_legacy_output(run)
    output["dataInfo"] = run.notes
    output["tableNames"] = list(run.table_names)
    attrs = run_attrs(info, output)

    paths["nc"] = write_run(run, os.path.join(out_dir, base_name + ".nc"), attrs=attrs)
    log.info("  Saved: %s", paths["nc"])

    if info.get("saveRawConditionedData", False) and run.raw is not None:
        from .export_hf import write_hf
        hf_name = base_name.replace(f"_{info.get('avgPer', 30)}minAvg_", "_hf_", 1)
        paths["hf"] = write_hf(run, os.path.join(out_dir, hf_name + ".nc"), output=output)

    if info.get("saveCSV", False):
        paths["csv"] = _save_csv(output, out_dir, base_name)

    return paths


def _save_csv(output: Dict, out_dir: str, base_name: str) -> str:
    """Write the numeric output arrays to CSV files; returns the directory."""
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
    return csv_dir
