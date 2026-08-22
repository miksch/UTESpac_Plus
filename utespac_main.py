"""UTESpac – Utah Turbulence in Environmental Studies Process and Analysis Code.

Python port of UTESpac v5.0 (MATLAB original by Derek Jensen & Eric Pardyjak,
modified by Diane Wang).

Usage::

    # GPF, recalculate PF coefficients (default)
    python utespac_main.py

    # LPF
    python utespac_main.py --globalCalculation local

    # GPF, reuse existing PFinfo without recalculating
    python utespac_main.py --globalCalculation global --recalculateGlobalCoefficients false

or import and call ``run_utespac(info)`` programmatically.
"""

import os
import argparse
import warnings

# ---------------------------------------------------------------------------
# USER INFORMATION – edit this section before running
# ---------------------------------------------------------------------------

info: dict = {}

# Root folder containing one sub-directory per site (see data/README.md).
# Defaults to <repo>/data; set to the repo root to process legacy site* folders.
info["rootFolder"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

info["UTESpacVersion"] = "5.0-Python"

# Averaging period in minutes (must divide evenly into 60)
info["avgPer"] = 30

# Save QC'd raw data tables as pickle
info["saveRawConditionedData"] = True

# Save NetCDF output
info["saveNetCDF"] = True

# Save CSV output
info["saveCSV"] = True

# Calculate TKE dissipation rate using structure function (slow)
info["calcDissipation"] = False

# useTrefHMP, avgSlowFreq, shiftzRef, zRefLowestSon are now set per-site in
# <siteFolder>/siteInfo.py so you do not need to change them here between runs.

# Compute extra statistics for transport efficiencies
info["storeExtraStats"] = True

# Detrending method: 'linear' or 'constant'
info["detrendingFormat"] = "linear"

# 'local'  – planar fit computed from individual CSV file only
# 'global' – multi-sector, multi-date-bin PF from all site data
info["PF"] = {
    "globalCalculation":              "global",
    "recalculateGlobalCoefficients":  True,
    "avgPer":                         30,
    "globalCalcMaxWind":              20,
    "globalCalcMinWind":              0.5,
}

# Reference specific humidity when no humidity measurement exists [g/kg]
info["qRef"] = 12

# --- Data Conditioning Options (Vickers & Mahrt 1997) ----------------------

info["spikeTest"] = {
    "maxRuns":                20,
    "windowSizeFraction":     1,
    "maxConsecutiveOutliers": 10,
    "maxPercent":             2,
    "spikeDef": {
        "u":               3.5,
        "v":               3.5,
        "w":               5.0,
        "Tson":            3.5,
        "fw":              3.5,
        "irgaCO2":         3.5,
        "irgaH2O":         3.5,
        "KH2O":            3.5,
        "cup":             3.5,
        "birdSpd":         3.5,
        "otherInstrument": 5.0,
    },
}

info["absoluteLimitsTest"] = {
    "u":       [-50,  50],
    "v":       [-50,  50],
    "w":       [-10,  10],
    "Tson":    [-20,  80],
    "fw":      [-20,  80],
    "irgaCO2": [0,  1500],
    "irgaH2O": [0,    50],
    "KH2O":    [0,    50],
    "cup":     [0,    50],
    "birdSpd": [0,    50],
}

info["windDirectionTest"] = {"envelopeSize": 20}
info["nanTest"]           = {"maxPercent":   55}

info["diagnosticTest"] = {
    "H2OminSignal":              0.7,
    "CO2minSignal":              0.7,
    "meanGasDiagnosticLimit":    0.1,
    "meanSonicDiagnosticLimit":  50,
    "meanLiGasDiagnosticLimit":  220,
}

# --- Sensor Name Templates -------------------------------------------------
# The wildcard '*' replaces the sensor height in the header variable name.

template = {
    "u":                  "Ux_*",
    "v":                  "Uy_*",
    "w":                  "Uz_*",
    "Tson":               "T_Sonic_*",
    "sonDiagnostic":      "diagnostic_*",
    "fw":                 "FW_*",
    "RH":                 "RH_*",
    "T":                  "Temp_*",
    "P":                  "Pressure_*",
    "irgaH2O":            "H2O_*",
    "irgaH2OsigStrength": "H2Osig_*",
    "irgaCO2":            "CO2_*",
    "irgaCO2sigStrength": "CO2sig_*",
    "irgaGasDiag":        "gas_diag_*",
    "LiH2O":              "LiH2O_*",
    "LiCO2":              "LiCO2_*",
    "LiGasDiag":          "Li_gas_diag_*",
    "KH2O":               "KH2O_H2O_*",
    "cup":                "cup_*",
    "birdSpd":            "wbSpd_*",
    "birdDir":            "wbDir_*",
}

# ---------------------------------------------------------------------------

def _display_run_summary(info: dict, data_files: list, table_names: list,
                          confirm: bool = True) -> None:
    """Display run summary table for both GPF and LPF modes.

    For LPF (confirm=True): asks 'Okay to begin analysis?' before returning.
    For GPF (confirm=False): displays summary only; find_global_pf handles confirmation.
    """
    pf_settings = info.get("PF", {})
    mode = pf_settings.get("globalCalculation", "local")
    mode_label = ("global (multi-sector GPF)" if mode == "global"
                  else "local (per-file, no sector binning)")
    recalc = pf_settings.get("recalculateGlobalCoefficients", True)
    site_name = os.path.basename(info.get("siteFolder", "unknown"))

    print("\n" + "=" * 60)
    print(f"  UTESpac — {'Global' if mode == 'global' else 'Local'} Planar Fit "
          f"({'GPF' if mode == 'global' else 'LPF'}) Mode")
    print("=" * 60)
    print(f"  Site              : {site_name}")
    print(f"  Tables            : {', '.join(table_names)}")
    print(f"  Files to process  : {len(data_files)}")
    print(f"  Averaging period  : {info.get('avgPer', 30)} min")
    print(f"  Detrending        : {info.get('detrendingFormat', 'linear')}")
    print(f"  PF mode           : {mode_label}")
    if mode == "global":
        print(f"  Recalculate PF    : {recalc}")
        print(f"  PF wind range     : {pf_settings.get('globalCalcMinWind', 0.5)}–"
              f"{pf_settings.get('globalCalcMaxWind', 12)} m/s  (periods used for GPF regression)")
    print("=" * 60)

    if confirm:
        ans = input("\nOkay to begin analysis? (yes/no): ").strip().lower()
        if ans not in ("yes", "y"):
            print("Analysis cancelled.")
            raise SystemExit(0)


def run_utespac(info: dict = None, tmpl: dict = None,
                site: str = None, dates=None) -> None:
    """Execute the full UTESpac processing pipeline.

    Parameters
    ----------
    info : dict  UTESpac configuration.
    tmpl : dict  Sensor name templates.
    site : str   Site folder name (non-interactive, e.g. ``'siteFire1'``).
    dates : None / 'all' / int / list
        Date rows to process.  ``None`` / ``'all'`` = all dates without prompt.
    """
    if info is None:
        from __main__ import info as _info
        info = _info
    if tmpl is None:
        from __main__ import template as _tmpl
        tmpl = _tmpl

    from utespac.find_files       import find_files
    from utespac.find_instruments import find_instruments
    from utespac.find_global_pf   import find_global_pf
    from utespac.load_data        import load_data
    from utespac.find_serial_date import find_serial_date
    from utespac.condition_data   import condition_data
    from utespac.avg              import avg
    from utespac.wind_stats       import wind_stats
    from utespac.sonic_rotation   import sonic_rotation
    from utespac.fluxes           import fluxes
    from utespac.save_data        import save_data

    # 1. Find files
    headers, data_files, table_names, info = find_files(info, site=site, dates=dates)

    # 2. Find instruments
    sensor_info = find_instruments(headers, tmpl, info)

    # 3. Run summary display + PF setup
    pf_info = None
    if dates == "prompt":
        is_gpf = info["PF"]["globalCalculation"] == "global"
        # GPF: show summary only (find_global_pf handles its own confirmation)
        # LPF: show summary and ask for confirmation
        _display_run_summary(info, data_files, table_names, confirm=not is_gpf)
    if info["PF"]["globalCalculation"] == "global":
        pf_info = find_global_pf(info, tmpl, sensor_info)

    # 4. Main processing loop
    for i, data_files_row in enumerate(data_files):
        try:
            # Load
            data, data_info = load_data(
                data_files_row, i + 1, len(data_files), info, table_names
            )

            # Convert timestamps
            data, data_info, info = find_serial_date(data, data_info, info)

            # QC
            data, output = condition_data(data, info, table_names, tmpl, headers)

            # Average
            output = avg(data, info, table_names, output, headers, sensor_info)

            # Wind statistics
            output = wind_stats(output, sensor_info, table_names, info)

            # Planar fit + yaw rotation
            rotated, pf_only, output, data_info = sonic_rotation(
                output, data, sensor_info, info, data_info, table_names, pf_info
            )

            # Fluxes
            output, raw = fluxes(
                data, rotated, pf_only, info, output, sensor_info, table_names
            )

            # Save
            save_data(info, output, data_info, headers, table_names, raw, tmpl)

        except Exception as exc:
            warnings.warn(f"Problem with date row {i + 1}: {exc}")
            import traceback; traceback.print_exc()

    print("\nProcessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UTESpac processing pipeline")
    parser.add_argument(
        "--globalCalculation",
        choices=["global", "local"],
        default=info["PF"]["globalCalculation"],
        help="Planar-fit mode: 'global' (GPF) or 'local' (LPF). "
             f"Default: {info['PF']['globalCalculation']}",
    )
    parser.add_argument(
        "--recalculateGlobalCoefficients",
        type=lambda x: x.lower() not in ("false", "0", "no"),
        default=info["PF"]["recalculateGlobalCoefficients"],
        metavar="{true,false}",
        help="Recompute GPF coefficients (default: true). "
             "Pass 'false' to reuse existing PFinfo.",
    )
    args = parser.parse_args()

    info["PF"]["globalCalculation"]             = args.globalCalculation
    info["PF"]["recalculateGlobalCoefficients"] = args.recalculateGlobalCoefficients

    run_utespac(info, template, dates="prompt")
