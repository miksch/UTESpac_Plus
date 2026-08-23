# UTESpac Python

Python port of UTESpac (Utah Turbulence in Environmental Studies Process and Analysis Code).  
Original MATLAB code by Derek Jensen & Eric Pardyjak, modified by Diane Wang.

## Installation

```bash
pip3 install numpy scipy matplotlib
pip3 install xarray netCDF4
```

## Running the pipeline

`raw_processing` provides sample code for generating formatted input for UTESpac package. Each site lives under `data/<SITE>/` with its `siteInfo.toml` (site facts: sonics, heights, bearings, slope geometry, elevation, latitude) and the 48-h inputs in `data/<SITE>/utespac/` (see `data/README.md`). Processing settings are packaged TOMLs in `utespac/config/` (`run.toml`, `qc.toml`, `pf.toml`, `flux.toml`); a `config/<stage>.toml` in the working directory overrides them, and the command-line flags override both.

```bash
python utespac_main.py                                  # prompts for site and dates, GPF with the planar-fit prompts
python utespac_main.py --site VAC001 --pf local --dates all --detrend constant
python utespac_main.py --site VAC001 --pf global --reuse-pf --no-prompts
```

The local planar fit (`--pf local`) has to be run before the global one for a site: `find_global_pf` reads the LPF products. From Python:

```python
from utespac import RunConfig, run_utespac, ScriptedPFSelection
config = RunConfig.from_config(rootFolder="data", pf={"globalCalculation": "global"},
                               flux={"detrendingFormat": "constant"})
result = run_utespac(config, site="VAC001", dates="all", prompter=ScriptedPFSelection())
result.ok, [d.paths for d in result.dates]
```

## Coherent-structure analysis (`ec_coherent`)

A sibling package that reads the `utespac-hf-1` high-frequency netCDF the pipeline writes with `saveRawConditionedData` and adds one analysis netCDF per file (`<Site>_coherent_<PF>_<Det>_<date>.nc`, a group per module). Settings: `ec_coherent/config/ec_coherent.toml` (overridable like the pipeline TOMLs); science notes: `library/writeups/ec_preprocess.md`, `ec_spectra.md`; plan: `testbed/2026-08-12_ec_coherent_gameplan.md`. Modules landed so far: `spectra` (spectra, cospectra, quadrature spectra, ogives).

```bash
python -m ec_coherent.cli data/VAC001/output/VAC001_hf_GPF_ConstDet_2023_07_06.nc           # all records
python -m ec_coherent.cli data/VAC001/output/VAC001_hf_*.nc --records 0-5 --modules spectra
python testbed/scripts/ec_spectra_vac001.py                                                 # closure + Kaimal figure
```

---

## Utility scripts

### Convert siteInfo.m → siteInfo.py

Converts a MATLAB `siteInfo.m` site configuration file into the Python equivalent that
`find_files()` expects.  Run from the `UTESpac_Python/` directory.

```bash
# Convert an explicit .m file
python3 convert_siteinfo.py UTESpac_MATLAB/siteGill/siteInfo.m

```
The script handles scalar values, numeric arrays (`[1 2 3]` or `[1, 2, 3]`), string cell
arrays (`{'name'}`), and preserves inline `%` comments as `#` comments.

---

### Generate AmeriFlux BASE data

`generate_ameriflux.py` loads the GPF run files (`*_30minAvg_GPF_LinDet_*.nc`) from `siteIRGA` and `siteGill`,
aggregates slow meteorology and radiation from the 1-min data files, and writes a
half-hourly AmeriFlux BASE CSV.  Edit the path constants at the top of the script before
running.

```bash
python3 generate_ameriflux.py
```

The output CSV is written to `ameriflux_output/` and follows the
[AmeriFlux BASE format](https://ameriflux.lbl.gov/half-hourly-hourly-data-upload-format/):
comma-delimited, `TIMESTAMP_START` / `TIMESTAMP_END` in `YYYYMMDDHHMM` local standard time,
missing values as `-9999`.

Variables included:

| Group | Variables |
|---|---|
| Turbulent fluxes | `H`, `LE`, `FC`, `TAU`, `USTAR` (×5 EC heights) |
| Wind | `WS`, `WD` (×5 heights) |
| Stability | `MO_LENGTH`, `ZL`, `TKE` (×5 heights) |
| Sonic temperature | `T_SONIC`, `T_SONIC_SIGMA` (×5 heights) |
| Gas scalars | `CO2`, `CO2_SIGMA`, `H2O`, `H2O_SIGMA`, `FH2O` (×5 heights) |
| Velocity variances | `U_SIGMA`, `V_SIGMA`, `W_SIGMA` (×5 heights) |
| Wind direction QC | `WD_FILTER` (×5 heights): 0=clean sector, 1=tower-disturbed |
| Quality flags | `TAU_SSITC_TEST`, `H_SSITC_TEST`, `LE_SSITC_TEST`, `FC_SSITC_TEST` (×5 heights) |
| Slow met | `TA`, `RH`, `VPD` (×4 HMP heights), `PA` |
| Radiation | `SW_IN`, `SW_OUT`, `LW_IN`, `LW_OUT`, `NETRAD`, `ALB` (×2 rad heights) |

**Note on SSITC flags:** SSITC flags are interpreted as diagnostic indicators of nonstationarity
and similarity-theory departure, rather than as direct indicators of instrument failure or
unusable observations over forested complex terrain.
