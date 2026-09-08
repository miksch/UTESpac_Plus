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
python utespac_main.py --site <SITE> --pf local --dates all --detrend constant
python utespac_main.py --site <SITE> --pf global --reuse-pf --no-prompts
```

The local planar fit (`--pf local`) has to be run before the global one for a site: `find_global_pf` reads the LPF products. From Python:

```python
from utespac import RunConfig, run_utespac, ScriptedPFSelection
config = RunConfig.from_config(rootFolder="data", pf={"globalCalculation": "global"},
                               flux={"detrendingFormat": "constant"})
result = run_utespac(config, site="MySite", dates="all", prompter=ScriptedPFSelection())
result.ok, [d.paths for d in result.dates]
```

## Outputs

One netCDF per site and processing date, `data/<SITE>/output/<SITE>_<avgPer>minAvg_<PF>_<Det>_<date>.nc` (format `utespac-run-3`). The averaged products are groups under `products/`, on `(time, height)`:

`sensible_heat`, `sensible_heat_lateral`, `sensible_heat_snsp`, `momentum`, `tke`, `sigma`, `correlation`, `obukhov`, `scaling`, `eta`, `delta_flux`, `delta_time`, `transport`, `dissipation`, `skewness`, `scalar_flux_lateral`, `latent_heat`, `co2_flux`, `flux_qc`, `temperature`, `humidity`.

Variable names use `[A-Za-z0-9_]` only, lowercase except the field's flux and scale symbols (`H`, `LE`, `Fc`, `Tau`, `TKE`, `L`, `Lv`), and read operands, then statistic, then qualifiers, with the frame last — `_raw` unrotated sonic axes, `_pf` planar fit + yaw, `_tilt` planar-fit frame: `w_theta_v_cov_pf`, `u_w_cov_pf`, `w_sigma_pf`, `Tau_pf`, `ustar_pf`, `LE_wpl_pf`, `Fc_wpl_pf`, `rho_air_moist`. Units and the readable formula are CF attributes (`units`, `long_name`), never part of the name. `utespac/names.py` is the single source of truth; `utespac-run-2` files are read through a rename shim.

The same file carries the `wind` (direction, speed, shadow flag, sector bounds), `rotation` (period means and fit records), `sensors`, `periods/<table>` and `flags/<table>` groups. `saveCSV` writes one CSV per product group under `output/csv/`, with `<name>_<height>` headers (`w_theta_v_cov_pf_10`). `saveRawConditionedData` writes the high-frequency file `<SITE>_hf_<PF>_<Det>_<date>.nc` (`utespac-hf-2`), which speaks the same vocabulary (`u_pf`, `ts`, `theta_v`, `rho_h2o`, `wind_dir`).

## Coherent-structure analysis (`ec_coherent`)

A sibling package that reads the `utespac-hf-2` high-frequency netCDF the pipeline writes with `saveRawConditionedData` and adds one analysis netCDF per file (`<Site>_coherent_<PF>_<Det>_<date>.nc`, a group per module). Settings: `ec_coherent/config/ec_coherent.toml` (overridable like the pipeline TOMLs); science notes: the `library/writeups/ec_*.md` topic notes (`ec_preprocess`, `ec_spectra`, `ec_mrd`, `ec_quadrant`, `ec_ramps`, `ec_ampmod`, `ec_scales`, `ec_coherent_flux`), one per module.

```bash
python -m ec_coherent.cli data/<SITE>/output/<SITE>_hf_GPF_ConstDet_<date>.nc      # all records
python -m ec_coherent.cli data/<SITE>/output/<SITE>_hf_*.nc --records 0-5 --modules spectra
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

`generate_ameriflux.py` is a supported standardized output: it reads a site's run files
through `utespac.run_io`, addressing the products by group and variable name, and writes a
half-hourly AmeriFlux BASE CSV to `ameriflux_output/`. One column block per sonic height,
`V = 1` the lowest.

```bash
python generate_ameriflux.py --site <SITE>                               # GPF_ConstDet
python generate_ameriflux.py --site <SITE> --qualifier LPF_LinDet --site-id US-xSITE
python generate_ameriflux.py --site <SITE> --met-dir <dir of 1-min met files>
```

The CSV follows the
[AmeriFlux BASE format](https://ameriflux.lbl.gov/half-hourly-hourly-data-upload-format/):
comma-delimited, `TIMESTAMP_START` / `TIMESTAMP_END` in `YYYYMMDDHHMM` local standard time,
missing values as `-9999`. The column names are AmeriFlux's; the module docstring lists the
`group/variable` each is built from.

| Group | Variables |
|---|---|
| Turbulent fluxes | `H`, `LE`, `FC`, `TAU`, `USTAR` |
| Wind | `WS`, `WD` |
| Stability | `MO_LENGTH`, `ZL`, `TKE` |
| Sonic temperature | `T_SONIC`, `T_SONIC_SIGMA` |
| Gas scalars | `CO2`, `CO2_SIGMA`, `H2O`, `H2O_SIGMA`, `FH2O` |
| Velocity variances | `U_SIGMA`, `V_SIGMA`, `W_SIGMA` |
| Wind direction QC | `WD_FILTER`: 0=clean sector, 1=tower-disturbed |
| Quality flags | `TAU_SSITC_TEST`, `H_SSITC_TEST`, `LE_SSITC_TEST`, `FC_SSITC_TEST` |
| Slow met | `PA` (from the run file); `TA`, `RH`, `VPD` with `--met-dir` |
| Radiation | `SW_IN`, `SW_OUT`, `LW_IN`, `LW_OUT`, `NETRAD`, `ALB` with `--met-dir` |

The 1-min met path also needs the site's column mapping (`HMP_COLS`, `RAD_COLS` at the top of
the script); without it only `PA` and the turbulence columns are written.

**Note on SSITC flags:** SSITC flags are interpreted as diagnostic indicators of nonstationarity
and similarity-theory departure, rather than as direct indicators of instrument failure or
unusable observations over forested complex terrain.
