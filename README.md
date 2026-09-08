# UTESpac_Plus

Python port of UTESpac (Utah Turbulence in Environmental Studies Process and Analysis
Code), with a coherent-structure analysis package (`ec_coherent`) and AmeriFlux exporters
built on the same run files.  
Original MATLAB code by Derek Jensen & Eric Pardyjak. Ported to Python by Diane Wang.

Currently being made more python-friendly by Matt Miksch. This fork is still under development and should not be used for processing.

## Installation

Python 3.10 or newer.

```bash
pip install -e .                # numpy, scipy, matplotlib, xarray, netCDF4
pip install -e ".[raw,test]"    # + pandas (raw_processing, AmeriFlux) and pytest
```

`pyproject.toml` is the authoritative dependency list; `requirements.txt` is the older
minimal pin set kept for the MATLAB-era workflow.

## Repository layout

| Path | What lives there |
|---|---|
| `utespac/` | the processing pipeline — stages, flux kernels, run-file I/O, `names.py` |
| `ec_coherent/` | coherent-structure analysis over the high-frequency products |
| `raw_processing/` | logger-side scripts that turn raw TOA5/CSV into UTESpac inputs |
| `tests/` | pytest suite (`pytest`, or `pytest tests/test_flux_engine.py`) |
| `testbed/` | exploratory scripts and gameplans, not part of the package |
| `library/` | reference notes and bibliography — `writeups/*.md`, `references.bib`, `index.md` |
| `UTESpac_MATLAB/` | the original MATLAB source, kept for reference |
| `BUGFIXES.txt` | numbered log of fixes and behaviour changes against the MATLAB original |
| `data/` | the per-site data tree (untracked; see below) |

## The data tree

`data/` holds one folder per site and is gitignored — nothing under it ships with the
repo. A site folder carries:

```
data/<SITE>/
  siteInfo.toml   site facts: sonics, heights, bearings, slope geometry, elevation, latitude
  scripts/        per-site raw_processing wrappers (column maps, date spans)
  raw/            as logged, one folder per raw table; read-only to the pipeline
  utespac/        formatted 48-h inputs written by raw_processing:
                  <PREFIX>_<table>_header.dat + <PREFIX>_<table>_<d1>000000_<d2>000000.txt
  output/         the netCDF products (run files, high-frequency files, csv/ on request)
  PFinfo.json     global planar-fit coefficients (utespac.pf_info.PFTable)
```

A site is discovered by the presence of `siteInfo.toml` (a legacy `siteInfo.py` still
loads). `data/fixtures/` holds the mini site trees the regression tests build against.

## Running the pipeline

Processing settings are packaged TOMLs in `utespac/config/` (`run.toml`, `qc.toml`,
`pf.toml`, `flux.toml`); a `config/<stage>.toml` in the working directory overrides them,
and the command-line flags override both.

```bash
python utespac_main.py                                  # prompts for site and dates, GPF with the planar-fit prompts
python utespac_main.py --site <SITE> --pf local --dates all --detrend constant
python utespac_main.py --site <SITE> --pf global --reuse-pf --no-prompts
```

Other flags: `--root` (data root, default `<repo>/data`), `--avg-per` (averaging period in
minutes), `--run-config` (a `run.toml` replacing the packaged one), `-q/--quiet`.
`--dates` takes `all` or row selectors like `1 3 4:7`.

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

`scaling` carries the Monin-Obukhov scales alongside the integrated stability functions
`psi_m` and `psi_h`.

The same file carries the `wind` (direction, speed, shadow flag, sector bounds), `rotation` (period means and fit records), `sensors`, `periods/<table>` and `flags/<table>` groups. `saveCSV` writes one CSV per product group under `output/csv/`, with `<name>_<height>` headers (`w_theta_v_cov_pf_10`). `saveRawConditionedData` writes the high-frequency file `<SITE>_hf_<PF>_<Det>_<date>.nc` (`utespac-hf-2`), which speaks the same vocabulary (`u_pf`, `ts`, `theta_v`, `rho_h2o`, `wind_dir`) and carries one column per gas-analyser level, ordered by `Sensors.heights_of`.

## Coherent-structure analysis (`ec_coherent`)

A sibling package that reads the `utespac-hf-2` high-frequency netCDF the pipeline writes with `saveRawConditionedData` and adds one analysis netCDF per file (`<Site>_coherent_<PF>_<Det>_<date>.nc`, a group per module). Settings: `ec_coherent/config/ec_coherent.toml` (overridable like the pipeline TOMLs); science notes: the `library/writeups/ec_*.md` topic notes (`ec_preprocess`, `ec_spectra`, `ec_mrd`, `ec_quadrant`, `ec_ramps`, `ec_ampmod`, `ec_scales`, `ec_coherent_flux`), one per module.

```bash
python -m ec_coherent.cli data/<SITE>/output/<SITE>_hf_GPF_ConstDet_<date>.nc      # all records
python -m ec_coherent.cli data/<SITE>/output/<SITE>_hf_*.nc --records 0-5 --modules spectra
```

---

## Utility scripts

### Convert siteInfo.m → siteInfo.py

A migration helper for sites that still only exist as MATLAB configuration: it converts a
`siteInfo.m` into the legacy `siteInfo.py` form, which `utespac.site_config` still loads.
New sites should be written as `siteInfo.toml` directly. Run from the repository root.

```bash
python convert_siteinfo.py UTESpac_MATLAB/siteGill/siteInfo.m   # explicit .m file
python convert_siteinfo.py siteGill/                            # folder holding siteInfo.m
python convert_siteinfo.py                                      # every site*/siteInfo.m under UTESpac_MATLAB/
python convert_siteinfo.py --dry-run siteGill/                  # print instead of writing
```

The script handles scalar values, numeric arrays (`[1 2 3]` or `[1, 2, 3]`), string cell
arrays (`{'name'}`), and preserves inline `%` comments as `#` comments. It asks before
overwriting an existing `siteInfo.py` unless `--force` is given.

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

---

### Generate AmeriFlux high-frequency data

`generate_ameriflux_hf.py` reads the `*_hf_*.nc` high-frequency files and writes the
[AmeriFlux HF upload format](https://ameriflux.lbl.gov/data/how-to-upload-data/uploading-high-frequency-data/):
one CSV per 30-min period, `<SITE_ID>_HF_<start>_<end>.csv`, zipped flat per site type into
`ameriflux_hf_output/` with the individual CSVs removed afterwards. Columns are
`TIMESTAMP` (`YYYYMMDDHHMMSS.cc`, local standard time), the planar-fit `U/V/W_1_{V}_1`,
`T_SONIC_1_{V}_1`, `H2O_IU_1_{V}_1`, `CO2_IU_1_{V}_1` and `PA_1_1_1`; `V = 1` is the lowest
level. `SITE_ID` and `PF_TYPE` are constants at the top of the script.

H2O and CO2 carry the `_IU` qualifier because they are submitted as densities (g m⁻³,
mg m⁻³) rather than AmeriFlux's standard mole fractions — per AmeriFlux, `_IU` in an HF
upload should be cleared with the data team before submission.

---

## Tests

```bash
pytest                              # the whole suite
pytest tests/test_flux_engine.py    # one module
```

`tests/KNOWN_DIVERGENCES.md` records where the Python results intentionally differ from
the MATLAB original; `BUGFIXES.txt` is the numbered change log behind those differences.
The pinned-fixture test runs only when a fixture tree is present under `data/fixtures/`
and skips otherwise.

---

## Terms of use

**There is no license on this repository, and the absence of a `LICENSE` file is
deliberate.** UTESpac_Plus is not licensed for use, redistribution, or modification by
others at this time.

The code has three layers of authorship: the MATLAB original by Derek Jensen and Eric
Pardyjak, the Python port by Diane Wang, and the continued development here. The MATLAB
original carries no copyright notice and no statement of terms, so it is treated as all
rights reserved by its authors and possibly their institution; the port under `utespac/`
is a derivative work of it. `ec_coherent/` and the AmeriFlux exporters are original to
this repository, but they read the products the ported pipeline writes and are not usable
independently of it. None of this is one party's alone to license.

The intent is a permissive open-source release, most likely BSD 3-Clause, once terms are
settled with the MATLAB authors and with Diane Wang. Until then, contact them before
using any part of this. See [`NOTICE`](NOTICE) for the full provenance, copyright, and
third-party statement.

`library/` (published literature) and `data/` are excluded from version control and are
not redistributed here.
