# Labeled run model (2026-08-22)

Board: migration / "Labeled inter-stage model". User rulings 2026-08-22:
this comes before the ec_coherent build-out so the architecture it
inherits is settled; the in-memory model is **xarray end-to-end** (option
(a) of the three put to the user: xarray; numpy dataclasses + pandas at the
edges; xarray only in ec_coherent). `xarray` 2026.7.0 was installed into
`UTESpac_Plus` on that ruling and is a `utespac` dependency from here on.

## What exists and what is missing

After migration steps 2–5 the stage numerics are typed pieces
(`averaging`, `rotation.rotate_sonics` → `RotationResult`,
`flux.reference`/`levels`/`engine`/`tables`), the configuration is a frozen
`RunConfig`, and the boundary reads/writes labeled netCDF
(`labeled.write_netcdf`, `PFinfo.json`, `export_hf`). What still carries
the MATLAB data model is the glue between stages: `pipeline.run_utespac`
passes `data` (list of 2-D arrays with a datenum column 0), `headers`,
`sensor_info` (arrays of `[table, col, height, bearing, manufacturer]`),
`info` (dict) and the `output` dict (`<name>` matrices + `<name>Header`
lists) through the wrappers `condition_data`, `avg`, `wind_stats`,
`sonic_rotation`, `fluxes`, `save_data`. Every wrapper re-derives column
positions from `sensor_info` and rebuilds header lists. That is what this
task replaces.

## Model

`utespac/model.py`:

- `Sensor` (frozen dataclass): `field` (template key, e.g. `"u"`,
  `"Tson"`, `"irgaH2O"`), `table`, `column` (header label, e.g.
  `"Ux_10.85"`), `height` [m], `orientation` and `manufacturer` (sonic
  `u` entries only, from the site's `[[sonics]]`). `Sensors` is a list
  with lookups (`by_field`, `at(field, height)`, `heights(field)`); it
  replaces `sensor_info`.
- `Run` (dataclass): `site` (the resolved site facts), `config`
  (`RunConfig`), `sensors`, `pf_table` (`PFTable` or None), `notes`
  (the `dataInfo` strings), and the data:
  - `tables: dict[name, xr.Dataset]` — one high-frequency Dataset per
    logger table, dim `time` (`datetime64[ns]`, the Campbell stamps, period
    end convention), one variable per header column named by its label,
    attrs `scan_hz`, `source_file`. QC writes in place (absolute limits,
    spikes interpolated) as the legacy `condition_data` did.
  - `flags: dict[name, xr.Dataset]` — per table, dim `time` (period end),
    variables `spike` and `nan` with an extra dim `column` (the table's
    labels), boolean. Replaces `<table>SpikeFlag`/`<table>NanFlag`.
  - `periods: dict[name, xr.Dataset]` — per-table period means, dim
    `time` (period end), variables by column label (propeller direction
    vector-averaged). Replaces `output[<table>]`.
  - `wind: xr.Dataset` — dims `(time, height)`: `direction`, `speed`,
    `shadow_flag`; attrs per height with the sector bounds. Replaces
    `spdAndDir`.
  - `rotation: xr.Dataset` — dims `(time_hf, height, component)` for
    `rotated` (planar fit + yaw) and `pf` (planar fit only), plus the
    per-height fit records as attrs/variables (`b0, b1, b2, pitch, roll`
    for LPF; the applied `PFRecord`s for GPF); period means as
    `rotated_mean`/`pf_mean` on `(time, height, component)`. Replaces
    `rotatedSonic`/`PFSonic` and the `rotated_sonic_data`/`pf_sonic_data`
    arrays.
  - `products: dict[name, xr.Dataset]` — one Dataset per flux table
    (`H`, `Hlat`, `tau`, …, `fluxQC`, `CO2flux`, `derivedT`,
    `specificHum`), dims `(time, height)`, variables by `tables.py` column
    key (`Ts_w`, `tau_PF`, …) with the display label and units as
    variable attrs; time-only variables (`rho`, `cp`) on `(time)`. Built
    from `PeriodResult`s directly; `FluxTable` becomes the adapter that
    renders the legacy matrix + header from a Dataset.
  - `raw: xr.Dataset` — the high-frequency products on `(time_hf,
    height)` (`uPF`, `vPF`, `wPF`, `u_tilt`, …, `sonTs`, `Theta_v_son`,
    `WD`, `spd`) and `(time_hf, height_h2o)` / `(time_hf, height_co2)`
    for the hygrometer/CO2 series, `P` on `(time_hf)`. Replaces the raw
    dict; `export_hf`'s A.2 file becomes a view of it.

Kernels keep taking numpy: the stages unwrap `.values`, call the existing
numerics, and wrap the results; xarray is the data model and the I/O, not
the arithmetic.

Time: `datetime64[ns]` everywhere inside the model. Period arithmetic
(`averaging.n_periods`, `period_bounds`) gets datetime64 forms; the datenum
functions in `campbell_date` stay as the boundary shim for the legacy
pickle and for reading old products.

## Boundary

Products persist as one netCDF per run with groups
(`/tables/<name>` is *not* persisted — it is the input; `/flags/<name>`,
`/periods/<name>`, `/wind`, `/rotation`, `/products/<name>`, `/raw`),
written with `xr.Dataset.to_netcdf(group=…)`, global attributes as
`labeled.run_attrs` gives them today, `utespac_format = "utespac-run-2"`.
The legacy pickle (`<site>_30minAvg_…pkl` / `_raw_`) is written from the
model by an adapter (`model.to_legacy_output(run)` → the `output` dict,
`model.to_legacy_raw(run)`) for as long as the fixture pins it; the
`utespac-averaged-1` netCDF (step 3) is superseded by the run file and its
writer/reader stay only to read files already on disk.

Readers: `get_data` keeps returning the legacy dict (tests and
`find_global_pf` use it) by reading the run file and flattening;
`load_products(root, site, …) -> dict[name, xr.Dataset]` concatenates the
run files of a site along `time` (`xr.concat(join="outer")`), which
retires the `get_data` NaN-block/label-alignment logic once nothing reads
pickles.

## Stages on the model

1. `load_run(config, site, files) -> Run` — `load_data` + `find_serial_date`
   → Datasets with `datetime64` time; `find_instruments` → `Sensors`.
2. `condition(run)` — QC in place; `run.flags`.
3. `average(run)` — `run.periods`.
4. `wind(run)` — `run.wind`.
5. `rotate(run)` — `rotate_sonics` on `Sensors` → `run.rotation`.
6. `flux(run)` — `reference_state`, `build_level`, `compute_period` →
   `run.products`, `run.raw`.
7. `save_run(run)` — the run netCDF, the legacy pickle through the adapter,
   CSV.

`pipeline.run_utespac` calls these; the wrappers `condition_data`, `avg`,
`wind_stats`, `sonic_rotation`, `fluxes`, `save_data` are deleted once the
pipeline no longer calls them (their numerics already live in the typed
pieces).

## Order of work (each step ends with the fixture green)

- S1 `model.py`: `Sensor`/`Sensors`, `Run`, converters from the legacy
  structures (`tables_from_legacy(data, headers, table_names)`,
  `sensors_from_legacy(sensor_info, headers, table_names)`) and the
  adapters back (`to_legacy_output`, `to_legacy_raw`, `to_legacy_data`);
  `averaging` datetime64 forms. Tests on synthetic data.
- S2 `pipeline.run_utespac` builds a `Run` after `load_data`/
  `find_serial_date` and calls the existing wrappers through the adapters;
  outputs unchanged.
- S3 stage by stage natively on the model — condition → average → wind →
  rotate → flux → save — deleting each wrapper as it goes; the
  `output` dict exists only inside `to_legacy_output`.
- S4 boundary: the run netCDF writer/reader, `get_data` on it,
  `load_products`, `export_hf` as a view; the pickle last (its own ledger
  row and re-pin when it goes).

## Record

S1–S3 landed 2026-08-22. `utespac/model.py`: `Sensor`/`Sensors`, `Run`
(with the kernel-side accessors `hf`, `hf_time`, `period_mean`, `flag`,
`period_time`), the converters in both directions for every piece and
`run_from_legacy`/`to_legacy_output`; a legacy→model→legacy round trip on
the fixture pipeline reproduces every array, header and raw field exactly.
`utespac/stages.py`: `load_run`, `condition` (over the `qc_table` kernel
factored out of `condition_data`), `average`, `wind`, `rotate` (on
`rotate_sonics`), `flux` (on `flux.reference`/`levels`/`engine`/`tables`,
now reading the `Run` through `Sensors`); `pipeline.run_utespac` calls
them and writes through `to_legacy_output`/`raw_to_legacy`, `keep_runs=True`
keeps each date's `Run` on its `DateResult`. The legacy wrappers
`fluxes.py`, `sonic_rotation.py`, `avg.py`, `wind_stats()` and
`condition_data()` are deleted (`simple_avg`/`stp_dn` stay as thin
legacy names; `find_global_pf` still uses `stp_dn` and `get_data`).
Verification: fixture pins unchanged (both configurations); VAC001 date 1
GPF ConstDet regenerated — every numeric array within 1e-11 of the
pre-migration product (float resolution of the datenum round trip), the
R/skew header differences being the parity retirement. Suite 161 passed.
`averaging.n_periods_dt64` is the datetime64 form of the day-span rule.

Left: S4 (the run netCDF writer/reader, `get_data` on it,
`load_products`, `export_hf` as a view of `run.raw`, the pickle's fate);
`find_global_pf` and `get_data` still work on the legacy dict — they read
products across dates, which is S4's reader.

DECIDE: when S4 lands, does the legacy pickle stay as an output option or
go? (default: goes, with the fixture re-pinned to the run file; `get_data`
reads run files and, for the old VAC001 products, pickles.)
A:
