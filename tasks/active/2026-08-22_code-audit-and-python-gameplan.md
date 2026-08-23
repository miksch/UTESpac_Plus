# UTESpac_Plus code audit and Python migration gameplan (2026-08-22)

Requested before starting the ec_coherent gameplan: audit the current code for
theoretical mistakes, weigh the de-MATLAB roadmap in
[ec_coherent_utespac_integration_notes.md](../../testbed/ec_coherent_utespac_integration_notes.md)
Part B, plan the migration, and name the validation data to assemble. Method:
full read of `fluxes.py`, `sonic_rotation.py`, `find_global_pf.py`,
`pf_coefficients.py`, and every science helper (`calc_dissipation_rate`,
`calc_snsp_angle`, `calc_ssitc_flags`, `get_virtual_pot_temp`,
`rh_to_spec_hum`, `nandetrend`, `find_eta`, `find_delta_*`, `condition_data`,
`wind_stats`, `avg`, `simple_avg`, `stp_dn`), with the MATLAB originals
consulted wherever a result looked wrong, to separate port errors from
inherited ones. The pytest suite passes on this tree (66 passed, 1 skipped).

Labels follow dopli: statements below are read directly from the code cited;
physics claims against the literature are marked where they rest on memory and
need the source-extraction pass the ec_coherent gameplan already mandates.

## Science findings

### 1. Global planar fit rotates with the wrong coefficients — inherited MATLAB bug, not in BUGFIXES.txt

`PF_coefficients` solves the Wilczak normal equations and returns
`[b0, b1, b2]` — MATLAB `PF_coefficients.m` (`g = [sw suw svw]'`,
`linsolve(H,g)`) and Python
[pf_coefficients.py:60-66](../../utespac/pf_coefficients.py#L60-L66)
agree, and `findGlobalPF.m:378-380` displays the stored vector with exactly
that layout. But when the stored coefficients are applied,
`sonicRotation.m:68-69` reads

```
b1 = localCoef(1);   % actually b0  (units m/s, not a slope)
b2 = localCoef(2);   % actually b1
```

and the Python port reproduces this deliberately at
[sonic_rotation.py:129-131](../../utespac/sonic_rotation.py#L129-L131), with a
comment noting it matches MATLAB. So every globally planar-fit dataset — in
both languages — is rotated using the intercept b0 as the pitch slope and b1
as the roll slope. b0 is a w offset of order 0.01–0.1 m/s being used as a
dimensionless slope; a b0 of 0.05 fabricates roughly 3° of pitch, and the true
pitch lands in the roll slot. Errors of degrees in tilt propagate to u*, w-based
fluxes, L, and everything downstream, with wind-direction-dependent sign.

Three aggravating details. The local-PF path is indexed correctly
([sonic_rotation.py:179](../../utespac/sonic_rotation.py#L179)), so LPF
outputs are fine. The interactive pitch/roll display in `find_global_pf` uses
the correct indices, so the numbers the user approves on screen are not the
numbers applied. And the MATLAB-parity regression suite cannot catch it,
because both implementations are identically wrong.

Action: fix the indexing to `coef[1], coef[2]`, re-run rotation and fluxes for
all GPF-processed data, and record the divergence from MATLAB in the
known-divergence ledger (below). This also amends the integration notes' claim
that "the rotation work is already done": the raw `uPF/vPF/wPF` pickles from
GPF runs must be regenerated before the ec_coherent converter consumes them.

### 2. The planar-fit offset b0 is never removed from w

Wilczak's method subtracts the instrument offset b0 from measured w before
rotating; neither `sonicRotation.m` nor
[sonic_rotation.py:133](../../utespac/sonic_rotation.py#L133) does (the plane
coefficients are applied as a pure rotation). Detrended covariances are
unaffected, but mean(wPF) retains a bias of order b0, which matters for the
ec_coherent preprocessing step whose stated verification is "mean(w) per
window is confirmed near zero." Confirm the exact Wilczak prescription on
extraction (the paper is in `library/`); the fix is one subtraction at
rotation time.

### 3. Sonic-temperature humidity conversion uses 0.61 where sonic physics gives 0.51, and the Schotanus covariance term is missing

`theta_son_air = (T_son + 273.15)/(1 + 0.61 q) − 273.15`
([fluxes.py:475](../../utespac/fluxes.py#L475)). The speed-of-sound derivation
gives Ts ≈ T(1 + 0.51 q), and the MATLAB history shows the original had 0.51:
`fluxes.m:575` carries the 0.51 version commented out with the active line
"modified by Diane" to 0.61. 0.61 is the virtual-temperature coefficient;
using it treats Ts as exactly Tv. (Coefficient values from memory — confirm
against Schotanus et al. 1983 / Kaimal & Gaynor 1991 before changing; add
both papers to the library request list.)

Separately, rescaling by the mean humidity only corrects the mean: the
Schotanus correction to the flux is w'T' = w'Ts' − 0.51 T̄ w'q', and that
covariance term is never subtracted even on siteIRGA where 20 Hz IRGA humidity
is available at every sonic level. The column labeled `T_air'wPF'` is therefore
a rescaled buoyancy flux, not a temperature flux. The error is proportional to
the latent heat flux; largest for midday moist-surface periods.

### 4. WPL corrections are driven by w'θv' instead of the temperature flux

The WPL terms for LE (W/m² outputs), the KH2O oxygen correction, and the CO2
flux all use `kinSenFlux` = mean(wPF′·θv′)
([fluxes.py:981,996-1001,1009,1127-1130](../../utespac/fluxes.py#L981)),
matching `fluxes.m:1073`. Webb-Pearman-Leuning requires the actual temperature
flux w'T'. Using the sonic virtual flux overstates the temperature term by the
embedded humidity contribution — the same missing Schotanus term as finding 3,
now leaking into LE and Fc. The code is also internally inconsistent: the
kinematic WPL column at
[fluxes.py:992-993](../../utespac/fluxes.py#L992-L993) uses `TairP` while the
W/m² columns two lines later use `kinSenFlux`. Fixing finding 3 (a proper
w'T' from Schotanus) and feeding it to every WPL term resolves both.

### 5. The dissipation-rate estimator is not a fit

[calc_dissipation_rate.py:63-72](../../utespac/calc_dissipation_rate.py#L63-L72)
locates the single lag (among the first 20) where D_LL(r) numerically equals
r^(2/3) — the minimum of |log D_LL − log r^(2/3)| — and evaluates ε there.
That crossing point depends on the units of r and on ε itself; it is not an
inertial-subrange fit and can land on any lag. The standard estimate averages
the compensated function D_LL/(C₂ r^(2/3)) over a lag window chosen inside the
inertial subrange, then raises to 3/2. Current ε output should not be used.
The windowing and NaN conventions the ec_coherent gameplan wants to reuse from
this module are unaffected; only the estimator needs replacing, validated
against a synthetic Kolmogorov signal and against σw/u* similarity.

### 6. Slope geometry is hardcoded to one site

`calc_snsp_angle` computes `kesi = phi − 30.0` with the comment "French
Meadows downslope direction is 30° from north"
([calc_snsp_angle.py:19](../../utespac/calc_snsp_angle.py#L19), same in
MATLAB), and `fluxes.py` swaps the tilt axes on assignment
([fluxes.py:336-337](../../utespac/fluxes.py#L336-L337)) because at that site
the PF v-axis points downslope (MATLAB `fluxes.m:736-737` comments confirm the
swap is deliberate). The slope-normal heat flux `wTHv_vert` — and through it
the Obukhov length
([fluxes.py:857-859](../../utespac/fluxes.py#L857-L859)) and every SSITC
stability input — silently assumes French Meadows geometry. With slope angle 0
the expression degenerates safely to w′θv′, so flat sites are unaffected; any
other sloped site gets wrong numbers with no warning. The downslope aspect and
the axis identification belong in `SiteInfo` next to the existing `angle`.

### 7. Smaller items

- `Vtheta_fw` uses the tower-reference humidity `q_ref_fast` where MATLAB
  (`fluxes.m:566`) uses the level-local `qRefFastLocal`
  ([fluxes.py:496](../../utespac/fluxes.py#L496)) — a port deviation, and the
  only parity break found in this audit.
- Two saturation-vapor-pressure formulas coexist: a constant-Lv
  Clausius-Clapeyron in
  [rh_to_spec_hum.py:32](../../utespac/rh_to_spec_hum.py#L32) and a different
  closed form in
  [get_virtual_pot_temp.py:52](../../utespac/get_virtual_pot_temp.py#L52), so
  q_ref and virtual-theta paths disagree at the ~1 % level. `rh_to_spec_hum`
  also drops the (P − 0.378e) denominator. Pick one formula with a citation.
- The IRGA-derived reference humidity divides by air density computed from
  total pressure with Rd
  ([fluxes.py:183-184](../../utespac/fluxes.py#L183-L184)) — labeled dry-air
  density but computed from moist P, biasing q_ref by ~0.5 %.
- CO2 ppm ([fluxes.py:1047-1049](../../utespac/fluxes.py#L1047-L1049)) is a
  moist-air molar ratio; AmeriFlux expects dry mole fraction. Fine internally,
  worth a header note or a conversion at export.
- `find_delta_time` divides ejection/sweep counts by the full window length
  including NaN samples
  ([find_delta_time.py:25-26](../../utespac/find_delta_time.py#L25-L26)),
  deflating both fractions in gappy windows.
- `nandetrend` fits the line on NaN-compressed indices
  ([nandetrend.py:29-39](../../utespac/nandetrend.py#L29-L39)), distorting the
  abscissa across interior gaps. Mostly moot because despiking interpolates
  first, but fitting on the true index is the correct replacement (same
  MATLAB-inherited behavior).
- The above-canopy ITC reference applies 2.0|ζ|^{1/8} on the stable side as
  well as the unstable
  ([calc_ssitc_flags.py:173-183](../../utespac/calc_ssitc_flags.py#L173-L183));
  Foken's table treats the stable side differently. Mark `[ASSUMED]` and
  confirm against Foken & Wichura on extraction.
- `sigma` columns use `np.nanstd` with ddof=0 where MATLAB `std` uses N−1 —
  negligible at n=36 000, but it is the kind of convention the divergence
  ledger should record (`_ss_dev` already uses ddof=1).
- `find_eta`'s "transport efficiency" is total/downgradient flux — a
  self-consistent definition, but not the literature's usual
  |counter-gradient|/downgradient exuberance; the ec_coherent quadrant module
  cross-check needs to match this definition, not the textbook one.

### What checked out

The planar-fit rotation matrix construction is algebraically correct
(verified: P maps the fitted plane normal to (0,0,1);
[sonic_rotation.py:252-270](../../utespac/sonic_rotation.py#L252-L270)). The
WPL "external fluctuation" algebra for both H2O and CO2 reduces exactly to the
Webb form given a temperature-flux input (the input is the problem, finding
4). The Obukhov length formula, yaw rotation, TKE (rotation-invariant),
Foken-style steady-state deviation in `_ss_dev`, the Vickers & Mahrt-style
despiking cascade, the vector wind-direction averaging, and the manufacturer
axis corrections are all sound. Unit handling on Lv, LE, ppm, and the
mmol→g/mg conversions is consistent.

## Verdict on the de-MATLAB roadmap (integration notes Part B)

The diagnosis and the strangler-fig strategy hold up: the root-cause ordering
(labeled data first, datenums second), migrating from the I/O boundary inward,
writing ec_coherent natively in the target idiom, and gating every stage on
the regression suite are all the right calls, and the file-by-file claims
spot-checked accurate. Three amendments:

1. Correctness before structure. The roadmap treats MATLAB parity as the
   safety net, but findings 1, 3, 4, and 5 are cases where parity preserves
   wrong physics — the port is faithful, and that is the problem. B.8's
   `matlab_compat` flag should gate science behavior too: the compat path
   reproduces MATLAB (for parity testing), the clean path carries the fixes,
   and a known-divergence ledger (one file, one entry per deliberate
   divergence with magnitude and reason) replaces "max_rel < 0.01 everywhere"
   as the acceptance statement. `testkit` already supports per-field
   tolerance overrides
   ([test_regression.py:28-30](../../tests/test_regression.py#L28-L30));
   the ledger is its documentation.
2. "Rotation is already done" is no longer true for GPF data (finding 1).
   The A.2 converter and everything in ec_coherent that consumes
   `uPF/vPF/wPF` must run on regenerated pickles, or stamp provenance saying
   which rotation they carry.
3. The golden-file comparison only runs where MATLAB outputs happen to sit on
   disk (it skips otherwise). Before migration begins, commit one pinned
   30-min fixture per site class with expected values, so every migration step
   is tested on any machine, not just this one.

## Migration gameplan

Ordered so that no step invalidates a previous one's baseline. Each step ends
with the regression suite green and, where outputs change, a ledger entry and
regenerated goldens.

1. Science fixes (now, before any restructuring). Fix the GPF coefficient
   indexing; subtract b0; correct the sonic-temperature coefficient and add
   the Schotanus covariance term where HF humidity exists; feed the corrected
   w'T' to all WPL terms; replace the dissipation estimator; move slope
   aspect/axis into `SiteInfo`; fix the `Vtheta_fw` local-humidity port
   deviation and the small items worth fixing. Each behind `matlab_compat`
   where it breaks parity. Sources first per dopli practice: Wilczak 2001,
   Vickers & Mahrt 1997, and Foken & Wichura 1996 are already in `library/`;
   add Schotanus et al. 1983, Kaimal & Gaynor 1991, and Webb et al. 1980 to
   the request list (currently absent).
2. Config and UI (roadmap B.3/B.4). Frozen `RunConfig`, structured returns,
   prompts moved to the CLI wrapper, `logging` in place of `print`. Pure
   mechanics, no numeric change, immediately makes the pipeline callable from
   tests and from ec_coherent.

   Config follows dopli's packaged-TOML pattern (`dopli/src/dopli/config/`),
   continued here as `utespac/config/`: one commented TOML per stage, the
   stage dataclass remaining the schema and the source of embedded defaults,
   the TOML mirroring those defaults with comments (provenance labels
   inline, `[CITED]`/`[ASSUMED]`/`[DERIVED]` as in dopli's `wave.toml`) and
   overriding them when present — partial files fine. Resolution order, per
   dopli's `config.resolve`: per-call keyword > explicit dataclass/dict >
   `config/<name>.toml` in the working directory > packaged TOML > dataclass
   default; loading via `importlib.resources`, `[section]` tables cosmetic
   and flattened. The split against `siteInfo.toml` stays sharp: site facts
   (heights, bearings, slope geometry, elevation) live in `siteInfo.toml`;
   processing knobs (spike/absolute-limit thresholds, diagnostic limits,
   detrend format, PF settings, SSITC sub-averaging, output flags) move from
   the `info` dict into stage TOMLs — `qc.toml`, `pf.toml`, `flux.toml` —
   each mirrored by its dataclass with a `from_config` classmethod.
   ec_coherent's `config.py` adopts the same mechanism from the start.
3. Labeled I/O boundary (B.1 boundary + B.2 + B.6). The A.2 netCDF converter
   (plus `lat`/`lon` on `SiteInfo`) as the first labeled artifact;
   `save_data`/`get_data` emit and read labeled structures; datetime64 at the
   boundary with `campbell_date` surviving as the legacy shim; `PFinfo`
   re-persisted with explicit dims instead of string keys.

   Landed 2026-08-22. `utespac/labeled.py` is the boundary: `tables()` turns
   every 2-D field of an averaged output into a `LabeledTable` (datetime64
   time, column labels, heights parsed from the label or the nested table
   header), `to_frames()` gives DataFrames, `write_netcdf()`/`read_netcdf()`
   persist the dict as a CF-time netCDF (one `time` dimension; per field a
   `<f>_column` string coordinate, `<f>_height`, the data variable; the
   header/flag bookkeeping in variable attributes so the read-back is the
   legacy dict again; site, run and git provenance as global attributes via
   `run_attrs`). `save_data` writes that file in place of the old
   attribute-less dump; `get_data(fmt="nc")` / `get_frames()` read it; the
   `.nc` of a product is the pickle's twin (VAC001 date 1 GPF ConstDet: zero
   differences after the round trip, 0.33 MB). `campbell_date` gained the
   vectorized `matlab_datenum_to_datetime64` / `datetime64_to_matlab_datenum`
   (millisecond rounding, lossless for logger timestamps); the serial-datenum
   loops stay as the legacy shim the stages still use. `utespac/pf_info.py`
   holds `PFTable` (records of height, date window, sector, b0/b1/b2);
   `find_global_pf` writes `PFinfo.json` beside `PFinfo.pkl` and reads the
   JSON first; `PFTable.from_legacy/to_legacy` reproduce the `cm_/day_/
   degrees_` dict exactly, so `sonic_rotation` is untouched until step 4.
   `utespac/export_hf.py` is the A.2 converter (`python -m utespac.export_hf
   --site VAC001 --pf GPF --det ConstDet`): dims `time`/`height` (ascending),
   `u v w` (PF + yaw), `*_tilt`, `Ts`, `theta_v`, `WD`, `spd`, fine-wire, `rhov`
   [g m⁻³] and `rhoCO2` [mg m⁻³] on their own height dims (the raw pickle now
   carries `z_h2o`/`z_co2`), `P`, a `record` dimension with u*, L, wdir, the
   tower-shadow/spike/NaN flags and the SSITC flags from the sibling averaged
   pickle, a `planar_fit` group (GPF from `PFTable`, LPF from the `dataInfo`
   strings), and the A.2 global attributes; VAC001 date 1 GPF: 135 MB at
   float32, 9 s. `SiteInfo.longitude` added (VAC001 −121.9105 from the EddyPro
   metadata). Acceptance: VAC001 date 1 GPF ConstDet re-run through the
   pipeline, averaged pickle bit-identical to its predecessor, raw pickle
   identical apart from the two new height vectors; tests
   `test_labeled.py`, `test_pf_info.py`, `test_campbell_date.py`,
   `test_export_hf.py`; suite 132 passed, 1 skipped. xarray is not a
   dependency (not in the env); the core stays numpy/scipy, netCDF4 and
   pandas are used at the boundary only.
4. Stage-by-stage core migration (B.1 inward): `avg`/`simple_avg` →
   `wind_stats` → `sonic_rotation` → `fluxes` last, each against the pinned
   fixtures. Splitting `fluxes.py` goes with its migration: reference-scalar
   assembly, per-window covariance engine, WPL, Obukhov/SSITC, and output
   assembly as separate testable functions, which is also where the stride
   arithmetic and header reconstruction die. Consolidation (B.5) falls out
   here.

   Pinned fixture (landed 2026-08-22): `tests/fixtures/vac001_1hz/` is a
   one-day mini site carved from VAC001 2023-07-08 — the IRGASON/fine-wire
   table decimated 20 Hz → 1 Hz, real samples 08:00–16:00 (16 periods),
   NaN rows for the rest of the day so the file is the complete day the
   stages' day-span windowing expects, gzipped to 1.4 MB; the day's 30-min
   T/RH table; `siteInfo.toml`; `PFinfo.json`. `build_fixture.py build`
   carves it (needs `data/VAC001`), `pin` runs LPF LinDet and GPF ConstDet
   through `run_utespac` and stores every 2-D averaged field with its
   header plus the raw output strided by 50 in `expected/`;
   `tests/test_pinned_fixture.py` re-runs both (6 s) and requires the same
   field set, identical headers and NaN patterns, values to rtol 1e-6. A
   re-pin is deliberate: only with a ledger row for the change. The 1 Hz
   decimation is a regression choice, not a science one — it keeps the
   fixture committable; ec_coherent's closure tests take their 20 Hz
   window from `data/` locally. A multi-sonic fixture waits on the
   French Meadows inputs (board: parity set).

   Stages 1–3 of step 4 landed 2026-08-22. `utespac/averaging.py` holds
   the period arithmetic once (`n_periods` from the day span — the one
   place that knows the datenum convention — `period_bounds`,
   `block_mean`/`block_last`/`block_average`, `block_mean_rows`,
   `vector_mean_direction`); `avg`, `simple_avg` (timestamp-column
   detection and spacing checks only) and `stp_dn` are wrappers over it,
   bit-identical on the fixture. `wind_stats.py` exposes
   `wind_direction_speed` / `sonic_axes` / `shadow_sector` / `shadow_flag`,
   and the three copies of the manufacturer axis convention
   (`wind_stats`, `find_global_pf._compute_direction`, the raw `WD`/`spd`
   in `fluxes`) now call the one function. `utespac/rotation.py` carries
   the rotation numerics: `PlanarFit` (fit / matrix / apply / describe),
   `pf_matrix`, `apply_planar_fit`, `sector_mask`, `fit_sectors`,
   `apply_global_fit` (date window × sector per `PFRecord`), `yaw_rotate`
   (per period, on the shared block split), `SonicSeries`,
   `rotate_sonics` → `RotationResult` (rotated, pf_only, bounds, fits /
   records / skipped by height, headers, `averaged()`); `sonic_rotation`
   is the legacy wrapper (accepts a `PFTable` or the `cm_/day_/degrees_`
   dict, converts with `PFTable.from_legacy`), `pf_coefficients` the keyed
   form of `fit_sectors`. One deliberate change, in the ledger: the yaw
   segments come from `period_bounds` instead of `searchsorted(t, t0 +
   k·dt)`; old-vs-new A/B on the fixture differed in exactly the 6 period
   edges where the float form was one sample late (planar fits, `PFSonic`,
   and every other row identical), the fixture was re-pinned, and the 20
   Hz / 10 Hz serial-date grids show the two splits coincide, so the
   48-h products are unchanged (VAC001 date 1 GPF ConstDet regenerated
   through the migrated stages: averaged and raw pickles bit-identical to
   their predecessors). Tests: `test_averaging.py`, `test_wind_stats.py`,
   `test_rotation.py`; suite 154 passed, 1 skipped.

   The `fluxes` split landed 2026-08-22 as the package `utespac/flux/`:
   `reference.py` (`reference_state` → `ReferenceState`: P, T, q, ρ_d,
   ρ_v, ρ, T_v per period at the lowest sonic, q on the fast axis, the
   barometer samples for the ppm conversion), `levels.py` (`build_level`
   → `LevelInputs`: one sonic's raw/planar-fit/tilt winds, sonic and
   derived temperatures, fine-wire, hygrometer and CO2 with their
   per-period flags, the level's own pressure and HMP humidity, and the
   `specificHum`/`derivedT` columns it contributes), `engine.py`
   (`compute_period` → `PeriodResult`: every covariance, WPL term, Obukhov
   length, SSITC flag and raw per-sample series of one period, values
   keyed by table and column name, `FluxOptions` for the run settings) and
   `tables.py` (`TABLE_SPECS`: the legacy column order of every output
   table as `(key, label template)` pairs — the duplicate `R_wPF_CO2` and
   the `skew_Theata_v` label kept under their own keys until B.8 —
   `FluxTable`/`FluxTables` with `set`/`set_many` by key, `trimmed()` as
   the old `_trim_both`, `store()` in the legacy order with the
   `storeExtraStats`/`CO2flux` rules). `fluxes.py` is now the ~200-line
   orchestrator (reference state, per-sonic `build_level`, per-period
   `compute_period`, raw-product bookkeeping, `derivedT`); the stride
   arithmetic (`c_H = 3 + ii*12`, the 13-wide sigma stride, the 17-wide
   R allocation) and the 200 lines of header reconstruction are gone.
   Verification: fixture LPF/GPF pinned values within rtol 1e-9; an
   old-vs-new A/B on the fixture inputs under eight option variants
   (default, `matlabCompat`, `calcDissipation`, `storeExtraStats` off,
   `useTrefHMP` off, `shiftzRef`, slope geometry, raw off) — identical
   keys, headers and NaN patterns, values to ≤ 1e-13 absolute; VAC001
   date 1 GPF ConstDet regenerated on the 20 Hz data: 44 of 57 averaged
   arrays bit-identical, 13 within 5e-13, raw identical except the two
   WPL external-fluctuation arrays within 1e-12. The residual is
   summation order (the reductions now run on single columns, which numpy
   sums pairwise, where `simple_avg` summed `[series, t]` stacks
   sequentially) — a ledger row, not a numeric change; the fixture
   tolerance is set above it (rtol 1e-9). Tests `test_flux_tables.py`,
   `test_flux_engine.py`. Not covered by any fixture and ported by
   transcription only: KH2O and LI-COR hygrometer paths, multi-sonic
   towers, `birdDir` vector averaging — the parity set (board) is what
   tests those. Left in step 4: the pipeline calling `rotate_sonics` and
   the flux pieces directly is step 5 work (retire the legacy wrappers
   with the other parity artifacts).
5. Retire parity (B.8): drop the compat flag, the gap columns, the duplicate
   R column and the `skew_Theata_v` typo in one commit; re-baseline testkit
   to clean output.

   Done 2026-08-22 without a decision needed: `find_global_pf` writes
   `PFinfo.json` only (`save_pf_info` returns `{"json": path}`);
   `load_pf_info` still reads a `PFinfo.pkl` left by an earlier run and
   logs that the JSON is the current form. The rest of step 5 removes the
   ability to compare against MATLAB outputs column by column
   (`matlabCompat` reproduces the legacy physics for parity runs; the
   duplicate R column and the `skew_Theata_v` label keep `testkit`'s
   positional comparison aligned), while the parity set on the board
   (MATLAB `.mat` + PFinfo + run settings for the three sites) is still
   pending and no MATLAB comparison has been run on this tree.

   DECIDE: retire MATLAB parity now, or keep it until the parity set has
   been run once? (a) retire now: drop `matlabCompat` from `RunConfig`,
   `sonic_rotation`, `flux.levels`/`flux.engine` and `get_data`, delete the
   `R_wPF_CO2_dup` column and rename `skew_Theata_v` in `TABLE_SPECS`,
   drop `tests/test_regression.py` + `utespac/testkit.py`'s MATLAB
   loaders, re-pin the fixture — one commit, the MATLAB comparison is
   then only possible from git history; (b) keep until the parity set
   (board: validation) has been assembled and run, then retire in one
   commit with the measured "magnitude: to measure" ledger rows filled
   in; (c) never assemble the parity set (the FM sites' MATLAB outputs are
   off-repo and the VAC001 IOP was never processed in MATLAB) and retire
   now, accepting that the port is validated only by the EddyPro
   comparison and the audit.
   (default: (b) — nothing is blocked on it; the flag costs one boolean
   and two branches.)
   A:

ec_coherent development starts after step 1 (it must not consume pre-fix GPF
pickles) and can proceed in parallel with steps 2-3, which it does not depend
on beyond the converter.

## VAC001 test dataset (landed 2026-08-22)

The user supplied EddyPro output, 30-min logger tables, and raw 20 Hz TOA5
for VAC001 (2023-07-06 to 07-21); they now live under `data/VAC001/` in the
new per-site layout (`data/README.md`), with `raw_processing/VAC001_*`
writing the 48-h inputs to `data/VAC001/utespac/` and
`testbed/scripts/run_vac001.py` / `compare_vac001_eddypro.py` as the
runner and comparison. Two things surfaced on the first run:

- The legacy `siteInfo.py` had `tower = "001"` (the tower name), which
  `wind_stats` read as a 1° bearing; with sonic orientation 215 the
  tower-shadow sector landed on the prevailing SW wind and flagged every
  period, so the local planar fit had no data and every PF product was NaN.
  `data/VAC001/siteInfo.toml` now carries `tower = 180` `[ASSUMED]`
  (tower directly behind the boom-mounted head; shadow sector centred on
  35°).

  Sonic azimuth 215° — user ruling 2026-08-22, matching EddyPro metadata
  `instr_1_north_offset=215.0` and `siteInfo.toml`; UTESpac wind direction
  matches EddyPro's to 0.2° median. The logger's `sonic_azimuth` column
  read 315 from 2023-05-17 through the IOP (215 before, 0/180 after the
  07-20 12:00 program reload) and is wrong; ignore it. IOP wind is 180–240°
  (SW sea-breeze regime, 96 % of periods). No period falls in the
  tower-wake sector, so the `[ASSUMED]` `tower = 180` has no effect on this
  IOP's results; sonic height stays 10.85 m (legacy siteInfo) versus
  EddyPro's rounded 11.00 m.

- The `card_convert/` upload is byte-identical to `raw/` (same 17 files,
  `cmp` clean); it was left in the old `siteVAC001_20230706_20230720/`
  folder and can be deleted.

Full-record comparison (LPF per 48-h window, linear detrend; 720 joined
30-min periods 2023-07-06 to 07-21; EddyPro reference: single-sector planar
fit from 07-06..07-08, block averaging, WPL, spectral corrections; figure
`testbed/scratch/vac001_LPF_vs_eddypro.png`, table
`vac001_LPF_vs_eddypro.csv`, both from `testbed/scripts/`):

| quantity | slope | intercept | bias | r² |
|---|---|---|---|---|
| u* | 1.009 | −0.010 | −0.004 m/s | 0.999 |
| σw | 1.012 | −0.009 | +0.002 m/s | 1.000 |
| TKE | 1.001 | −0.040 | −0.04 m²/s² | 0.999 |
| wind speed | 0.997 | 0.001 | −0.01 m/s | 1.000 |
| LE (WPL, wPF′) | 0.967 | 2.1 | −3.9 W/m² | 0.994 |
| Fc (WPL, wPF′) | 0.930 | 0.32 | +0.71 µmol m⁻² s⁻¹ | 0.996 |
| H from Θv′wPF′ | 1.018 | 10.3 | +10.4 W/m² | 0.978 |
| w′Ts′ kinematic vs EddyPro `w/ts_cov` | 1.022 | 0.009 K m/s | +0.009 | 0.977 |
| z/L (clipped ±2) | 1.133 | −0.03 | −0.03 | 0.758 |
| H from unrotated w′Ts′ | 1.213 | 11.8 | +12.4 W/m² | 0.977 |
| H from fine-wire | 0.368 | 51.5 | — | 0.028 |

Per-window LPF tilt was stable (pitch 4.3–5.2°, roll 1.2–2.4°; EddyPro's
B1/B2 give 4.57°/1.69°). L sign agreement 95.4 %.

The H excess is the missing Schotanus term, measured: regressing
(H_UTESpac − H_EddyPro) on LE gives slope 0.0599 W/m² per W/m² with
intercept −0.4 (r = 0.79); the theoretical single Schotanus term
0.51·T̄·cp/Lv is 0.062. Day/night split of the H difference is +14.6 / +2.1
W/m². Applying that term to UTESpac's H with its own LE collapses the
disagreement to bias −0.6 W/m², RMSE 7.9 (from 16.8), slope 0.986, r² 0.992.
Finding 3 is therefore confirmed in field data, and the expected size of
the fix is established. The EddyPro project file shows why EddyPro lands on
the corrected side: `col_ts=12`, i.e. it was fed the logger's
`T_SONIC_corr` (humidity-corrected sonic temperature; raw rows confirm
T_SONIC − T_SONIC_corr = 0.51·q·T), and its reported sonic temperature is
T_SONIC − 1.27 K. Side effect recorded against the reference: EddyPro's
`air_temperature` is a further 1.7 K low (it humidity-corrected an already
corrected channel), which biases its air density by ~0.6 % — small, but it
is the kind of q-channel oddity the 2026-08-22 ruling warned about.

The fine-wire failed 07-12 to 07-17 (its all-NaN columns were trimmed from
those files, which is how the `get_data` blanking defect on the board was
found); the fine-wire row above is therefore not a meaningful test.

Constant-detrend variant (block averaging, matching EddyPro's
`detrend_meth=0`; files `*_LPF_ConstDet_*`): LE slope 0.985 / r² 1.000
(linear: 0.967 / 0.994), Fc 0.937 / 0.999, TKE 1.004 / 1.000, u* 1.010,
raw w′Ts′ covariance r² 0.984 (linear: 0.977); the H Schotanus offset is
unchanged (+11.2 W/m², slope 1.026). So the linear-vs-block choice
accounted for most of the LE slope shortfall; what remains (≈1.5 %) is the
size of EddyPro's spectral correction factors (LE_scf, co2_scf ≈ 1.01)
plus the WPL temperature term carrying w′θv′ instead of w′T′ (finding 4).
Figures: `testbed/scratch/vac001_LPF_{LinDet,ConstDet}_vs_eddypro.png`.

Planar-fit coefficients (`testbed/scripts/pf_vac001_eddypro.py`, fitting
the same LPF 30-min means `find_global_pf` regresses on, single sector):

| fit | N | b0 | b1 | b2 | pitch | roll |
|---|---|---|---|---|---|---|
| EddyPro planar_fit file | 714 | 0.0395 | −0.0800 | 0.0295 | 4.58° | 1.69° |
| UTESpac whole IOP, 0.5–20 m/s | 710 | 0.0425 | −0.0811 | 0.0295 | 4.64° | 1.69° |
| UTESpac 07-06..07-08 only | 143 | 0.0514 | −0.0827 | 0.0214 | 4.73° | 1.23° |

EddyPro's file states a 2-day determination period but its sector
numerosity (714) shows it fit the whole record (`pf_subset=0`), and the
whole-IOP UTESpac fit reproduces it to 0.06° in pitch and 0.00° in roll.
The legacy coefficient indexing (finding 1) would have applied pitch −2.94°
and roll −4.73° to this sonic — 7.6° and 6.4° off. Removing b0 takes the
mean rotated w over the point cloud from +0.042 m/s to 0.000 (finding 2;
Wilczak 2001 eqs. 35–39, pp. 139–140, read from the PDF on 2026-08-22:
u_p = P(u_m − c), b0 = c3). Figure:
`testbed/scratch/vac001_planar_fit_3d.png` (point cloud with the fitted,
legacy-applied, and EddyPro planes). Both fixes landed in
`utespac/sonic_rotation.py` behind `info["matlabCompat"]` (default off);
`tests/test_planar_fit.py` pins the contract.

q-variance check (the 2026-08-22 EddyPro caveat): UTESpac's σ_H2O equals a
direct 20 Hz standard deviation of the raw TOA5 `H2O_density` (slope 0.999,
intercept −0.0001 g/m³, r² 1.000, 95 clean 30-min blocks). EddyPro's
`h2o_var` is a constant factor 41.3 (= 1/air molar volume) above the g/m³
conversion that assumes mixing-ratio units; read as **(mmol m⁻³)²**, i.e.
molar-density variance, it matches UTESpac (slope 0.998, r² 1.000, N 714)
and the raw std (slope 1.000). `co2_var` is the same: (mmol m⁻³)², ×44.01
→ mg/m³ matches UTESpac's σ_CO2 (slope 1.06, r² 0.98). So EddyPro's q′
second moments are computed correctly and mislabelled; the caveat reduces
to "convert `h2o_var`/`co2_var` as molar-density variances".

GPF fluxes with the corrected rotation (single sector, whole IOP, block
averaging; `testbed/scripts/run_vac001_gpf.py` drives `find_global_pf`'s
prompts from a script; files `*_GPF_ConstDet_*`), the like-for-like
configuration against EddyPro:

| quantity | slope | intercept | bias | RMSE | r² |
|---|---|---|---|---|---|
| u* | 1.006 | −0.006 | −0.001 m/s | 0.005 | 1.000 |
| σw | 1.011 | −0.007 | +0.002 | 0.006 | 1.000 |
| TKE | 1.004 | −0.008 | +0.002 | 0.013 | 1.000 |
| LE (WPL, wPF′) | 0.989 | 0.39 | −1.6 W/m² | 3.0 | 1.000 |
| Fc (WPL, wPF′) | 0.940 | 0.30 | +0.63 µmol m⁻² s⁻¹ | 0.98 | 0.999 |
| H from Θv′wPF′ | 1.028 | 11.1 | +11.1 W/m² | 15.7 | 0.985 |
| w′Ts′ vs `w/ts_cov` | 1.031 | 0.010 K m/s | +0.010 | 0.014 | 0.984 |

Schotanus/WPL fix in code (2026-08-22, same GPF configuration re-run with
`T_air'wPF'` = w′Ts′ − 0.51·T̄·w′q′ from the IRGASON humidity and the WPL
terms driven by that w′T′; `library/writeups/sonic_temperature_flux.md`):

| quantity | slope | intercept | bias | RMSE | r² |
|---|---|---|---|---|---|
| H from T_air′wPF′ (Schotanus) | 1.001 | −0.04 | −0.03 W/m² | 0.75 | 1.000 |
| H from Θv′wPF′ (buoyancy, unchanged) | 1.028 | 11.1 | +11.1 W/m² | 15.7 | 0.985 |
| LE (WPL, wPF′) | 0.984 | 0.47 | −2.4 W/m² | 4.2 | 1.000 |
| Fc (WPL, wPF′) | 0.975 | −0.04 | +0.10 µmol m⁻² s⁻¹ | 0.32 | 1.000 |

H split by sign and time of day (VAC001 is advective: EddyPro H < 0 in
63 % of all periods and 30 % of 08–18 h periods): bias −0.16 W/m² (RMSE
0.52) where H < 0, +0.18 (1.03) where H > 0; +0.24 (0.95) by day, −0.24
(0.55) by night. EddyPro consumed the logger's `T_SONIC_corr`, so this is
the same correction applied on both sides; the residual is at the level of
the detrending/averaging conventions. LE sits 1.4 % below EddyPro's LE; the
EddyPro spectral-correction factor `LE_scf` has median 1.0105, which
UTESpac does not apply, leaving ~0.5 % inside the q′ caveat. The earlier
0.989 slope had the buoyancy-driven WPL term inflating LE by ~0.5 %. Fc
moved from 0.940 to 0.975 with the bias down from +0.63 to +0.10 µmol
m⁻² s⁻¹. (EddyPro's `un_LE` / `un_co2_flux` are pre-WPL as well as
pre-spectral — Fc vs `un_co2_flux` has slope 0.71 — so they are not a
reference for the WPL columns.)

Against the 48-h LPF runs this halves the u* and LE residuals (RMSE 0.011
→ 0.005 m/s, 4.4 → 3.0 W/m²); what remains on LE/Fc is the size of
EddyPro's spectral correction plus the WPL temperature term (finding 4),
and H still carries the Schotanus offset (finding 3). Figure:
`testbed/scratch/vac001_GPF_ConstDet_vs_eddypro.png`.

Before/after for finding 1, same PFinfo applied with the legacy MATLAB
indexing (`run_vac001_gpf.py --compat --reuse-pf`, linear detrend; joined
table `testbed/scratch/vac001_GPF_LinDet_vs_eddypro.csv`, figure
`vac001_GPF_LinDet_vs_eddypro.png`; the pickles were deleted afterwards so
they cannot be mistaken for real output):

| quantity | legacy indexing: slope / bias | fixed: slope / bias |
|---|---|---|
| u* | 1.176 / +0.123 m/s | 1.006 / −0.001 |
| σw | 1.136 / +0.098 m/s | 1.011 / +0.002 |
| H (Θv′wPF′) | 1.322 / +13.0 W/m² | 1.028 / +11.1 |
| LE | 1.204 / +43.0 W/m² | 0.989 / −1.6 |
| Fc | 1.197 / −0.63 µmol | 0.940 / +0.63 |

That is the damage a 7.6° pitch / 6.4° roll error does on this sonic:
every PF-based flux 18–32 % high. Any existing GPF product from either
code base is affected at this order (site-dependent; it scales with b0 and
the true tilt).

Energy-balance closure (`testbed/scripts/closure_vac001.py`, figure
`testbed/scratch/vac001_closure_GPF_ConstDet.png`). Available energy from
the user's `_soil_corr` slow table: `NETRAD` (assembled outside this repo)
and the rebuilt ground heat flux `ghf_avg` (plate + storage; daytime
medians Rn 544, G 63, plate 45, storage 20 W/m²). Both are provisional per
the user, so the absolute ratios are indicative; the relative ranking is
what the dataset can support. GPF/block-average UTESpac, 715 periods:

| H + LE variant | Σ ratio, all | Σ ratio, day (Rn > 50) | OLS slope, day | r², day |
|---|---|---|---|---|
| UTESpac H(Θv′wPF′) + LE, as computed | 0.952 | 0.872 | 0.720 | 0.75 |
| UTESpac H − Schotanus(LE) + LE | 0.897 | 0.828 | 0.683 | 0.74 |
| EddyPro H + LE | 0.904 | 0.834 | 0.692 | 0.74 |
| logger EasyFlux H + LE (slow table) | 0.936 | 0.863 | 0.710 | 0.70 |

(vs Rn − ghf_avg; with G_plate_avg only, every ratio drops by ~0.05; with
no G, by ~0.11.) Two readings. First, once the Schotanus term is applied,
UTESpac and EddyPro close the budget to within 0.006 of each other — the
same conclusion as the direct flux comparison, now against an independent
reference. Second, closure cannot adjudicate the Schotanus correction: the
uncorrected w′Ts′ flux is the buoyancy flux and is larger, so it "closes
better" (0.872 vs 0.828) exactly as any overestimate of H would; the
~0.83 daytime / ~0.90 all-periods closure of the corrected fluxes is the
ordinary non-closure range for a grassland tower, and the residual (day
mean ≈ 75 W/m², r² 0.74) is dominated by the provisional Rn/G rather than
by the turbulent side. If Rn or G are revised, re-run the script; nothing
in the ranking depends on their absolute level.

## Validation data to assemble

For "did the port break anything" (parity):

- The MATLAB `.mat` outputs (averaged + rawFlux) for the exact input days on
  disk for all three sites, together with the `PFinfo` actually used and the
  MATLAB run's settings (detrend format, avgPer, spike thresholds). These
  drive `testkit`; they exist for siteGill/siteIRGA per the repo layout —
  confirm coverage for siteVAC001 and for both LPF and GPF modes.
- From those, one pinned clean 30-min window per site class (one 20 Hz
  multi-height, one 10 Hz single-height) committed as a small fixture — the
  same windows the ec_coherent gameplan wants for its closure tests, so pick
  them once.

For "did the corrections make the data better" (the port being faithful means
parity data cannot answer this — an external reference is needed):

- An EddyPro run on a few days of the same raw data — ruled the external
  reference (user ruling 2026-08-22; prior UTESpac runs are explicitly not
  the correction reference, they validate only the port). siteGill's Li-7500
  GHG archives are directly EddyPro-ingestible; EddyPro applies planar fit,
  Schotanus, and WPL in their published forms, so H, LE, Fc, u*, and L from
  EddyPro versus the fixed pipeline (and versus the unfixed one) is the
  single most decisive check for findings 1, 3, and 4. Caveat from the same
  ruling: EddyPro has shown unexplained problems with q' variances in the
  user's experience, so treat its H2O second moments (σ_q, and anything
  built on w'q' variance normalisation) as suspect — anchor the comparison
  on H, u*, L, and the momentum quantities, and cross-check the humidity
  channel between EddyPro, the fixed pipeline, and raw-counts sanity before
  trusting either side's q' statistics.
- Radiation and soil-heat-flux data for any of the sites, if logged: energy
  balance closure (H + LE vs Rn − G) before/after the corrections. The
  corrections should move closure in a consistent direction; midday moist
  periods (small Bowen ratio) are where findings 3 and 4 are largest, so
  select those days deliberately.
- A 3D planar-fit figure per height/sector/date-bin: the binned
  (ū, v̄, w̄) averages that fed the regression, scattered against the plane
  w = b0 + b1·u + b2·v, rendered for both the coefficients as stored and as
  currently applied. Findings 1 and 2 become visible directly — the applied
  plane misses the point cloud, and the cloud floats b0 above the fixed
  plane. Needs only the existing averaged pickles and `PFinfo.pkl`.
- Per-sector mean(wPF) and pitch/roll tables from the pre-fix GPF outputs:
  after the coefficient fix, mean(wPF) per sector should collapse toward b0
  and the applied pitch/roll should match what `find_global_pf` displayed.
  This needs no new data, only the existing averaged pickles.
- For the dissipation fix: a synthetic signal with prescribed ε (exact
  target), plus σw/u* Monin-Obukhov similarity on real data as a plausibility
  band. No archival data needed.
- Ambient CO2 plausibility (~420 ppm) per period — already proven useful (it
  caught the 38 ppm bug in BUGFIXES.txt item 10), cheap to keep as a
  standing check.

What previous processing is *not* useful for: validating the science fixes
against old MATLAB outputs — both sides carry the same physics, so agreement
is expected and meaningless. Old outputs validate the port; external
references validate the corrections. Keep the two piles separate.
