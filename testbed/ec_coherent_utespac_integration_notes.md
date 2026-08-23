# EC Coherent-Structure Analysis × UTESpac_Plus — Integration & De-MATLAB Notes

Companion to `ec_coherent_structure_analysis_gameplan(1).md`. That gameplan was written
without knowledge of this repository; this document maps its assumptions onto what
UTESpac_Plus actually produces, states what must be adapted, and lays out a
de-MATLAB-ification roadmap for the existing library. Both documents together are the
input for the implementation chat. **No code has been changed to produce this document.**

---

## Part A — Integration

### A.1 Gap analysis: gameplan §0–§1 assumptions vs UTESpac reality

| Gameplan assumes | UTESpac reality | Consequence |
|---|---|---|
| Per-location **netCDF** of raw HF series, dims `(time, height)`, CF datetime | HF data persisted **only** as `<Site>_raw_<PF>_<Det>_<YYYY_MM_DD>.pkl` — a dict of `(n_pts, nSonics)` numpy arrays + 1-D `t` (MATLAB serial datenum) + 1-D `z` (heights, m). The existing netCDF writer (`utespac/save_data.py:92`) dumps only the **averaged** output, with auto-named dims and zero attributes. | Need a reader/adapter or (recommended) a converter to the gameplan's §1 schema. See A.2. |
| Rotation applied in-memory from stored PF coefficients (b0,b1,b2) | **Rotation is already applied upstream.** `raw["uPF"/"vPF"/"wPF"]` are planar-fit **plus per-averaging-period yaw** rotated HF series (`utespac/fluxes.py:650-652`). `raw["u_tilt"/"v_tilt"/"w_tilt"]` are PF-only (no yaw). Coefficients also persist in `<siteFolder>/PFinfo.pkl` as `cm_<h*100> → day_<d0>to<d1> → degrees_<lo>_to_<hi> → [b0,b1,b2]`; the PF matrix is reconstructible via `sonic_rotation._build_pf_matrix` (`utespac/sonic_rotation.py:252-270`). | Gameplan §0 step 1 becomes a **verification step**, not a transform. The analysis package consumes `uPF/vPF/wPF` directly. Caveat: **unrotated u,v,w are not saved** in the raw pkl, so a per-window double-rotation cross-check cannot be done from the raw file alone (see A.3). |
| Read QC flags; do not re-despike | Despiking already happened, and it **interpolates over spikes in place** (`utespac/condition_data.py:95-146`) — the raw pkl is gap-filled, not merely flagged. Per-averaging-window booleans `<table>SpikeFlag` / `<table>NanFlag` live in the **averaged** pkl (`condition_data.py:156,162`), not the raw pkl. SSITC quality flags (`fluxQC`) also live in the averaged pkl. | Flags must be joined from the averaged pkl by window timestamp. The "windows failing QC are processed but flagged" policy carries over unchanged. |
| Per-window ancillaries `ustar, L, wdir` available | Present in the averaged pkl: `tau` (contains u*), `L`, `spdAndDir`, `derivedT`, etc. — each a 2-D matrix with a parallel `<name>Header` list of strings like `"51.5m:u"`. `utespac/testkit.py:155-167` (`get_header`) is the existing helper for unwrapping headers. | Join averaged pkl on window timestamps (col 0 of each matrix is a serial datenum). |
| Global attrs: `site_id, lat, lon, sampling_frequency_hz, flux_averaging_s, canopy_height, zi` | `SiteInfo` dataclass (`utespac/site_config.py:42-66`) has heights (`sonics: list[SonicLevel]`), `tableScanFrequency` [Hz], `canopyHeight`, `siteElevation`, `displacementHeight`, tower bearing, slope angle. **No lat/lon field exists anywhere. No z_i** (gameplan §7.3–7.4 already anticipated the missing z_i). | Add two optional fields (`latitude`, `longitude`) to `SiteInfo` — a two-line dataclass change plus per-site values; `from_mapping` already warns on unknown keys so old site files remain valid. This is the **only** change to `utespac/` core the integration strictly needs. |
| Scalars `Ts/T, q, co2` per height | HF available in raw pkl: `sonTs` (sonic T), `Theta_v_son` (virtual potential T), `rhov` [g m⁻³], `rhoCO2` [mg m⁻³] (unit note at `generate_ameriflux_hf.py:28-30`), plus already-detrended `rhovPrime`, `rhoCO2Prime` and "external-reference" variants. Also `WD`, `spd`, `P` ([time, kPa]). | Use `rhov` as the humidity scalar (density, not specific humidity — document it). **Do not** use the stored `*Prime` perturbations: they are locked to the pipeline's detrend setting; the gameplan requires selectable detrend, so recompute perturbations per window from the raw series. |
| Uniform time grid | Yes: `raw_processing` writes 48-h files on a uniform grid, first sample 1/hz after midnight (Campbell convention, `raw_processing/common.py:147-171`); gaps are NaN. Sampling: siteGill 10 Hz (1 sonic @ 51.5 m, canopy 19.3 m), siteIRGA 20 Hz (4 sonics @ 4.42/6.35/13.94/32.18 m), siteVAC001 20 Hz. | 30-min windows align exactly to the grid. FFT/wavelet code still needs a NaN policy per window (reject above threshold; else fill) since despiking/gap-filling is not exhaustive. |

**Bottom line:** everything the gameplan needs exists, but in pickles keyed by
MATLAB-serial-datenum floats, split across a raw pkl (HF series) and an averaged pkl
(flags + ancillaries), with site metadata in `SiteInfo`. The rotation work is already
done. The main integration artifact is an I/O bridge.

### A.2 Recommended integration architecture

**Recommendation: keep `ec_coherent` standalone as designed in the gameplan, and add a
converter script that assembles the gameplan's §1 netCDF from UTESpac artifacts.**
Precedent already in-repo: `generate_ameriflux_hf.py` does exactly this shape of job
(raw pkl → external HF format), including vectorized datenum→datetime conversion
(`generate_ameriflux_hf.py:59-69`).

Two options were considered:

1. **(Recommended) Converter script** `export_hf_netcdf.py` (or `ec_coherent/ingest/utespac.py`
   exposed as a CLI): reads `_raw_*.pkl` + matching averaged `.pkl` + `SiteInfo`, writes
   one CF-compliant netCDF per raw file (per site per day), consumed by `ec_coherent.io`
   via `xarray.open_mfdataset`. Pros: `ec_coherent` stays exactly as the gameplan
   designed it (portable to non-UTESpac data); the netCDF is self-describing where the
   pickles are schema-fragile; the converter doubles as the first "labeled-data boundary"
   artifact of the de-MATLAB roadmap (Part B). Cons: one more file format on disk
   (~1.7 GB/day for siteIRGA at f4 — acceptable; use zlib compression and one file per day).
2. **Direct pkl reader inside `ec_coherent.io`**: skip netCDF, yield windows straight
   from the pickles. Pros: no intermediate files. Cons: couples the analysis package to
   pickle internals (dict keys, datenum floats, header-string parsing), which Part B
   wants to retire; harder to share data outside this repo.

Either way, the **dependency direction is one-way**: `ec_coherent` may import from
`utespac` (readers, `campbell_date`, `site_config`), never the reverse. Keep
`ec_coherent/` as a sibling top-level package so `utespac` core keeps its minimal
numpy/scipy dependency set; `xarray`, `netCDF4`, `pywt` belong to `ec_coherent` (and the
converter) only.

**Converter spec (for the implementation chat):**

- **Inputs:** glob `<site>/output/*_raw_<PF>_*.pkl` (reuse or mirror the matching logic
  in `utespac/get_data.py:76-95`, which already handles both averaged and raw pkls);
  the sibling averaged pkl (same base name with `_<avgPer>minAvg_`); `load_site_info()`.
- **Dims/coords:** `time` (datetime64, from `raw["t"]` via
  `campbell_date.matlab_datenum_to_datetime` or the vectorized pattern in
  `generate_ameriflux_hf.py:63-69`), `height` (from `raw["z"]`, sorted ascending —
  note `SiteInfo.ascending` can be False, e.g. siteGill).
- **Variables:** `u ← uPF`, `v ← vPF`, `w ← wPF`, `Ts ← sonTs`, `theta_v ← Theta_v_son`,
  `q_rho ← rhov` [g m⁻³], `co2_rho ← rhoCO2` [mg m⁻³], `P` (interp to `time`), `WD`, `spd`.
  Optionally `u_tilt/v_tilt/w_tilt` (PF-only) for diagnostics.
- **Per-window ancillaries** (dim `record` = window start): `ustar`, `L`, `wdir`,
  `spike_flag`, `nan_flag`, `ssitc_*` pulled from the averaged pkl by header lookup +
  timestamp match. This satisfies gameplan §1's "optional per-window ancillaries" and
  §3.5's "report z/L alongside".
- **Global attrs:** `site_id` (site folder name), `lat`/`lon` (from new `SiteInfo`
  fields; write NaN + warning until populated), `sampling_frequency_hz`
  (`SiteInfo.tableScanFrequency` entry for the fast table), `flux_averaging_s`
  (`avgPer*60`), `canopy_height`, `elevation`, `displacement_height`,
  `rotation = "planar_fit+yaw (applied upstream by UTESpac)"`, `pf_type` (GPF/LPF),
  `detrend_upstream` (linear/constant — what the pipeline used; the analysis package
  applies its own selectable detrend regardless), `source_files`, `utespac_version`,
  `git_commit`.
- **PF coefficients:** copy the applied `[b0,b1,b2]` per height/sector/date-bin out of
  `PFinfo.pkl` into a small `/planar_fit` group (or attrs) for provenance, even though
  rotation is pre-applied.

### A.3 Gameplan amendments, module by module

- **§0.1 Rotation** — becomes: verify `mean(w) ≈ 0` per window on `wPF`; no transform.
  The optional **double-rotation cross-check cannot be run from current raw pkls**
  (unrotated u,v,w are not saved). If that cross-check matters, the implementation chat
  should propose adding unrotated `u/v/w` (or just the yaw angle series) to the `raw`
  dict in `fluxes.py` — a small, additive UTESpac change to be made deliberately, not
  silently.
- **§0.2 Detrend** — implement fresh (block-mean / linear / filter) on the netCDF
  series. `utespac/nandetrend.py` is the NaN-tolerant reference (including its >90 %-NaN
  all-NaN rule); either import it or reimplement with `scipy.signal.detrend` + explicit
  NaN handling. Do not consume the stored `*Prime` arrays (fixed upstream detrend).
- **§0.3 QC pass-through** — flags come from the averaged pkl (via converter), one flag
  set per window per table, not per sample. Note in provenance that spikes were
  *interpolated* upstream (Vickers & Mahrt 1997 style, `condition_data.py`), so
  extreme-value statistics (ramp amplitudes, quadrant tails) see slightly smoothed data.
- **§0.4 Taylor** — Ū per window from `spd` (already tower-relative corrected) or
  `hypot(mean(uPF), mean(vPF))`; after yaw rotation mean(vPF)≈0 so mean(uPF) suffices.
- **§3.1 Spectra** — sampling frequency differs per site (10 vs 20 Hz); carry `fs` from
  the netCDF attr, never hardcode. 30 min → 36 000 samples @20 Hz, 18 000 @10 Hz.
- **§3.2 MRD** — 2^M trim: 32 768 of 36 000 @20 Hz (retains 91 %), 16 384 of 18 000
  @10 Hz (91 %). Record trim policy per window as planned.
- **§3.3 Quadrant/octant** — UTESpac already computes quadrant-adjacent per-window
  metrics: `delta_flux_ctrb`, `delta_time_ctrb` (`utespac/find_delta_flux.py`,
  `find_delta_time.py` — ejection−sweep asymmetry of a covariance) and transport
  efficiency `eta` (`find_eta.py`). The new quadrant module at hole size H=0 should
  **cross-validate** against these columns in the averaged pkl — a free correctness test.
  Octant triplet (w′,T′,q′): use `rhov`′ for q′; available at 4 heights on siteIRGA, one
  (Li-7500) on siteGill.
- **§3.4 Ramps** — `utespac/calc_dissipation_rate.py` already implements a
  structure-function computation (third-order, for ε); reuse its windowing/NaN
  conventions when building S_n(r) for Van Atta. TKE ramps: e(t) from uPF′,vPF′,wPF′ —
  all three components are in the raw pkl, so the Mangan et al. TKE detection needs
  nothing extra.
- **§3.5 / §3.6 cutoffs** — `spectral_gap` default stands; z per level from `height`
  coord; z_i absent exactly as the gameplan's locked decision #3–4 assumed. z/L per
  window from the joined `L` ancillary.
- **§3.7 Coherent flux** — quadrant-derived estimator can also be checked against
  `delta_flux_ctrb` at H=0 as above.
- **Output file (§2)** — unchanged; write it next to UTESpac output as
  `<site>/output/<Site>_coherent_<...>.nc`. Naming should echo UTESpac's base-name
  convention (`save_data.py:40`) so files sort together.
- **Testing** — follow the repo's existing pattern: pytest + a small pinned regression
  window (see `tests/test_regression.py`, `utespac/testkit.py`). Pick one clean 30-min
  window per site (one 20 Hz multi-height, one 10 Hz single-height) as fixtures; assert
  Parseval/sum-closure identities (∫S=σ², ΣD_xy=⟨x′y′⟩, ΣS_i,0=1) — these are free,
  strong invariants for every module.

### A.4 Data inventory quick reference (for the implementation chat)

**Raw pkl** (`<Site>_raw_<PF>_<Det>_<date>.pkl`, only when
`info["saveRawConditionedData"]=True`): `t` (n,), MATLAB serial datenum; `z` (nSonics,);
`uPF,vPF,wPF,u_tilt,v_tilt,w_tilt,WD,spd,sonTs,Theta_v_son` all (n, nSonics);
`fwThPrime,fwTh,fwT` (fine-wire, if present); `rhov,rhovPrime,rhovextenalPrime`
(n, nH2O); `rhoCO2,rhoCO2Prime,rhoCO2extenalPrime` (n, nCO2); `P` (n, 2)=[t, kPa].
Built at `utespac/fluxes.py:624-667,920-932,1015-1024,1150-1158`.

**Averaged pkl**: dict of (nWindows × cols) matrices each with `<name>Header` string
list; col 0 = window serial datenum. Of interest here: `tau` (u*), `L`, `spdAndDir`,
`tke`, `sigma`, `H`, `LHflux`, `CO2flux`, `fluxQC` (SSITC), `delta_flux_ctrb`,
`delta_time_ctrb`, `eta`, `<table>SpikeFlag`, `<table>NanFlag`. Reader:
`utespac/get_data.py` (concatenates multiple days, handles both pkl kinds).

**PFinfo.pkl**: `pf_info["cm_<h*100>"]["day_<d0>to<d1>"]["degrees_<lo>_to_<hi>"] = [b0,b1,b2]`
(`utespac/find_global_pf.py:352-380,487`). Matrix reconstruction:
`utespac/sonic_rotation.py:252-270`.

**SiteInfo** (`utespac/site_config.py`): `sonics` (height/orientation/manufacturer/
hmp_height), `tableNames`, `tableScanFrequency`, `canopyHeight`, `displacementHeight`,
`siteElevation`, `tower`, `angle`, `ascending`, SSITC settings. Loaded from
`siteInfo.toml` (preferred) or `siteInfo.py`.

**Sites on hand**: Gill (formerly `siteGill20250723_20250828`, no longer in the repo; 10 Hz, Gill @ 51.5 m, canopy 19.3 m,
Li-7500 gas), IRGA (formerly `siteIRGA20250723_20250828`, no longer in the repo; 20 Hz, 4 IRGASONs @ 4.42–32.18 m),
`siteVAC001_20230706_20230720` (20 Hz).

---

## Part B — De-MATLAB roadmap for `utespac/`

The library is a faithful port of ~33 MATLAB `.m` files (kept in-repo under
`UTESpac_MATLAB/` for reference). Real modernization has already happened —
`SiteInfo`/`SonicLevel` dataclasses with TOML loading, the `testkit` comparison engine,
a pytest suite, an argparse CLI, tower-profile height-keyed lookup — but the compute
core still carries the MATLAB data model. Items below are ordered by leverage;
each names the pattern, where it lives, and what it should become.

### B.1 Root cause: unlabeled parallel arrays → adopt labeled data (xarray/pandas)

Everything else follows from this. The pipeline's currency is 2-D float matrices whose
columns mean something only by position, with names in separate `<name>Header` lists
(33 such assignments across `avg.py`, `wind_stats.py`, `sonic_rotation.py`, `fluxes.py`,
`get_data.py`, `find_global_pf.py`). `fluxes.py` is the extreme case: ~20 preallocated
NaN matrices addressed by hand-computed strides (`c_H = 3 + ii*12`,
`num_sig_vars = 13 if has_fw else 12` — a stride chosen to leave a NaN gap column
matching MATLAB's memory layout, `fluxes.py:224-227`), then ~200 lines
(`fluxes.py:1240-1412`) rebuilding header strings, including a preserved MATLAB typo
(`"skew_Theata_v"`, `fluxes.py:1338`) and a preserved duplicate column
(`fluxes.py:1281`).

**Target:** `xarray.Dataset` with `time`/`height` coords as the canonical in-memory and
on-disk model (HF data), and either xarray or pandas DataFrames for per-window products.
Column-stride arithmetic, `<name>Header` bookkeeping, and the label-reconstruction code
in `save_data._save_csv` all disappear. `sensor_info` (ndarrays whose columns are
`[table_idx, col_idx, height, bearing, manufacturer]`, `find_instruments.py:33-81`)
becomes a list of small dataclasses/records — `SonicLevel` already shows the idiom.

**Strategy (strangler fig, boundaries inward):** (1) convert at the I/O boundary first —
`save_data`/`get_data` emit and read labeled structures (the A.2 converter is the first
such artifact); (2) all *new* code (`ec_coherent`) is written natively labeled;
(3) migrate `utespac` stage by stage (`avg` → `wind_stats` → `sonic_rotation` →
`fluxes`), each behind the existing regression suite. Never a big-bang rewrite —
`testkit` + `tests/test_regression.py` are the safety net that makes incremental
migration cheap.

### B.2 Time: retire MATLAB serial datenums

Every persisted timestamp is a float in days since year 0 (`MATLAB_EPOCH = 719529.0`,
`campbell_date.py:7`). Symptoms: `simple_avg.py:46-49` *identifies* the timestamp column
by checking whether a float falls between the datenums for 2000 and 2030; averaging math
in fractional days (`dt_days = avg_per/(24*60)` in four modules); filename dates parsed
by fixed character offsets (`find_files.py:99-105`); per-element Python-loop datetime
conversion (`campbell_date.py:45-71`); PF date bins stored as strings
`'day_730485to730515'` parsed back with regex (`sonic_rotation.py:87-90`).

**Target:** `numpy.datetime64`/`pandas.DatetimeIndex` end-to-end. Datenum conversion
survives only as a boundary shim in `campbell_date.py` for reading legacy pkls and for
MATLAB-parity testing in `testkit`. With labeled time (B.1), the magic-column detection
and the days-arithmetic go away together; windowing becomes `resample`/`groupby`.

### B.3 Config and signatures: one typed config, structured returns

`info` is a ~30-key nested dict created at module scope (`utespac_main.py:28-144`),
mutated by nearly every stage (`find_files.py:58`, `find_serial_date.py:46`,
`site_config.apply_to`), and even pulled from `__main__` globals when omitted
(`utespac_main.py:198-201`). Functions take 5–7 positional args and return bare tuples
of 2–4 items (`sonic_rotation` returns 4, `find_files` returns 4, `fluxes` returns 2).

**Target:** extend the `SiteInfo` precedent to a frozen `RunConfig` dataclass (QC
thresholds, PF settings, detrend, output flags) built once in `utespac_main` — no
mid-pipeline mutation; values a stage *derives* (e.g. `info["date"]`) travel in that
stage's return object instead. Multi-returns become small dataclasses
(`RotationResult(rotated, pf_only, ...)`). Delete the `__main__` fallback. This also
makes the pipeline callable as a library (notebooks, `ec_coherent` cross-checks, tests)
without touching module globals.

### B.4 Separate UI from library

`input()` prompts are embedded in library modules: 8 in `find_global_pf.py`
(sector boundaries, date barriers, confirmations — several with `# MATLAB:` comments),
site/date selection in `find_files.py:55,139` and `get_data.py:62`, and a confirm +
`SystemExit` inside `_display_run_summary` (`utespac_main.py:179`). Progress is 100 %
`print()` (no `logging` anywhere), including physical values printed mid-computation
(`fluxes.py:212-215`, `sonic_rotation.py:182`), and the main loop swallows all
exceptions via `warnings.warn` + traceback (`utespac_main.py:265-267`).

**Target:** all decisions become config/arguments (sector boundaries and date barriers
belong in `SiteInfo`/`RunConfig` or a saved PF-setup file — they are *site facts*, not
runtime answers); prompting lives only in the CLI entry point as a thin wrapper that
fills the config. Replace `print` with the `logging` module (per-module loggers; INFO
progress, DEBUG diagnostics). Let per-file failures raise a typed error collected by the
driver, not a blanket `except Exception`. This is also what `ec_coherent`'s batch CLI
needs — it cannot block on stdin.

### B.5 Consolidate one-function-per-file modules

~35 modules mirror MATLAB `.m` files one-to-one (`find_eta.py` is 25 lines/one function;
similarly `find_delta_flux.py`, `find_delta_time.py`, `stp_dn.py`, `strfndw.py`,
`nandetrend.py`, `campbell_date.py`, ...). **Target grouping:** `times.py` (campbell/serial),
`qc.py` (condition_data, consec_flag_removal, nan rules), `averaging.py` (avg,
simple_avg, stp_dn), `rotation.py` (sonic_rotation, pf_coefficients, find_global_pf
numerics), `turbulence.py` (find_eta, find_delta_*, calc_dissipation_rate, strfndw →
private helper), `io.py` (load/save/get_data, import_header). Keep the old module names
as one-line re-export shims for a deprecation window so tests and user scripts don't
break.

### B.6 Persistence: self-describing formats instead of pickle

Primary artifacts are pickles of dict-of-matrices (direct `.mat` analog), with metadata
encoded in underscore-joined filenames (`save_data.py:40`) and globbed back with
MATLAB-translated patterns (`get_data.py:75-83`). `PFinfo.pkl` is a nested dict whose
*keys* encode data (`cm_5150`, `day_730485to730515`) that consumers re-parse with
regex/arithmetic. The netCDF writer emits no attributes, no real dimensions, no CF time.

**Target:** netCDF (via the B.1 labeled model) for HF and averaged output — the A.2
converter defines the schema; CSV export stays for humans. `PFinfo` becomes a small
netCDF/JSON with real dims `(height, date_bin, sector)` and explicit `b0,b1,b2`
variables plus bin-edge coordinates — no string-key parsing. `get_data`'s defensive
NaN-block insertion on shape mismatch (`get_data.py:158-179`) is subsumed by
`xarray.concat(..., join="outer")`.

### B.7 Vectorization and shared primitives

The window-slicing idiom `bp = np.round(np.linspace(0, n, N+1))` is repeated in
`avg.py:34`, `simple_avg.py:73`, `condition_data.py:99`, `fluxes.py:222` — extract one
window-iterator (or replace with `resample` once time is datetime64; B.2). The
computational core is a scalar-per-cell double loop (`for ii in sonics: for jj in
windows:`, `fluxes.py:695-848`); with labeled windows, most covariances vectorize with
`groupby().cov()`-style reductions. Smaller items: `consec_flag_removal`'s shifted-stack
loop → convolution/`sliding_window_view`; per-column `for c in range(n_cols)` QC loops
in `condition_data.py:86-145` → broadcasting; per-element datetime loops (B.2);
`interp1d` imported inside a function body (`condition_data.py:28`). Keep `nandetrend`'s
*semantics* (NaN preservation, >90 %-NaN rule) even if the internals change — they are
QC policy, not incidental.

### B.8 MATLAB byte-parity artifacts behind a compat flag

Some code exists solely to match MATLAB output byte-for-byte: the NaN gap-column stride
(`fluxes.py:224-227`), the duplicate column (`fluxes.py:1281`), the preserved header typo
`skew_Theata_v` (`fluxes.py:1338`), `.mat`-style glob patterns, and 1-based `rows`
arguments (`get_data.py:95`, `find_files.py:150`). While MATLAB regression comparison is
still active, gate these behind an explicit `matlab_compat=True` option so the clean
path can exist in parallel; when parity testing is retired, delete them in one commit.
`testkit`'s pinned expected values should be re-baselined to the clean output at that
point.

### B.9 Sequencing

1. **Now (with `ec_coherent` work):** A.2 converter + `lat`/`lon` in `SiteInfo`;
   `ec_coherent` written natively in the target idiom (xarray, datetime64, typed config,
   logging, no prompts) — it becomes the reference for what migrated `utespac` code
   should look like.
2. **Short term:** B.3 (RunConfig, structured returns) + B.4 (prompts out of library,
   logging) — mechanical, low-risk, immediately improves scriptability.
3. **Medium term:** B.1/B.2 boundary-first migration (`save_data`/`get_data` → labeled;
   then `avg`/`wind_stats`; then `sonic_rotation`; `fluxes` last), each step gated by
   `tests/test_regression.py`. B.5/B.6 fall out along the way.
4. **Last:** B.8 removal once MATLAB parity is formally retired.
