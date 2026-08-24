# tasks history -- recently closed

Rolling log of roughly the last 15 closures; older records live only in
archive/. Format: `- [DONE|DROPPED YYYY-MM-DD] <one line> -- <link>`.

- [DONE 2026-08-24] ec_coherent build-out (steps 0-7 of the gameplan):
  the `ec_coherent` package (io, preprocess, spectra, mrd, quadrant/octant,
  ramps, ampmod, scales, coherent_flux, config, cli), 8 library notes with
  sources extracted, every DECIDE slot ruled, validation figure + script per
  module, and the step-7 close-out -- CLI pipeline test, all 24 VAC001 HF
  files processed, closure checker green over the set (worst error 2.9e-6),
  suite 242 green -- record in
  [archive/ec-coherent/2026-08-23_ec-coherent-buildout.md](archive/ec-coherent/2026-08-23_ec-coherent-buildout.md).
- [DONE 2026-08-23] `run_io` readers open each run file once:
  `read_run` and `load_products` take one `netCDF4.Dataset` handle and read
  every group through an `xr.backends.NetCDF4DataStore` on it, `run_files`
  sniffs `utespac_format` from the netCDF attributes without an xarray open.
  Measured on the VAC001 LPF LinDet 2023-07-06 run file: `read_run`
  0.88 -> 0.27 s, `run_files` over the 24 run files 3.5 -> 0.8 s; the Run
  read is identical field by field to the previous reader on a GPF and an
  LPF file, 164 tests pass.
- [DONE 2026-08-23] The 48 pre-netCDF VAC001 pickles (`VAC001_30minAvg_*.pkl`,
  `VAC001_raw_*.pkl`, ~14 GB) deleted by the user; the run files and HF
  netCDF regenerated the same day reproduce them (entry below).
  `data/VAC001/PFinfo.pkl` is still on disk (superseded by `PFinfo.json`,
  nothing reads it while the JSON exists).
- [DONE 2026-08-23] VAC001 products regenerated as the netCDF outputs:
  `run_vac001.py` (LPF, linear and constant detrend) and `run_vac001_gpf.py`
  (GPF, `PFinfo.json` reused) on all 8 dates with `saveRawConditionedData`
  -- 24 `utespac-run-2` run files and 24 `_hf_` HF netCDF in
  `data/VAC001/output`, the lone `utespac-averaged-1` file overwritten. The
  run files reproduce the pre-netCDF pickles field by field (max relative
  difference 7e-14, the duplicate `R` column and `skew_Theata_v` label being
  the parity retirement), the HF files equal the raw pickles to float32;
  `compare_vac001_eddypro.py` (GPF ConstDet joined table identical to the
  2026-08-22 one; H Tair'wPF' bias −0.03 W/m², LE 0.984, Fc 0.975) and
  `pf_vac001_eddypro.py` (b0 0.0425, b1 −0.0811, b2 0.0295) give the audit
  doc's numbers unchanged -- record in the audit doc "VAC001 test dataset";
  the pickle delete stays on the board.
- [DONE 2026-08-22] Labeled inter-stage model (closes the migration board
  line): `utespac/model.py` (`Sensors`, `Run` of xarray Datasets) and
  `utespac/stages.py` replace the MATLAB-shaped glue between stages;
  `utespac/run_io.py` writes/reads the `utespac-run-2` run netCDF, the only
  averaged product from here on (user ruling 2026-08-22: the end nc product,
  no pickles); the HF netCDF is written from the `Run` by
  `utespac.export_hf.write_hf`; `get_data` defaults to the run files,
  `load_products` gives them as Datasets; fixture pins unchanged -- record in
  [archive/migration/2026-08-22_labeled-run-model.md](archive/migration/2026-08-22_labeled-run-model.md).
- [DROPPED 2026-08-22] GPF regeneration for the French Meadows sites (Gill,
  IRGA) with the fixed rotation: the user will not process those sites;
  VAC001 (flat, so the slope-geometry keys stay unused) is the only site.
  The fix itself is in `utespac/rotation.py` and recorded in the audit doc,
  findings 1-2.
- [DONE 2026-08-22] De-MATLAB migration steps 4-5 (closes the migration
  board line). Step 4: pinned fixture `tests/fixtures/vac001_1hz/` +
  `test_pinned_fixture.py`; `utespac/averaging.py` behind `avg`/`simple_avg`/
  `stp_dn`; `wind_stats` primitives shared; `utespac/rotation.py`
  (`PlanarFit`, `rotate_sonics` → `RotationResult`, `sonic_rotation` on
  `PFTable`); the `fluxes` split into `utespac/flux/` (`reference`, `levels`,
  `engine`, `tables`) with `fluxes.py` as orchestrator; verified by the
  fixture, old-vs-new A/Bs and VAC001 date-1 GPF on 20 Hz data. Step 5:
  MATLAB parity retired on the user's ruling (EddyPro is the validation
  reference) -- `matlabCompat` gone from `RunConfig`/`run.toml`/rotation/
  flux/`get_data`/CLI, duplicate `R_wPF_CO2` column and `skew_Theata_v`
  label gone, `tests/test_regression.py`, `compare_outputs.py`,
  `run_test.py` and the `.mat` loaders in `testkit` deleted, `PFinfo.json`
  written alone (legacy pickle still read), fixture re-pinned; the
  parity-set validation line dropped -- record in
  [active/2026-08-22_code-audit-and-python-gameplan.md](active/2026-08-22_code-audit-and-python-gameplan.md)
  "Migration gameplan" steps 4-5 and [../tests/KNOWN_DIVERGENCES.md](../tests/KNOWN_DIVERGENCES.md).
- [DONE 2026-08-22] Next-chat handoff closed: items 1-4 landed or closed,
  item 5 steps 2-3 landed; the remaining migration steps 4-5 live on the
  migration board line, the user-side notes (card_convert copy deletable,
  `tower = 180` [ASSUMED], provisional Rn/G) in the archived doc --
  [archive/meta/2026-08-22_next-chat-handoff.md](archive/meta/2026-08-22_next-chat-handoff.md).
- [DONE 2026-08-22] Migration step 3 (labeled I/O boundary): `utespac/labeled.py`
  (labeled tables/DataFrames of the averaged output; CF netCDF writer/reader
  whose read-back is the legacy dict, written by `save_data`, read by
  `get_data(fmt="nc")`/`get_frames`), `campbell_date` datetime64 shims,
  `utespac/pf_info.PFTable` with `PFinfo.json` beside the pickle, the A.2
  HF converter `utespac/export_hf.py`, `SiteInfo.longitude`; VAC001 date-1
  GPF averaged pickle bit-identical, nc round trip zero diffs, HF file 135 MB
  in 9 s; suite 132 passed -- `tests/test_labeled.py`, `test_pf_info.py`,
  `test_campbell_date.py`, `test_export_hf.py`; record in
  [active/2026-08-22_code-audit-and-python-gameplan.md](active/2026-08-22_code-audit-and-python-gameplan.md)
  "Migration gameplan" step 3.
- [DONE 2026-08-22] Migration step 2 (config and UI): packaged stage TOMLs
  in `utespac/config/` with frozen dataclasses (`RunConfig`, `QCConfig`,
  `PFConfig`, `FluxConfig`; dopli resolution order; `to_info` bridge),
  `utespac.pipeline.run_utespac` returning `RunResult`, the planar-fit
  prompts moved behind `utespac.prompts` (scripted + console), `print` →
  `logging`, `utespac_main.py` reduced to the CLI; VAC001 date-1 GPF/LPF
  products bit-identical through the new API; suite 114 passed --
  `tests/test_config.py`, `tests/test_prompts.py`.
- [DONE 2026-08-22] Legacy `siteGill20250723_20250828` / `siteIRGA…` folders
  moved to `data/Gill` and `data/IRGA` (siteInfo at the site root, headers
  in `utespac/`; the IRGA 1-min header gained its `.dat`); no `site*` folder
  with tracked files remains at the repo root; `test_tower_profile` and
  `test_site_config` follow the new paths. Then removed outright at the
  user's request the same day (only headers and siteInfo, nothing to
  process here); `test_tower_profile` keeps the two profiles as inline
  fixtures. GPF regeneration for these sites stays on the rotation board
  line, to run where their data live.
- [DONE 2026-08-22] ITC σw/u* stable side beyond z/L = 0.4 (and without a
  latitude) on Pahlow et al. 2001 eq. 14, 1.1 + 0.9 (z/L)^0.6; the last
  `[ASSUMED]` in `calc_ssitc_flags._above_canopy_sigmaw` removed; PDFs of
  Pahlow2001, DeBruin1993, Nieuwstadt1984, Sorbjan1986 received --
  `library/writeups/itc_sigmaw.md`, `tests/test_ssitc_itc.py`, ledger row.
- [DONE 2026-08-22] ITC σw/u* reference sourced: unstable side cited to
  Foken 2008 Table 2.11 (= Foken et al. 2004 Table 9.1, Foken et al. 2012
  Table 4.2; 1.3 / 2.0(−z/L)^(1/8) at 0.032), stable side 0 ≤ z/L ≤ 0.4 on
  Thomas & Foken 2002 (Table 2.12, Coriolis form) with new
  `SiteInfo.latitude` (VAC001 38.300056 from EddyPro metadata); z/L > 0.4
  stays the MATLAB extension marked `[ASSUMED]` (no source on hand) --
  `library/writeups/itc_sigmaw.md`, `tests/test_ssitc_itc.py`.
- [DONE 2026-08-22] Schotanus temperature-flux correction and WPL driver
  (audit findings 3-4): `utespac/sonic_temperature.py` (0.51 coefficient,
  T' = Ts' − 0.51 T̄ q'), `fluxes.py` T_air columns from HF humidity and all
  WPL terms on that w'T' (`matlabCompat` keeps 0.61 + buoyancy flux); four
  sources extracted into `library/writeups/sonic_temperature_flux.md`.
  VAC001 GPF vs EddyPro: H bias +11.1 → −0.03 W/m² (RMSE 15.7 → 0.75), Fc
  slope 0.940 → 0.975 -- record in the audit doc "VAC001 test dataset" and
  [../tests/KNOWN_DIVERGENCES.md](../tests/KNOWN_DIVERGENCES.md).
- [DONE 2026-08-22] Minor-fixes batch (audit finding 7): `Vtheta_fw` on the
  level-local humidity, one e_sat formula (Stull 1988 eq. 7.5.2d) with the
  moist denominator in both humidity paths, IRGA q_ref from true dry-air
  density, `find_delta_time` over valid samples, `nandetrend` on the true
  index, `find_eta`/ppm/ddof conventions documented; ITC reference split off
  as a BLOCKED board line -- record in
  [../tests/KNOWN_DIVERGENCES.md](../tests/KNOWN_DIVERGENCES.md).
- [DONE 2026-08-22] Slope geometry into `SiteInfo` (`downslopeAspect`,
  `slopeAxis`; French Meadows fallback with a warning when `angle != 0`),
  `calc_snsp_angle` parameterized; keys added to the legacy FM siteInfo
  files -- audit finding 6, `tests/test_site_config.py`, `tests/test_utils.py`.
- [DONE 2026-08-22] Dissipation estimator replaced by the compensated
  inertial-subrange average (lags 0.1-2 s, C2 = 2.0); recovers a synthetic
  Kolmogorov signal's own D_LL epsilon to 1 %, VAC001 07-08 gives
  phi_eps = kappa z eps / u*^3 = 1.15 (IQR 1.07-1.27) near neutral -- audit
  finding 5, `tests/test_utils.py::TestDissipationRate`.
- [DONE 2026-08-22] `get_data` concatenates labeled fields by header label
  (union of columns, `matlab_compat=` keeps the MATLAB block-NaN); `derivedT`
  header trimmed with its data; VAC001 07-12..07-16 outputs regenerated, the
  16-day LPF load now has 53 all-NaN rows (gap periods) instead of 341 --
  `tests/test_get_data.py`, ledger in
  [../tests/KNOWN_DIVERGENCES.md](../tests/KNOWN_DIVERGENCES.md).
- [DONE 2026-08-22] Stand up dopli-style silos (`tasks/`, `testbed/scripts`
  + `scratch`) and the per-site `data/` tree; site discovery by `siteInfo.*`,
  inputs in `data/<SITE>/utespac/`; VAC001 raw/slow/EddyPro landed and
  processed to 48-h inputs -- record in
  [active/2026-08-22_code-audit-and-python-gameplan.md](active/2026-08-22_code-audit-and-python-gameplan.md)
  "VAC001 test dataset".
