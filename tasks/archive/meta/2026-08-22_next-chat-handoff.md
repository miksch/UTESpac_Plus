# Handoff: after the audit and the VAC001 validation pass

Opened 2026-08-22 at the close of the audit session; updated the same day
after the second session landed items 1, 2, 4 and step 2 of item 5 below. State of the tree:
the audit doc
[2026-08-22_code-audit-and-python-gameplan.md](../../active/2026-08-22_code-audit-and-python-gameplan.md)
holds the findings, the VAC001-vs-EddyPro results, and the migration plan;
[board.md](../../board.md) holds the queue;
[tests/KNOWN_DIVERGENCES.md](../../../tests/KNOWN_DIVERGENCES.md) is the
known-divergence ledger the audit asked for (one row per deliberate
difference from MATLAB, with the expected magnitude). Landed so far: the
`data/` per-site tree and `tasks/`/`testbed/` silos, the planar-fit
rotation fix, VAC001 processed end to end with five validation scripts in
`testbed/scripts/`, label-aligned `get_data`, the new dissipation
estimator, slope geometry in `SiteInfo`, the finding-7 fixes, the
Schotanus/WPL temperature flux, and the sourced ITC σw/u* reference.
Suite: 114 passed, 1 skipped.

How to run anything here: repo root, `conda run -n UTESpac_Plus python
testbed/scripts/<script>.py` (conda at
`C:\Users\mdmiksch\AppData\Local\miniconda3\condabin\conda.bat`; set
`PYTHONIOENCODING=utf-8`, otherwise conda's output re-print dies on
non-ASCII after the script has finished). The pipeline on VAC001:
`run_vac001.py` (LPF) / `run_vac001_gpf.py` (GPF, prompts scripted,
`--reuse-pf` keeps `PFinfo.pkl`); comparisons:
`compare_vac001_eddypro.py --pf GPF --det ConstDet`,
`plot_vac001_eddypro.py`, `pf_vac001_eddypro.py`, `closure_vac001.py`. A
full LPF run of the 16-day record takes ~30 min; three days take ~8.

## 1. Schotanus and WPL temperature flux — DONE 2026-08-22

Sources (all four PDFs now in `library/`, rows in `index.md`, entries in
`references.bib`; Schotanus 1983 and Webb 1980 are OCR scans with garbled
equation typography — render the pages if an equation form is in doubt):
Schotanus1983 (10.1007/BF00164332), Kaimal1991 (10.1007/BF00119215),
Webb1980 (10.1002/qj.49710644707), Liu2001 (10.1023/A:1019207031397). The
note [library/writeups/sonic_temperature_flux.md](../../../library/writeups/sonic_temperature_flux.md)
carries the equations with page/equation loci and the symbol each one
lands on.

Code: `utespac/sonic_temperature.py` (`SONIC_HUMIDITY_COEFF = 0.51`,
`air_temperature_from_sonic`, `air_temperature_perturbation`);
`fluxes.py` rescales `theta_son_air` with 0.51, and inside the H2O block
rebuilds the `T_air'w'`/`T_air'wPF'` columns as w′Ts′ − 0.51·T̄·w′q′ with
q′ = ρv′/ρ̄ from the level's hygrometer, then drives `kin_sen_flux` (LE
W/m², KH2O O₂, CO2 WPL) and the sample-wise `rhov_ext`/`rhoc_ext` with
that w′T′. `info["matlabCompat"]` restores 0.61 and the buoyancy-flux
driver. The crosswind term of Schotanus eq. 8 / Liu eq. 12 is not applied
(CSAT3/IRGASON heads correct it internally, Kaimal & Gaynor eq. 4). Where
no high-frequency humidity exists the columns fall back to the
mean-humidity rescale, which corrects the mean only.

VAC001 GPF ConstDet re-run and compared (`compare_vac001_eddypro.py --pf
GPF --det ConstDet`, 712 periods): H `T_air'wPF'` vs EddyPro H bias
−0.03 W/m², RMSE 0.75, slope 1.001, r² 1.000 (buoyancy column unchanged at
+11.1 / 15.7). Split as required for the advective site: H < 0 (450
periods) bias −0.16, RMSE 0.52; H > 0 (262) +0.18, 1.03; day 08–18 h
+0.24, 0.95; night −0.24, 0.55. Fc slope 0.940 → 0.975, bias +0.63 →
+0.10 µmol m⁻² s⁻¹. LE slope 0.989 → 0.984 (bias −2.4 W/m² on a 180 W/m²
mean): 1 % of that is EddyPro's spectral correction (`LE_scf` median
1.0105), not applied in UTESpac; the buoyancy-driven term had been
inflating LE by ~0.5 %. Numbers in the audit doc "VAC001 test dataset".
The LPF ConstDet and LinDet outputs were regenerated with the same code
after the GPF run so every VAC001 product carries the correction.

Not done here, noted for later: a config option to consume the logger's
`T_SONIC_corr` directly (EasyFlux does the same correction on the logger).

## 2. `get_data` blanking whole days — DONE 2026-08-22

`get_data` now concatenates any 2-D field that carries a
`<field>Header`/`<field>header` by label: the output columns are the
union across files and a file missing a label contributes NaN in that
column only. Unlabeled fields and `matlab_compat=True` keep the MATLAB
shape check. `fluxes.py` trims `derivedTheader` together with the
all-NaN columns of `derivedT` (it had not, which left the header wider
than the data on the outage days). The VAC001 07-12, 07-14, 07-16 outputs
(LPF LinDet, LPF ConstDet, GPF ConstDet, plus raw) were regenerated with
that header fix; the 16-day LPF ConstDet load is 768 rows with 53 all-NaN
rows (genuine gaps) where the old path gave 341. `tests/test_get_data.py`
pins the behaviour.

## 3. Bring the other sites into `data/` and regenerate their GPF products — closed 2026-08-22

The legacy `siteGill…`/`siteIRGA…` folders held only headers and
`siteInfo` (no 48-h inputs, outputs or PFinfo). They were moved to
`data/Gill` / `data/IRGA` and then, at the user's request, removed from
the repo altogether; `tests/test_tower_profile.py` keeps both tower
profiles as inline fixtures. What remains of the item is off-repo work:
the GPF regeneration with the fixed rotation (every earlier GPF product
off at the order the VAC001 before/after showed, u* +18 %, LE +20 %) and
the per-height/sector 3D planar-fit figure, wherever those sites'
`siteInfo` and inputs live (board: rotation line, 3D-figure line). Earlier
in the session the user's "don't reprocess siteGill and such" was recorded
here as deferring the whole item; the user clarified that only the
reprocessing was meant.

## 4. The remaining audit fixes — DONE 2026-08-22

Dissipation (finding 5): `calc_dissipation_rate` averages
D_LL/(C2 r^(2/3)) over lags 0.1–2 s (C2 = 2.0, r = ū τ) and raises to
3/2; `calc_structure_function` takes `max_lag`, so only the needed lags
are computed. Validation: on a synthetic series with
E11 = 0.49 ε^(2/3) k^(−5/3) the estimate matches the ε implied by the
signal's own discrete D_LL to 0.1 % (the 13 % shortfall against the
nominal ε is Nyquist truncation of the synthesis, shown by computing
D_LL from the discrete spectrum); on VAC001 07-08 (GPF raw uPF, 96
periods, all near neutral, u* 0.9–1.2 m/s) φ_ε = κ z ε / u*³ has median
1.15 (IQR 1.07–1.27). `info["calcDissipation"]` defaults False, so no
stored VAC001 output carries the new ε yet.

Slope geometry (finding 6): `SiteInfo` gained `downslopeAspect` (fall-line
direction from north, deg) and `slopeAxis` ("u" or "v": the planar-fit
horizontal axis that points downslope). `calc_snsp_angle(phi, alpha,
downslope_aspect)` takes the aspect as an argument; `fluxes.py` assigns
`u_tilt` to the along-slope PF component from `slopeAxis` (the old
unconditional swap was the `"v"` case) and warns when `angle != 0` with
either key unset, then uses the French Meadows values.

Finding-7 batch: `Vtheta_fw` uses `q_ref_fast_local` (MATLAB
`fluxes.m:566`); `sat_vapor_pressure` in `rh_to_spec_hum.py` (Stull 1988
eq. 7.5.2d, p. 276) is the single e_sat used by `rh_to_spec_hum` (now
q = 0.622e/(P − 0.378e)) and `get_virtual_pot_temp`; the IRGA-derived
reference humidity is ρ_v/(ρ_d + ρ_v) with ρ_d from P − e;
`find_delta_time` divides by valid samples; `nandetrend` fits on the true
index; `find_eta` and the CO2 ppm header state their conventions; the
`sigma` ddof=0 convention is commented at the site and in the ledger.
The above-canopy ITC reference is now sourced
([library/writeups/itc_sigmaw.md](../../../library/writeups/itc_sigmaw.md)):
the unstable side (1.3 for −0.032 < z/L < 0, 2.0(−z/L)^(1/8) below) is
Foken 2008 Table 2.11 = Foken et al. 2004 Table 9.1 = Foken et al. 2012
Table 4.2; the stable side 0 ≤ z/L ≤ 0.4 uses Thomas & Foken 2002's
Coriolis form (0.21 ln(z+ f/u*) + 3.1, Table 2.12) through the new
`SiteInfo.latitude` (VAC001: 38.300056 from
`eddypro/metadata/vac_001_2023_iop.metadata`; the legacy FM `siteInfo.py`
files have none yet). z/L > 0.4, and the stable side without a latitude,
now use Pahlow, Parlange & Porté-Agel 2001 eq. 14 (σw/u* = 1.1 + 0.9
(z/L)^0.6, Boundary-Layer Meteorol. 99, 225-248, DOI
10.1023/A:1018909000098; PDF in `library/`), so nothing on the reference
is `[ASSUMED]` any more; the 15 % step between the two sourced forms at
z/L = 0.4 is recorded in the note and the ledger. `tests/test_ssitc_itc.py` pins
the current branches.

None of these are behind `matlabCompat`; the parity consequences are
rows in the ledger, with "magnitude: to measure" where only the parity
set (board: validation, first item) can put a number on them.

## 5. Migration steps 2–5 and ec_coherent — steps 2 and 3 DONE 2026-08-22, next: step 4

Closed 2026-08-22 after step 3 landed (labeled I/O boundary: `utespac/labeled.py`,
`utespac/pf_info.py` + `PFinfo.json`, `utespac/export_hf.py`, datetime64
shims, `SiteInfo.longitude`; record in the audit doc "Migration gameplan"
step 3, board line under migration). This handoff is archived; steps 4–5
stay on the migration board line, nothing else from here is open.

Step 2 (config and UI) is in: `utespac/config/` holds `run.toml`, `qc.toml`,
`pf.toml`, `flux.toml` (commented, mirroring the dataclass defaults;
`[ASSUMED]` on the inherited thresholds); `utespac/run_config.py` has the
frozen `RunConfig` / `QCConfig` / `PFConfig` / `FluxConfig` with
`from_config` (kwarg > dataclass/dict > `config/<name>.toml` in the cwd >
packaged TOML > default) and `RunConfig.to_info()`, which renders the
legacy `info` dict the stages still read — the bridge step 4 retires.
`utespac/pipeline.run_utespac(config, site=, dates=, prompter=)` returns a
`RunResult` (per-date status and written paths); it also accepts a legacy
`info` + `template`. Every `input()` left the core: `find_global_pf` takes a
prompter (`utespac/prompts.py`: `ScriptedPFSelection` is the default —
single sector, all dates — and `ConsolePFPrompter` carries the interactive
session with the figures); `find_files`/`get_data` require the site.
`print` became `logging.getLogger("utespac")`. `utespac_main.py` is the CLI
(site/date prompts, flags, `--no-prompts`, `--reuse-pf`, `--run-config`).
`testbed/scripts/run_vac001*.py` use the API (no `builtins.input`
monkeypatch). Acceptance: VAC001 date 1 GPF ConstDet (averaged + raw) and
LPF ConstDet regenerated through the new API are bit-identical to the
products already on disk; suite 114 passed, 1 skipped. Site facts stayed in
`siteInfo.toml`; the SSITC sub-averaging, canopy and displacement settings
are still site keys (they are per-site by nature).

Next is step 3 (labeled I/O boundary): the netCDF converter (`lon` to add to
`SiteInfo`, `latitude` is there), `save_data`/`get_data` on labeled
structures, datetime64 at the boundary with `campbell_date` as the legacy
shim, `PFinfo` with explicit dims. Step 4 (stage-by-stage core migration,
`fluxes.py` split) wants the pinned fixtures from the parity-set board line
first. ec_coherent Phase 0 (library notes) can run in parallel and its
converter is step 3's.

## User-side items noted

- `siteVAC001_20230706_20230720/card_convert/` is a byte-identical copy of
  `data/VAC001/raw/fast/` (3.9 GB) and can be deleted; the old folder's
  `siteInfo.py` is superseded by `data/VAC001/siteInfo.toml`.
- VAC001 `tower = 180` is `[ASSUMED]` (no effect on this IOP); sonic
  height kept at 10.85 m against EddyPro's rounded 11.00 m.
- Closure used provisional Rn (assembled outside the repo) and the rebuilt
  `ghf_avg`; `closure_vac001.py` re-runs in seconds if either is revised.
- VAC001 `2023_07_06` LPF ConstDet (averaged + raw) was re-written by the
  smoke run of the edited `fluxes.py`. For this site (RH path, one level,
  angle 0) the only numeric difference from its predecessor is the e_sat
  formula in q_ref, which moves q by ~1 % and the quantities built on it
  (ρ, θ_v, T_air, `specificHum`) by ~1e-4 relative; H, LE, u* and the
  WPL terms are untouched. The 07-08..07-20 files predate the edit.
