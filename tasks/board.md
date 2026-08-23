# tasks board -- the active queue

One line per open task; format and lifecycle in [README.md](README.md).
Rationale, measurements and rulings live in the linked docs, not here.

## utespac-core

- [PENDING 2026-08-22] GPF coefficient indexing + b0 removal: fix landed in
  `utespac/sonic_rotation.py` (legacy behaviour behind `info["matlabCompat"]`),
  pinned by `tests/test_planar_fit.py`, verified on VAC001 against EddyPro's
  planar fit. Remaining: regenerate GPF outputs/raw pickles for the
  French Meadows sites (Gill, IRGA) -- nothing of theirs is in the repo
  any more (user removed the Gill/IRGA folders 2026-08-22), so this runs
  wherever their `siteInfo` and 48-h inputs live, reusing each site's
  PFinfo.pkl; and add `matlabCompat` to the testkit parity runs. --
  EFFORT M (reprocessing), RISK low. Source: findings 1-2 in
  [active/2026-08-22_code-audit-and-python-gameplan.md](active/2026-08-22_code-audit-and-python-gameplan.md).
## migration

- [ACTIVE 2026-08-22] De-MATLAB migration steps 2-5. Step 2 landed
  2026-08-22 (packaged TOMLs + `RunConfig`, `run_utespac`/`RunResult`,
  prompts behind `utespac/prompts.py`, `logging`, thin CLI). Step 3 landed
  2026-08-22: `utespac/labeled.py` (labeled tables/DataFrames of the
  averaged output, CF netCDF writer/reader that is the pickle's twin;
  `save_data` writes it, `get_data(fmt="nc")`/`get_frames` read it),
  datetime64 shims in `campbell_date`, `PFinfo.json` via
  `utespac/pf_info.PFTable` beside the legacy pickle, the A.2 HF converter
  `utespac/export_hf.py` (`python -m utespac.export_hf`), `SiteInfo.longitude`;
  VAC001 date-1 GPF averaged pickle bit-identical, suite 132 passed.
  Step 4 stages 1-3 landed 2026-08-22 against the pinned fixture
  (`tests/fixtures/vac001_1hz`): `utespac/averaging.py` (period
  arithmetic once; `avg`/`simple_avg`/`stp_dn` wrappers), `wind_stats`
  primitives shared by `find_global_pf` and `fluxes`, `utespac/rotation.py`
  (`PlanarFit`, `rotate_sonics` → `RotationResult`, `sonic_rotation` as
  the wrapper on `PFTable`); VAC001 date-1 GPF products bit-identical,
  suite 154 passed. The `fluxes` split landed the same day as
  `utespac/flux/` (`reference`, `levels`, `engine`, `tables`; `fluxes.py`
  the orchestrator), verified by the fixture, an eight-variant old-vs-new
  A/B (≤ 1e-13) and VAC001 date-1 GPF on 20 Hz data (≤ 5e-13, summation
  order only; ledger row). Step 4 is complete; `PFinfo.pkl` is no longer
  written (step 5, safe part). Remaining: the rest of step 5 — retire
  the compat flag, the duplicate R column and `skew_Theata_v`, the MATLAB
  loaders in `testkit`, the legacy stage wrappers (pipeline on
  `rotate_sonics` and the flux pieces directly), re-pin the fixture — is
  BLOCKED on the DECIDE slot in the gameplan doc (retire MATLAB parity
  now vs after the parity set has run once; default: after). -- EFFORT M,
  RISK med. Detail:
  [active/2026-08-22_code-audit-and-python-gameplan.md](active/2026-08-22_code-audit-and-python-gameplan.md)
  "Migration gameplan" steps 4-5.

## validation

- [PENDING 2026-08-22] Assemble parity set: MATLAB .mat + PFinfo + run
  settings for all three sites (both PF modes). The committed pinned
  fixture landed 2026-08-22 as `tests/fixtures/vac001_1hz/` (one VAC001
  day at 1 Hz, LPF and GPF expected outputs, `tests/test_pinned_fixture.py`);
  what remains is the MATLAB side for `test_regression.py`. -- EFFORT S,
  RISK low. Detail: audit doc "Validation data to assemble".
- [ACTIVE 2026-08-22] 3D planar-fit validation figure: landed for VAC001
  (single sector, whole IOP) in `testbed/scripts/pf_vac001_eddypro.py`;
  still to do per height/sector/date-bin on the multi-sonic sites once
  they are migrated into `data/`. -- EFFORT S, RISK low. Source: findings
  1-2 in
  [active/2026-08-22_code-audit-and-python-gameplan.md](active/2026-08-22_code-audit-and-python-gameplan.md).
- [ACTIVE 2026-08-22] External-reference validation against EddyPro (ruled
  2026-08-22: EddyPro, not previous UTESpac runs; caveat -- EddyPro q'
  variances have shown unexplained issues, anchor on H/u*/L/momentum).
  VAC001 2023 IOP is the dataset (`data/VAC001/`, runner + comparison in
  `testbed/scripts/`). Done: LPF (linear/block) and corrected-GPF flux
  comparison, planar-fit coefficients, q-variance units, legacy-indexing
  before/after, energy-balance closure (Rn/G provisional), post-Schotanus
  re-comparison (H bias −0.03 W/m², Fc slope 0.975). Remaining: re-run
  closure if Rn/G are revised. -- GAIN decisive check that
  corrections improve the data, EFFORT S, RISK low. Detail: audit doc
  "VAC001 test dataset".
## ec-coherent

- [PENDING 2026-08-22] ec_coherent build-out per
  [../testbed/2026-08-12_ec_coherent_gameplan.md](../testbed/2026-08-12_ec_coherent_gameplan.md).
  Unblocked 2026-08-22: the GPF coefficient fix landed and every VAC001
  product (LPF and GPF, raw and averaged) has been regenerated with it;
  the A.2 converter (`utespac.export_hf`) exists. Only VAC001 pickles are
  in `data/`; any FM-site pickle from before the fix must not be consumed.
  -- EFFORT L, RISK med.
