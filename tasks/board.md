# tasks board -- the active queue

One line per open task; format and lifecycle in [README.md](README.md).
Rationale, measurements and rulings live in the linked docs, not here.

## migration

- [ACTIVE 2026-08-22] Labeled inter-stage model (B.1 inward of the I/O
  boundary). S1-S3 landed 2026-08-22: `utespac/model.py` (`Sensors`,
  `Run` of xarray Datasets with datetime64 time, converters to/from the
  legacy structures) and `utespac/stages.py` (`load_run` → `condition` →
  `average` → `wind` → `rotate` → `flux` on the `Run`); the pipeline calls
  them and the wrappers `fluxes.py`/`sonic_rotation.py`/`avg.py`/
  `wind_stats()`/`condition_data()` are gone; fixture pins unchanged,
  VAC001 date-1 GPF within float resolution. Remaining (S4): the run
  netCDF writer/reader with groups, `get_data`/`find_global_pf` reading it,
  `load_products` across dates, `export_hf` as a view of `run.raw`, the
  legacy pickle's fate (DECIDE in the doc); gated by
  `tests/test_pinned_fixture.py`. User ruling
  2026-08-22: this comes before the ec_coherent build-out, so the
  architecture is settled first; in-memory model xarray end-to-end (user
  ruling 2026-08-22; xarray added to the env and pyproject). -- EFFORT L,
  RISK med. Source: integration notes B.1/B.2; the migration record in
  [active/2026-08-22_code-audit-and-python-gameplan.md](active/2026-08-22_code-audit-and-python-gameplan.md)
  "Migration gameplan" steps 4-5. Detail:
  [active/2026-08-22_labeled-run-model.md](active/2026-08-22_labeled-run-model.md).

## validation

- [BLOCKED 2026-08-22] 3D planar-fit validation figure per
  height/sector/date-bin: landed for the VAC001 10.85 m sonic (single
  sector, whole IOP) in `testbed/scripts/pf_vac001_eddypro.py`; the
  per-height form waits on the other VAC001 levels, which the user has
  not supplied yet (the French Meadows sites will not be processed --
  user ruling 2026-08-22). -- EFFORT S, RISK low. Source: findings 1-2 in
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
  the A.2 converter (`utespac.export_hf`) exists; VAC001 is the only site
  (user ruling 2026-08-22). Ordered after the labeled inter-stage model
  (migration line) so the architecture it builds on is settled. --
  EFFORT L, RISK med.
