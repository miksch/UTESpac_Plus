# tasks board -- the active queue

One line per open task; format and lifecycle in [README.md](README.md).
Rationale, measurements and rulings live in the linked docs, not here.

## utespac-core

- [PENDING 2026-08-22] GPF coefficient indexing + b0 removal: fix landed in
  `utespac/rotation.py` (`PlanarFit.apply`, `apply_global_fit`), pinned by
  `tests/test_planar_fit.py`, verified on VAC001 against EddyPro's
  planar fit. Remaining: regenerate GPF outputs/raw pickles for the
  French Meadows sites (Gill, IRGA) -- nothing of theirs is in the repo
  any more (user removed the Gill/IRGA folders 2026-08-22), so this runs
  wherever their `siteInfo` and 48-h inputs live, reusing each site's
  PFinfo (the legacy pickle is still read; the run writes PFinfo.json). --
  EFFORT M (reprocessing), RISK low. Source: findings 1-2 in
  [active/2026-08-22_code-audit-and-python-gameplan.md](active/2026-08-22_code-audit-and-python-gameplan.md).
## migration

- [PENDING 2026-08-22] Labeled inter-stage model (B.1 inward of the I/O
  boundary). The stage numerics live in typed pieces (`averaging`,
  `rotation.rotate_sonics` -> `RotationResult`, `flux.reference`/`levels`/
  `engine`/`tables`), but the pipeline still passes the legacy `output`
  dict, `sensor_info` arrays and datenum time between stages through the
  wrappers `avg`/`wind_stats`/`sonic_rotation`/`fluxes`. Remaining: a
  labeled run object (tables + datetime64 time + sensor records) that the
  pipeline hands from stage to stage, the wrappers retired, `save_data`
  writing from it; gated by `tests/test_pinned_fixture.py`. -- EFFORT L,
  RISK med. Source: integration notes B.1/B.2; the migration record in
  [active/2026-08-22_code-audit-and-python-gameplan.md](active/2026-08-22_code-audit-and-python-gameplan.md)
  "Migration gameplan" steps 4-5.

## validation

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
