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

- [PENDING 2026-08-22] De-MATLAB migration steps 2-5 (RunConfig/logging with
  dopli-style packaged config TOMLs in `utespac/config/`; labeled I/O
  boundary + netCDF converter; stage-by-stage core migration with fluxes.py
  split; retire parity artifacts). Gated on the utespac-core science fixes
  landing first. -- EFFORT L, RISK med. Detail:
  [active/2026-08-22_code-audit-and-python-gameplan.md](active/2026-08-22_code-audit-and-python-gameplan.md)
  "Migration gameplan".

## validation

- [PENDING 2026-08-22] Assemble parity set: MATLAB .mat + PFinfo + run
  settings for all three sites (both PF modes); commit one pinned clean
  30-min fixture per site class. -- EFFORT S, RISK low. Detail: audit doc
  "Validation data to assemble".
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
## meta

- [ACTIVE 2026-08-22] Next-chat handoff: items 1 (Schotanus/WPL), 2
  (`get_data`) and 4 (audit fixes) landed 2026-08-22; item 3 closed by the
  user removing the Gill/IRGA folders (GPF regen for them runs off-repo);
  item 5 (migration steps 2-5, ec_coherent) picked up 2026-08-22, step 2. Detail:
  [active/2026-08-22_next-chat-handoff.md](active/2026-08-22_next-chat-handoff.md).

## ec-coherent

- [BLOCKED 2026-08-22] ec_coherent build-out per
  [../testbed/2026-08-12_ec_coherent_gameplan.md](../testbed/2026-08-12_ec_coherent_gameplan.md).
  Blocked on the GPF coefficient fix (must not consume pre-fix `uPF/vPF/wPF`
  pickles); Phase 0 library work can proceed. -- EFFORT L, RISK med.
