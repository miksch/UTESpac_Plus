# tasks board -- the active queue

One line per open task; format and lifecycle in [README.md](README.md).
Rationale, measurements and rulings live in the linked docs, not here.

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

(build-out closed 2026-08-24, see history.md; no open ec-coherent tasks)
