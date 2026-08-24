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
  [archive/audit/2026-08-22_code-audit-and-python-gameplan.md](archive/audit/2026-08-22_code-audit-and-python-gameplan.md).
- [BLOCKED 2026-08-24] Energy-balance closure re-run
  (`testbed/scripts/closure_vac001.py`): waits on the user finalizing
  Rn (`NETRAD`, assembled off-repo) and the ground heat flux in the
  `_soil_corr` slow table; the 2026-08-22 numbers are indicative only.
  Nothing in the flux ranking depends on their level. -- EFFORT S, RISK
  low. Source: audit doc "VAC001 test dataset" in
  [archive/audit/2026-08-22_code-audit-and-python-gameplan.md](archive/audit/2026-08-22_code-audit-and-python-gameplan.md).
## ec-coherent

(build-out closed 2026-08-24, see history.md; no open ec-coherent tasks)
