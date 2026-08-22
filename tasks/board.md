# tasks board -- the active queue

One line per open task; format and lifecycle in [README.md](README.md).
Rationale, measurements and rulings live in the linked docs, not here.

## utespac-core

- [ACTIVE 2026-08-22] GPF coefficient indexing + b0 removal: fix landed in
  `utespac/sonic_rotation.py` (legacy behaviour behind `info["matlabCompat"]`),
  pinned by `tests/test_planar_fit.py`, verified on VAC001 against EddyPro's
  planar fit. Remaining: regenerate GPF outputs/raw pickles for siteGill/
  siteIRGA after their migration into `data/`, and add `matlabCompat` to
  the testkit parity runs. -- EFFORT M (reprocessing), RISK low. Source:
  findings 1-2 in
  [active/2026-08-22_code-audit-and-python-gameplan.md](active/2026-08-22_code-audit-and-python-gameplan.md).
- [BLOCKED 2026-08-22] Schotanus/SND temperature-flux correction (0.51
  coefficient, subtract 0.51·T̄·w'q' where HF humidity exists) and feed the
  corrected w'T' to all WPL terms (LE, Fc, KH2O O2). -- GAIN unbiased H, LE,
  Fc, EFFORT M, RISK med. Blocked on source PDFs: Schotanus et al. 1983,
  Kaimal & Gaynor 1991, Webb et al. 1980 (ask-for-sources). Field evidence
  in hand (VAC001 vs EddyPro: H excess = 0.060×LE, theory 0.062; applying
  the term takes H bias 10.4 → −0.6 W/m²). Detail: findings 3-4 and "VAC001
  test dataset" in the audit doc.
- [PENDING 2026-08-22] Replace the dissipation-rate estimator with a
  compensated inertial-subrange fit; validate on synthetic ε and σw/u*
  similarity. -- EFFORT S, RISK low. Source: finding 5 in the audit doc.
- [PENDING 2026-08-22] Move slope geometry (downslope aspect, tilt-axis
  identification) from hardcoded French Meadows values into `SiteInfo`. --
  EFFORT S, RISK low. Source: finding 6 in the audit doc.
- [PENDING 2026-08-22] Minor-fixes batch: `Vtheta_fw` level-local humidity
  (port parity), unify e_sat formulas, dry-air density in IRGA q_ref branch,
  `find_delta_time` NaN denominator, `nandetrend` true-index fit, document
  ppm/eta/ITC conventions. -- EFFORT S, RISK low. Source: finding 7 in the
  audit doc.

- [PENDING 2026-08-22] `get_data` blanks whole days: `fluxes` trims all-NaN
  columns per file, so a sensor dead for a file (VAC001 fine-wire 07-12 to
  07-17) changes the matrix width and `get_data`'s MATLAB-style shape check
  replaces that file's entire `H`/`sigma`/`derivedT` block with NaN.
  Concatenate by header label (or stop trimming behind `matlab_compat`).
  -- GAIN no silent data loss in multi-day loads, EFFORT S, RISK low.
  Found via `compare_vac001_eddypro.py` (now loads per file by label).

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
  before/after, energy-balance closure (Rn/G provisional). Remaining: re-run
  closure if Rn/G are revised; repeat the comparison after the Schotanus/WPL
  fix lands in code (expected: H bias → ~0). -- GAIN decisive check that
  corrections improve the data, EFFORT S, RISK low. Detail: audit doc
  "VAC001 test dataset".
- [PENDING 2026-08-22] Migrate legacy `siteGill*`/`siteIRGA*` folders into
  `data/<SITE>/` (headers + siteInfo to `utespac/`; tests in
  `test_tower_profile.py` reference the old paths). -- EFFORT S, RISK low.

## meta

- [PENDING 2026-08-22] Next-chat handoff: sequenced steps (Schotanus/WPL →
  `get_data` blanking → other sites into `data/` + GPF regen → remaining
  audit fixes → migration/ec_coherent). Detail:
  [active/2026-08-22_next-chat-handoff.md](active/2026-08-22_next-chat-handoff.md).

## ec-coherent

- [BLOCKED 2026-08-22] ec_coherent build-out per
  [../testbed/2026-08-12_ec_coherent_gameplan.md](../testbed/2026-08-12_ec_coherent_gameplan.md).
  Blocked on the GPF coefficient fix (must not consume pre-fix `uPF/vPF/wPF`
  pickles); Phase 0 library work can proceed. -- EFFORT L, RISK med.
