# ec_coherent build-out -- working doc

Opened 2026-08-23 on the board's ec-coherent line. Plan of record:
[../../../testbed/2026-08-12_ec_coherent_gameplan.md](../../../testbed/2026-08-12_ec_coherent_gameplan.md)
with the companion
[../../../testbed/ec_coherent_utespac_integration_notes.md](../../../testbed/ec_coherent_utespac_integration_notes.md).
This doc tracks what has landed against the gameplan's build order, the
rulings still open, and what each session found when the sources were
read. Science lives in the library notes, not here.

## Input on disk

24 `utespac-hf-1` files `data/VAC001/output/VAC001_hf_<PF>_<Det>_<date>.nc`
(8 dates x LPF LinDet, LPF ConstDet, GPF ConstDet), one sonic at 10.85 m,
20 Hz, 96 records per file. `time` starts one sample after midnight
(Campbell convention), `record` is the END of each 30-min window, so the
samples of record r are `(r - 1800 s, r]` -- 36 000 samples exactly.
Ancillaries on `record`: `ustar`, `L`, `wdir`, `wind_flag`, `spike_flag`,
`nan_flag`, `ssitc_*`. `rhov` and `rhoCO2` sit on their own height dims
(`height_rhov`, `height_rhoCO2`), both 10.85 m here.

## Build order -- status

| Step | Gameplan item | Status |
|---|---|---|
| 0 | Phase 0 for the first modules (sources read, notes) | preprocess + spectra notes written 2026-08-23 from fresh extraction |
| 1 | `io` + `preprocess` (A.2 converter already in `utespac.export_hf`) | landed 2026-08-23 |
| 2 | `spectra` | landed 2026-08-23 |
| 3 | `mrd`, `quadrant`/`octant` | complete 2026-08-23: Haar MRD (Howell & Mahrt recursion + FHT, Vickers gap algorithm), quadrant hole analysis (both normalizations), octants (Li & Bo triplet); all 7 sources read, notes ec_mrd + ec_quadrant; two recollections resolved (worked example confirmed, 5th-order polynomial refuted), hole attribution corrected to Willmarth & Lu 1972 (DECIDE items below) |
| 4 | `ramps` (wavelet first) | complete 2026-08-23: wavelet detector (Ts, u), structure-function detector (Van Atta/Spano linearized + Paw U 2005 two-lag), TKE trigger + IQA (Mangan 2022); Chen 1997 read, finite-microfront fit not adopted (DECIDE below) |
| 5 | `ampmod`, `scales` | complete 2026-08-23: Mathis 2009 decoupling (single-point, u_l and w_l modulators per Salesky & Anderson 2018), band fractions at the Balakumar & Adrian 0.1&pi;&delta;/&pi;&delta; cuts; all 8 sources read, notes ec_ampmod + ec_scales |
| 6 | `coherent_flux` | complete 2026-08-23: wavelet estimator (TF2007 eqs. 3-7 triple decomposition on the ramp-module event sets, u' default + Ts' alongside per ruling) and quadrant estimator (ejection+sweep above the hole, TF2007 §3.3); EC flux error eq. 12b; Thomas2007 + Turner1994 (rendered scan) read, CB93b re-read; note ec_coherent_flux; Turner K-rms estimator added as option 2026-08-24 (ruling, addendum below) |
| 7 | `cli` + full-file regression | complete 2026-08-24: CLI pipeline test (`tests/test_ec_cli.py`), all 24 VAC001 HF files processed, closure checker green over the full set (section below) |

## Findings on extraction (2026-08-23)

- "Vickers & Mahrt (2003) discuss the block-vs-linear trade-off for
  spectra" -- **not confirmed**: the paper has no passage on detrending
  (searched the full text for detrend/trend). Sources for the detrend
  modes found on hand the same day (user question): Moncrieff, Clement,
  Finnigan & Meyers (2004), Handbook of Micrometeorology ch. 2 (block /
  linear / recursive RC filter, eqs. 2.8-2.23, Table 2.2 transfer
  functions, §2.4 comparison) and Rebmann et al. (2012), Eddy Covariance
  ch. 3 §3.2.3 (eqs. 3.11-3.13); bib entries `Moncrieff2004`,
  `Rebmann2012` (chapter DOIs Crossref-verified). The exponential
  (recursive) mode the user asked for is Moncrieff eq. 2.23 and landed as
  `detrend="recursive"`; the earlier zero-phase Butterworth mode is
  renamed `butterworth` and stays `[ASSUMED]`. Block remains the default.
  Rannik & Vesala (1999) and Culf (2000), the comparisons both chapters
  cite, are not on hand.
- Ogive attribution: Foken & Wichura (1996) define the ogive (eq. 10,
  p. 89) and attribute the convergence test to Oncley et al. (1990), whom
  we do not hold; Desjardins et al. (1989) use the "cumulative
  contribution of the cospectral estimates" for filter-loss assessment
  (pp. 61-62, Fig. 5), not as an averaging-time test. Recorded in the
  spectra note; citing Foken & Wichura for the definition is correct,
  citing Desjardins for the averaging-time test is not.
- Kaimal et al. (1972) neutral curves eq. 21a-g read off the rendered
  p. 579 (the OCR layer garbles the exponents); the curve forms and their
  stated validity range 0.01 < f < 4.0 are in the spectra note.
- Stull (1988) 8.6.1-8.6.2 (pp. 312-314) give the folded one-sided
  discrete spectrum and the density S = E/Δn; 1.4 (pp. 5-7) the
  wavenumber form κ = f/M and the Willis & Deardorff σ_M < 0.5 M
  validity criterion for Taylor's hypothesis.

## What landed 2026-08-23 (steps 1-2)

- `ec_coherent/` package: `config.py` (+ packaged `config/ec_coherent.toml`,
  dopli resolution order), `io.py` (`open_hf`, `iter_windows`,
  `init_output`, `write_group`, `read_group`, `output_path`),
  `preprocess.py` (`fill_gaps`, `detrend` block/linear/recursive/
  butterworth with `utespac.nandetrend` semantics, `recursive_filter`, `taylor_*`, `rotation_check`,
  `prepare`), `spectra.py` (`spectrum`, `ogive`, `log_bins`/`log_bin`,
  `kaimal_neutral`, `run`), `cli.py` (`python -m ec_coherent.cli <hf
  files> [--records] [--modules] [--config]`). Registered in
  `pyproject.toml`.
- Tests `tests/test_ec_preprocess.py`, `test_ec_io.py`,
  `test_ec_spectra.py` on a synthetic HF file (`tests/ec_helpers.py`):
  exact Parseval closure with boxcar, ogive(0) = covariance, half-line
  snapped log bins whose band sum is exact, Kaimal eq. 21 hand values,
  NaN policy and scalar height mapping through the module. 184 tests
  pass.
- Validation run `testbed/scripts/ec_spectra_vac001.py` on
  `VAC001_hf_GPF_ConstDet_2023_07_06.nc` (96 records, 15 s): band-sum /
  variance = 1.000 for u, w, Ts on every record, ogive(f_min)/cov_uw =
  1.0000; all 96 records are near-neutral (|z/L| < 0.05, u* 0.8-1.3
  m/s) and the record-median n S_u/u*², n S_w/u*² and -n Co_uw/u*²
  follow the Kaimal et al. (1972) eq. 21 curves over 0.01 < f < 4
  (figure `testbed/scratch/ec_spectra_vac001.png`); the u'w' ogives
  plateau at the low-frequency end (30 min is adequate for momentum),
  several w'Ts' ogives do not (small H at this site, see the VAC001
  advective caveat); σ_M/U = 0.34-0.42 (Taylor valid, Stull eq. 1.4d);
  |w̄|/σ_w median 0.022, max 0.066 after PF + yaw. Output
  `VAC001_coherent_GPF_ConstDet_2023_07_06.nc` next to the HF file.
- Taper finding: a single-segment Hann taper returned 0.74-1.03 of the
  Ts variance on six windows (block detrend), boxcar closes exactly;
  boxcar is the default (spectra note, deviation register).

## Step 4, wavelet path (2026-08-23)

- Read: Collineau & Brunet 1993 I (transform eq. 4, wavelet variance
  eq. 8, duration scale eq. 22, Table I, zero-crossing method §4.3) and II
  (calibration §4.1, time localization §5.2, conditional averages eq. 3,
  triple decomposition §7.2), Gao, Shaw & Paw U 1989 (ramp and microfront
  description), Thomas & Foken 2007 §3.1-3.2 (highest-frequency variance
  peak after a 6.2 s low-pass, Mexican hat at D_e, flux contribution eqs.
  1-7). Note: `library/writeups/ec_ramps.md`.
- Landed: `ec_coherent/ramps.py` (`mhat`, `ramp_wavelet`, `haar`, `cwt`,
  `wavelet_variance`, `duration_scale`, `zero_crossings`, `refine_times`,
  `detect`, `run`), `[ramps]` config, `ramps` in the CLI,
  `tests/test_ec_ramps.py`, `testbed/scripts/ec_ramps_vac001.py`; suite
  192 green.
- Findings: (1) CB Table II geometry confirmed -- D is the event length,
  half the pattern period; (2) the MHAT zero-crossing at a0 lags an ideal
  microfront by 0.35 a0; RAMP/HAAR re-timing removes it (DECIDE below);
  (3) VAC001: u' gives a clean single-peak scalogram (D ≈ 11 s, ~80
  events/30 min, spacing ≈ 2D); Ts' has no ramp-scale peak in most
  records (small-scale shoulder at 3 s, trend hump at 60-260 s), so
  temperature-based detection at this site rests on the `D_min_s` choice
  -- recorded in the note; the coherent-flux module should probably
  condition on u' (or w') events here rather than Ts' as Thomas & Foken do.
- Not done: Van Atta / Paw U / Spano / Chen (structure-function
  detector), Mangan 2022 (TKE, IQA) -- unread, no code.

## Step 4, structure-function + TKE (2026-08-23, second part)

- Read: Van Atta 1977 (decomposition eq. 2.5, moment identities 2.9,
  cubic 2.13, period 2.15; pp. 167-168 rendered), Paw U et al. 1995 (SR
  concept, eq. 1), Spano et al. 1997 (S^n eq. 3, cubic eqs. 4-6, l+s
  eq. 7, flux eq. 2, lag set and l+s > 10r constraint), Paw U et al. 2005
  (full moment eqs. 2a-c, two-lag d/s split eqs. 3-5; pp. 460-461
  rendered), Chen et al. 1997 (finite microfront; read, fit not adopted),
  Mangan et al. 2022 (u_TKE trigger §4, IQA eqs. 2-4). Note extended:
  `library/writeups/ec_ramps.md`.
- Extraction finding: Van Atta's printed eq. 2.11 higher-order polynomial
  coefficients (n = 3, 4, 5) disagree with direct integration of his own
  ramp model; Paw U 2005 eqs. 2a-c match the derivation (re-derived
  independently, asserted brute-force in `tests/test_ec_ramps.py`). Only
  the leading linear terms -- which all agree -- enter the linearized
  method.
- Landed: `structure_function`, `vanatta` (linearized + two-lag),
  `sr_flux`, `utke`, `utke_lp`, `detect_tke`, `iqa` in
  `ec_coherent/ramps.py`; `[ramps]` config extended (sr_*, tke_*);
  outputs on `sr_lag` axis + TKE event set in the `/ramps` group;
  `testbed/scripts/ec_sr_vac001.py`; suite 196 green.
- VAC001 GPF ConstDet 2023-07-06 (96 records, z = 10.85 m, alpha = 1):
  SR flux tracks the covariance tightly (r = 0.94-0.96 per lag) but
  overestimates ~2.2x (median F_SR/w'Ts' at r = 0.5 s), i.e. an
  effective alpha ~ 0.45 -- inside the spread Spano reports for fitted
  alpha; amplitude sign follows stability (a < 0 at night). Linearized
  l+s median 29 s, matching the u' wavelet spacing (~20 s); two-lag split
  d ~ 4 s, s ~ 20 s. |S^3(r)/r| is flat-to-declining over r = 0.25-1 s
  (lags sit at or above Chen's t_m -- no small-lag rolloff). TKE trigger:
  median 16 events/30 min (one per ~2 min, as in Mangan), median IQA
  bulk-sweep fraction 0.54; triggers land on the major structures where
  the u' wavelet set (~80/30 min) is much denser. Figure
  `testbed/scratch/ec_sr_vac001.png`.

## SR under advection -- Castellvi & Snyder 2009 sanity check (2026-08-23)

User supplied the PDF (Crossref-verified 2026-08-23; French et al. 2012
turned out to be on hand too -- next section). Read and
extracted: their SR runs calibration-free under regional advection by
computing the weighting factor from similarity (Castellvi 2004),
alpha = [(k/pi)(z-d)/z^2 * tau * u* / phi_h(zeta)]^1/2 (their eq. 3,
inertial branch), with phi_h from Hogstrom (their eq. 5 prints "116",
a misprint for 11.6 -- confirmed against Foken 2008 on hand). Their
site is the VAC001 regime: light winds, high temperatures, stable
daytime cases dominating "due to regional advection"; they get slopes
0.89-1.10, R^2 >= 0.86 against EC without calibration. Landed:
`phi_h`, `alpha_castellvi` in ec_coherent/ramps.py, Castellvi2009 in
references.bib/index.md, note section in ec_ramps.md, check wired into
testbed/scripts/ec_sr_vac001.py; suite 197 green.

Sanity check on VAC001 GPF ConstDet 2023-07-06 (r = 0.5 s, d = 0
assumed, EC u* and L ancillaries): alpha_C median 0.61 (IQR 0.48-0.71)
against the 0.45 the fixed-alpha run implied; applying alpha_C per
record moves the median F_SR/w'Ts' from 2.23 to 1.39 and tightens the
correlation from 0.96 to 0.99 -- the similarity alpha explains most of
the overestimate and nearly all of the scatter. The residual 1.39
plausibly reflects d = 0, the un-modelled roughness sub-layer branch,
and their iterative u*/zeta solution (we use EC values directly). Their
lag rule (r above the first global max of S^3(r)/r) is satisfied by our
lag set: |S^3(r)/r| declines over 0.25-1 s (figure panel d).

## SR under advection -- French et al. 2012 sanity checks (2026-08-23)

The PDF was in library/ all along (2023 file timestamp defeated the
recency check that reported it missing); title-checked, French2012 in
references.bib/index.md, gap note removed, section added to ec_ramps.md.
Their SR is exactly our landed configuration (sample-lag Van Atta cubic,
alpha = 1, z = measurement height, lags including our set), tested at a
strongly advective irrigated site (BEAREX08) against 9 EC stations.
Their three transferable findings, run as checks on VAC001 GPF ConstDet
2023-07-06 (ec_sr_vac001.py, no new module code):

- Lag: they find an RMSE optimum at r = 1.0 s with slope near 1,
  "calibration of SR fluxes based on lag is a more important
  consideration than the alpha height-dependent term". VAC001: the
  median F_SR/w'Ts' declines monotonically 2.42 / 2.23 / 2.16 / 2.04
  over r = 0.25/0.5/0.75/1.0 s -- same direction, 1.0 s is our best lag
  too, but lag choice alone closes far less of the gap than their
  z = 2.25 m case; at z = 10.85 m the similarity alpha (Castellvi
  section above) is the dominant correction.
- Sign fidelity (their key SR advantage under advection: S^3 sign flags
  H < 0): VAC001 sign(a) == sign(w'Ts') in 97-100% of valid records per
  lag (100% at r = 0.5 and 1.0 s), through the site's diurnal H sign
  flips. SR is a reliable stability/advection flag here.
- q-dominance does NOT transfer: they find p ~ 0 so "computation using
  only the third order structure function would be sufficient". VAC001:
  a_q = cbrt(-10 S^3) tracks the full-cubic a tightly in ranking
  (r = 0.98) but median a_q/a = 0.56 -- the p term (2nd/5th order
  structure functions) matters at this site and the full cubic stays.

## Step 5, ampmod + scales (2026-08-23)

- Read (all on hand since 2026-08-17; PyMuPDF text, no scans): Mathis,
  Hutchins & Marusic 2009 (decoupling procedure fig. 4, R eq. 5.1,
  envelope eq. 4.5, single-point §6.2, phase-scrambled null test §7.1.1,
  cutoff robustness §7.1.3), Talluru et al. 2014 (eq. 3.4a-c: u_L
  modulator for all components), Salesky & Anderson 2018 (eq. 1.6
  generalisation, sharp spectral filter at z_i, w_l the "buoyancy proof"
  modulator, fluxes (uw)_s and (wθ)_s as signals), Salesky & Anderson
  2020 (AHATS tower cutoff T = δ/U with δ = 1000 m assumed, insensitive),
  Kim & Adrian 1999 (VLSM from the premultiplied-spectrum peak; Taylor
  underestimates the largest wavelengths), Balakumar & Adrian 2007 (band
  boundaries k_x δ = 20 and 2, i.e. 0.1πδ and πδ, one decade; VLSM carry
  40-65 % KE, 30-50 % stress), Wang & Zheng 2016 (QLOA ASL: cutoffs
  0.3δ/3δ ≈ the same decade; LSM peak indistinguishable in ASL spectra;
  VLSM fraction 20-30 % at z/δ < 0.01), Hutchins et al. 2012 (SLTEST:
  neutral ASL "behaves precisely as" the lab boundary layer; δ = 60 m by
  statistical consistency). Notes: `library/writeups/ec_ampmod.md`,
  `ec_scales.md` (bib entries already present).
- Landed: `ec_coherent/ampmod.py` (`lowpass_sharp`, `envelope`,
  `am_coefficient`, `spectral_gap`, `cutoff_frequency`, `run` ->
  `/amplitude_mod`), `ec_coherent/scales.py` (`band_edges`,
  `band_fractions`, `run` -> `/scale_separation`), `[ampmod]`/`[scales]`
  config (cutoff_mode = spectral_gap | delta | scaled per locked
  decision 3; delta_m default 1000 m [ASSUMED] after Salesky 2020), both
  modules in the CLI and default module list;
  `tests/test_ec_ampmod.py` (envelope recovers B+m; exact filter
  partition; constructed AM -> R > 0.95; phase-scrambled -> R ~ 0; gap
  finder; cutoff modes; module runs), `tests/test_ec_scales.py` (band
  fractions sum to 1 exactly; two-sine localisation; delta-mode edges).
  Suite 209 green.
- VAC001 GPF ConstDet 2023-07-06 (96 records, z = 10.85 m), figure
  `testbed/scratch/ec_ampmod_scales_vac001.png`
  (`testbed/scripts/ec_ampmod_scales_vac001.py`): spectral gap found in
  68/96 records (median λ_c 648 m, IQR 366-1000; fallback to the
  assumed-δ 1000 m elsewhere). R_uL_uS median +0.38 (IQR 0.29-0.50) --
  positive as the sources find near the wall (z/δ ~ 0.01 here);
  R_uL_TsS +0.26; the w_l-modulator coefficients are small and scattered
  (R_wL_uS median -0.07) with no visible z/L trend over the file's
  near-neutral range (|z/L| < 0.05) -- consistent with Salesky &
  Anderson's w_l modulation being a convective-regime feature. Cutoff
  sweep 200-3000 m: R_uL_uS rises smoothly 0.33 -> 0.53, no sign or
  shape change (the registered deviation's sensitivity validation;
  matches Mathis §7.1.3). Scale bands (gap cutoffs): u' variance
  small/lsm/vlsm = 0.34/0.45/0.17 medians -- the VLSM fraction sits at
  the low end of Wang & Zheng's 20-30 % for z/δ < 0.01, reasonable for
  30-min windows that resolve only the first few VLSM-band lines; w'Ts'
  flux fractions 0.50/0.34/0.10 with blow-ups where the covariance
  denominator crosses zero (the site's H sign flips -- expected,
  fractions of a near-zero total).

## Step 3, mrd + quadrant/octant (2026-08-23)

- Read: Howell & Mahrt 1997 (recursion eqs. 1-7, FHT appendix + eq. 15
  identity, sampling error eqs. 11-14, 2^M interpolation eq. 8), Vickers &
  Mahrt 2003 (residual formulation, Table 1 worked example, down-grid
  eq. 8, gap algorithm §4, gap model eqs. 12-14), Wallace 2016 (history),
  Lu & Willmarth 1973 (§4.3 RMS hole), Raupach 1981 (S/T/ΔS eqs. 1-7,
  Gram-Charlier eqs. 8-12, H 0-20), Li & Bou-Zeid 2011 (S/D eqs. 4-6,
  η eqs. 8-9, ΔS/ΔD eqs. 16-17), Li & Bo 2019 (hole eq. 9, octants
  eq. 11, near-neutral duration fractions). Notes:
  `library/writeups/ec_mrd.md`, `ec_quadrant.md`.
- Extraction findings: (1) the Vickers & Mahrt worked numeric example
  EXISTS (Table 1, 8-sample series; now the exact test target); (2) the
  recalled fifth-order-polynomial gap fit is REFUTED -- their algorithm is
  a first-peak scan with a 1 % accumulative-flux leveling rule; (3) the
  hyperbolic hole originates with Willmarth & Lu 1972 (flux-normalized),
  not Lu & Willmarth 1973 -- but the recalled H·σ_u·σ_w form IS Lu &
  Willmarth 1973's (§4.3), with Raupach 1981 noting the two conventions
  differ by ρ (both landed as `hole_norm`, bib entry `Willmarth1972`
  Crossref-verified, PDF not held); (4) both MRD sources map onto 2^M by
  interpolation, not the gameplan's trim (DECIDE below); (5) every read
  octant source uses the (u', w', scalar) triplet -- the gameplan's
  (w', T', q') appears in none (DECIDE below).
- Landed: `ec_coherent/mrd.py` (`to_grid`, `fht`, `mr_spectrum`,
  `smooth121`, `gap_scale`, `run` -> `/mrd`), `ec_coherent/quadrant.py`
  (`quadrant_stats`, `derived_h0`, `octant_stats`, `run` -> `/quadrant`,
  `run_octant` -> `/octant`), `[mrd]`/`[quadrant]` config, both in the
  CLI and default module list, `tests/test_ec_mrd.py`,
  `test_ec_quadrant.py` (Vickers Table 1 exact; FHT = direct block means;
  sum closure; H=0 fractions sum to 1; hole-norm ρ equivalence; octant
  closure; and the A.3 cross-check: delta_S/delta_D/eta reproduce
  utespac `find_delta_flux`/`find_delta_time`/`find_eta` exactly,
  asserted against those functions). Suite 228 green.
- VAC001 GPF ConstDet 2023-07-06 (96 records, z = 10.85 m), figure
  `testbed/scratch/ec_mrd_quadrant_vac001.png`
  (`testbed/scripts/ec_mrd_quadrant_vac001.py`): MRD closure to file
  precision (max |ΣD − cov| ~ 3e-7, f4 storage); D_uw peaks at τ ≈ 7-13 s;
  gap detected in 95/96 (wTs, median 410 s, IQR 205-410) and 94/96 (uw
  stress magnitude, median 410 s, IQR 410-819) -- on the dyadic grid,
  bracketing Vickers' 540 s neutral 10-m value; per-record spread is
  large, as they warn. Quadrants at H=0: S medians Q1 −0.20, Q2 +0.66,
  Q3 −0.18, Q4 +0.71 -- the near-symmetric matched-layer picture (Raupach
  η≈0.2: S2 ≈ S4 ≈ 0.6, interactions ≈ −0.1); duration fractions
  0.167/0.329/0.191/0.314 vs Li & Bo's near-neutral
  0.19/0.296/0.2/0.314; eta_uw median 0.73 (IQR 0.72-0.74), eta_wTs 0.57
  with blow-ups where w'Ts' crosses zero (the site's H sign flips);
  contributions vanish by H ≈ 8-10 as in Raupach fig. 6. Octants (52
  records with w'Ts' > 0): O2 +0.51/O8 +0.57 of u'w' (sum 1.08), O2
  +0.64/O8 +0.67 of w'Ts' -- the Li & Bo hot-ejection/cold-sweep
  dominance. Full coherent file regenerated with all 7 groups.
- A.3 cross-validation against the real averaged product
  (`VAC001_30minAvg_GPF_ConstDet_2023_07_06.nc`, all 96 records matched
  on time): delta_S_uw ~ S_wPFuPF, delta_D_uw ~ D_wPFuPF, eta_uw ~
  eta_wPFuPF, delta_S_wTs ~ S_wPFThetav, eta_wTs ~ eta_wPFThetav all
  agree with median |diff| = 0.0000 (r = 0.997-1.000; max |diff|
  0.001-0.03 from the averaged file's Theta_v vs our Ts and the f4 HF
  storage).

## Step 6, coherent_flux (2026-08-23)

- Tooling first (user ruling): the dopli CLAUDE.md conda/shell rules
  adopted verbatim (blessed `cmd.exe //c ... conda.bat run` invocation,
  no Python through PowerShell, standalone deletions), and
  `testbed/scripts/extract_paper.py` ported from dopli (bibkey/fragment
  resolution against the abbreviated index.md rows incl. mid-string
  `...`, thin-text-layer warning, `--render`, `--check-index`;
  extractions cached in gitignored `library/extracted/`).
- Read: Thomas & Foken 2007 in full (triple decomposition eqs. 1-7,
  operator window ±D_e, F_ej/F_sw halves with F_cs = F_ej + F_sw exact,
  quadrant estimator eqs. 8-9 with the r_xy sign rule, 0.8-1.2
  representativeness gate, EC flux error eqs. 10-12, results 0.16/0.26
  and the quadrant 3-4x finding); Turner & Leclerc 1994 (image scan,
  5 pages rendered and read: Haar family, K-rms coefficient threshold
  eq. 5, strong/weak reconstruction; no flux fractions in the paper);
  Collineau & Brunet 1993 II re-read for eqs. 3-7 (30-s window, their
  0.26/0.40 contributions). Note: `library/writeups/ec_coherent_flux.md`.
- Landed: `ec_coherent/coherent_flux.py` (`conditional_average`,
  `flux_contribution`, `quadrant_fraction`, `run` -> `/coherent_flux`),
  `[coherent_flux]` config (event_signals u+Ts per ruling, window
  duration|fixed, holes 0/0.5/1), CLI + default module list,
  `tests/test_ec_coherent_flux.py` (pattern recovery, F_ej+F_sw = F_cs
  exact, coherent-covariance recovery on a sawtooth train, eq. 12b
  collapse under block detrend, quadrant delta_S cross-check, module
  runs). Suite 236 green. Full coherent file regenerated, 8 groups.
- VAC001 GPF ConstDet 2023-07-06 (u' events, N >= 2 and gate-valid
  records; figure `testbed/scratch/ec_coherent_flux_vac001.png`,
  script `ec_coherent_flux_vac001.py`): conditional averages show the
  textbook pattern (u' zero-crossing at the event, w' and Ts' moving
  together against it, positive <w'><Ts'> humps). F_cs/F_tot median
  0.22 (u'w') and 0.14 (w'Ts') -- momentum at TF2007's scalar level and
  the scalar fraction lower, the reverse of their Ts-conditioned
  ordering (our events are u'-conditioned and the site's H is small
  with sign flips). Sweep/ejection ~ 1.0 (their above-canopy <1).
  Quadrant estimator medians 1.61/1.41/1.11 (wTs, L = 0/0.5/1): the
  H = 0 value is the downgradient fraction 1/eta ~ 1.75 of the earlier
  quadrant step, so the wavelet-to-quadrant gap is ~8-12x here, far
  beyond their 3-4x -- countergradient flux at this advective site
  inflates the quadrant estimator's numerator. EC flux error: median
  |err| ~ 0.1%, inside their <4%. Degenerate case found and recorded:
  6 records where the u' variance peak sits on the trend-hump scale
  give N = 1 and F_cs = F_tot identically (note's deviation register;
  script drops N < 2).

## Step 6 addendum, Turner estimator (2026-08-24)

Ruling: adopt the Turner & Leclerc threshold reconstruction as an option,
not the default. Landed: `haar_coefficients`, `haar_reconstruct`,
`turner_split`, `turner_fraction` in coherent_flux.py (redundant dyadic
Haar via cumulative sums; reconstruction corr 0.9998 on a 36 000-sample
AR(0.95) series), `turner = false` / `turner_K = 4.0` config, tests
(closure, K-monotonicity, isolated-event recovery); suite 238 green. The
strong-strong covariance fraction is the [NOVEL] construction (sign-off
in this ruling), deviation register updated. VAC001 wTs finding: median
fraction 0.64/0.31/0.11/0.01/0.00 at K = 1/1.5/2/3/4 -- at K = 2 it
matches a correlated Gaussian pair (~0.09), so by the K-rms criterion
this record set has almost no coefficients outside the Gaussian bulk and
their K = 4 default passes nothing; validation script prints the sweep
and draws the K = 4 medians in panel c.

## Step 7, cli + full-file regression (2026-08-24)

- Landed: `tests/test_ec_cli.py` -- the full default pipeline through
  `cli.main()` on the synthetic file (all 8 groups written, provenance
  attrs, and the cross-group closure identities: spectra band sum = var,
  MRD sum = cov, quadrant/octant H=0 fractions sum to 1, scale bands sum
  to 1, F_ej + F_sw = F_cs), plus the `--records`/`--modules`/`--out`
  argv paths (`--records` keeps the full record axis, unselected records
  NaN). Suite 242 green.
- Full-file run: all 24 VAC001 HF files (8 dates x 3 PF/detrend variants)
  processed with the packaged defaults, ~70 s/file; 24 coherent files, 8
  groups each. `testbed/scripts/ec_full_regression_vac001.py` checks
  every output (groups, provenance attrs, 7 closure identities) and
  prints one row per file: PASS over all 24, worst error 2.9e-6 (f4
  storage). Valid coherent-flux records 89-96 per full day; 42 on the
  half-day 2023-07-20 files.
- Checker finding: the 07-20 files' absent half-day is all-NaN in every
  group (correct behaviour); xarray's default `skipna=True` turns those
  all-NaN closure sums into 0, so the checker sums with `skipna=False`
  -- absent records read NaN and are excluded rather than failing.

## Open decisions


Step-3 rulings received (user, 2026-08-23) and implemented the same day:
MRD grid stays `trim` with the VM2003 `interp` option kept, and windows
up to ~2 h are supported (M follows the window; 2-h regression test
added); octants now run a *list* of triplets -- `octant_triplets`
defaults to (u,w,Ts) + (w,Ts,rhov), variables tagged per triplet
(`flux_frac_uwTs_wTs`, `flux_frac_wTsrhov_wTs`, ...), any signals
accepted (v, rhoCO2 anticipated; `wrhoCO2` quadrant pair also
available). VAC001: the (w,Ts,rhov) space puts the w'Ts' flux in the
warm-moist-updraft (+0.73) and cool-dry-downdraft (+0.76) octants.
Suite 229 green; coherent file regenerated. Rulings recorded in the
ec_mrd and ec_quadrant deviation registers.

Earlier rulings received (user, 2026-08-23) and implemented the same day:
refine stays "none" (option kept per site), u' event set with Ts'
alongside for coherent_flux, Chen finite-microfront not adopted, and
`sr_alpha_mode = "fixed" | "castellvi" | "fit"` landed in config/ramps
with `sr_alpha_Ts` stored in the output (fixed remains the default,
`sr_d` added for the castellvi branch; VAC001 output regenerated,
suite 197 green). Rulings recorded in the ec_ramps note's deviation
register.

Both issues plotted on real data 2026-08-23 (user request):
`testbed/scripts/ec_ramps_issue_vac001.py` →
`testbed/scratch/ec_ramps_issues_vac001.png`. Panel a: record 21 (11:00,
the day's largest w'Ts'), Ts ramps clearly visible in the trace, the Ts
detector finds 7 events at D = 104 s while the u detector's 74 events land
on the visible ramps -- Ts ramps visible but not detectable through the Ts
scalogram (panel b: no interior Ts peak between the small-scale shoulder
and the trend hump). Panels c-d: pooled over all 6585 u / 1737 Ts events
of the day, the (zero-crossing − RAMP-refined)/a0 lag has median 0.00 and
±a0 scatter (only 16 % agree within 0.1 a0) -- the +0.35 a0 lag is an
ideal-train artifact, absent on real data; where Ts detection works
(record 18) the zero-crossings sit on the visible drops (+0.4 s median).
Both findings recorded in the ec_ramps note.

DECIDE: `refine` default for the wavelet detector -- keep the MHAT
zero-crossing time as the paper does (`none`, landed), or re-time each
event at the RAMP extremum within ±a0 (`ramp`), which puts ideal
microfronts to one sample but on real data adds ±a0 scatter with no
systematic correction (figure panel c). (default: `none`; the option
stays available)
A: i want this to be repeatable and usable for other sites, so keep none for now but allow users to refine for their site

DECIDE: detection signal for VAC001 -- Ts' (Thomas & Foken's choice;
visible ramps but usually no ramp-scale scalogram peak here, figure
panels a-b) or u' (clean scalogram, events land on the visible Ts ramps)
as the event set the coherent-flux module conditions on. (default: u',
with Ts' reported alongside)
A: default is fine


DECIDE: adopt the Chen et al. (1997) finite-microfront model (nonlinear
fit of their eqs. 14-15 to S^3(dt)/dt over all lags, yielding M, tau, t_f
per record)? Their Table I says the linearized method under-recovers
M/tau by 2-2.5x against it; on VAC001 the linearized flux already sits
2.2x HIGH of the covariance, so adopting it would move further away and
the empirical alpha absorbs either bias. (default: not adopted; the
S^3(r)/r diagnostic is stored so the small-lag rolloff is checkable)
A: default is fine.

DECIDE: sr_alpha for VAC001 -- keep 1.0 and treat sr_flux as an
uncalibrated diagnostic (default), or fit alpha against the measured
w'Ts' (2023-07-06 gives ~0.45) and store it as [SITE-TUNED]? EC flux is
measured here, so SR is never the flux of record. A third option exists
since the Castellvi & Snyder 2009 extraction: the similarity
alpha_castellvi per record (no fit; VAC001 check gives median 0.61,
ratio 1.39, r 0.99 -- section above). (default: alpha = 1,
uncalibrated)
A: make sr_alpha=1.0 the default but give options for castellvi and empirical fit

DECIDE: MRD grid mode -- the gameplan trims each window to the leading
2^15 = 32768 of 36000 samples (91 %, 1638.4 s of 1800; landed default
`grid = "trim"`), but both sources map arbitrary records onto 2^M by
linear interpolation instead (Howell & Mahrt 1997 eq. 8 up-grids, Vickers
& Mahrt 2003 eq. 8 down-grids "so all the data points ... are used");
`grid = "interp"` implements the Vickers down-grid (dt × 36000/32768
≈ 0.0549 s, scales stretched accordingly, stored in the output attrs).
(default: trim, per the gameplan)
A: Have an option to use the VM2003 approach but trim is fine. Also keep in mind that longer time periods (usually up to 2 hours) may be analyzed

DECIDE: octant triplet -- landed default (u', w', Ts') per Li & Bo 2019
eq. 11 (every read octant source uses velocity pair + one scalar; O2/O8
dominance confirmed on VAC001). The gameplan's (w', Ts', q') scalar-
dissimilarity triplet is available via `octant_triplet = ["w","Ts","rhov"]`
but appears in no read source ([ASSUMED] if used; Li & Bou-Zeid 2011 do
scalar dissimilarity with quadrants, not octants). (default: u, w, Ts)
A: I want the w', Ts', q' quadrant analysis as well. v and CO2 may also be used in the future

DECIDE: Welch segmenting vs full-record periodogram as the default
spectral estimator. The landed default is the full-record periodogram
with boxcar taper and half-line-snapped log-binning (exact Parseval
closure, keeps the 1/1800 Hz end for the ogive); Welch is available via
`nperseg`, Hann via `taper`. (default: keep full-record boxcar)
A: keep the defaults (user, 2026-08-23).

DECIDE: install `pywt` into the UTESpac_Plus env for step 4 (ramps,
wavelet detector). (default: yes, when step 4 starts)
A: yes (user, 2026-08-23).
