# ec_coherent build-out -- working doc

Opened 2026-08-23 on the board's ec-coherent line. Plan of record:
[../../testbed/2026-08-12_ec_coherent_gameplan.md](../../testbed/2026-08-12_ec_coherent_gameplan.md)
with the companion
[../../testbed/ec_coherent_utespac_integration_notes.md](../../testbed/ec_coherent_utespac_integration_notes.md).
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
| 3 | `mrd`, `quadrant`/`octant` | pending (Howell & Mahrt 1997, Vickers & Mahrt 2003; Wallace 2016, Lu & Willmarth 1973, Raupach 1981, Li & Bou-Zeid 2011, Li & Bo 2019 to read) |
| 4 | `ramps` (wavelet first) | wavelet detector landed 2026-08-23 (Ts, u; `pywt` installed); structure-function (Van Atta) detector and TKE (Mangan 2022) pending -- sources unread |
| 5 | `ampmod`, `scales` | pending |
| 6 | `coherent_flux` | pending |
| 7 | `cli` + full-file regression | `cli` runs the landed modules over one file; extend per module |

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

## Open decisions

DECIDE: `refine` default for the wavelet detector -- keep the MHAT
zero-crossing time as the paper does (`none`, landed), or re-time each
event at the RAMP extremum within ±a0 (`ramp`), which puts ideal
microfronts to one sample but combines the paper's two schemes. (default:
`none`; the option stays available)
A:

DECIDE: detection signal for VAC001 -- Ts' (Thomas & Foken's choice,
ill-posed here) or u' (clean scalogram) as the event set the coherent-flux
module conditions on. (default: u', with Ts' reported alongside)
A:


DECIDE: Welch segmenting vs full-record periodogram as the default
spectral estimator. The landed default is the full-record periodogram
with boxcar taper and half-line-snapped log-binning (exact Parseval
closure, keeps the 1/1800 Hz end for the ogive); Welch is available via
`nperseg`, Hann via `taper`. (default: keep full-record boxcar)
A: keep the defaults (user, 2026-08-23).

DECIDE: install `pywt` into the UTESpac_Plus env for step 4 (ramps,
wavelet detector). (default: yes, when step 4 starts)
A: yes (user, 2026-08-23).
