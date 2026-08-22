# EC coherent-structure analysis gameplan (rev. 2026-08-12)

Revision of `ec_coherent_structure_analysis_gameplan(1).md`, rewritten under the
publication-fidelity practice developed in the dopli repository
(`dopli/LLM_PRACTICE.md`). The technical scope is unchanged. What changes is the
epistemic status of the content: the original presented from-memory methodology
and citations as settled, and this revision separates what is planned from what
is verified. The UTESpac mapping in
[ec_coherent_utespac_integration_notes.md](ec_coherent_utespac_integration_notes.md)
remains the companion document; its amendments are folded in below where they
change the science plan.

## Scope

A standalone package (`ec_coherent`) consumes per-location high-frequency
netCDF files -- assembled from UTESpac pickles by the converter specified in
integration notes A.2 -- and writes a separate analysis netCDF per location.
Nothing writes back to the raw series. EC flux computation and planar-fit
rotation are already done upstream. Seven analyses: spectra/cospectra/ogives,
multiresolution decomposition (MRD), quadrant and octant analysis, ramp
detection on temperature, streamwise velocity, and TKE, amplitude modulation,
LSM/VLSM scale separation, and coherent-structure flux fractions.

Interpretations locked in the original discussion: "utke" means ramp/structure
detection extended to u and TKE, and "VSLM" was a typo for VLSM (very-large-
scale motion).

## Epistemic status of this document

Every methods statement below was drafted from general knowledge of the
boundary-layer literature, not from the papers themselves. Under the
source-fidelity rule that makes this a working plan, not implementable
methodology. Before any module is coded:

1. its source PDFs are in hand, supplied or approved by the user;
2. the relevant sections are freshly extracted and read;
3. a library note exists mapping equations to citations (verbatim quote plus
   locus: section, page, equation number) to code symbols;
4. science-critical constants in that note carry provenance labels.

The label vocabulary is dopli's: `[CITED]` (taken directly from a source,
bibkey plus locus), `[DERIVED]` (derivation shown in the note), `[ASSUMED]`
(placeholder without source backing, visibly marked), `[SITE-TUNED]` (adjusted
against local data, validation linked), `[NOVEL]` (not in the literature;
requires explicit user sign-off before implementation). Unlabelled
science-critical values are treated as `[ASSUMED]`. Each module's library note
keeps a deviation register: what the paper does, what the code does instead,
why, and what validation supports the substitution.

Statements below that are known paraphrase risks are marked **confirm on
extraction**. Such a mark means "read this before coding it" -- the marked
statement carries no evidentiary weight on its own.

## Silos to stand up in this repo

Adopt the dopli knowledge-silo structure at the point the package is created:

- `library/` -- one note per module, following the conventions of
  `dopli/library/README.md` (provenance labels, deviation register, `$` math,
  relative links, code linked by file and symbol name). `library/references.bib`
  holds bibkeys; `library/papers/` holds the gitignored PDFs with an
  `index.md` mapping filename to bibkey.
- Gameplans and handoffs live in dated files (this one); completed-work records
  are immutable once archived.
- Assistant memory holds workflow facts only. Science lives in library notes
  where it can be checked against sources.

## Phase 0 -- assemble the library (gates all coding)

The reference list at the end of this document is a request list, not a
bibliography. For each module, the user supplies or approves the minimum
reading set before that module is implemented. Identifier resolution was
completed 2026-08-17: every entry except Van Atta 1977 carries a DOI or ISBN
verified against a live Crossref or publisher record, and the verification
pass caught one real citation error (Collineau & Brunet Part II, corrected in
the list). Van Atta 1977 predates its journal's online archive (which starts
at vol. 48, 1996), so a scan had to be sourced.

PDF acquisition completed 2026-08-17: the user supplied all sources
(including a Van Atta 1977 scan) in `library/`, each PDF was checked against
its citation by automated title extraction or, for the two image-only scans
(Antonia 1979, Turner & Leclerc 1994), by visual inspection of the rendered
title page. One wrong file was caught and replaced (the Thomas & Foken 2007
companion paper had been downloaded instead of the flux-contribution paper;
both are now on hand). Curated `library/references.bib` (40 entries, cleaned
from the Zotero export) and `library/index.md` (filename-to-bibkey map with
per-file check status) are the tracked artifacts; PDFs are gitignored.
Kaimal & Finnigan 1994 was not acquired -- Stull 1988 fills the textbook
slot. What remains before each module is coded is its fresh-extraction read
and library note: the content of every entry is still unverified.

## Shared preprocessing

One function, reused by every module, operating per flux-averaging window
(30 min, matching the upstream `flux_averaging_s`). All modules consume
perturbation series computed here; the stored `*Prime` arrays are never used
because they are locked to the pipeline's upstream detrend setting.

Rotation is already applied upstream: `uPF/vPF/wPF` carry planar fit plus
per-window yaw (integration notes A.1). The preprocessing step is therefore a
verification -- mean(w) per window is confirmed near zero -- rather than a
transform. A double-rotation cross-check cannot be run from current raw
pickles because unrotated series are not saved; if it matters, adding them to
the raw dict is a deliberate UTESpac change to propose separately.

Detrending is selectable: block mean (the Reynolds default), linear, or
recursive filter. The recollection that Vickers & Mahrt (2003) discuss the
block-vs-linear trade-off for spectra: **confirm on extraction**.

QC flags are joined from the averaged pickle by window timestamp; windows
failing QC are processed but flagged. Spikes were interpolated in place
upstream, so ramp amplitudes and quadrant tails see slightly smoothed extremes
-- recorded in output provenance, not correctable downstream.

Taylor's hypothesis converts frequency to wavelength via λ = Ū/f and to
wavenumber via k = 2πf/Ū, with Ū the per-window mean horizontal wind (after
yaw rotation, mean(uPF) suffices). Ū is stored with every scale-resolved
output, and the breakdown at large λ/z is noted there.

Conventions: fluxes positive upward; perturbations u′, w′, T′, c′;
TKE e = ½(u′² + v′² + w′²). Numerics are implemented from the extracted
equations; standard primitives are named explicitly (`numpy.fft`,
`scipy.signal.{welch,csd,hilbert,butter,sosfiltfilt}`, `pywt`), and any adapted
third-party implementation cites its URL and commit in the docstring.

## Input and output

Input is the converter netCDF (integration notes A.2): dims `(time, height)`,
CF datetimes, per-window ancillaries (`ustar`, `L`, flags) on a `record` dim,
provenance globals. Sampling frequency is 10 Hz at siteGill and 20 Hz at
siteIRGA/siteVAC001; it is read from the file attribute, never hardcoded. The
humidity scalar is `rhov`, a vapor density in g m⁻³ rather than a specific
humidity, and outputs that consume it say so in their metadata.

Output is one netCDF per location with a group per analysis, so heterogeneous
dimensions coexist cleanly. Shared dims `record` (window start) and `height`;
CF-1.10 attributes; provenance globals (`source_file`, `git_commit`,
`created`, `taylor_hypothesis`, `detrend_method`, cutoff source flags).

| Group | Extra dims | Core variables |
|---|---|---|
| `/spectra` | `frequency` | `Suu,Svv,Sww,STT,Sqq`; `Co_wT,Co_wq,Co_uw`; `Qu_*`; `ogive_*`; `U_mean` |
| `/mrd` | `mr_scale` | MR (co)spectra `D_ww,D_wT,D_uw,...`; `tau_scale`; `gap_scale` |
| `/quadrant` | `quadrant`(4), `hole` | `Suw_frac`, `dur_frac`, `count`, `exuberance` |
| `/octant` | `octant`(8), `hole` | `flux_frac_wT`, `flux_frac_wq`, `count` |
| `/ramps` | (per record) | `amp`, `period`, `duration`, `microfront_time`, `direction`, `sr_flux`, per signal (T, u, e) |
| `/amplitude_mod` | (per record) | `R_AM_u`, `R_AM_w`, `R_AM_T`, `cutoff_lambda`, `cutoff_f`, cutoff source |
| `/scale_separation` | `scale_band`(3) | `var_frac`, `flux_frac_uw`, `flux_frac_wT`, band cutoffs |
| `/coherent_flux` | (per record) | `F_coh_frac_wT`, `F_coh_frac_uw`, `n_structures`, `method` |

## Modules

### Spectra, cospectra, ogives

Per window: detrend, taper (Hann default -- `[ASSUMED]` as common practice
until a source states a preference; the taper and its variance-correction
factor are stored), then one-sided PSD/CSD via Welch segment averaging with
log-binning. Normalization is fixed by the closure identities: the integral of
S_xx over frequency recovers σ_x², and the integral of the cospectrum recovers
the covariance. These identities are also the module's regression test. The
pre-multiplied form f·S(f) on a log-f axis feeds the scale-separation module.
Kaimal et al. (1972) surface-layer reference curves are overlaid for QC once
that paper's curve forms are extracted. The ogive is the cospectrum integrated
from the high-frequency end down to f₀; a low-frequency plateau indicates the
averaging window captures the flux-carrying scales. The attribution of the
ogive test to Desjardins et al. (1989) and Foken & Wichura (1996):
**confirm on extraction**.

Minimum reading set: Kaimal et al. 1972; Foken & Wichura 1996; one textbook
treatment (Kaimal & Finnigan 1994 or Stull 1988) for conventions.

### Multiresolution decomposition

Orthogonal Haar dyadic decomposition on N = 2^M samples per window: 32 768 of
36 000 at 20 Hz, 16 384 of 18 000 at 10 Hz (91 % retained either way; the trim
policy is recorded per window). For each mode m, segment means on windows of
length 2^m are removed and the MR (co)spectrum D_xy(m) is the covariance
carried at that scale. The sum over modes equals the total covariance --
Reynolds averaging holds at every scale, no periodicity assumed -- and that
sum closure is the module's free invariant test. Implementation follows the
Howell & Mahrt (1997) recursion as extracted from the paper. Two
recollections to check before coding: that Vickers & Mahrt (2003) contains a
worked numeric example usable as a validation target, and that their gap-scale
detection fits a fifth-order polynomial in log-τ to locate the post-peak zero
crossing. **Confirm both on extraction**; if the worked example does not
exist, validation falls back to sum closure plus synthetic signals with known
scale content.

Minimum reading set: Howell & Mahrt 1997; Vickers & Mahrt 2003.

### Quadrant and octant analysis

Quadrants of (u′, w′): outward interaction, ejection, inward interaction,
sweep. A hyperbolic hole of size H excludes events with |u′w′| < H·σ_u·σ_w;
recalled attribution of the hole to Lu & Willmarth (1973): **confirm on
extraction**. Reported per window and per H: stress fraction, duration
fraction, event count, and exuberance (counter-gradient over down-gradient
sums). Stress fractions at H = 0 sum to 1, the module's invariant. H sweeps
0 to 8 in steps to be fixed in config -- the range is `[ASSUMED]` until a
source motivates it. Scalar quadrants (w′,T′) and (w′,c′) reuse the machinery.

UTESpac already computes ejection-sweep asymmetries (`delta_flux_ctrb`,
`delta_time_ctrb`) and transport efficiency (`eta`); the H = 0 output must
reproduce those columns from the averaged pickle, a cross-validation that
costs nothing (integration notes A.3).

The octant module partitions on three signs, default triplet (w′, T′, q′)
with q′ from `rhov` perturbations -- available at four heights on siteIRGA and
one on siteGill. The default triplet is `[ASSUMED]` (configurable) until the
octant sources are read; it was chosen for scalar-dissimilarity diagnostics.

Minimum reading set: Wallace 2016 (review, orients the rest); Lu & Willmarth
1973; Raupach 1981; for octants Li & Bou-Zeid 2011 and Li & Bo 2019.

### Ramp detection on T, u, and TKE

Ramps -- gradual rise, sharp microfront -- are the surface-layer signature of
coherent structures. Two detectors run independently on T′, u′, and e, and
both emit results.

The structure-function detector recovers ramp amplitude from the real root of
the Van Atta cubic, with coefficients built from second-, third-, and
fifth-order structure functions, then period and microfront time from the
structure-function shape, then a surface-renewal flux from amplitude, period,
and measurement height. Every specific in that sentence is a paraphrase from
memory: the cubic's coefficient set, the order of moments used, and the flux
formula are implemented only from fresh extractions of Van Atta 1977, Spano
et al. 1997, and Paw U et al. 1995, with the finite-microfront refinement of
Chen et al. 1997 if adopted. **Confirm all on extraction.** The existing
third-order structure-function code in `utespac/calc_dissipation_rate.py`
supplies the windowing and NaN conventions to reuse.

The wavelet detector locates microfronts at extrema of a Haar (or Mexican-hat)
covariance transform, yielding event times, spacing, duration, and direction.
Its event set feeds the coherent-flux module.

Extension to TKE follows Mangan et al. (2022): the recalled rationale is that
some structures advect past a single sensor with cross-stream-dominated
velocity signatures and are missed by u,w-only detection, and that their
integrated quadrant analysis couples ejection, sweep, and quiescent phases in
time with two-lag structure-function pre-selection. That summary is exactly
the kind of accumulated paraphrase the fidelity rules exist to catch: the
library note quotes Mangan's actual identification criterion and IQA
definition before any of it is coded. All three velocity components are in the
raw pickle, so the TKE series needs nothing new upstream.

Minimum reading set: Van Atta 1977; Paw U et al. 1995; Spano et al. 1997;
Chen et al. 1997; Collineau & Brunet 1993 (both parts); Mangan et al. 2022;
Gao, Shaw & Paw U 1989 for the expected T-u ramp phase relation.

### Amplitude modulation

The signal is split at cutoff frequency f_c = Ū/λ_c into large-scale and
small-scale parts with zero-phase filters; the small-scale envelope comes from
the analytic signal (`scipy.signal.hilbert`), is low-passed at f_c, and the AM
coefficient R is the correlation of the large-scale signal with that filtered
envelope. R is computed for small scales modulated by large-scale u′ and by
large-scale w′, and for scalar (T′) modulation, with z/L reported alongside
from the joined `L` ancillary.

Deviation to register at implementation time: the source literature (Mathis
et al. 2009 and successors) sets the large/small cutoff in outer units, δ or
z_i. No z_i measurement exists at these sites, so the default cutoff is the
MRD gap scale, falling back to the pre-multiplied-spectrum valley, with a
`scaled` mode (user multiple of measurement height z) as the alternative --
locked decision 3. This substitution is defensible but is not what the cited
papers do; the register entry carries a sensitivity check of R against λ_c as
its validation.

Minimum reading set: Mathis, Hutchins & Marusic 2009; Salesky & Anderson 2018
and 2020 (ABL stability dependence); Talluru et al. 2014 for the
all-component treatment.

### LSM/VLSM separation

Band cutoffs come from the pre-multiplied streamwise spectrum, bimodal in
high-Reynolds wall flows: inner peak, outer peak, valley between. u′ (and
optionally w′, T′) is band-passed into small-scale, LSM, and VLSM bands, and
per-band variance and flux fractions are reported. The same missing-z_i
deviation as the amplitude-modulation module applies and shares its register
entry; cutoffs and their source flag are stored per record. Recalled
outer-unit ranges (LSM of order 2-3δ, VLSM larger) are not used, even as QC
bounds, until read from the sources.

Minimum reading set: Kim & Adrian 1999; Balakumar & Adrian 2007; Wang & Zheng
2016 for the atmospheric surface layer; Hutchins et al. 2012 for
lab-atmosphere reconciliation.

### Coherent-structure flux fractions

Two estimators, both written with a `method` label. The wavelet estimator
conditionally averages w′ and the scalars about the microfronts detected by
the ramp module and accumulates covariance inside the conditioned windows,
reporting the coherent fraction of ⟨w′T′⟩, ⟨w′q′⟩, and ⟨u′w′⟩ plus event
count. The quadrant estimator takes the ejection-plus-sweep contribution at a
chosen hole size. The two differ by construction and the literature reports
them separately, so no reconciliation is attempted. The quadrant estimator at
H = 0 has the same `delta_flux_ctrb` cross-check as the quadrant module.

Minimum reading set: Thomas & Foken 2007; Turner & Leclerc 1994; Collineau &
Brunet 1993 (shared with ramps).

## Package layout and dependencies

```
ec_coherent/
  io.py            # netCDF reader/writer, window iterator, CF metadata
  preprocess.py    # verification, detrend, Taylor helpers
  spectra.py
  mrd.py
  quadrant.py      # quadrant + octant
  ramps.py         # structure-function + wavelet detectors
  ampmod.py
  scales.py
  coherent_flux.py
  config.py        # cutoff modes, window length, hole sizes, defaults
  cli.py           # run all / subset over a location file
```

Sibling top-level package; may import from `utespac`, never the reverse.
Dependencies `numpy`, `scipy`, `xarray` + `netCDF4`, `pywt`; optional `numba`
for MRD/quadrant inner loops. Written natively in the target idiom of the
de-MATLAB roadmap (labeled data, datetime64, typed config, logging, no
prompts) so it doubles as the reference for migrated `utespac` code.

## Build order

0. Phase 0 for the first modules: sources in hand, `references.bib` and
   `library/papers/index.md` started, note skeletons created.
1. `io` + `preprocess` + the A.2 converter (plus `lat`/`lon` on `SiteInfo`,
   the one core change the integration needs).
2. `spectra` -- provides the cutoff inputs consumed by amplitude modulation,
   scale separation, and coherent flux.
3. `mrd`, then `quadrant`/`octant` (independent of each other).
4. `ramps`, wavelet path first since the coherent-flux module reuses it.
5. `ampmod`, `scales`.
6. `coherent_flux`.
7. `cli` plus full-file regression.

Testing follows the repo's existing pattern (pytest, pinned fixtures per
`tests/test_regression.py` and `utespac/testkit.py`): one clean 30-min window
per site class -- one 20 Hz multi-height, one 10 Hz single-height -- with the
closure identities asserted for every module (spectral integrals recover
variances, MRD modes sum to covariances, quadrant fractions sum to one, H = 0
results match the UTESpac asymmetry columns).

## Validation artifacts

Every science-affecting landing ships with a figure the user can inspect
(matplotlib static; plotly only where interaction genuinely helps), and the
generating script lands in the repo so the result is reproducible. Where a
source paper provides a worked example or canonical figure, reproducing it is
the module's acceptance test. Negative results -- an invariant that fails, a
detector that disagrees with the UTESpac cross-check, a recalled validation
target that turns out not to exist in the paper -- are recorded as prominently
as positive ones.

## Acceptance criteria

- No module is implemented before its minimum reading set is extracted and
  its library note exists with quotes and loci.
- Every science-critical constant in the notes carries a provenance label or
  a deviation-register entry; the missing-z_i cutoff substitution is
  registered with its sensitivity validation.
- Closure-identity tests pass for every module on the pinned fixtures, and
  the H = 0 quadrant output reproduces `delta_flux_ctrb`/`delta_time_ctrb`.
- Any `[NOVEL]` element has recorded user sign-off before implementation.
- The analysis netCDF carries full provenance attributes.

## Locked decisions (carried from the original)

1. "utke" means ramp/TKE-based structure detection per Mangan et al. (2022);
   "VSLM" read as VLSM. Interpretation decisions, not science claims.
2. Octant default triplet (w′, T′, q′), configurable -- `[ASSUMED]` pending
   the octant sources.
3. Cutoff policy for amplitude modulation and scale separation: both
   `spectral_gap` and `scaled` modes, default `spectral_gap` -- a registered
   deviation from the outer-unit convention, forced by the absence of z_i.
4. z_i/δ are absent from these files; outer-unit values, once read from
   sources, serve only as QC context.

## Reference request list (identifiers verified, content not)

Every identifier below was confirmed against a live Crossref or publisher
record on 2026-08-17 (three parallel verification passes; fetch URLs in the
session record). "Verified" here means the DOI/ISBN resolves and its metadata
matches the citation -- no PDF has been read, so no methods content is
verified. Entries are grouped by the module whose minimum reading set they
belong to; a few serve more than one module and are listed once.

Preprocessing and QC:

- Taylor 1938, Proc. R. Soc. Lond. A 164, 476-490. doi:10.1098/rspa.1938.0032
- Wilczak, Oncley & Stage 2001, Boundary-Layer Meteorol. 99, 127-150.
  doi:10.1023/A:1018966204465
- Vickers & Mahrt 1997, J. Atmos. Oceanic Technol. 14, 512-526.
  doi:10.1175/1520-0426(1997)014<0512:QCAFSP>2.0.CO;2 (upstream despiking
  context -- the raw pickles were conditioned this way)

Spectra, cospectra, ogives:

- Kaimal, Wyngaard, Izumi & Coté 1972, Q. J. R. Meteorol. Soc. 98, 563-589.
  doi:10.1002/qj.49709841707
- Foken & Wichura 1996, Agric. For. Meteorol. 78, 83-105.
  doi:10.1016/0168-1923(95)02248-1
- Desjardins, MacPherson, Schuepp & Karanja 1989, Boundary-Layer Meteorol.
  47, 55-69. doi:10.1007/BF00122322 (journal DOI; a book-chapter duplicate
  10.1007/978-94-009-0975-5_5 exists -- use the journal one)
- Kaimal & Finnigan 1994, Atmospheric Boundary Layer Flows, Oxford Univ.
  Press. ISBN 978-0-19-506239-7; ebook doi:10.1093/oso/9780195062397.001.0001
- Stull 1988, An Introduction to Boundary Layer Meteorology, Kluwer/Springer.
  ISBN 978-90-277-2769-5; ebook doi:10.1007/978-94-009-3027-8 (either
  textbook suffices for conventions)

Multiresolution decomposition:

- Howell & Mahrt 1997, Boundary-Layer Meteorol. 83, 117-137.
  doi:10.1023/A:1000210427798
- Vickers & Mahrt 2003, J. Atmos. Oceanic Technol. 20, 660-672.
  doi:10.1175/1520-0426(2003)20<660:TCGATF>2.0.CO;2

Quadrant and octant analysis:

- Wallace 2016, Annu. Rev. Fluid Mech. 48, 131-158.
  doi:10.1146/annurev-fluid-122414-034550 (review; read first)
- Wallace, Eckelmann & Brodkey 1972, J. Fluid Mech. 54, 39-48.
  doi:10.1017/S0022112072000515
- Lu & Willmarth 1973, J. Fluid Mech. 60, 481-511.
  doi:10.1017/S0022112073000315
- Raupach 1981, J. Fluid Mech. 108, 363-382. doi:10.1017/S0022112081002164
- Volino & Simon 1994, J. Turbomach. 116, 752-758. doi:10.1115/1.2929469
- Li & Bou-Zeid 2011, Boundary-Layer Meteorol. 140, 243-262.
  doi:10.1007/s10546-011-9613-5
- Li & Bo 2019, J. Wind Eng. Ind. Aerodyn. 189, 1-10.
  doi:10.1016/j.jweia.2019.03.013

Ramp detection and surface renewal:

- Van Atta 1977, Arch. Mech. 29, 161-171. NO DOI -- the journal's online
  archive starts at vol. 48 (1996). A scan must be sourced (interlibrary
  loan or author archive). The only entry without a resolvable identifier;
  see the decision note below the list.
- Antonia, Chambers, Friehe & Van Atta 1979, J. Atmos. Sci. 36, 99-108.
  doi:10.1175/1520-0469(1979)036<0099:TRITAS>2.0.CO;2
- Gao, Shaw & Paw U 1989, Boundary-Layer Meteorol. 47, 349-377.
  doi:10.1007/BF00122339
- Collineau & Brunet 1993, Part I, Boundary-Layer Meteorol. 65, 357-379.
  doi:10.1007/BF00707033
- Collineau & Brunet 1993, Part II, Boundary-Layer Meteorol. 66, 49-73.
  doi:10.1007/BF00705459 (the prior draft's "vol. 65, 381-410" was wrong --
  caught by the verification pass)
- Paw U, Qiu, Su, Watanabe & Brunet 1995, Agric. For. Meteorol. 74, 119-137.
  doi:10.1016/0168-1923(94)02182-J
- Spano, Snyder, Duce & Paw U 1997, Agric. For. Meteorol. 86, 259-271.
  doi:10.1016/S0168-1923(96)02420-3
- Chen, Novak, Black & Lee 1997, Boundary-Layer Meteorol. 84, 99-124.
  doi:10.1023/A:1000338817250
- Paw U, Snyder, Spano & Su 2005, in Micrometeorology in Agricultural
  Systems, Agronomy Monograph 47, 455-483. doi:10.2134/agronmonogr47.c20;
  ISBN 978-0-89118-158-3
- Mangan, Oldroyd, Paw U, Clay, Drake, Kelley & Suvocarev 2022, Integrated
  Quadrant Analysis: A New Method for Analyzing Turbulent Coherent
  Structures, Boundary-Layer Meteorol. 184(1), 45-69.
  doi:10.1007/s10546-022-00694-w (volume/pages now confirmed)
- Torrence & Compo 1998, Bull. Am. Meteorol. Soc. 79, 61-78.
  doi:10.1175/1520-0477(1998)079<0061:APGTWA>2.0.CO;2
- Blackwelder & Kaplan 1976, J. Fluid Mech. 76, 89-112.
  doi:10.1017/S0022112076003145 (VITA alternative; optional)

Amplitude modulation:

- Mathis, Hutchins & Marusic 2009, J. Fluid Mech. 628, 311-337.
  doi:10.1017/S0022112009006946
- Hutchins & Marusic 2007, J. Fluid Mech. 579, 1-28.
  doi:10.1017/S0022112006003946
- Marusic, Mathis & Hutchins 2010, Science 329, 193-196.
  doi:10.1126/science.1188765 (context)
- Talluru, Baidya, Hutchins & Marusic 2014, J. Fluid Mech. 746, R1.
  doi:10.1017/jfm.2014.132
- Salesky & Anderson 2018, J. Fluid Mech. 856, 135-168.
  doi:10.1017/jfm.2018.711
- Salesky & Anderson 2020, Phys. Rev. Lett. 125, 124501.
  doi:10.1103/PhysRevLett.125.124501

LSM/VLSM separation:

- Kim & Adrian 1999, Phys. Fluids 11, 417-422. doi:10.1063/1.869889
- Balakumar & Adrian 2007, Phil. Trans. R. Soc. A 365, 665-681.
  doi:10.1098/rsta.2006.1940
- Wang & Zheng 2016, J. Fluid Mech. 802, 464-489. doi:10.1017/jfm.2016.439
- Hutchins, Chauhan, Marusic, Monty & Klewicki 2012, Boundary-Layer
  Meteorol. 145, 273-306. doi:10.1007/s10546-012-9735-4

Coherent-structure flux fractions:

- Thomas & Foken 2007, Boundary-Layer Meteorol. 123, 317-337.
  doi:10.1007/s10546-006-9144-7
- Turner & Leclerc 1994, Conditional Sampling of Coherent Structures in
  Atmospheric Turbulence Using the Wavelet Transform, J. Atmos. Oceanic
  Technol. 11, 205-209.
  doi:10.1175/1520-0426(1994)011<0205:CSOCSI>2.0.CO;2 (registered title is
  longer than the prior draft's)

Decision note on Van Atta 1977: if a scan cannot be obtained, the fallback
is to implement the cubic from its restatement in Spano et al. 1997 or the
Paw U et al. 2005 chapter and cite the method accordingly -- a deliberate,
registered substitution, not a silent one. That choice is the user's under
the ask-for-sources rule.
