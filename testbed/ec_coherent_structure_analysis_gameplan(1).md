# Eddy-Covariance Coherent-Structure Analysis — Gameplan

Scope: post-processing analyses that consume an existing per-location netCDF of raw high-frequency series (multiple heights) and write a **separate** analysis netCDF. Nothing writes back to the raw series. EC flux computation and planar fit are assumed already done and available (rotation coefficients stored). Interpretations of two of your terms, flag if wrong: **utke** = streamwise-velocity + TKE ramps; **VSLM** = VLSM (very-large-scale motion).

Nothing here is restricted content — this is standard published boundary-layer turbulence analysis. No changes to your request are needed.

---

## 0. Conventions & shared preprocessing

All modules operate on **perturbation series** derived per flux-averaging window (default 30 min), never on the stored raw series. Shared preprocessing (one function, reused everywhere):

1. **Rotation.** Apply the stored planar-fit coefficients (b0, b1, b2) to (u,v,w) so w-mean → 0 in the fitted plane (Wilczak et al. 2001). Optionally a per-window double rotation for cross-checking. Rotation is applied in-memory only.
2. **Detrend.** Selectable: block mean (Reynolds), linear, or recursive-filter. Default block mean; expose linear for spectra where low-frequency leakage matters (Vickers & Mahrt 2003 discuss the trade-off).
3. **Despiking flag pass-through.** Read QC flags from the raw file if present; do not re-despike (that would modify the series). Windows failing QC are still processed but flagged.
4. **Taylor's hypothesis.** Frequency↔wavelength via λ = Ū/f, wavenumber k = 2πf/Ū, with Ū the per-window mean horizontal wind (Taylor 1938). Store Ū with every spectral/scale output; note breakdown for large λ/z.

Sign convention: fluxes positive upward. Perturbations denoted u′, w′, T′, c′. TKE e = ½(u′²+v′²+w′²).

**Code sourcing policy.** Numerics implemented directly from the cited equations. Where a standard library primitive is used it is named explicitly (`numpy.fft`, `scipy.signal.{welch,csd,hilbert,butter,sosfiltfilt}`, `pywt` for wavelets). If any third-party implementation is adapted, its URL/commit is cited in a docstring at that function.

---

## 1. Input file schema (raw, read-only)

Expected/negotiable — adapt reader to what your files actually carry.

- **Dimensions:** `time` (high-freq, e.g. 10–20 Hz), `height`.
- **Coordinate vars:** `time` (CF datetime), `height` (m).
- **Data vars (per height):** `u,v,w` (m s⁻¹), `Ts` sonic temperature or `T` (K), optional `q,co2,...`. Optional per-window ancillaries: `ustar`, `L` (Obukhov), `wdir`, `pf_b0,pf_b1,pf_b2`, QC flags.
- **Global attrs:** `site_id`, `lat`, `lon`, `sampling_frequency_hz`, `flux_averaging_s`, `canopy_height`, `bl_depth` or `zi` if available.

Reader yields an iterator of (window, height) perturbation arrays + metadata.

## 2. Output file schema (analysis, write)

One netCDF4 file per location, mirroring input location. Use **groups** per analysis so heterogeneous dims coexist cleanly. Shared dims: `record` (window start-time), `height`. CF-1.10 attributes, `units`, `long_name`, `cell_methods`, provenance globals (`source_file`, `git_commit`, `created`, `taylor_hypothesis`, `detrend_method`).

| Group | Extra dims | Core variables |
|---|---|---|
| `/spectra` | `frequency` | `Suu,Svv,Sww,STT,Sqq`; `Co_wT,Co_wq,Co_uw`; `Qu_*`; `ogive_wT,ogive_wq,ogive_uw`; `U_mean` |
| `/mrd` | `mr_scale` | `D_ww,D_wT,D_uw,...` (MR co/spectra), `tau_scale`, `gap_scale` |
| `/quadrant` | `quadrant`(4), `hole` | `Suw_frac`, `dur_frac`, `count`, `exuberance` |
| `/octant` | `octant`(8), `hole` | `flux_frac_wT`, `flux_frac_wq`, `count` |
| `/ramps` | (scalar per record) | `amp`, `period`, `duration`, `microfront_time`, `direction`, `sr_flux`, plus per-signal for T, u, e |
| `/amplitude_mod` | (scalar; opt `height2`) | `R_AM_u`, `R_AM_w`, `R_AM_T`, `cutoff_lambda`, `cutoff_f` |
| `/scale_separation` | `scale_band`(3) | `var_frac`, `flux_frac_uw`, `flux_frac_wT`, `lambda_cut_LSM`, `lambda_cut_VLSM` |
| `/coherent_flux` | (scalar per record) | `F_coh_frac_wT`, `F_coh_frac_uw`, `n_structures`, `method` |

---

## 3. Modules

### 3.1 Spectral analysis — spectra, cospectra, ogive

**Method.** Per window: detrend → taper (Hann default; store window and its variance-correction factor) → one-sided PSD/CSD. Use Welch segment-averaging (`scipy.signal.welch`/`csd`) for stable estimates, then log-bin. Normalize so integrals recover (co)variance:

- PSD: ∫₀^∞ S_xx(f) df = σ_x²
- Cospectrum Co_xy(f)=Re{cross-spectrum}, Quadrature Qu_xy(f)=−Im{...}; ∫₀^∞ Co_xy df = ⟨x′y′⟩
- Pre-multiplied form f·S(f) (log-f axis) for peak identification, feeds §3.6.
- Optional Kaimal surface-layer reference curves for QC overlay (Kaimal et al. 1972).

**Ogive.** Cumulative cospectral integral from high→low frequency:
Og_xy(f₀) = ∫_{f₀}^{∞} Co_xy(f) df.
Convergence/plateau at low f indicates the averaging window captures the flux-carrying scales; non-monotonicity flags meso-scale contamination (ogive test: Desjardins et al. 1989; Foken & Wichura 1996).

Citations: Kaimal et al. 1972; Kaimal & Finnigan 1994; Stull 1988; Taylor 1938; Foken & Wichura 1996.

### 3.2 Multiresolution decomposition (MRD)

**Method.** Orthogonal Haar-basis dyadic decomposition; requires N = 2^M samples (trim/pad per window, record which). For each mode m (1…M), remove segment means on windows of length 2^m; the MR (co)spectrum D_xy(m) is the covariance carried at scale 2^m, with Σ_m D_xy(m) = ⟨x′y′⟩ — i.e. Reynolds averaging holds at every scale and periodicity is not assumed. Timescale τ_m = 2^m / f_s (→ length via Ū).

Implement the recursion of **Howell & Mahrt (1997)**; validate against the worked numeric example in **Vickers & Mahrt (2003, Fig. 1)**. **Gap-scale detection:** fit the MR cospectrum vs log-τ (5th-order polynomial per Vickers & Mahrt) and locate the post-peak gap/zero-crossing → `gap_scale`, the turbulent/meso separation time.

Citations: Howell & Mahrt 1997; Vickers & Mahrt 2003.

### 3.3 Quadrant & octant analysis

**Quadrant (u′,w′).** Q1 outward interaction (u′>0,w′>0), Q2 ejection (u′<0,w′>0), Q3 inward interaction (u′<0,w′<0), Q4 sweep (u′>0,w′<0). Hyperbolic hole size H excludes weak events (|u′w′| < H·σ_u·σ_w). Stress fraction:

S_{i,H} = (1/⟨u′w′⟩)·⟨u′w′·I_{i,H}⟩,  with Σ_i S_{i,0} = 1.

Also duration fraction, event count, and exuberance = (Σ counter-gradient)/(Σ down-gradient). Repeat for scalar quadrants (w′,T′) and (w′,c′). Sweep H from 0 to ~8.

Citations: Wallace, Eckelmann & Brodkey 1972; Lu & Willmarth 1973 (hole size); Raupach 1981 (rough-wall / atmospheric); review: Wallace 2016.

**Octant.** Three-variable sign partition, **default (w′,T′,q′)** (configurable), → 8 classes, with per-octant contributions to ⟨w′T′⟩ and ⟨w′q′⟩; diagnoses scalar/scalar and momentum/scalar transport dissimilarity.

Citations: Volino & Simon 1994; Li & Bou-Zeid 2011; Li & Bo 2019.

### 3.4 Ramp detection — temperature and u/TKE

Ramps (gradual ramp + sharp microfront) are the surface-layer signature of coherent structures; detect on T′, u′, and e independently.

Two detectors, both emitted:

1. **Structure-function / surface-renewal.** S_n(r)=⟨(θ(t+r)−θ(t))^n⟩. Ramp amplitude `a` from the real root of the Van Atta cubic a³ + p·a + q = 0, with p,q built from 2nd/3rd/5th-order structure functions; ramp period and microfront time from the S_n(r) shape; surface-renewal flux from a, total ramp period, and measurement height. Implement coefficients directly from Van Atta 1977 / Spano et al. 1997; scalar-renewal flux per Paw U et al. 1995; two-scale/finite-microfront refinement per Chen et al. 1997.
2. **Wavelet.** Haar (or Mexican-hat) covariance transform; local extrema of the wavelet coefficient locate microfronts → event times, spacing, duration, direction (warm-to-cool vs cool-to-warm). Feeds §3.7 conditional sampling.

**u / TKE ramps (utke).** Same detection machinery applied to u′ and to e. Follow **Mangan et al. (2022)**: identify the ejection phase of coherent structures from **TKE** rather than from u′/w′ alone, because some structures are cross-stream (v′) dominated as they advect past a single sensor and are missed by u,w-only detection. Their integrated quadrant analysis (IQA) couples ejection→sweep→quiescent phases in time and preserves the structure trajectory from Eulerian data; probable-ramp windows are pre-selected with the van Atta **two-lag** structure-function amplitude/duration estimate (van Atta 1977; Paw U et al. 2005). Expect u-ramps largely anti-phase with T-ramps (sweep = high-u, cool; Gao et al. 1989). Emit ramp stats for T, u, and e, plus IQA event trajectories/strengths.

Citations: Van Atta 1977; Antonia et al. 1979; Paw U et al. 1995, 2005; Spano et al. 1997; Chen et al. 1997; Gao, Shaw & Paw U 1989; Collineau & Brunet 1993 (Parts I & II); Mangan et al. 2022 (IQA, TKE identification); Torrence & Compo 1998; Blackwelder & Kaplan 1976 (VITA alt.).

### 3.5 Amplitude modulation (LSM modulation of small scales)

**Cutoff policy (both supported).** `cutoff_mode ∈ {spectral_gap, scaled}`. **Default `spectral_gap`** — λ_c from the §3.2 MRD gap scale (fallback §3.1 pre-multiplied-spectrum valley) — because z_i/δ scaling requires a mixed-layer-depth measurement not typically available here (no Doppler lidar; z_i not in the raw files). `scaled` mode uses a user multiple of measurement height z (or δ/z_i if ever supplied). λ_c stored per record with its source flag.

**Method.** f_c = Ū/λ_c. Split the signal:
- large scale u_L = low-pass(u′, f_c); small scale u_S = high-pass(u′, f_c) (zero-phase `sosfiltfilt`).
- envelope via analytic signal: E(t) = |u_S(t) + i·𝓗{u_S}(t)| (`scipy.signal.hilbert`).
- E_L = low-pass(E, f_c).

AM coefficient:
R = [⟨u_L·E_L⟩ − ⟨u_L⟩⟨E_L⟩] / (σ_{u_L}·σ_{E_L}).

Compute R for the small scales modulated by large-scale u′ **and** by large-scale w′; also modulation of small-scale T′ (scalar AM). Report stability (z/L) alongside.

Citations: Mathis, Hutchins & Marusic 2009 (Hilbert-envelope method); Hutchins & Marusic 2007; Marusic, Mathis & Hutchins 2010; Talluru et al. 2014 (all-component AM); Salesky & Anderson 2018 & 2020 (ABL stability dependence, w-modulation, flux-gradient link).

### 3.6 VLSM / LSM separation

**Cutoff policy (both supported), default spectral.** Identify band cutoffs from the **pre-multiplied streamwise spectrum** f·S_uu (bimodal in high-Re wall flows): inner peak → LSM, outer peak → VLSM, spectral valley → λ_cut1/λ_cut2. Literature outer-unit reference (LSM ~ O(2–3δ), VLSM ~ >3δ up to ~10–20δ or z_i) is used only for QC/sanity, not as the default cutoff, since z_i/δ are **not typically available** in these files. **Fallback when the spectral valley is ambiguous:** a user-set multiple of measurement height z (`scaled` mode). Store both cutoffs and their source flag per record.

**Method.** Band-pass u′ (and, if desired, w′,T′) into small-scale / LSM / VLSM via zero-phase filters at the two cutoffs; report per-band variance fraction and flux (⟨u′w′⟩, ⟨w′T′⟩) fraction.

Citations: Kim & Adrian 1999; Balakumar & Adrian 2007; ASL specifics: Wang & Zheng 2016; Hutchins et al. 2012 (lab↔atmosphere reconciliation).

### 3.7 Fluxes carried by coherent structures

**Method (two estimators, both emitted with `method` label):**
1. **Wavelet conditional sampling.** Using §3.4 wavelet event set, conditionally average w′ and (T′,c′,u′) about detected microfronts; the coherent flux is the covariance accumulated inside the conditioned windows; report F_coh/F_total for ⟨w′T′⟩, ⟨w′q′⟩, ⟨u′w′⟩ and event count.
2. **Quadrant-derived.** Coherent contribution from ejection+sweep (Q2+Q4) at a chosen hole size, from §3.3.

The two differ by construction (sampling strategy) — expose both rather than reconcile, matching the literature.

Citations: Thomas & Foken 2007; Turner & Leclerc 1994; Collineau & Brunet 1993.

---

## 4. Package layout & dependencies

```
ec_coherent/
  io.py            # netCDF reader/writer, window iterator, CF metadata
  preprocess.py    # rotation, detrend, Taylor helpers  (§0)
  spectra.py       # §3.1
  mrd.py           # §3.2
  quadrant.py      # §3.3 (quadrant + octant)
  ramps.py         # §3.4 (structure-function + wavelet)
  ampmod.py        # §3.5
  scales.py        # §3.6
  coherent_flux.py # §3.7
  config.py        # cutoffs, window length, hole sizes, defaults
  cli.py           # run all / subset over a location file
```

Deps: `numpy`, `scipy`, `xarray`+`netCDF4`, `pywt` (wavelets). Optional `numba` for the MRD/quadrant inner loops.

## 5. Build order

1. `io` + `preprocess` (unblocks everything; validate rotation/detrend on one window).
2. `spectra` (gives cutoffs/gap consumed by §3.5–3.7).
3. `mrd`, `quadrant`/`octant` (independent, quick wins).
4. `ramps` (wavelet path first — reused by §3.7).
5. `ampmod`, `scales`.
6. `coherent_flux`.
7. `cli` + full-file regression test against one known window.

---

## 6. References (DOIs verified against publisher records where marked ✓; others given as journal/volume/pages)

- Taylor, G.I. 1938. The spectrum of turbulence. *Proc. R. Soc. Lond. A* 164, 476–490. https://doi.org/10.1098/rspa.1938.0032
- Kaimal, Wyngaard, Izumi, Coté 1972. Spectral characteristics of surface-layer turbulence. *Q. J. R. Meteorol. Soc.* 98, 563–589. https://doi.org/10.1002/qj.49709841707 ✓
- Wallace, Eckelmann, Brodkey 1972. The wall region in turbulent shear flow. *J. Fluid Mech.* 54, 39–48.
- Lu, Willmarth 1973. Measurements of the structure of the Reynolds stress in a turbulent boundary layer. *J. Fluid Mech.* 60, 481–511. https://doi.org/10.1017/S0022112073000315 ✓
- Blackwelder, Kaplan 1976. On the wall structure of the turbulent boundary layer. *J. Fluid Mech.* 76, 89–112.
- Van Atta, C.W. 1977. Effect of coherent structures on structure functions of temperature in the atmospheric boundary layer. *Arch. Mech.* 29, 161–171.
- Antonia, Chambers, Friehe, Van Atta 1979. Temperature ramps in the atmospheric surface layer. *J. Atmos. Sci.* 36, 99–108. https://doi.org/10.1175/1520-0469(1979)036<0099:TRITAS>2.0.CO;2 ✓
- Raupach, M.R. 1981. Conditional statistics of Reynolds stress in rough-wall and smooth-wall turbulent boundary layers. *J. Fluid Mech.* 108, 363–382.
- Gao, Shaw, Paw U 1989. Observation of organized structure in turbulent flow within and above a forest canopy. *Boundary-Layer Meteorol.* 47, 349–377. https://doi.org/10.1007/BF00122339 ✓
- Desjardins, MacPherson, Schuepp, Karanja 1989. An evaluation of aircraft flux measurements of CO₂, water vapor and sensible heat. *Boundary-Layer Meteorol.* 47, 55–69.
- Collineau, Brunet 1993. Detection of turbulent coherent motions in a forest canopy. Parts I & II. *Boundary-Layer Meteorol.* 65, 357–379 / 381–410.
- Kaimal, Finnigan 1994. *Atmospheric Boundary Layer Flows.* Oxford Univ. Press.
- Turner, Leclerc 1994. Conditional sampling of coherent structures using the wavelet transform. *J. Atmos. Oceanic Technol.* 11, 205–209.
- Volino, Simon 1994. An application of octant analysis to turbulent and transitional flow data. *J. Turbomach.* 116, 752–758.
- Paw U, Qiu, Su, Watanabe, Brunet 1995. Surface renewal analysis: a new method to obtain scalar fluxes. *Agric. For. Meteorol.* 74, 119–137.
- Foken, Wichura 1996. Tools for quality assessment of surface-based flux measurements. *Agric. For. Meteorol.* 78, 83–105. https://doi.org/10.1016/0168-1923(95)02248-1 ✓
- Chen, Novak, Black, Lee 1997. Coherent eddies and temperature structure functions… Part I: ramp model with finite microfront time. *Boundary-Layer Meteorol.* 84, 99–124.
- Spano, Snyder, Duce, Paw U 1997. Surface renewal analysis for sensible heat flux density using structure functions. *Agric. For. Meteorol.* 86, 259–271.
- Paw U, Snyder, Spano, Su 2005. Surface renewal estimates of scalar exchange. In *Micrometeorology in Agricultural Systems*, Agronomy Monograph 47, ASA-CSSA-SSSA, 455–483. (van Atta two-lag structure-function ramp estimate)
- Howell, Mahrt 1997. Multiresolution flux decomposition. *Boundary-Layer Meteorol.* 83, 117–137. https://doi.org/10.1023/A:1000210427798
- Torrence, Compo 1998. A practical guide to wavelet analysis. *Bull. Am. Meteorol. Soc.* 79, 61–78.
- Kim, Adrian 1999. Very large-scale motion in the outer layer. *Phys. Fluids* 11, 417–422. https://doi.org/10.1063/1.869889 ✓
- Wilczak, Oncley, Stage 2001. Sonic anemometer tilt correction algorithms. *Boundary-Layer Meteorol.* 99, 127–150. (planar fit)
- Vickers, Mahrt 2003. The cospectral gap and turbulent flux calculations. *J. Atmos. Oceanic Technol.* 20, 660–672. https://doi.org/10.1175/1520-0426(2003)20<660:TCGATF>2.0.CO;2 ✓
- Balakumar, Adrian 2007. Large- and very-large-scale motions in channel and boundary-layer flows. *Phil. Trans. R. Soc. A* 365, 665–681. https://doi.org/10.1098/rsta.2006.1940 ✓
- Thomas, Foken 2007. Flux contribution of coherent structures… in a tall spruce canopy. *Boundary-Layer Meteorol.* 123, 317–337. https://doi.org/10.1007/s10546-006-9144-7 ✓
- Hutchins, Marusic 2007. Evidence of very long meandering features in the log region… *J. Fluid Mech.* 579, 1–28.
- Mathis, Hutchins, Marusic 2009. Large-scale amplitude modulation of the small-scale structures in turbulent boundary layers. *J. Fluid Mech.* 628, 311–337. https://doi.org/10.1017/S0022112009006946 ✓
- Marusic, Mathis, Hutchins 2010. Predictive model for wall-bounded turbulent flow. *Science* 329, 193–196. https://doi.org/10.1126/science.1188765 ✓
- Li, Bou-Zeid 2011. Coherent structures and the dissimilarity of turbulent transport of momentum and scalars… *Boundary-Layer Meteorol.* 140, 243–262. https://doi.org/10.1007/s10546-011-9613-5 ✓
- Hutchins, Chauhan, Marusic, Monty, Klewicki 2012. Towards reconciling the large-scale structure of turbulent boundary layers in the atmosphere and laboratory. *Boundary-Layer Meteorol.* 145, 273–306.
- Talluru, Baidya, Hutchins, Marusic 2014. Amplitude modulation of all three velocity components in turbulent boundary layers. *J. Fluid Mech.* 746, R1. https://doi.org/10.1017/jfm.2014.132 ✓
- Wallace, J.M. 2016. Quadrant analysis in turbulence research: history and evolution. *Annu. Rev. Fluid Mech.* 48, 131–158.
- Wang, Zheng 2016. Very large scale motions in the atmospheric surface layer: a field investigation. *J. Fluid Mech.* 802, 464–489.
- Mangan, M.R., et al. 2022. Integrated Quadrant Analysis: a new method for analyzing turbulent coherent structures (TKE-based coherent-structure identification). *Boundary-Layer Meteorol.* https://doi.org/10.1007/s10546-022-00694-w ✓ (DOI verified; confirm exact volume/pages at implementation)
- Salesky, Anderson 2018. Buoyancy effects on large-scale motions in convective atmospheric boundary layers… *J. Fluid Mech.* 856, 135–168. https://doi.org/10.1017/jfm.2018.711 ✓
- Li, Bo 2019. An application of quadrant and octant analysis to the atmospheric surface layer. *J. Wind Eng. Ind. Aerodyn.* 189, 1–10.
- Salesky, Anderson 2020. Coherent structures modulate atmospheric surface layer flux-gradient relationships. *Phys. Rev. Lett.* 125, 124501. https://doi.org/10.1103/PhysRevLett.125.124501 ✓

## 7. Resolved decisions (locked)

1. **utke** = streamwise-velocity + TKE ramps. TKE-based coherent-structure identification per Mangan et al. (2022) — TKE outperforms u/w-only detection because some structures are cross-stream dominated at a single sensor. (VSLM in the original request = VLSM.)
2. **Octant default triplet = (w′, T′, q′)** (configurable).
3. **Cutoff policy for §3.5 AM and §3.6 VLSM/LSM:** both `spectral_gap` and `scaled` modes implemented; **default `spectral_gap`** (MRD gap / pre-multiplied-spectrum valley) because z_i is hard to obtain without Doppler lidar.
4. **z_i / δ not typically present** in the raw files → `scaled` mode falls back to a user multiple of measurement height z; outer-unit (δ, z_i) values used only for QC when available.
