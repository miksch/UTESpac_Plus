# ec_coherent coherent-structure flux fractions

Step 6 of the gameplan ("Coherent-structure flux fractions"): the fraction
of $\overline{w'T_s'}$, $\overline{w'q'}$, $\overline{u'w'}$ carried by
coherent structures, per 30-min record. Two estimators, both labelled
`method`: conditional averages about the ramp-module event times (triple
decomposition), and the ejection-plus-sweep quadrant contribution at a
hole size. Written 2026-08-23 from fresh extractions of Thomas2007 (text
layer) and Turner1994 (image scan, pages rendered and read), plus the
Collineau1993a extraction re-read for the conditional-average equations.
Code in [ec_coherent/coherent_flux.py](../../ec_coherent/coherent_flux.py);
the event detectors live in [ec_ramps.md](ec_ramps.md).

## Conditional average and triple decomposition -- Collineau & Brunet (1993 II) [@Collineau1993a]

Definition (eq. 3, p. 62): conditional averages are "sampled on a 30 s
long window centered at detection points", and

$$\langle f(t)\rangle = \frac{1}{N}\sum_{i=1}^{N} f(t + t_i),$$

"$t_i$ is the detection time", $N$ the number of events. "The length of
30 s chosen for the averaging window is of the order of the period
between events" (p. 62). Localization in time is what matters for the
window centre: "the RAMP and HAAR wavelets are well localized in time,
whereas MHAT and WAVE are well localized in frequency" (p. 62) -- their
conditional averages use RAMP detection; ours inherit the ramps module's
zero-crossing times (deviation register).

Triple decomposition (§7.2, eqs. 4-7, pp. 69-70): $F = \bar F + f_l +
f_s$, "$f_l$ a perturbation due to the large-scale motion and $f_s$ the
remaining small-scale fluctuation". Conditional averaging of $f' = f_l +
f_s$ gives $\langle f'\rangle = f_l$ "under the assumption that $f_s$ is
uncorrelated with the detected large-scale motion ($\langle f_s\rangle =
0$)", and for products (eq. 5)

$$\langle f'g'\rangle = \langle f'\rangle\langle g'\rangle + \langle f_s g_s\rangle.$$

Averaging over the detection window with $\widetilde{(\,)} =
\frac{1}{\Delta T}\int_{-\Delta T/2}^{+\Delta T/2}\langle\,\rangle\,dt$
(eq. 6) yields eq. 7,
$\widetilde{\langle f'g'\rangle} = \widetilde{\langle f'\rangle\langle
g'\rangle} + \widetilde{\langle f_s g_s\rangle}$: "the first term on the
right-hand side of Equation (7) represents the contribution to
$\overline{f'g'}$ from the organized motions" (p. 70). Checks they run
that we adopt: $\widetilde{\langle u'w'\rangle}/\overline{u'w'}$ and
$\widetilde{\langle w'T'\rangle}/\overline{w'T'}$ "reasonably close to 1
(respectively, 1.11 and 1.15 at $z/h$ = 1.24)" (p. 70). Their large-scale
contributions: 0.26 (stress) and 0.40 (heat) at $z/h$ = 1.24, 0.31 and
0.39 at $z/h$ = 0.82; "large-scale motions appear to be more efficient at
transporting heat than momentum" (p. 70). On the $\langle f_s\rangle = 0$
assumption: the small-scale contribution "appears as a roughly constant
background level ... the assumption of small- and large-scale motions
being uncorrelated is not entirely appropriate but seems reasonable
enough for our purpose" (pp. 70-71).

## The flux-contribution formalism -- Thomas & Foken (2007) [@Thomas2007]

Same decomposition, notation $x = \bar x + x_l + x_t$ (eqs. 1-2, p. 321),
$\langle x'\rangle = \langle x_l\rangle$ (eq. 3), and (eq. 4)
$\langle x'y'\rangle = \langle x'\rangle\langle y'\rangle + \langle x_t
y_t\rangle$. Their window operator (eq. 5, p. 321) is scale-specific:

$$\widetilde{\langle x\rangle} = \frac{1}{2D_e}\int_{-D_e}^{+D_e}\langle x'(t)\rangle\,dt,$$

"The window of the conditional averages is centered at the moment of
detection $t_i$ and spans $2D_e$", $D_e$ the event duration from the
wavelet-variance peak (§3.1: Morlet variance spectrum after a low-pass
"removing all fluctuations with an event duration $<6.2$ s", then a MHAT
transform at $D_e$ for the moments of occurrence $t_i$; $D = \frac12
f^{-1}$ citing Collineau & Brunet 1993a). Equation 6/7:
$\widetilde{\langle x'y'\rangle} = \widetilde{\langle x'\rangle\langle
y'\rangle} + \widetilde{\langle x_t y_t\rangle}$, written $F(x,y)_{tot} =
F(x,y)_{cs} + F(x,y)_t$; "$F(x,y)_{cs}$ reflects the sole contribution of
the coherent structures at their specific temporal scale, and not the
entire exchange during the occurrence of coherent structures ... as e.g.,
quadrant analysis does" (pp. 321-322). Ejection/sweep split: "applying
the averaging operator in Eq. 5 within the borders of $[-D_e, 0]$ and
$[0, +D_e]$, respectively, whereas $F_{cs} = F_{ej} + F_{sw}$" (p. 322)
-- with the $\frac{1}{2D_e}$ prefactor kept, so the identity is exact.
Fluxes: $\tau = F(w,u)$, $H = F(w,T_s)$, $C = F(w,c_{CO_2})$, $\lambda E
= F(w,q)$ (p. 322).

Event signal: they sample all fluxes "according to the moments of
occurrence $t_i$ at the characteristic event duration $D_e$ for the sonic
temperature data", for three stated reasons (Ts ramps pronounced day and
night, available at all heights, literature comparability) (p. 322). We
default to $u'$ events instead (deviation register -- the VAC001 Ts
scalogram has no ramp-scale peak in most records; ruling 2026-08-23).

Quality gates (§4.2, p. 325): discard records whose "event duration
$D_e$ determined for one interval was significantly different when
compared to its neighbouring intervals" and those where "the total flow
of the conditional average was not representative for the flow of the
entire time series, i.e., $F_{tot}\overline{x'y'}^{-1} < 0.8$ and
$F_{tot}\overline{x'y'}^{-1} > 1.2$". We store the ratio and a validity
flag rather than dropping records.

Quadrant analysis as the second estimator (§3.3, p. 322): quadrants of a
scatter plot of $x$ and $y$, hyperbolic hole "$L = x'y'(\sigma_x
\sigma_y)^{-1}$", flux fraction per quadrant $S_i = \langle
xy\rangle_i/\overline{xy}$ (eq. 8) with the indicator-function average
(eq. 9); "quadrant analysis provides an estimate for the entire exchange
during the occurrence of coherent structures, including high-frequency
turbulence and potential mesoscale activity". Sweep/ejection quadrants
"S4 and S2, respectively, for downward directed net fluxes, and by S3 and
S1, respectively, for upward directed net fluxes. In this study, the sign
of the correlation coefficient $r_{xy}$ was used to determine the
direction of the flux" (p. 322).

Their results, the comparison targets: quadrant values "three to four
times larger" than the conditional/wavelet estimator for momentum, "up to
a factor of 2" for buoyancy (p. 325); low-passing the series at 0.1 Hz
before quadrant analysis moves the two "by approximately 20%" together
(p. 326-327); mean $F_{cs}F_{tot}^{-1}$ over all heights 0.16 (momentum)
and 0.26 (scalars); transport efficiency (flux contribution over time
fraction) 0.30 and 0.53, time fraction 0.39-0.62 (p. 327). Antecedents
they compare to: Collineau & Brunet 0.33/0.35, Antonia et al. 0.28/0.44
(p. 328).

EC flux error (§4.3, eqs. 10-12, p. 329): with $\Delta x$ "the assumed
error in calculating $\bar x_R$ due to the presence of coherent
structures",

$$\overline{x'y'}_{corr} = \overline{x'y'} - \frac{1}{n-1}\sum_{i=1}^{n-1}\left(x'\Delta y + y'\Delta x - \Delta y\,\Delta x\right) = \overline{x'y'} - \Delta\overline{x'y'},$$

with "$\Delta x \approx \bar x_R - \widetilde{\langle x'\rangle}$" (the
printed locus; the operative content is $\Delta x = -\widetilde{\langle
x'\rangle}$ once $\bar x_R$ is the removed mean -- `[DERIVED]` in the
code, where block detrend gives $\sum x' = 0$ and the sum collapses to
$-\frac{n}{n-1}\Delta x \Delta y$). Result: "the relative flux error did
not exceed 4% in the majority of cases", no systematic bias (p. 329).

## Wavelet conditional sampling -- Turner & Leclerc (1994) [@Turner1994]

Read (rendered scan) as the third named source; adopted as the optional
third estimator `method = "turner"` (ruling 2026-08-24, off by default).
Haar family $\Psi^{(m)}(x - x') = 2^{-m/2}
\Psi^{(0)}\left(\frac{x-x'}{2^m}\right)$ (eq. 1), coefficients $W(m,x)$
(eq. 3), reconstruction (eq. 4, "specifically for use with the Haar
basis"). The sampling rule (eq. 5, p. 206): coefficients are split at "some
multiple of the root-mean-square of wavelet coefficients at that scale,
for the entire record", $|W(m,x)| > K\left[\frac{1}{N}\sum W^2(m,x)
\right]^{1/2}$; inverse-transforming the two sets separately yields
"strong" and "weak" signal components whose sum is the original signal
(minus the long-term trend, p. 208). "Unlike other methods, there is no
preferred scale" (p. 205). $K = 4$ "is the default value for this paper";
$K = 2$ admits background, $K = 6$ weakens genuine events (p. 208, figs.
3-4). No flux fractions are computed in the paper (temperature only); the
strong-signal covariance fraction below is our construction on their
sampling scheme (deviation register).

## What the code does

`coherent_flux.py`: `conditional_average` (eq. 3 sampler on a lag grid,
NaN-padded at the record edges), `flux_contribution` (the eq. 5-7
pipeline for one pair and one event set: $F_{tot}$, $F_{cs}$, $F_t$,
$F_{ej}$, $F_{sw}$, $F_{tot}/\overline{x'y'}$ ratio and 0.8-1.2 validity
flag, eq. 12b flux error), `quadrant_fraction` (estimator 2: ejection +
sweep flux fractions above each hole from
[quadrant.py](../../ec_coherent/quadrant.py) `quadrant_stats`, quadrants
chosen sign-aware per the $r_{xy}$ rule), `haar_coefficients` /
`haar_reconstruct` / `turner_split` / `turner_fraction` (estimator 3,
`turner = true`: the eq. 1-5 redundant dyadic Haar transform via
cumulative sums, per-scale $K\,$-rms split, strong-signal covariance
fraction), `run` (the `/coherent_flux` group: per event signal and pair
`F_frac_<pair>_<sig>` = $F_{cs}/F_{tot}$ with the pieces,
`n_structures_<sig>`, window half-width, flux error; quadrant estimator
on a `hole` axis; `F_frac_turner_<pair>` and its fidelity ratio when
enabled; `method` labels on every variable). Events are re-detected with
the ramps-module detector under the `[ramps]` config so the module runs
standalone (same convention as ampmod's internal cutoff).

## Deviation register

- **Event set and duration scale.** Thomas & Foken detect on $T_s$ with a
  Morlet variance spectrum for $D_e$, then a MHAT transform at $D_e$ for
  $t_i$ `[CITED]` §3.1. We reuse the ramps module detector (MHAT variance
  peak $a_0$ with the smallest-scale-peak rule and $D_{min}$ = 6.2 s,
  zero-crossings at $a_0$) and condition on $u'$ events by default with
  $T_s'$ alongside -- the VAC001 ruling (2026-08-23): visible Ts ramps
  usually produce no ramp-scale scalogram peak here. Validation: the
  conditional-average patterns (fig-2 analog) and the $F_{tot}/cov$ gate.
- **Window.** `window = "duration"` (default): half-width $D_e$ of the
  event signal's record `[CITED]` TF2007 eq. 5. `window = "fixed"`:
  configurable `window_s` (CB93b's 30 s, chosen as "the order of the
  period between events") for cross-checks.
- **Records are kept, flagged.** TF2007 discard $F_{tot}/cov$ outside
  0.8-1.2 and $D_e$-outlier records; we store `valid_<pair>_<sig>` and the
  ratio instead and leave rejection to the analysis stage. Their
  $D_e$-neighbour test is not implemented (single-file records; the
  stored $D_e$ series makes it checkable).
- **Flux error simplification.** Eq. 12b is evaluated with $\Delta x =
  -\widetilde{\langle x'\rangle}$ `[DERIVED]` (equivalent to their
  $\bar x_R - \widetilde{\langle x\rangle}$ statement under our block
  detrend where the record mean is removed exactly).
- **Quadrant estimator reuse.** Estimator 2 sums the two downgradient
  quadrant flux fractions (ejection: downgradient with $w' > 0$; sweep:
  $w' < 0$) from `quadrant_stats` -- sign-aware, equivalent to TF2007's
  $r_{xy}$-sign quadrant selection `[CITED]` §3.3, and identical at
  $H = 0$ to the S2+S4 / S1+S3 choice. Hole grid default (0, 0.5, 1)
  `[CITED]` TF2007 figs. 3-5. At $H = 0$, $F_{ej} - F_{sw}$ equals the
  quadrant module's `delta_S` (cross-check in the tests).
- **The N = 1 degeneracy.** With a single event the conditional average
  *is* the sampled series, so $F_{cs} = F_{tot}$ identically and the
  fraction is 1 regardless of the data -- the estimator carries no
  information. On VAC001 this happens where the $u'$ variance peak lands
  on the trend-hump scale ($D_e \approx$ 380-430 s, one surviving
  zero-crossing; 6 of 96 records on 2023-07-06). These records pass the
  0.8-1.2 gate (the ratio is exactly 1), which is why TF2007's
  $D_e$-neighbour outlier test exists. No threshold is imposed in the
  module (`n_structures` is stored); the validation script excludes
  $N < 2$ from its statistics.
- **Turner & Leclerc estimator: adopted as an option, covariance
  construction `[NOVEL]`.** Their paper decomposes single signals and
  computes no flux. `turner_fraction` splits *each* series of a pair by
  its own per-scale $K\,$-rms threshold (as they do for temperature; no
  cross-signal coefficient masking) and reports $\overline{x_s y_s}$
  over the reconstructed covariance -- user sign-off 2026-08-24, off by
  default (`turner = false`, `turner_K` = 4 their default `[CITED]`
  p. 208). Implementation notes: coefficients via cumulative sums,
  dyadic scales $m = 1..\lfloor\log_2 n\rfloor$, positions whose support
  passes the record end are zero (invalid, excluded from the rms);
  reconstruction fidelity on a 36 000-sample AR(0.95) series: corr
  0.9998, variance ratio 0.998 (tests). The rms in eq. 5 is over the
  valid coefficients only `[ASSUMED]` (the paper's $N$ is not defined
  for the edge region). The absolute fraction is strongly
  $K$-dependent (their figs. 3-4; monotonicity asserted in the tests)
  -- comparisons across sites must fix $K$. VAC001 (GPF ConstDet
  2023-07-06, $w'T_s'$): the median fraction is 0.64 / 0.31 / 0.11 /
  0.01 / 0.00 at $K$ = 1 / 1.5 / 2 / 3 / 4 -- at $K = 2$ it matches
  what a correlated Gaussian pair gives (~0.09, tests), i.e. by the
  $K\,$-rms criterion this record set has almost no coefficients
  outside the Gaussian bulk, and at their $K = 4$ default the strong
  set is empty. Their near-surface temperature record (isolated sharp
  features, their fig. 2) is a different regime.
