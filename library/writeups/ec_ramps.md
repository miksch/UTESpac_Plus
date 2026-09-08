# ec_coherent ramp detection

Step 4 of the gameplan ("Ramp detection on T, u, and TKE"): the wavelet
detector (Collineau & Brunet zero-crossing method), the structure-function
detector (Van Atta / Spano / Paw U), and the TKE trigger (Mangan et al.
2022). Written 2026-08-23 from fresh `pdftotext` extractions; equations of
the scanned or garbled pages (Van Atta pp. 167-168, Spano pp. 260-261,
Paw U et al. 2005 pp. 460-461) read off the rendered pages. Code in
[ec_coherent/ramps.py](../../ec_coherent/ramps.py).

## What a ramp is -- Gao, Shaw & Paw U (1989) §3.1 p. 353, §3.2 pp. 355-356 [@Gao1989]

"The variation of temperature follows distinct ramp patterns which are
characterized by a gradual rise terminated by a sharp drop of about 1.5 °C
over 1 to 3 s" (unstable, $L$ = −138 m, p. 353); "Inverse temperature
ramps also occurred during stable conditions ... composed of a gradual
temperature decrease, followed by a sharp temperature rise, and are
essentially mirror images of the ramps described earlier" (p. 353). The
flow field: "The thermal field is composed of warm and cold regions
separated by a narrow microfront with a dramatic temperature decrease of
1-2 °C occurring over 2-4 s ... the vector flow field shows a weak upward
motion before the microfront arrives. Close to the frontal region, the wind
rapidly shifts to a strong downward motion. The dramatic sweep preceded by
the relatively gentle ejection" (pp. 355-356). So the microfront is the
sharp edge; its sign follows the sign of the mean scalar gradient.

## The transform -- Collineau & Brunet (1993) Part I [@Collineau1993]

Wavelet family (eq. 3, p. 359) and transform with $p = 1$ (eq. 4, p. 360;
§3.3 pp. 367-368 for the choice $p = 1$):

$$ T_1(a, b) = \frac{1}{a}\int_{-\infty}^{+\infty} h(t)\, g\!\left(\frac{t-b}{a}\right) dt, $$

"equal to the covariance between the wavelet and the input signal"
(p. 367; eq. 16 p. 368 with Gamage & Hagelberg 1991). Basic wavelets of
Table I (p. 359): HAAR ($1$ on $(-0.5, 0]$, $-1$ on $(0, 0.5]$), RAMP
($2x+1$ on $(-0.5,0]$, $2x-1$ on $(0,0.5]$), MHAT ($(1-x^2)e^{-x^2/2}$),
WAVE ($x e^{-x^2/2}$); "wavelet duration unit" $D_g$: HAAR ≈ 0.674,
RAMP 1/2, MHAT $\pi/\sqrt2$, WAVE $\pi$. MHAT is "second
derivative-like", the others first derivative-like; "Threshold in jump
detection": HAAR yes, RAMP yes, MHAT no, WAVE yes.

Implemented by `mhat` (the Table I definition, no normalisation) and
`cwt` (eq. 4 as a discrete convolution, $dt = 1/f_s$, $a$ in seconds)
`[CITED]`.

## Duration scale from the wavelet variance -- Part I §2.4, §3.4, §4.1

"We introduce a wavelet variance $W_p(a)$, obtained by integrating the
wavelet coefficients over the translation parameter $b$:
$W_p(a) = \int |T_p(a,b)|^2\, db$" (eq. 8, p. 364); "the wavelet scalogram
displays the distribution of energy along the scales $a$" (p. 364). "if the
wavelet scalogram of a given function $h$ has a peak at $a_0$, then a
characteristic time scale $D$ of the input signal can be defined such as
$D = \tfrac12\,(2\pi/\omega_g)\,a_0 = a_0 D_g$" (eq. 22, p. 369) -- "the
mean duration of the events contributing most to the signal energy"
(p. 369). Test on synthetic ramps/steps of length 100 (§4.1, Table II,
pp. 372-373): "the duration scale $D$ turns out to be very close to the
length of the elementary pattern ... within 7% for the ramp pattern (3-4%
with HAAR and RAMP)" (p. 373).

Implemented by `wavelet_variance` (eq. 8, discrete sum times $dt$) and
`duration_scale` ($D = a_0 D_g$, $D_g = \pi/\sqrt2$ for MHAT) `[CITED]`.
The scale grid (log-spaced, `a_min_s`–`a_max_s`, `n_scales_per_decade`) is
`[ASSUMED]`. Which peak: Part II §3.1 (p. 52) notes that for temperature "a
secondary peak is visible at large values of the dilation factor. This is
due to the existence of slow trends"; Thomas & Foken (2007) §3.1
(pp. 320-321) [@Thomas2007] take "the spectral peak with the highest
frequency in the wavelet variance spectrum ... after the time series had
been passed through a low-pass filter removing all fluctuations with an
event duration <6.2 s" (their variance uses the complex Morlet wavelet;
$D = \tfrac12 f^{-1}$). `duration_scale` therefore searches
$W_1(a)$ only over scales with $D = a D_g \ge$ `D_min_s` (default 6.2 s,
Thomas & Foken's cut, `[CITED]` for their spruce site and carried here as
the default) and returns the smallest-scale interior local maximum
(`peak = "smallest_scale"`, following Thomas & Foken), with `peak =
"global"` as the Collineau & Brunet reading; a maximum on the edge of the
searched grid is not a peak and gives NaN (no events). The Morlet variance
itself is not implemented (deviation register). Measured on the validation
site's GPF ConstDet reference day (96 records, block detrend): without the `D_min_s` cut
the smallest-scale maximum of the `Ts'` scalogram sits on the small-scale
turbulence shoulder at $a$ = 1.5-3 s (158 "events" per 30 min); with it,
79 records have an interior peak and the `Ts'` peak is most often the
large-scale hump at $a$ = 60-260 s ($D$ median 90 s, 7 events per 30 min)
because the scalogram has no ramp-scale maximum in between -- the
temperature signal at this near-neutral, small-|H| site does not carry the
dominant ramp scale Collineau & Brunet and Thomas & Foken saw above
forest. `u'` behaves as the papers describe, one broad peak: $a_0$ ≈ 5 s,
$D$ median 10.7 s, ≈ 80 events per 30 min, mean spacing 20 s ≈ 2 D
(Collineau & Brunet: $D$ 13-20 s for $u'$, spacing 29 s for $T'$ at
$D$ = 12 s). Figure and script: the ramps validation script under
`testbed/scripts/` (development tree).

## Detection -- the zero-crossing method, Part I §4.3 pp. 374-375; Part II §3.2.1 pp. 54-55

"The significance of the wavelet variance peaks as seen in 4.1 leads us to
choose the corresponding $a_0$-scale to process detection-aimed wavelets,
whatever the basic wavelet ... The detection function is thus defined as
the wavelet coefficients $T_1(a_0, b)$" (§4.3.1, p. 375). "we support the
use of the zero-crossing method, using the MHAT wavelet ... The jump
detection procedure then consists of detecting the zero-crossings in the
MHAT wavelet coefficients with a certain slope sign and with the value
$a_0$ given by the wavelet variance peak. On the one hand, when applied to
temperature data, a negative slope is characteristic of ramp descendance.
On the other hand, major increases in the streamwise windspeed component
correspond to a positive slope (Part II)" (§4.3.3, p. 375). First
derivative-like wavelets need "a threshold on the wavelet transform
peaks. Usually, this threshold is determined empirically ... a Gordian knot
in the detection process" (§4.3.2, p. 375).

Part II calibration (§4.1, pp. 55-56): against 60 visually identified
temperature ramps in one half hour, "with the scale $a_0$, corresponding to
the variance peak, the number of events detected is 62, very close to the
reference value of 60"; durations $D$ for temperature 12.2 s and 12.4 s at
the two heights (Table III, p. 54), $w'$ about half of $T'$ (p. 54); the
mean interval between detected events 29 s (Fig. 5, p. 60).

Implemented by `zero_crossings` and `detect` `[CITED]`: zero-crossings of
$T_1(a_0, b)$ with the configured slope sign (`slope`: `negative` for the
unstable temperature ramp, `positive` for $u$; `auto` picks the sign from
the window's $\overline{w'T'}$ -- positive flux → negative slope, negative
flux → positive slope, following Gao et al.'s mirror-image ramps
`[DERIVED]` from Gao 1989 p. 353 and Collineau & Brunet p. 375). Events
within $3a_0$ of either window edge are dropped (`edge_scales`,
`[ASSUMED]`: the convolution is zero-padded there). The detection time is
the linearly interpolated crossing.

## Time localization of the zero-crossing

Part II (p. 58-59, 64-65): "Errors in zero-crossings (MHAT) are due to too
good a localization in frequency of the Laplacian of a Gaussian: this
method misses very close events, while it tends to add irrelevant events
during quiescent periods"; and on conditional averages "because of the weak
localization in time of the associated wavelet, the sharpness of the peaks
obtained from the zero-crossing method is less marked ... WAG, HAAR and RAMP
are the most successful averaging techniques, given their good localization
in time. In what follows, all conditional averages will be performed by
the RAMP method." Measured here on an ideal train (ramp length $L$, flat
interval $L$, $L$ = 10 and 30 s, with and without 10 % noise,
`tests/test_ec_ramps.py`): the MHAT zero-crossing at $a_0$ lands
$0.35\,a_0$ after the microfront, every time; re-timing each event to the
extremum of $|T_1(a_0, b)|$ of the RAMP or HAAR wavelet within $\pm a_0$
puts it on the microfront to one sample. `refine_times` implements that
re-timing behind `refine = "ramp" | "haar"`, default `"none"`: the paper
uses the first derivative-like wavelets with a threshold for detection,
not as a re-timing step after a threshold-free detection. Ruled (user,
2026-08-23): `"none"` stays the default for cross-site repeatability;
the option remains for per-site use.

On real data the ideal-train lag does not survive (measured 2026-08-23 on
all events of the validation site's GPF ConstDet reference day; the
ramp-issue diagnostic script and its figure under `testbed/scripts/` and
`testbed/scratch/`, development tree, panel c): over 6585 u and
1737 Ts events the median of (zero-crossing − RAMP-refined)/$a_0$ is 0.00,
the distribution spreads over the full $\pm a_0$ search window, and only
16 % of u events agree within $0.1\,a_0$ -- on real turbulence the RAMP
extremum within $\pm a_0$ is frequently a different nearby feature, so
re-timing adds scatter rather than removing a bias. Panel d (record 18,
09:30, where Ts detection works, $a_0$ = 4.2 s): the zero-crossings land
on the visible sharp drops, a median +0.4 s after the refined times. The
$+0.35\,a_0$ lag is a property of the ideal isolated-ramp geometry, not of
these data; it supports keeping `refine = "none"` as the default.

## The structure-function detector -- Van Atta (1977) [@Atta1977]

Decomposition (eq. 2.5, p. 166): $\theta = \theta_T + \theta_R$, a random
turbulent part plus a coherent organized part, assumed statistically
independent; with local isotropy of the turbulent part
($\langle(\Delta\theta_T)^n\rangle = 0$ for odd $n$), eq. 2.9 (p. 166)
gives

$$ \langle(\Delta\theta)^2\rangle = \langle(\Delta\theta_T)^2\rangle + \langle(\Delta\theta_R)^2\rangle,\quad
   \langle(\Delta\theta)^3\rangle = \langle(\Delta\theta_R)^3\rangle,\quad
   \langle(\Delta\theta)^5\rangle = 10\langle(\Delta\theta_T)^2\rangle\langle(\Delta\theta_R)^3\rangle + \langle(\Delta\theta_R)^5\rangle. $$

Uniform ramp model (p. 167): ramps of amplitude $a$ and length $l$
separated by quiet periods $s$. For $r \ll l$ "the linear term in $r$
dominates the behavior of all the ramp structure functions, and as a good
approximation" (eq. 2.12, p. 167, rendered)
$\langle(\Delta\theta_R)^n\rangle = (-1)^n a^n r/(l+s)$. Combining with
eq. 2.9 yields the cubic for the ramp amplitude (eq. 2.13, p. 167):

$$ a^3 + \left(10\langle(\Delta\theta)^2\rangle - \frac{\langle(\Delta\theta)^5\rangle}{\langle(\Delta\theta)^3\rangle}\right) a + 10\langle(\Delta\theta)^3\rangle = 0, $$

"Each calculation yielded only one positive real root for $a$, so that no
ambiguity was encountered" (p. 168); and the ramp period (eq. 2.15,
p. 168, rendered): $(l+s) = -a^3 r / \langle(\Delta\theta)^3\rangle$.
Taylor's hypothesis is used "in the form $r = -U\tau$" (§3.1, p. 168).

Extraction finding: Van Atta's printed full-polynomial moments (eq. 2.11,
p. 167) disagree with a direct integration of his own Fig. 1 model beyond
the leading term -- the derivation gives, with $x = r/l$,
$n{=}3$: $a^3r[-1 + \tfrac32 x - \tfrac12 x^3]/(l+s)$ (VA prints
$-1 + \tfrac52 x - 2x^2 + \tfrac12 x^3$), $n{=}4$: last coefficient
$\tfrac35$ (VA $\tfrac45$), $n{=}5$: last coefficient $\tfrac23$ (VA
$\tfrac56$). Paw U et al. (2005) eqs. [2a-c] match the direct derivation,
which was repeated independently here (`[DERIVED]`, and asserted in
`tests/test_ec_ramps.py` against brute-force moments of a synthetic ramp
train). The leading linear terms -- all the linearized method uses --
agree in every source.

## Practical form and flux -- Spano et al. (1997) [@Spano1997]

Structure function (eq. 3, p. 261, rendered):

$$ S^n(r) = \frac{1}{m-j}\sum_{i=1+j}^{m}(T_i - T_{i-j})^n, $$

$m$ samples per interval, $j$ the sample lag, time lag $r = j/f$. The
cubic (eqs. 4-6): $a^3 + pa + q = 0$ with
$p = 10S^2(r) - S^5(r)/S^3(r)$, $q = 10S^3(r)$, solved "for the real
roots"; the inverse ramp frequency (eq. 7): $l+s = -a^3 r/S^3(r)$. In the
time-lag form the signs are self-consistent: unstable ramps give
$S^3 < 0$, hence $a > 0$ and $l+s > 0$; stable (mirror-image) ramps give
$S^3 > 0$ and $a < 0$ -- the sign of $a$ carries the flux direction, no
prior stability information needed.

Flux (eq. 2, p. 260, rendered): $H = \rho c_p \frac{a}{l+s} z$ "when
measurements are taken well above the canopy top (i.e. no $\alpha$ factor
is needed)", $z$ the measurement height; at canopy height eq. 1 inserts a
weighting factor $\alpha$ ($\alpha = 0.5$ for tall canopies, from the
assumed linear heating profile, p. 260). Constraints and choices: "the
time lag ($r$) must be much less than $l+s$"; records were dropped "when
the length of $l+s$ was less than 10 times $r$" (p. 261); lags 0.25,
0.50, 0.75, 1.00 s were used at 8 Hz, with $r = 0.5$ s at intermediate
heights giving slopes closest to 1 (Tables 1-3); the fitted $\alpha$
increases with lag and decreases with measurement height (Figs. 4, 6, 8).
Implemented as `structure_function`, `vanatta`, `sr_flux` in kinematic
form $\alpha\, a\, z/(l+s)$ (K m s$^{-1}$; multiply by $\rho c_p$ for
W m$^{-2}$), $\alpha = 1$ by default for this tower `[CITED]` Spano
eq. 2.

## Two-lag solution for d and s -- Paw U et al. (2005) [@PawU2005]

The chapter's full moment equations (eqs. [2a-c], p. 460, rendered; $v =
r/d$, $d$ the ramp duration, $s$ the quiet gap):

$$ S^2 = \frac{a^2 r}{d+s}P_2,\quad S^3 = -\frac{a^3 r}{d+s}P_3,\quad S^5 = -\frac{a^5 r}{d+s}P_5, $$
$$ P_2 = 1 - \tfrac13 v^2,\quad P_3 = 1 - \tfrac32 v + \tfrac12 v^3,\quad
   P_5 = 1 - \tfrac52 v + \tfrac{10}{3}v^2 - \tfrac52 v^3 + \tfrac23 v^5. $$

"One can determine the ramp dimensions $d$ and $s$ without the linear
assumptions of Van Atta (1977) by using more than one lag" (p. 460): with
$R \equiv S^3(br)/S^3(r)$ and $b = 2$ (eq. [3a], p. 460, verified here by
substituting eq. [2b]):

$$ v^3 - \frac{12-3R}{16-R}\,v + \frac{4-2R}{16-R} = 0, $$

whose root in $(0, 1)$ gives $d = r/v$; the corrected amplitude cubic
(eq. [4]) is the Van Atta cubic with $p \to p\,P_3/P_5$ and $q \to
q\,P_2/P_5$; and $s = -\frac{a^3 r}{S^3(r)}P_3 - d$ (eq. [5], p. 461).
All five equations re-derived here from the ramp geometry and confirmed
(`[DERIVED]` + `[CITED]`). Flux: eq. [15] (p. 465) is Spano's form with
$F(z)$ the height-dependent calibration factor. Lag guidance against the
microfront time: "when $r > 2t_f$ for tall and $r > 4t_f$ for short
canopies, the complete Van Atta (1977) formulations are adequate"
(p. 462). Implemented as `two_lag` (lag pair $r$, $2r$).

## Finite microfront time -- Chen et al. (1997) [@Chen1997], read, not adopted

Chen et al. replace the instantaneous drop with a finite microfront time
$t_f$ (their Fig. 4, $s = 0$) and fit their eqs. (14)-(15) for $n = 3$ to
$\overline{(\Delta T)^3}/\Delta t$ over all lags (Marquardt-Levenberg,
$r^2$ 0.72-0.99). Findings carried here: (1) measured
$\overline{(\Delta T)^3}/\Delta t$ versus $\Delta t$ "generally reaches a
well-defined maximum at some $\Delta t = t_m$" and the decline below
$t_m$ "is a signature of the finite microfront time" (pp. 104, 118) --
so lags well below $t_m$ underestimate $|S^3|$; (2) against the
finite-microfront fit, the VA linearized method "overestimated $M$ by
10-30%" and $\tau$ "by a factor of 2-4" (p. 111, Table I: $M$ ratios
1.08-1.43, $\tau$ ratios 2.27-3.68), so the SR flux factor $M/\tau$ comes
out 0.39-0.48 of the finite-microfront value -- a bias the empirical
$\alpha$ calibration absorbs; (3) their microfront times: $t_f \approx$
0.02-0.04 s (bare soil), 0.06-0.11 s (mulch), 0.18-0.3 s (forest)
(p. 462 of the Paw U 2005 restatement). The nonlinear fit is not
implemented (ruled not adopted, user 2026-08-23); the code reports
$S^3(r)/r$ over the configured lags so $t_m$ is visible per record.

## SR under regional advection -- Castellví & Snyder (2009) [@Castellvi2009]

Full-season test over two rice fields (Sacramento Valley; "no rainfall,
light winds, high temperatures, clear skies, and regional advection of
sensible heat flux are the typical weather conditions", p. 549; stable
daytime cases outnumber unstable ones, "a typical climate feature in
Sacramento Valley due to regional advection", p. 551). The flux is
Spano's form, $H = \rho c_p (\alpha z) A_T/\tau$ (eq. 2, p. 547), but
$\alpha$ is not a fitted constant: from the Castellví (2004)
SR-similarity combination (eq. 3, p. 547, rendered), for measurements in
the inertial sub-layer ($z > z^*$)

$$ \alpha = \left[\frac{k}{\pi}\,\frac{z-d}{z^2}\,\tau\,u_*\,\phi_h^{-1}(\zeta)\right]^{1/2}, $$

with the $z \le z^*$ branch replacing $z-d$ by $z^*-d$; $\zeta =
(z-d)/L$, and $\phi_h$ from Högström (1988)/Foken (2006) (eq. 5, p. 547):
$\phi_h = 0.95 + 7.8\zeta$ for $0 \le \zeta \le 1$,
$\phi_h = 0.95(1-11.6\zeta)^{-1/2}$ for $-2 \le \zeta \le 0$. Misprint
recorded: eq. 5 prints "116$\zeta$"; the Högström coefficient is 11.6,
confirmed against Foken (2008) Table (on hand) [@Foken2008] -- 11.6 is
implemented. With this $\alpha$, "regardless of the stability conditions
and measurement height above the canopy, sensible heat flux estimates
using SR analysis gave results that were similar to those measured with
the eddy covariance method" (abstract): slopes 0.89-1.10, $R^2 \ge$
0.86, Rmse $\le$ 13 W m$^{-2}$ (Table 1, p. 551), without calibration
against a sonic.

Lag selection (p. 550): the shortest usable lag "is one that produces
the first global maximum of $S^3(r)/r$" ($r_{1G}$, the Chen $t_m$;
$S^3$ from the step-drop model "is unrealistic for $r < r_{1G}$"), and
they linearize $S^3(r) = -A^3 r/\tau$ over lags from
$r_{ini} = r_{1G} + 1/f$ up to $\approx 0.01 L_r$. Ramps not well formed
(H near 0) and amplitude-sign/flux-sign disagreements were the main
error contributors but "fall within the measurement error" (p. 551).

Implemented as `phi_h` and `alpha_castellvi` (inertial-sublayer branch;
the $z \le z^*$ branch and the roughness-sublayer-depth machinery of
their eqs. 6-14 are not implemented -- our sonic sits at 10.85 m, far
above any local canopy). Validation-site sanity check (advective site, same
regime): see the validation script and task doc -- the similarity
$\alpha$ computed from the measured $u_*$, $\zeta$ and the SR $\tau$ is
compared against the empirical $\alpha \approx 0.45$ that the fixed
$\alpha = 1$ run implied.

## SR under strong local advection -- French et al. (2012) [@French2012]

BEAREX08 (Texas High Plains, irrigated cotton amid dryland; "strongly
advective events were those when mid-day H fluxes become dominantly
negative, leading to LE fluxes exceeding net radiation", p. 92 -- the
validation site's regime). Their SR is exactly the configuration landed here: the
sample-lag Van Atta cubic (their eqs. 1-5) with "$\alpha$ was set to 1.0
and $z$ to measurement height" (Snyder's interpretation, p. 94), 20 Hz
thermocouples at 2.25 m, tested against 9 EC stations. Findings carried
as checks:

- Lag selection dominates: "selection of lags that were too short or too
  long significantly affected estimation accuracy" -- RMSE minimum near
  the 1.0 s lag, ~20 W m$^{-2}$ better than shorter/longer lags; "the
  best linear agreement at 1 s also closely corresponded to the ideal
  slope of 1, indicating that calibration of SR fluxes based on lag is a
  more important consideration than the $\alpha$ height-dependent term"
  (pp. 95-96). Too-short lags distort the ramp *duration* under strong
  advection (0.5 s gave >15 s vs 10 s at 1 s on their strongly advective
  day, Fig. 5); amplitude is stable for lags > 0.25 s.
- Sign fidelity is SR's advantage under advection: "ramp amplitude
  changes sign in agreement with the sign of H" (p. 94; their Fig. 9
  uses the sign of $S^3$ alone to flag $H < 0$); "the SR approach was
  likely to correctly identify the direction of H flux" even when the
  magnitude is off (p. 97).
- The cubic is $q$-dominated: "the p coefficient generally was close to
  zero and thus usually unimportant ... 2nd and 5th order structure
  functions played a minor role in SR analysis, while the 3rd order
  function was crucial"; "computation using only the third order
  structure function would be sufficient" (pp. 98, 103). Three-real-root
  ambiguity "did not arise"; amplitudes are "highly uncertain" near
  dawn/dusk when $q \to 0$ (p. 99).
- Performance: weakly advective mid-day H to $\sim$35 W m$^{-2}$;
  strongly advective $\sim$60 W m$^{-2}$, mid-day R$^2$ collapsing at
  the wettest site (their Table 5) while transition/night times favour
  SR over flux variance.

No new code from this source; it validates the landed formulation and
supplies the diagnostics run in the SR validation script under
`testbed/scripts/` (development tree; per-lag ratio, sign-agreement
fraction, $q$-only amplitude check).

## TKE detection and IQA -- Mangan et al. (2022) [@Mangan2022]

Motivation: "large-eddy-simulation studies suggest that the cross-stream
velocity component is important for maintaining a microfront ...
therefore all three velocity components are crucial to coherent
structures" (§4, p. 52); "The temperature signal cannot indicate the
presence of a coherent structure when the temperature vertical gradient
is weak whereas the $u_{TKE}$ method is not limited by the temperature
gradient" (p. 66) -- directly the validation site's situation (visible-but-
undetectable Ts ramps, task doc).

Detection signal: $u_{TKE} = \sqrt{u'^2 + v'^2 + w'^2}$ ("the square
root of the sum of the high-frequency velocity variances, which is also
twice the square root of turbulence kinetic energy", §4 p. 52 --
i.e. $\sqrt{2e}$). Low-pass: "integrating the signal over a moving time
window" of $\Delta t = 10$ s, "analogous to a moving average multiplied
by the window time interval"; "a 10-s integration time appears
appropriate" for both their sites (eq. 5, p. 52). Printed inconsistency
recorded: eq. 5's integrand is $u'^2+v'^2+w'^2$ (unrooted) while the text
and Fig. 3 filter $u_{TKE}$; implemented as the centred moving mean of
$u_{TKE}$ -- the constant factor $\Delta t$ and the choice do not affect
the threshold test, which is relative.

Trigger: continuous MHAT (Torrence & Compo 1998 [@Torrence1998]) applied
to $u_{TKE,LP}$ at one fixed scale -- "the selected scale of the wavelet
was 10. This corresponds to a wave with a period of approximately 40 s"
(p. 52; 30 s for their grass site). Wave amplitudes "from minimum to
maximum" are averaged per 30-min period to $\bar A_i$, corrected for
height and mean $u_{TKE}$ by dataset-specific regressions (eqs. 6-11,
Table 1); "the amplitude of the wavelet must surpass $1.25\bar A_i$"
(p. 54), and the event starts "at the minimum of the wavelet
coefficient's wave" (p. 55), i.e. in the weak ejection phase before the
microfront. Caveat: "this method may not work well under low $u_{TKE}$
periods" ($< 0.3$ m s$^{-1}$, p. 54). One sonic here, so the multi-height
regression collapses: $\bar A$ is the per-record mean wave amplitude
(deviation register).

IQA (eqs. 2-4, p. 49): cumulative $X_i = X_{i-1} + u_i'\Delta t$ (same
for $Y, Z$), integration constant reset to zero at each trigger; "bulk
sweeps are defined as periods when $Z_i < 0$, and bulk ejections ... $Z_i
> 0$" (p. 50); an event runs trigger to trigger. Implemented as `utke`,
`utke_lp`, `detect_tke`, `iqa`; the `/ramps` group stores the TKE event
set and per-event bulk-sweep time fraction, the trajectories themselves
are recomputable from the events.

The surface-renewal concept itself is Paw U et al. (1995)
[@KyawThaPawU1995]: $H = \rho c_p \frac{dT}{dt}\frac{V}{A}$ (their eq. 1,
p. 121) with the parcel volume-to-area ratio the sensing height; that
paper estimates $dT/dt$ from the band-pass-filtered trace with a
regression factor $\alpha$ (their eq. 3) and points to structure
functions as future work (p. 135); the structure-function realisation is
Spano's, above.

## Outputs (`/ramps` group)

Wavelet detector, per record, height and signal (`ts`, `u`): `a0_*` (s),
`D_*` (s, eq. 22), `n_events_*`, `mean_spacing_*` (s, mean interval
between consecutive detections), `event_time_*` (s from window start,
`(record, height, event)` padded with NaN), the wavelet variance `W_*` on
the `scale` axis, and attributes `wavelet = "mhat"`, `D_g`, `peak`,
`slope_*`, `edge_scales`.

Structure-function detector, per record, height, signal and `sr_lag` (s):
`sr_a_*` and `sr_period_*` (linearized Van Atta $a$ and $l+s$),
`sr_d_*`, `sr_s_*`, `sr_a2_*` (two-lag $d$, $s$ and P-corrected $a$,
lags $r$ and $2r$), `sr_S3_rate_*` ($S^3(r)/r$, the Chen $t_m$
diagnostic), and for `ts` the kinematic SR flux `sr_flux_ts`
$= \alpha\,a\,z/(l+s)$ (K m s$^{-1}$, linearized $a$, $l+s$) with the
applied factor stored as `sr_alpha_ts`. `sr_alpha_mode` (ruled, user
2026-08-23) selects $\alpha$: `fixed` (the `sr_alpha` value, default 1,
`[CITED]` Spano eq. 2), `castellvi` (per record from the ancillary
$u_*$ and $L$ via `alpha_castellvi`, `[CITED]` Castellví & Snyder 2009
eq. 3, with `sr_d` the displacement, `[ASSUMED]` 0), or `fit`
(`[SITE-TUNED]`: one least-squares factor per lag and file against the
measured $\overline{w'T_s'}$). Lags with $l+s < 10r$ are NaN (Spano's
constraint), as is $l+s$ longer than the window itself (`[ASSUMED]` cap;
near-neutral records with $S^3 \approx 0$ otherwise return periods of
10^4-10^5 s).

TKE trigger, per record and height: `n_events_e`, `mean_spacing_e`,
`event_time_e`, `sweep_frac_e` (per-event bulk-sweep time fraction from
IQA, on the `event` axis), `A_mean_e` (mean MHAT wave amplitude of
$u_{TKE,LP}$), and attributes `tke_lp_s`, `tke_a_s`, `tke_thresh`.

## Deviation register

| Paper | Code | Why | Validation |
|---|---|---|---|
| Collineau & Brunet: scalogram peak $a_0$ (one peak for velocities, a secondary trend peak for $T$); Thomas & Foken: highest-frequency peak after a 6.2 s low-pass, Morlet variance | smallest-scale local maximum of the MHAT variance above `a_min_s` (default), global maximum optional; no Morlet | keeps one wavelet for variance and detection; the low-pass is replaced by `a_min_s` | synthetic ramp train recovers its period (`tests/test_ec_ramps.py`); validation-site figure |
| continuous $b$ over the record | detections within $3a_0$ of the edges dropped | zero-padded convolution | `[ASSUMED]` |
| Thomas & Foken: low-pass (<6.2 s) before the variance | `D_min_s` bounds the peak search instead; scalogram itself unfiltered | one transform for variance and detection | validation-site numbers above; `[CITED]` value, site-specific |
| MHAT zero-crossing time as the event time | optional `refine` to the RAMP/HAAR extremum, default off | 0.35 $a_0$ lag measured on ideal ramps | ruled: `none` default for repeatability, option per site (user 2026-08-23) |
| slope sign chosen by the analyst per signal | `slope="auto"` from the sign of $\overline{w'T'}$ | validation-site runs pass through both stabilities unattended | `[DERIVED]`, recorded per record |
| -- | scale grid, `a_min_s`, `a_max_s` | -- | `[ASSUMED]`, config |
| Spano: 8 Hz thermocouples over crops, lags 0.25-1.0 s | same lag set at 20 Hz sonic Ts at 10.85 m | lag sensitivity is stored per record on the `sr_lag` axis | site validation script; `[ASSUMED]` transferability |
| Spano/Paw U: $\alpha$ fit against eddy covariance per site; Castellví: similarity $\alpha$ with $z^*$ machinery | `sr_alpha_mode`: `fixed` (default 1), `castellvi` (inertial branch only, `sr_d` displacement), `fit` (per lag and file) | ruled (user 2026-08-23): fixed default, other modes selectable; EC flux is the flux of record here | validation site: fixed gives ratio 2.23, castellvi 1.39 at r 0.99 (task doc) |
| Chen: nonlinear fit of the finite-microfront model | linearized + two-lag only; $S^3(r)/r$ reported so $t_m$ is visible | fit not adopted (DECIDE, task doc); their Table I quantifies the bias | -- |
| Mangan: $\bar A_i$ from multi-height + $u_{TKE}$ regressions (eqs. 6-11) | $\bar A$ = per-record mean wave amplitude, threshold $1.25\bar A$ | one sonic, no reference height; the threshold factor is theirs | `[SITE-TUNED]` at their sites; validation-site event counts vs u-wavelet detector |
| Mangan: eq. 5 integrand printed as $u'^2{+}v'^2{+}w'^2$ | centred moving mean of $u_{TKE} = \sqrt{u'^2{+}v'^2{+}w'^2}$ | text and Fig. 3 filter $u_{TKE}$; constant factors cancel in the relative threshold | note, TKE section |
