# ec_coherent amplitude modulation

Step 5 of the gameplan, first half ("Amplitude modulation"): the degree to
which the large-scale motions modulate the amplitude of the small-scale
fluctuations, quantified by the Hilbert-transform decoupling procedure of
Mathis, Hutchins & Marusic (2009). Written 2026-08-23 from fresh PyMuPDF
extractions of Mathis2009, Talluru2014, Salesky2018, Salesky2020 (all four
rendered cleanly; no scanned pages). Code in
[ec_coherent/ampmod.py](../../ec_coherent/ampmod.py). The band-separation
sibling is [ec_scales.md](ec_scales.md); the missing-$z_i$ cutoff deviation
is registered here and shared by both.

## The phenomenon -- Mathis et al. (2009) §3.2 pp. 317-318 [@Mathis2009]

At the near-wall point, "when this negative large-scale excursion occurs,
the amplitude of the small-scale fluctuations $u_S^+$ is significantly
reduced ... for a positive large-scale excursion, the opposite scenario is
true ... the low-wavenumber motions associated with the footprints of
superstructure-type events influence the near-wall $u$ fluctuations in a
manner akin to a pure amplitude modulation" (pp. 317-318). Their scale
decomposition (Table 3, p. 317): large scales $\lambda_x/\delta > 1$,
small scales $\lambda_x/\delta < 1$, a cutoff "carefully selected from the
pre-multiplied energy-spectra map" between the inner and outer spectral
peaks (§3.2 p. 317; the outer peak at $z/\delta = 0.06$,
$\lambda_x = 6\delta$).

## The decoupling procedure -- Mathis et al. (2009) §4-5 pp. 318-321

Hilbert transform (eq. 4.1), analytic signal $Z(t) = x(t) + iX(t)$
(eq. 4.4), envelope

$$ A(t) = \sqrt{x^2(t) + X^2(t)} \quad \text{(eq. 4.5)} $$

"represents the envelope of the original real-valued signal" (p. 319). The
procedure (figure 4, p. 321): short-wavelength pass filter the raw signal
($\lambda_x/\delta < 1$) to get $u_S^+$; Hilbert transform for the envelope
$E(u_S^+)$; "we filter the envelope at the same cutoff as the large-scale
signal ($\lambda_x/\delta > 1$). Hence, a filtered envelope $E_L(u_S^+)$
describing the modulation of small-scale structures is obtained" (p. 320);
then

$$ R = \frac{\overline{u_L^+\, E_L(u_S^+)}}
       {\sqrt{\overline{u_L^{+2}}}\ \sqrt{\overline{E_L(u_S^+)^2}}}
   \quad \text{(eq. 5.1, p. 321)} $$

"where $\sqrt{\overline{u^2}}$ denotes the root mean square value" -- i.e.
the correlation coefficient of the large-scale signal with the filtered
envelope. Single-point form (§6.2 pp. 323-324): "the filtered envelope is
now correlated to the local large-scale component $u_{iL}^+$ ... the
single-point analysis seems to provide a reasonable estimate" (their
two-point $R = 0.33$ vs single-point $0.25$ at $z^+ = 15$). We have one
sonic per height, so the single-point form is what runs; the fluctuating
envelope (mean removed) enters the correlation, as in Salesky & Anderson's
primed notation below.

Robustness (§7.1, pp. 326-329): the estimate is "only weakly dependent on
the choice of cutoff wavelength" -- ten cutoffs $\lambda_x/\delta$ = 0.2-4
change the magnitude but not the form of $R(z^+)$ (fig. 11); local vs
global convection velocity changes $R$ by ~5 % (fig. 10). Validation
target (§7.1.1, pp. 326-327): a phase-scrambled synthetic signal with the
same spectrum "effectively exhibits a zero level" of $R$ -- amplitude
modulation lives in the phases. Both are tests in
`tests/test_ec_ampmod.py`.

## All velocity components -- Talluru et al. (2014) §3.2-3.3 [@Talluru2014]

Spectral cutoff filter at $\lambda_x = \delta$, "shown by Hutchins &
Marusic (2007) to be effective at separating the inner and outer peaks ...
Mathis et al. (2009) shows that the filter size has very little bearing on
the observed modulation" (p. R1-7). AM coefficients (eq. 3.4a-c, p. R1-9):
$R_u, R_v, R_w$ correlate the *streamwise* large-scale $u_L$ with the
filtered envelopes $E_L(u_s), E_L(v_s), E_L(w_s)$ -- one modulator, several
modulated signals. "The small-scale $u$, $v$, $w$ and unfiltered Reynolds
shear stress ($-uw$) components are modulated in a very similar fashion"
(p. R1-8); the three $R$ profiles share their zero crossing (fig. 4).

## The convective ABL -- Salesky & Anderson (2018) §1.2, §3.5 [@Salesky2018]

The generalisation we implement (their eq. 1.6, p. 142):

$$ R_{b_l,a_s} = \frac{\overline{b'_l\, E'_l(a_s)}}
   {\sqrt{\overline{b_l^{\prime 2}}}\ \sqrt{\overline{E_l^{\prime 2}(a_s)}}} $$

for any large-scale modulator $b_l$ and small-scale signal $a_s$
(single-point: $z_{ref} = z$, advective lag $\delta\tau = 0$). Filtering:
"We here employ a sharp spectral filter at cutoff scale $z_i$" (p. 141),
the small scale by subtraction, $a_s = a - a_l$ (p. 141). They compute
$R$ for $a_s \in \{u_s, w_s, \theta_s, (uw)_s, (w\theta)_s\}$ and
$b_l \in \{u_l, w_l\}$ (§1.3 p. 143) -- the instantaneous-flux series are
decomposed like any other signal. Findings that shape our defaults: as
$-z_i/L$ grows, "the degree of amplitude modulation due to $u_l$ ...
decreases until it is negligible. However, amplitude modulation due to
$w_l$ was found to be significant for all stabilities considered, as long
as there was a sufficient separation between the inner and outer peak"
(conclusions iii, p. 160); $w_l$ is "a 'buoyancy proof' modulator"
(p. 158). Sign convention: near the ground $R_{u_l,u_s} > 0$,
$R_{w_l,u_s} > 0$ in updrafts ("small-scale turbulence is excited in
updraft regions, and suppressed in downdraft regions", p. 157). Caveat
carried into the code: for their most convective case "there is no
separation between the outer and inner peak and the concept of amplitude
modulation is no longer meaningful" (p. 156) -- so the cutoff source is
stored per record and a missing gap is a flagged fallback, not an error.

Their convergence remark (§2.2 p. 145: ~120 large-eddy turnover times to
converge AM coefficients in the CBL; Hutchins et al. 2009 needed
$5\times10^3$-$10^4$ in neutral flows) means a single 30-min window gives
a noisy $R$; record-to-record scatter is expected and the validation
script reports medians.

## Tower practice for the cutoff -- Salesky & Anderson (2020) [@Salesky2020]

On the AHATS single-tower time series: "The time-series measurements of
streamwise velocity are low-pass filtered at timescale $\mathcal{T}$,
where $\mathcal{T} U(z)/\delta = 1$, which is equivalent (in frequency) to
demarcation at $\lambda_x/\delta = 1$ ... We assumed a boundary layer
depth of $\delta = 1000$ m; results were not significantly sensitive to
choice of $\delta$" (p. 124501-3). This is the precedent for our
`delta` cutoff mode (assumed outer scale, Taylor-converted per window).
Their modulation parameter $\alpha(x,t) = u'_l/u_\tau$ (eq. 2) and the
generalized gradient $\phi_{m,l} = \phi_m(\zeta)(1 + 0.10\,\alpha)$
(eqs. 5-6, AHATS fit $C_1 = 0.10$) are recorded here for context; not
implemented.

## What the code does

`ampmod.py`: `lowpass_sharp` (FFT coefficients at $f > f_c$ zeroed --
the sharp spectral filter; $x = x_l + x_s$ exactly), `envelope`
(eq. 4.5 via `scipy.signal.hilbert`), `am_coefficient` (eq. 5.1 /
eq. 1.6 with fluctuating envelope), `cutoff_frequency` (mode logic
below), `run` (the `/amplitude_mod` group: $R$ for every configured
modulator x signal pair, flux series $u'w'$ and $w'Ts'$ included,
plus `cutoff_lambda`, `cutoff_f`, `cutoff_source`, `zeta` = $z/L$ from
the EC ancillary, `U_mean`). All `[CITED]` except as registered below.

## Deviation register

- **Cutoff scale without $z_i$/$\delta$** (shared with
  [ec_scales.md](ec_scales.md); locked decision 3 of the gameplan). The
  sources set the large/small cutoff in outer units: $\lambda_x = \delta$
  (Mathis2009 Table 3; Talluru2014 p. R1-7), $\lambda_x = z_i$
  (Salesky2018 p. 141). No $z_i$ measurement exists at our sites. Modes
  implemented: `spectral_gap` (default) -- the interior minimum of the
  pre-multiplied $u$ spectrum between its outer and turbulent maxima,
  per window, which is where the sources say the cutoff must sit
  ("provided that the filter length corresponds with the spectral plateau
  separating the inner and outer peaks", Salesky2018 p. 141), falling
  back to the `delta` value when no interior minimum exists (source flag
  stored per record); `delta` -- $\lambda_c$ = `am_delta_m`, default
  1000 m `[ASSUMED]`, the Salesky2020 AHATS assumption quoted above;
  `scaled` -- $\lambda_c$ = `am_z_mult` $\times z$ `[ASSUMED]`.
  Validation: the Mathis §7.1.3 cutoff-sensitivity sweep is repeated on
  VAC001 by the validation script (R vs $\lambda_c$ over a decade).
  The gap detector itself is `[ASSUMED]` in its particulars (no source
  prescribes an algorithm): log-binned premultiplied spectrum at 10
  bins/decade, 3-bin smoothing, the two largest interior maxima at
  least half a decade apart, dip at least 10 % below the smaller
  maximum. Mathis §7.1.3's weak cutoff dependence is what makes these
  choices tolerable.
- **Single-point only.** The two-point form with the advective lag
  $\delta\tau$ (Salesky2018 eq. 1.7) needs a second measurement level;
  VAC001 has one sonic. Single-point is Mathis §6.2's validated
  approximation (0.25 vs 0.33 above).
- **Envelope-count caution, not implemented.** Schlatter & Örlü (2010b)
  critiqued the metric (envelope of a broadband carrier correlates with
  the signal itself); Talluru2014 (p. R1-2) note that "at sufficient Re
  all [alternative methods] produce results that are consistent with the
  method of Mathis et al. (2009)". We implement the Mathis form only; the
  phase-scrambled null test guards against the artefact their critique
  concerns. Baars et al. (2015) wavelet variant not on hand, not
  implemented.
- **30-min window ceiling.** The longest resolvable wavelength is
  $U T \approx$ 9-14 km at VAC001 winds; the large-scale signal below a
  1000-m cutoff holds only ~5-15 Fourier modes per window, and $R$ from
  one window is far from converged (Salesky2018 §2.2). Reported per
  record; interpret medians.
