# ec_coherent multiresolution decomposition (MRD)

Step 3 of the gameplan, first half. Written 2026-08-23 from fresh PyMuPDF
extractions of Howell1997 and Vickers2003 (both render cleanly; the
Howell1997 equations were read against the rendered pages where the text
layer scrambles sub/superscripts). Code in
[ec_coherent/mrd.py](../../ec_coherent/mrd.py).

## The decomposition -- Howell & Mahrt (1997) [@Howell1997]

Time series $w_i$, $\phi_i$ of $2^M$ points. Block (nonoverlapping) means
over windows of $2^m$ points (their eq. 1):
$\bar w_n(2^m) = 2^{-m}\sum_{i=(n-1)2^m}^{n2^m-1} w_i$,
$n = 1..2^{M-m}$. The flux at Reynolds averaging length $2^m$ points is
the record mean of the products of deviations from the local window means
(eq. 2); it is zero at $m = 0$ (one-point averages) and the full
covariance at $m = M$. The telescoping difference between consecutive
averaging lengths (eq. 5) gives the MR cospectrum for averaging length
$2^m$ (eq. 7):

$$\langle \tilde w(2^m)\tilde\phi(2^m)\rangle = \frac{1}{2^{M-m}}
\sum_{n=1}^{2^{M-m}} [\bar w_{2n}(2^{m-1}) - \bar w_n(2^m)]
[\bar\phi_{2n}(2^{m-1}) - \bar\phi_n(2^m)]$$

i.e. the covariance carried by half-window means deviating from their
parent-window means; the two halves give equal products, so one half per
parent suffices. Summing $m = 1..M$ recovers the total record covariance
exactly (eq. 6) -- "Reynolds averaging holds at every scale", the module's
free invariant. "The averaging length corresponding to the peak in
multiresolution cospectra depends mostly on the width of the dominant
(local) flux events whereas the wavelength of the peak in Fourier
cospectra depends on the principal periodicity" (p. 117); Fourier
cospectra "usually peak at a wavelength that is larger" -- the
Fourier-Haar transfer function for a single sinusoid peaks at
$\lambda \approx 3a/2$ (p. 123, citing Mahrt & Howell 1994).

**Fast Haar Transform** (appendix): in-place, $O(N)$; per pass,
$AVG = (w_i + w_{i+2^{m-1}})/2$ and $DEL = (w_i - w_{i+2^{m-1}})/2$
replace the pair; the cospectrum is the mean product of same-scale
transform values (eq. 15), the sum over $m = 1..M$ the total covariance.

**Sampling error** (eqs. 11-12): each cospectrum value is the mean of
$2^{M-m}$ per-window fluxes, so its sample variance $s_F^2$ is free, and
the relative standard error is
$\epsilon = (s_F/\langle F\rangle)/\sqrt{2^{M-m}}$ -- "conﬁdence intervals
naturally accompany multiresolution cospectra" (p. 133). Eq. 14 inverts
this into a required record length; their BOREAS heat-flux answer
(~55 km at 10 %) matches the Lenschow & Stankov formula.

**$2^M$ grid** (§4.1, eq. 8): a record of $R$ points is linearly
interpolated onto a *denser* grid of $2^M$ points with
$2^{M-1} < R \le 2^M$, spacing $\Delta t\,(R-1)/(2^M-1)$ -- "in order to
avoid discarding points".

A spatially nonorthogonal (maximum-overlap) variant (eqs. 9-10) reduces
scale aliasing from window placement; orthogonal and nonorthogonal
cospectra agreed at the dyadic lengths in their data (fig. 4). Not
implemented; the orthogonal form carries the closure identity.

## Worked example, gap scale -- Vickers & Mahrt (2003) [@Vickers2003]

Same decomposition presented as successive residual removal (record mean,
then half-record means, then quarters, ...; their eqs. 1-3, Fig. 1);
their $D(m+1)$ is the second moment of the width-$2^m$ segment means of
the residual series -- identical to Howell's eq. 7 labelled by the parent
window. Frequency association $f = (\Delta t\, 2^m)^{-1}$; normalized
form their eq. 7.

**Worked numeric example -- confirmed** (§2a, Table 1; the gameplan's
"confirm on extraction" recollection was right). Series
$\{1,3,2,5,1,2,1,3\}$, $M = 3$: record mean 2.25; half-record means
$\pm 0.5 \Rightarrow D_w(3) = 0.25$; then $D_w(2) = 0.3125$,
$D_w(1) = 1.125$ (indices: $D_w(1)$ is the shortest scale, 2 points).
Sum $= 1.6875 =$ the variance about the record mean. This is
`tests/test_ec_mrd.py`'s exact validation target.

**$2^M$ grid** (§2b, eq. 8): opposite direction from Howell1997 -- the
record is interpolated onto a *coarser* grid, $M$ the largest integer
with $2^M < N$, $\Delta t = (N-1)\Delta t_o/(2^M-1) > \Delta t_o$, "all
the data points in the original series are used. An alternative might be
to select some portion of the original series and discard the remaining
portion."

**Gap detection -- the fifth-order-polynomial recollection is refuted.**
The gameplan's "confirm on extraction" item that their gap-scale fit is a
fifth-order polynomial in $\log\tau$ found nothing: no polynomial appears
anywhere. Their algorithm (§4): smooth each 1-h cospectrum with a 1-2-1
filter; scan from the shortest averaging timescale; the first peak is "a
decrease in magnitude with increasing scale"; the gap is where the
cospectra "either increase or level off", a leveling-off meaning "the
accumulative flux changes by 1% or less with an increase in timescale".
Momentum is scanned on the magnitude of the stress vector; the heat-flux
cospectrum "often changed sign in the gap region". Records with no
significant turbulence peak are excluded rather than assigned a gap.
Because a single 1-h gap estimate carries "significant random sampling
error", they average gap scales over the whole experiment before use, and
model the average as $\tau = a_r (z/z_r)^{1/3} f(R_b)$ with
$a_r = 540$ s at $z_r = 10$ m, $f = (1-50R_b)^{1/4}$ ($R_b<0$) and
$(1+100R_b)^{-1/2}$ ($R_b>0$), clamped to 30-1200 s (eqs. 12-14).
Physics: gap timescale grows with height ($\lesssim$ 20 m) and
instability, shrinks sharply with stability; unstable records often show
no clear gap ("large convective eddies and mesoscale motions overlap").

## What the code does

`mrd.py`: `to_grid` (trim or Vickers-interpolate a window onto $2^M$
samples), `fht` (in-place forward FHT, Howell appendix), `mr_spectrum`
(MR spectra/cospectra $D(m)$ from same-scale FHT products, eq. 15, with
the per-scale sample std and relative error, eqs. 11-12, 16), `gap_scale`
(the Vickers §4 scan on one smoothed cospectrum), `run` (the `/mrd`
group: `D_<pair>` on a `tau_scale` axis, relative errors, per-record
`gap_wTs` from $|D_{wTs}|$ and `gap_uw` from the $(D_{uw}, D_{vw})$
stress magnitude, closure diagnostics). Closure $\sum_m D_{xy}(m) =$
window covariance is the invariant test.

## Deviation register

- **Grid mode.** Both sources map $R$ onto $2^M$ by linear interpolation
  (Howell up-grids, Vickers down-grids); the gameplan's plan of record
  trims to the leading $2^{15} = 32768$ of 36 000 samples (91 %,
  1638.4 s of 1800). Landed as `grid = "trim" | "interp"`, default
  `trim`, `interp` the Vickers2003 eq. 8 down-grid `[CITED]`; the mode
  and effective $\Delta t$ are stored. Ruled (user, 2026-08-23): trim
  stays the default with the VM2003 option kept, and windows up to
  ~2 h may be analyzed -- $M$ follows the window length automatically
  (a 2-h test locks this in).
- **Per-record gap scales.** Vickers2003 use experiment-averaged gaps
  because single-record estimates are noisy; we store the per-record
  detected gap (with their 1 % leveling rule) and leave any averaging to
  the analysis layer. The task-doc validation run reports the spread.
- **Gap on 30-min windows.** Their records are 1 h; ours are 30 min, so
  the largest resolved averaging length is $2^{15}$ samples = 1638 s
  (trim) and mesoscale scales beyond that are outside the accounting
  `[ASSUMED]` transferability.
- **No $R_b$.** Their gap-scale *model* (eqs. 12-14) needs a bulk
  Richardson number from surface radiative temperature, which the
  validation site does not measure; the model is recorded here but not
  implemented. The
  detected gap is the deliverable.
