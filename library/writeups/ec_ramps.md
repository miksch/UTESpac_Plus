# ec_coherent ramp detection -- wavelet path

Step 4 of the gameplan ("Ramp detection on T, u, and TKE"), wavelet
detector first because the coherent-flux module reuses its event set.
Written 2026-08-23 from fresh `pdftotext` extractions (Collineau & Brunet
Table I read off the rendered page). Code in
[ec_coherent/ramps.py](../../ec_coherent/ramps.py). The structure-function
(Van Atta) detector and the TKE extension (Mangan et al. 2022) are not
covered here: their sources have not been read yet, and no code for them
exists.

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
itself is not implemented (deviation register). Measured on VAC001 GPF
ConstDet 2023-07-06 (96 records, block detrend): without the `D_min_s` cut
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
$D$ = 12 s). Figure `testbed/scratch/ec_ramps_vac001.png`
(`testbed/scripts/ec_ramps_vac001.py`).

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
not as a re-timing step after a threshold-free detection, so the
combination is a deviation awaiting the user's ruling (task doc).

On real data the ideal-train lag does not survive (measured 2026-08-23 on
all events of VAC001 GPF ConstDet 2023-07-06,
`testbed/scripts/ec_ramps_issue_vac001.py`, figure
`testbed/scratch/ec_ramps_issues_vac001.png` panel c): over 6585 u and
1737 Ts events the median of (zero-crossing − RAMP-refined)/$a_0$ is 0.00,
the distribution spreads over the full $\pm a_0$ search window, and only
16 % of u events agree within $0.1\,a_0$ -- on real turbulence the RAMP
extremum within $\pm a_0$ is frequently a different nearby feature, so
re-timing adds scatter rather than removing a bias. Panel d (record 18,
09:30, where Ts detection works, $a_0$ = 4.2 s): the zero-crossings land
on the visible sharp drops, a median +0.4 s after the refined times. The
$+0.35\,a_0$ lag is a property of the ideal isolated-ramp geometry, not of
these data; it supports keeping `refine = "none"` as the default.

## Outputs (`/ramps` group)

Per record, height and signal (`Ts`, `u`; TKE waits on Mangan 2022): `a0_*`
(s), `D_*` (s, eq. 22), `n_events_*`, `mean_spacing_*` (s, mean interval
between consecutive detections), `event_time_*` (s from window start,
`(record, height, event)` padded with NaN), the wavelet variance `W_*` on
the `scale` axis, and attributes `wavelet = "mhat"`, `D_g`, `peak`,
`slope_*`, `edge_scales`.

## Deviation register

| Paper | Code | Why | Validation |
|---|---|---|---|
| Collineau & Brunet: scalogram peak $a_0$ (one peak for velocities, a secondary trend peak for $T$); Thomas & Foken: highest-frequency peak after a 6.2 s low-pass, Morlet variance | smallest-scale local maximum of the MHAT variance above `a_min_s` (default), global maximum optional; no Morlet | keeps one wavelet for variance and detection; the low-pass is replaced by `a_min_s` | synthetic ramp train recovers its period (`tests/test_ec_ramps.py`); VAC001 figure |
| continuous $b$ over the record | detections within $3a_0$ of the edges dropped | zero-padded convolution | `[ASSUMED]` |
| Thomas & Foken: low-pass (<6.2 s) before the variance | `D_min_s` bounds the peak search instead; scalogram itself unfiltered | one transform for variance and detection | VAC001 numbers above; `[CITED]` value, site-specific |
| MHAT zero-crossing time as the event time | optional `refine` to the RAMP/HAAR extremum, default off | 0.35 $a_0$ lag measured on ideal ramps | DECIDE slot, task doc |
| slope sign chosen by the analyst per signal | `slope="auto"` from the sign of $\overline{w'T'}$ | VAC001 runs through both stabilities unattended | `[DERIVED]`, recorded per record |
| -- | scale grid, `a_min_s`, `a_max_s` | -- | `[ASSUMED]`, config |
