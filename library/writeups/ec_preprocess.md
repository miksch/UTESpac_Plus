# ec_coherent preprocessing -- windows, perturbations, Taylor's hypothesis, upstream QC

Shared step of every `ec_coherent` module (gameplan "Shared preprocessing").
Written 2026-08-23 from fresh `pdftotext` extractions (detrending section
added the same day from Moncrieff et al. 2004 and Rebmann et al. 2012),
with Taylor's eq. 7 and Stull's §1.4 read off rendered pages because the text layers garble
the equations. Code linked by file and symbol: the window iterator is
`iter_windows` in [ec_coherent/io.py](../../ec_coherent/io.py), the rest
is in [ec_coherent/preprocess.py](../../ec_coherent/preprocess.py).

## Averaging window and perturbations -- Reynolds decomposition; window from the upstream `flux_averaging_s`

The window is the upstream flux-averaging period (30 min, read from the
`flux_averaging_s` attribute of the `utespac-hf-2` file, never hardcoded).
`record` marks the END of each period (`utespac.export_hf`), so the
samples of record $r$ are $(r - T, r]$; `iter_windows` slices exactly
$T f_s$ samples per record (36 000 at 20 Hz).

Perturbations are recomputed per window from the stored series; the
upstream `*Prime` arrays are not used (they are locked to the pipeline's
detrend setting, integration notes A.3).

## Detrending -- Moncrieff, Clement, Finnigan & Meyers (2004), Handbook ch. 2 §2.1-2.4 [@Moncrieff2004]; Rebmann et al. (2012) §3.2.3 [@Rebmann2012]

Moncrieff et al. name "the three main types of operation available to
us, time averaging, detrending, and filtering" (p. 11) and give each with
its transfer function (Table 2.2, p. 15). `detrend` offers all three plus
one unsourced variant:

- `block` -- time averaging with mean removal, eqs. 2.8-2.9 (p. 11):
  $\bar w = \frac1T\int_0^T w\,dt$, $w' = w - \bar w$; "time averaging
  with mean removal (MR) obeys the Reynolds averaging conditions"
  (p. 12) `[CITED]`. The default `[ASSUMED]` as a choice; Rebmann et al.
  (p. 71): "It has the advantage over the alternatives that it dampens
  low-frequency parts of the turbulence signal to the least degree.
  However, when there is a need to remove a trend ... block averaging is
  not sufficient".
- `linear` -- "we find the line of best fit over the period, i. e. the
  linear trend, and subtract that" (§2.2, p. 12; eq. 2.14, least squares
  eqs. 2.17-2.18 p. 14) `[CITED]`. "linear detrending does not obey
  Reynolds averaging rules" (p. 14) -- the Leonard terms of eq. 2.16 are
  "in most practical cases ... small" (p. 17); its ramp transform "decay[s]
  with frequency $\omega$ like $1/|\omega|^2$ and so [has] contributions at
  all frequencies" (p. 14).
- `recursive` -- "the recursive digital filter that is an exact analog of
  a simple, single pole RC filter (Moore 1986, McMillen 1988)", eq. 2.23
  (p. 16):

$$ \tilde c_k = e^{-\Delta t/\tau_f}\,\tilde c_{k-1} + \left(1 - e^{-\Delta t/\tau_f}\right) c_k, \qquad c' = c - \tilde c, $$

  "where $\Delta t$ is the interval between samples and $\tau_f$ is the RC
  filter time constant"; "the earlier values having an exponentially
  decreasing influence on the current value of the filtered signal"
  (p. 17). Rebmann et al. eq. 3.12-3.13 (p. 71) write the same filter as
  $s_{AF,j} = \varphi s_j + (1-\varphi) s_{AF,j-1}$ with
  $\varphi = 1 - e^{-2\pi f_c/f_s}$, "sometimes falsely called running
  mean". Implemented by `recursive_filter` `[CITED]`; transfer function
  $(2\pi f\tau_f)^2/[1+(2\pi f\tau_f)^2]$ (Table 2.2). Moncrieff compare
  the three with $\tau_f$ = 40 s against $T$ = 1800 s (p. 17) -- an
  illustration, not a recommendation; the default `filter_tau_s` = 300 s
  is `[ASSUMED]`. Deviation: the paper's filter runs continuously across
  records ("to obtain a filtered record in a period $T$ requires access to
  a time series longer than $T$", p. 17); the windows here are independent,
  so the recursion starts at the window mean `[ASSUMED]` and the first
  $\sim\tau_f$ of each window carries the start-up transient. A causal
  filter also lags a trend by $\tau_f$, so $\overline{c'} \ne 0$ on
  trended windows (the Leonard terms of eq. 2.21).
- `butterworth` -- a zero-phase second-order Butterworth low-pass at
  $f_c = 1/(2\pi\tau_f)$ (`scipy.signal.sosfiltfilt`), kept because it
  has no phase lag; `[ASSUMED]`, not in either source.

On the choice (§2.4, p. 17-18): "mean removal gives the sharpest cut-off
... Linear detrending removes more low frequencies from the signal as
expected but also shows the oscillations at higher frequencies ... Finally
the RC filter has the least sharp cut-off but has a much more predictable
shape at the high frequency end of the spectrum." Rebmann et al. (p. 72):
"any covariance, irrespective of the detrending method used, must be
corrected for high-pass filtering losses" (Rannik & Vesala 1999, Culf
2000 are the comparisons cited; neither on hand). The gameplan's
recollection that Vickers & Mahrt (2003) [@Vickers2003] discuss the
block-versus-linear trade-off was checked on 2026-08-23 and is **not
confirmed**: that paper contains no passage on detrending. NaN handling
follows `utespac.nandetrend` (NaN positions preserved, a window with more
than 90 % NaN returns all-NaN) so the two packages agree on the edge
cases.

## Taylor's hypothesis -- Taylor (1938), eq. 7 and eq. 11 [@Taylor1938]; Stull (1988) §1.4 [@Stull1988]

Taylor, p. 478: "If the velocity of the air stream which carries the
eddies is very much greater than the turbulent velocity, one may assume
that the sequence of changes in $u$ at the fixed point are simply due to
the passage of an unchanging pattern of turbulent motion over the point,
i.e. one may assume that

$$ u = \phi(t) = \phi\!\left(\frac{x}{U}\right), \qquad (7) $$

where $x$ is measured upstream at time $t = 0$ from the fixed point where
$u$ is measured." The spectrum-correlation pair (eqs. 9-11, p. 478)
carries the spatial frequency as $2\pi n x / U$, i.e. frequency $n$ maps
to wavenumber $2\pi n/U$ and wavelength $U/n$. Stull, eq. 1.4c p. 6,
states the same in angular form, $\kappa = f/M$ with $\kappa = 2\pi/\lambda$,
$f = 2\pi/\mathbb{P}$, and gives the Willis & Deardorff (1976) validity
criterion (eq. 1.4d, p. 6) $\sigma_M < 0.5\,M$, "Taylor's hypothesis
should be satisfactory when the turbulence intensity is small relative to
the mean wind speed" (p. 7).

Implemented by `taylor_wavelength` ($\lambda = \bar U/f$) and
`taylor_wavenumber` ($k = 2\pi f/\bar U$) `[CITED]`; `mean_wind` gives
$\bar U$ per window as $\sqrt{\bar u^2 + \bar v^2}$ of the rotated
components (after the upstream yaw rotation $\bar v \approx 0$, so this is
$\bar u$ to rounding); `taylor_check` reports $\sigma_M/\bar U$ against the
0.5 criterion `[CITED]` Stull eq. 1.4d. $\bar U$ is stored with every
scale-resolved output. Kaimal et al. (1972) p. 565 use the same
conversion ("the conversion from one to the other is made through the use
of Taylor's hypothesis", $\kappa_1 = 2\pi n/U$, their eq. 1) [@Kaimal1972].

## Rotation is upstream -- Wilczak, Oncley & Stage (2001), eq. 39 and pp. 142-143 [@Wilczak2001]

The HF file holds `u, v, w` after the planar fit plus per-period yaw
rotation applied by `utespac.rotation`; the planar-fit coefficients sit in
the file's `planar_fit` group. Wilczak's fit is eq. 39 (p. 140),
$\bar w_m = b_0 + b_1 \bar u_m + b_2 \bar v_m$. On what remains per
window after the fit (pp. 142-143): "although the vertical velocity
averaged over the entire data set is zero, the mean vertical velocities
may be non-zero for individual data runs, in large part due to mesoscale
motions or due to sampling limitations. This residual mean vertical
velocity is subtracted for each run so that it does not contribute to the
Reynolds stress."

So the preprocessing step is a verification, not a transform:
`rotation_check` reports $\bar w$, $\bar v$ and $\bar w/\sigma_w$ per
window as diagnostics, and the per-window removal of the residual
$\bar w$ happens through the detrend step above exactly as Wilczak
prescribes `[CITED]`. A double-rotation cross-check is not possible from
the HF file (unrotated components are not stored; integration notes A.3).

## Upstream despiking -- Vickers & Mahrt (1997), §3a pp. 518-519 [@Vickers1997]

Recorded here because it shapes what the ramp and quadrant modules see.
The HF series were despiked upstream by `utespac.condition_data` in the
Vickers & Mahrt manner: "Any point in the window that is more than 3.5
standard deviations from the window mean is considered a spike. The point
is replaced using linear interpolation between data points. When four or
more consecutive points are detected, they are not considered spikes and
are not replaced. The entire process is repeated until no more spikes are
detected. During the second pass ... the threshold for spike detection
increases to 3.6 standard deviations and a like amount for each
subsequent pass. The record is hard flagged when the total number of
spikes replaced exceeds 1% of the total number of data points." (p. 518-
519; moving window $L_1$ = 5 min for tower data, p. 518.) The thresholds
"are somewhat arbitrary" (p. 519). UTESpac's actual settings are the
`[spikeTest]` block of `utespac/config/qc.toml` (per-variable thresholds,
window fraction, consecutive-outlier limit) and are not re-derived here.

Consequence, carried into every output's provenance (`despiking`
attribute copied from the HF file): spikes were interpolated in place, so
extreme-value statistics (ramp amplitudes, quadrant tails) see slightly
smoothed extremes. The per-window `spike_flag` / `nan_flag` /
`Tau_ssitc` / `H_ssitc` / `LE_ssitc` / `Fc_ssitc` / `Tau_ss` / `H_ss` /
`LE_ss` / `Fc_ss` ancillaries are joined by `iter_windows`; windows
failing QC are processed and flagged, never dropped.

## Missing samples -- `[ASSUMED]` policy

Gaps left by the upstream conditioning are NaN. `fill_gaps` rejects a
window whose NaN fraction exceeds `nan_max_frac` (default 0.1,
`[ASSUMED]`) and otherwise fills the NaNs by linear interpolation before
the FFT/wavelet steps, recording the filled fraction on the window. No
source on hand prescribes the threshold; it is a config value.

## Deviation register

| Paper | Code | Why | Validation |
|---|---|---|---|
| Moncrieff 2004: three operations, no default named | `detrend(method="block")` default; `linear`, `recursive`, `butterworth` selectable | block obeys Reynolds averaging (eq. 2.11); Vickers & Mahrt 2003 recollection refuted | `tests/test_ec_preprocess.py` (eq. 2.23 step response, NaN semantics) |
| RC filter runs continuously across records (p. 17) | recursion restarts at each window mean | windows are independent in the HF file | start-up transient noted; `[ASSUMED]` |
| -- | `butterworth` zero-phase mode | no phase lag; convenience | `[ASSUMED]`, unsourced |
| Wilczak subtracts the residual $\bar w$ per run | same, through `detrend` | -- | `rotation_check` diagnostics stored per record |
| -- | `fill_gaps` 10 % NaN threshold | upstream gap-filling is not exhaustive | `[ASSUMED]`, config |
