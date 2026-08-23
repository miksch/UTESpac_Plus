# ec_coherent spectra, cospectra and ogives

First analysis module of `ec_coherent` (gameplan "Spectra, cospectra,
ogives"). Written 2026-08-23 from fresh `pdftotext` extractions; Kaimal's
eq. 21 and Stull's eqs. 8.6.x read off rendered pages. Code in
[ec_coherent/spectra.py](../../ec_coherent/spectra.py), symbols named
below. Preprocessing (window, detrend, Taylor) is in
[ec_preprocess.md](ec_preprocess.md).

## Normalization -- Kaimal et al. (1972) eq. 1 [@Kaimal1972]; Stull (1988) §8.6.1-8.6.2 [@Stull1988]

Kaimal p. 565: "Taking the $u$ spectrum as an example

$$ \int_0^\infty F_u(\kappa_1)\,d\kappa_1 = \overline{u^2} = \int_0^\infty S_u(n)\,dn. $$

If $\kappa_1 = 2\pi n/U$ we have ... $\kappa_1 F_u(\kappa_1) = n S_u(n)$
(1)." And: "The spectral forms involving the product with $n$ will be
referred to as 'logarithmic' spectra and cospectra", plotted against the
dimensionless frequency $f = nz/U$ (p. 564, list of symbols).

Stull §8.6.1 pp. 312-313: the discrete spectral energy is folded,
$E_A(n) = 2|F_A(n)|^2$ for $n = 1 \dots n_f$ (the Nyquist term not doubled
for even $N$), so that $\sum E_A = \sigma_A^2$ (eq. 8.6.1b); §8.6.2
pp. 313-314: the density is $S_A(n) = E_A(n)/\Delta n$ and "can be
integrated over $n$ to yield the total variance" (eq. 8.6.2a-b).

Implemented by `spectrum` `[CITED]`: one-sided densities from
`scipy.signal.periodogram` / `scipy.signal.csd` with
`scaling="density"`, so that

$$ \int_0^{f_N} S_{xx}\,df = \sigma_x^2, \qquad \int_0^{f_N} \mathrm{Co}_{xy}\,df = \overline{x'y'}, $$

the two closure identities that `tests/test_ec_spectra.py` asserts
(exactly for a boxcar taper on the full record; to the taper's
variance-correction tolerance with Hann). The cospectrum is the real part
and the quadrature spectrum the imaginary part of the one-sided
cross-spectral density (`Co_*`, `Qu_*`). The pre-multiplied form
$f\,S(f)$ is what the figure script plots and what the scale-separation
module will consume.

## Estimator -- full-record periodogram, log-binned `[ASSUMED]`

Kaimal et al. computed their spectra "by dividing each 1-hr record into 16
consecutive blocks of 4,096 data points and constructing a composite
spectrum by averaging the 16 separate spectra. The composite spectrum was
then smoothed by averaging spectral estimates over frequency bands"
(p. 564) -- i.e. Welch segment averaging plus band averaging; no taper is
mentioned. The landed default is the full-record periodogram (one
segment, boxcar) followed by log-spaced band averaging (`log_bin`,
`n_bins_per_decade` in config): it keeps the lowest resolvable frequency
$1/T$ = 1/1800 Hz, which the ogive needs, and closes Parseval exactly.
Welch is available through `nperseg`, a Hann taper through `taper`.
Measured 2026-08-23 on six VAC001 windows (GPF ConstDet 2023-07-06,
records 0-5, block detrend): boxcar closes to 1.000 for every series; a
single-segment Hann taper returns 0.74-1.03 of the variance for `Ts`
(0.96-1.09 for `u`, `w`) because its variance correction holds only in
expectation and these nighttime `Ts` windows carry a residual trend. With
linear detrend the Hann ratios tighten to 0.94-1.06. Neither the taper
nor the bin count is sourced; both are `[ASSUMED]` and stored in the
output attributes (`taper`, `n_bins_per_decade`, `nperseg`). Decision
slot in the task doc.

## Neutral reference curves -- Kaimal et al. (1972) eq. 21a-g, p. 579, Section 7 [@Kaimal1972]

"At $z/L = 0$, where $\phi_\epsilon = G = H = K = 1$ and $\phi_h = 0.74$
we have" (read off the rendered page; $f = nz/U$):

$$ nS_u(n)/u_*^2 = 105 f/(1 + 33 f)^{5/3} \qquad (21a) $$
$$ nS_v(n)/u_*^2 = 17 f/(1 + 9.5 f)^{5/3} \qquad (21b) $$
$$ nS_w(n)/u_*^2 = 2 f/[1 + 5.3 f^{5/3}] \qquad (21c) $$
$$ nS_\theta(n)/T_*^2 = \begin{cases} 53.4 f/(1 + 24 f)^{5/3}, & f \le 0.15 \\ 24.4 f/(1 + 12.5 f)^{5/3}, & f \ge 0.15 \end{cases} \qquad (21d) $$
$$ -nC_{uw}(n)/u_*^2 = 14 f/(1 + 9.6 f)^{2.4} \qquad (21e) $$
$$ -nC_{w\theta}(n)/u_* T_* = \begin{cases} 11 f/(1 + 13.3 f)^{1.75}, & f \le 1.0 \\ 4.4 f/(1 + 3.8 f)^{2.4}, & f \ge 1.0 \end{cases} \qquad (21f) $$
$$ nC_{u\theta}(n)/u_* T_* = 40 f/(1 + 14 f)^{2.6} \qquad (21g) $$

"The above formulae are good approximations of the observed curves (see
Fig. 17 for plots). The only departure exceeding ±10 per cent are at the
low-frequency ends of the $v$ spectrum and the $u\theta$ cospectrum. ...
the equation fit the data extremely well in the range $0.01 < f < 4.0$."
(p. 579). $T_* = -\overline{w\theta}/u_*$ (p. 564: "Scaling temperature
$T_*$ is defined as $-\overline{w\theta}/u_*$").

Implemented by `kaimal_neutral` `[CITED]`, used only as a QC overlay in
`testbed/scripts/ec_spectra_vac001.py`; never an input to any estimate.
The overlay is drawn for $0.01 < f < 4$ only. VAC001 is a flat site with
the sonic at 10.85 m, so $z$ is the sonic height; $u_*$ is the per-record
`ustar` ancillary, $T_*$ is formed from the window's own $\overline{w'T'}$.

## Ogive -- Foken & Wichura (1996) eq. 10, p. 89 [@Foken1996]; Desjardins et al. (1989) pp. 61-62 [@Desjardins1989]

Foken & Wichura: "The necessary averaging time can be determined by the
use of an 'ogive' function. This function is defined as the cumulative
integral of the cospectrum (e.g., of the momentum flux) beginning at the
highest frequencies

$$ Og_{w'x'}(f_0) = \int_\infty^{f_0} Co_{w'x'}(f)\,df \qquad (10) $$

Oncley et al. (1990) have shown that this ogive function converges to a
constant value at a frequency which could be converted to the averaging
time of the measurement. While normally averaging times of about 20-30 min
are used in the surface layer, the averaging time should be tested by the
ogive function for measurements during stable stratification conditions
and at higher levels above the surface." (The integral bounds are as
printed; the sense is from the Nyquist end down to $f_0$.)

Attribution check (gameplan "confirm on extraction"): the definition and
the averaging-time use are Foken & Wichura's (citing Oncley et al. 1990,
not on hand). Desjardins et al. (1989) plot "the cumulative contribution
of the cospectral estimates as a function of $K \times Z$" to assess "the
percentage loss in the cospectral estimates which can result by high- and
low-pass filtering" (p. 61, Fig. 5 p. 62) -- a filter-loss use, not an
averaging-time test. Cite Foken & Wichura for the test; Desjardins only
for the cumulative-cospectrum idea.

Implemented by `ogive` `[CITED]`: trapezoidal integration of the unbinned
cospectrum from $f_N$ down to each bin frequency, so $Og(f \to 0)$ equals
the covariance (closure) and the low-frequency plateau is the adequacy
check. Stored as `ogive_*` on the binned frequency axis, normalised
forms left to the figure.

## Outputs (`/spectra` group)

Dims `(record, height, frequency)`; `frequency` is the mean line
frequency of each log bin, `frequency_edges` the bin edges. The edges
are snapped to the half-line points $(k + \tfrac12)\,\Delta f$ of the
FFT grid (`log_bins`), so every bin holds at least one line, its width is
$n_b\,\Delta f$, and the band sum $\sum_b S_b\,\Delta f_b$ equals the
variance exactly -- the closure survives the binning (VAC001 check
2026-08-23: band-sum/variance 1.000 for u, w, Ts on all 96 records).
`S_u, S_v, S_w, S_Ts, S_rhov, S_rhoCO2`; `Co_uw, Co_wTs, Co_wrhov,
Co_wrhoCO2` and the matching `Qu_*`; `ogive_*` for the same pairs;
`U_mean`, `var_*`, `cov_*` (the closure targets); `n_valid`,
`nan_filled_frac`, `taylor_ratio`; attributes `taper`, `nperseg`,
`n_bins_per_decade`, `detrend_method`, `taylor_hypothesis` ("lambda =
U/f, k = 2 pi f/U"), plus the provenance globals copied from the HF file.

## Deviation register

| Paper | Code | Why | Validation |
|---|---|---|---|
| Kaimal 1972: 16 x 4096-point segment averaging + band averaging | full-record periodogram + log-band averaging (default); Welch via `nperseg` | keep $1/T$ for the ogive; exact closure | closure tests; figure overlay against eq. 21 |
| no taper stated | boxcar default, Hann optional `[ASSUMED]`; stored in attrs | boxcar closes Parseval exactly; Hann measured 0.74-1.09 on VAC001 windows (above) | closure tests; figure |
| Kaimal curves are for the Kansas surface layer at $z/L=0$ | overlay only for $0.01<f<4$, QC use | stated validity range p. 579 | figure |
