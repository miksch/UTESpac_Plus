# ec_coherent LSM/VLSM scale separation

Step 5 of the gameplan, second half ("LSM/VLSM separation"): per-band
variance and flux fractions of the perturbation series, bands defined by
streamwise wavelength. Written 2026-08-23 from fresh PyMuPDF extractions of
Kim1999, Balakumar2007, Wang2016b, Hutchins2012 (all rendered cleanly; the
Balakumar JSTOR scan has a noisy text layer -- the loci quoted below were
read against the rendered layout where garbled). Code in
[ec_coherent/scales.py](../../ec_coherent/scales.py). The amplitude-
modulation sibling is [ec_ampmod.md](ec_ampmod.md); the missing-$z_i$
cutoff deviation is registered there and shared by both modules.

## VLSMs exist and are read off the premultiplied spectrum -- Kim & Adrian (1999) [@Kim1999]

Pipe flow, hot film, Taylor's hypothesis with the local mean: the
premultiplied spectrum is bimodal, "a low wave number mode that scales
with outer variables and a high wave number mode that scales with inner
variables" (p. 419-420); "We use the location of the maximum,
$2\pi/\Lambda_{max}$, to indicate the scale of the very large-scale
motions" (p. 420). $\Lambda_{max}$ reaches 12-14$R$ in the log layer,
dropping to ~2$R$ at the centreline (fig. 5, p. 420). Wavelengths from
frequency spectra via Taylor are *underestimates* for these scales
(p. 419) -- carried as a caveat on our per-window Taylor conversion.
Conceptual model: VLSMs as streamwise alignment of hairpin-packet LSMs
(§IV, p. 421).

## The band boundaries -- Balakumar & Adrian (2007) [@Balakumar2007]

Channel and ZPG boundary layer, following Guala et al. (2006): motions
were "termed 'LSMs' if their lengths range between $0.1\pi R$ and
$\pi R$, and VLSMs if their lengths were greater than $\pi R$, nominally.
Motions shorter than $0.1\pi R$ were attributed to the range of active
turbulent motion" (p. 666). "Following GHA06, we shall take $k_x h = 2$ or
$k_x\delta = 2$ to be the nominal dividing line between LSMs and VLSMs.
This corresponds to wavelengths of $\pi h$ or $\pi\delta$" (p. 671).
"A substantial portion of the kinetic energy (40-65 %) and the Reynolds
shear stress (30-50 %) is carried by VLSMs in pipe, channel and ZPGBL
flows. Thus, the dividing boundaries between VLSMs and LSMs (taken to be
$\pi\delta_0$) and LSMs and the main turbulent motions (taken to be
$0.1\pi\delta_0$) are nominal, but happen to be useful rules of thumb"
(conclusions, p. 677-678). Premultiplied peaks in their data: bimodal at
$\lambda \approx 1\delta_0$ and $\approx 6$-$7.5\delta_0$ inside the log
layer (fig. 1 inset, p. 670; fig. 2, p. 673). So the three bands are one
decade apart by construction: small $< 0.1\pi\delta$, LSM
$0.1\pi\delta$-$\pi\delta$, VLSM $> \pi\delta$ `[CITED]`.

## The ASL field precedent -- Wang & Zheng (2016) [@Wang2016b]

QLOA, $Re_\tau \sim O(10^6)$, sonic towers, neutral criteria
$|z/L| < 0.06$, $|\alpha| < 22°$ (eq. 3.2, p. 472), synoptic de-trend by
low-pass at 20$\delta$ (p. 472-473; changing to 16$\delta$/24$\delta$
changes little, fig. 20). VLSM energy "plotted against height using
low-pass filtering with a cutoff length of 3$\delta$" with LSM band-pass
"0.3$\delta$-3$\delta$ (Guala et al. 2006) [or] $\delta$-3$\delta$ (Lee,
Ahn & Sung 2015)" (p. 479) -- numerically the Balakumar $0.1\pi$-$\pi$
decade. Findings: VLSMs confirmed at $Re_\tau$ up to $4\times10^6$; their
TKE fraction grows log-linearly with height, 20-30 % at $z/\delta < 0.01$
to ~60 % at $z = 0.2\delta$ (p. 481, noting the 50-Hz sampling makes 60 %
"may be an overestimate", p. 485); the LSM premultiplied peak "is
difficult to distinguish in the current ASL data" (p. 483-484) -- the
reason our band edges do not rest on detecting two separate peaks. Their
$\delta$ was not measured but fitted through the Marusic et al. (2013)
turbulence-intensity formulation (p. 473); single-height VAC001 cannot do
that, hence the assumed-$\delta$ mode below.

## The ASL behaves like the lab -- Hutchins et al. (2012) [@Hutchins2012]

SLTEST, neutral hour selected by near-zero heat flux, ±30° acceptance
cone, steady winds ("at least 30 min, which at 5 m s$^{-1}$ equates to an
advection length of O(100) boundary-layer thicknesses", p. 283);
de-trending by subtracting the array-average low-passed at 1 km / 180 s
(p. 285). $\delta = 60$ m estimated by consistency with laboratory
statistics (p. 286). Conclusions (pp. 302-303): "the ASL under
near-neutral conditions behaves precisely as a canonical flat plate
turbulent boundary layer in the logarithmic region"; two-point
correlation maps "virtually indistinguishable" from the lab;
superstructures "commonly exceed 10$\delta$ in length ... of the
kilometre scale, and can take several minutes to advect past a stationary
measurement array" -- the 30-min-window resolution caveat below. Ramp-like
shear zones and roll modes present in the ASL as in the lab.

## What the code does

`scales.py`: `band_edges` (the two cutoff wavelengths per window from the
configured mode), `band_fractions` (per-band sums of the full-record
boxcar periodogram/cross-periodogram over the three wavelength bands --
the same estimator whose Parseval closure is exact in
[ec_spectra.md](ec_spectra.md), so the band fractions of variance and
covariance sum to 1 exactly: the module's invariant test), `run` (the
`/scale_separation` group: `var_frac_<x>` for u, w, Ts and
`flux_frac_uw`, `flux_frac_wTs` on a `scale_band` axis of small | lsm |
vlsm, plus the band-edge wavelengths and frequencies, cutoff source,
`zeta`, `U_mean`). Frequency-to-wavelength per window by Taylor
($\lambda = U/f$), with Kim & Adrian's underestimate caveat.

## Deviation register

- **Band edges without $\delta$/$z_i$** -- the shared entry in
  [ec_ampmod.md](ec_ampmod.md). For this module: `delta` mode sets the
  VLSM cut at $\pi\delta$ and the small|LSM cut at $0.1\pi\delta$
  (`[CITED]` ratios, Balakumar2007 pp. 666, 671; `scales_delta_m` default
  1000 m `[ASSUMED]` as in ec_ampmod); `spectral_gap` mode (default, per
  locked decision 3) takes the VLSM cut from the same per-window
  premultiplied-spectrum gap as ec_ampmod and the small|LSM cut one
  decade below it (the decade is the `[CITED]` Balakumar band width);
  `scaled` mode uses `scales_z_mult_small`/`scales_z_mult_vlsm` $\times z$
  `[ASSUMED]`. Sources flagged per record.
- **Bands from a 30-min window.** The lowest resolved frequency is
  1/1800 Hz, i.e. $\lambda \le U T \approx$ 9-14 km at VAC001 winds; a
  VLSM band starting at ~3 km holds only the first few spectral lines, so
  its fraction is a coarse, high-variance estimate (Hutchins2012:
  several minutes per structure). Fractions are of the *within-window*
  variance after the upstream/window detrend -- energy at scales longer
  than the window is outside the accounting by construction.
- **No w/Ts outer-scale literature adopted.** The sources are $u$-centred;
  the same band edges are applied to w', Ts' and the flux cospectra
  without a source stating band bounds for them `[ASSUMED]` (the
  Salesky2018 spectrograms show the w outer peak at the same wavelength
  as u's, p. 151, which is the nearest support on hand).
