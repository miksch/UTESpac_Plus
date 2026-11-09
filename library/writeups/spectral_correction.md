# Massman (2000) spectral correction

The analytical spectral correction of Massman (2000) [@Massman2000],
"frequency response corrections" in its title (p. 185), is written
from the extractions
`library/extracted/Massman2000.txt` and
`library/extracted/Massman2000_p4.txt` (Table 1), read on 2026-11-05.
The functions are in
[utespac/flux/spectral_correction.py](../../utespac/flux/spectral_correction.py);
the engine wiring is `_massman_factors` in
[utespac/flux/engine.py](../../utespac/flux/engine.py); the keys are
read and checked in [utespac/run_config.py](../../utespac/run_config.py)
`FluxConfig.from_config` and documented in
[utespac/config/flux.toml](../../utespac/config/flux.toml). The tests
are in
[tests/test_flux_spectral_correction.py](../../tests/test_flux_spectral_correction.py),
[tests/test_flux_engine.py](../../tests/test_flux_engine.py) and
[tests/test_config.py](../../tests/test_config.py). Each section names
the symbol that implements it.

Quotes are written in ASCII as in
[instrument_specs.md](instrument_specs.md): ligatures spelled out,
en-dash and minus sign as "-", "<=" and ">=" for the inequality signs,
curly apostrophes as straight ones; Greek letters as the extraction
gives them; a hyphen that only breaks a word across a line is dropped
`[HOUSE]`. The extraction page n is printed
page 184 + n `[DERIVED]` from the running heads ("===== PAGE 4 ====="
carries the printed "188"); loci give the printed page.

The flow the source was written for:

- **Massman (2000) [@Massman2000].** No field campaign. The method is
  tested against numerical integration of its own Eq. (3), "in terms of
  the flat terrain cospectra of Kaimal et al. (1972)" (p. 186), and
  "the Kaimal et al. (1972) flat terrain cospectra are used and no
  approximations are made to any transfer function." (p. 191)
  `[DERIVED]` from Sects. 2-3.
  - Scenarios: "Five different eddy covariance scenarios were tested
    for this study: momentum flux, virtual temperature" "flux (sonic
    thermometry), water vapor flux with an open path Krypton
    hygrometer, and both open- and closed-path CO2 systems." (p. 191).
  - Sonic and sampling: "the sonic anemometer has collocated vertical
    (lw) and horizontal (lu) paths of length 0.15 m; the block
    averaging period, Tb is 30 min"; the recursive high-pass constant
    "is 450 s (=Tb/4, after Rannik and Vesala, 1999)"; "the sampling
    frequency, fs, is 10 Hz"; "the measurement height above the zero
    plane displacement (z or z-d) is 5 m; and no anti-noise filter is
    applied" (p. 191).
  - Open-path scalar: "for the open path system the lateral separation
    is 0.30 m; the CO2 sensor path length is 0.20 m with an aspect
    ratio for volume averaging over a right circular cylinder
    (=[diameter]/[path length]) of 0.2 (Andreas, 1981); and no
    longitudinal separation is assumed." (p. 191). "The open-path
    instrument is taken to be the instrument developed at NOAA/ATDD
    (Auble and Meyers, 1991)." (p. 192). The closed-path CO2 time
    constant "is taken to be 0.1 s" (p. 191).
  - Regime: the approximations hold "for 0<=z/L<=2 and 0.1
    ms-1<=u<=20 ms-1" (p. 195), under the conditions of Horst (1997),
    "including the need for (i) a horizontally-homogeneous upwind
    fetch, (ii) the validity of cospectral similarity, (iii)
    sufficiently long averaging periods, and, preferably, (iv)
    relatively small corrections." (p. 195).

The instrument at hand is a CSAT3 sonic with a separate LI-7500DS, or
an IRGASON; neither is a source instrument. Their path lengths and
separations are in [instrument_specs.md](instrument_specs.md) rows C1
(CSAT3 paths), L1 (LI-7500DS optical path), L2-L4 (LI-7500DS
separations), I1, I2 (IRGASON optical and sonic paths) and I4
(IRGASON separation); they are not restated here. Table 1 states that
"The sonic anemometer is assumed to have orthogonal collocated axes."
(p. 188); the CSAT3 and IRGASON paths are inclined (C2, I3), so the
sonic line-averaging constant is carried over from an orthogonal sonic
`[DERIVED]` from Table 1 note and C2, I3. The LI-7500DS separations
are entered north, east and vertical (L2-L4); the code takes lateral
and longitudinal ones, so a geometry given in the LI-COR frame needs
the wind direction to be converted (register row 5).

## Equivalent time constants -- Table 1 [@Massman2000]

Each attenuation process is replaced by a first-order filter. The
constants used (Table 1, p. 188; "The horizontal wind speed is denoted
by u."):

| Process | Table 1 entry | Symbol | Label |
|---|---|---|---|
| Sonic line averaging, scalar flux | "Sonic anemometer line averaging (scalar flux)", "(lw/8.4u)" | `tau_sonic_line_scalar` | `[CITED]` |
| Scalar sensor line averaging | "Line averaging scalar sensor", "lscalar/4.0u" | `tau_scalar_line` | `[CITED]` |
| Lateral separation | "(llat/1.1u)" | `tau_lateral_separation` | `[CITED]` |
| Longitudinal separation | "Longitudinal separation without first order instrument, Qa=0", "(llon/1.05u)" | `tau_longitudinal_separation` with `tau_sensor=0` | `[CITED]` |
| Block averaging | "Tb/2.8" | `tau_block_average` | `[CITED]` |
| Linear detrending | "Tb/5.3" | `tau_linear_detrend` | `[CITED]` |

The Table 1 note adds: "The tube lag time, Lt/Ut, is usually removed
from consideration by digital time shifting. It is included here as
part of the longitudinal separation effects for completeness."
(p. 188). For closed-path systems the study states "Because most
closed path systems implement the digital time shift to remove the
tube lag time effects, this study will assume that Lt/Ut=0 as far as
the phase shift is concerned." (p. 189).

## Combined filter and attenuation -- Eqs. (9)-(11), Table 2 [@Massman2000]

The low-pass constants combine as the root sum of squares, Eq. (9)
(p. 190), "where M is the total number of instrument filters and the
τ is are their associated equivalent first-order time constants
(Table 1)" (p. 190), implemented by `combined_time_constant`
`[CITED]`. The peak frequency is f_x = n_x u/z, from "the
non-dimensional frequency (nx=fxz/u) corresponding to fx" (p. 189);
below Eq. (7), "for z/L<=0, a=7/8 and nx=0.085 and for z/L>0, α=1 and
nx=2.0-1.915/(1+0.5z/L)" (p. 189), the extraction rendering the first
α as a. Only n_x is taken from this passage, implemented by
`peak_frequency_nx` `[CITED]`; "Here z denotes the eddy covariance
measurement height, L is the Monin-Obukhov length" (p. 189).

The attenuation is Eq. (11) (p. 190) in the rows of Table 2,
"Recommended analytical formulae for spectral corrections to measured
momentum and scalar fluxes" (p. 191). For the row "Unstable
atmospheric conditions (z/L<=0)" (p. 191) the text states "for
unstable atmospheric conditions each of the factors a, b, and p can be
raised to the power of 0.925." (p. 190), and the Table 2 note gives
"the flat terrain neutral/unstable flat terrain cospectrum of Kaimal
et al. (1972), α=0.925." (p. 191); this branch is
`attenuation_unstable` `[CITED]`. The row "Stable atmospheric
conditions (0<z/L<=2)" with "Fast-response open path systems"
(p. 191) carries no exponent; this branch is `attenuation_stable`
`[CITED]`. The extrapolations beyond the tested range are register
rows 10-12 `[ASSUMED]`.

## Correction factor -- p. 192 [@Massman2000]

"(Note that the correction factor is the inverse of Eq. (3)" (p. 192).
`massman2000_factor` returns the inverse of the Table 2 ratio, true
over measured flux `[CITED]`. Only the scalar-flux branch exists; no
factor is formed for momentum.

## Engine inputs and config keys

With `spectralCorrection = "massman2000"`, `_massman_factors` calls
`massman2000_factor` once per period for H, LE and Fc. The inputs:

| Input | Source in the run | Label |
|---|---|---|
| `U` | the period mean of the streamwise wind `u_pf`, taken after the planar fit and the per-period yaw rotation, a vector mean (engine.py:622) | `[ASSUMED]`, rows 11 and 14 |
| `z_eff` | sonic height minus `displacementHeight` | `[CITED]` p. 191 |
| `zeta` | (z - d)/L of the period | `[DERIVED]`, rows 1 and 10 |
| `fs` | `scan_freq`; used only with the anti-noise filter, which is off | `[CITED]` p. 189 |
| `avg_period_s` | the averaging period, as T_b: "Tb denotes the block averaging period for calculating the fluxes" (p. 188) | `[CITED]` p. 188 |
| `detrend` | `detrendingFormat`: `"linear"` takes the Table 1 row "High pass filtering linear detrending", "Tb/5.3" (p. 188); `"constant"` removes the block mean only, the a -> inf limit of Eq. (11), with no trend term | `"linear"`: `[CITED]` Table 1, p. 188; `"constant"`: `[DERIVED]` (a -> inf) |
| `tau_sensor` | 0 | `[ASSUMED]`, row 3 |
| `path_len_sonic` | `sonicPathLength` (sonic vertical path) | `[CITED]` Table 1 |
| `path_len_scalar` | `irgaPathLength` for LE and Fc, in the Table 1 row "Line averaging scalar sensor", "lscalar/4.0u" (p. 188), not the volume-averaging row; `sonicPathLength` for H | `[ASSUMED]`, rows 2 (H) and 9 (LE, Fc) |
| `sep_lateral`, `sep_longitudinal` | `irgaSeparationLateral`, `irgaSeparationLongitudinal` for LE and Fc; 0 for H | `[ASSUMED]`, rows 5-7 and 12 |

The four geometry keys in `flux.toml` have no defaults. Under
`massman2000`, `FluxConfig.from_config` refuses a missing key, and a
negative value is refused in either mode (run_config.py:198-208);
`FluxOptions` repeats both checks (engine.py:75-82). `stages.flux`
passes the keys to `FluxOptions` (stages.py:379-383).

## Outputs

| Variable | Group | Unit | Meaning |
|---|---|---|---|
| `scf_H` | `flux_qc` | 1 | spectral correction factor of w'Ts', the leading covariance of `H_pf`, true over measured; not applied to the stored flux (row 13) |
| `scf_LE` | `flux_qc` | 1 | spectral correction factor of w'h2o', the leading covariance of `LE_wpl_pf`, true over measured, where the level has H2O; not applied (row 13) |
| `scf_Fc` | `flux_qc` | 1 | spectral correction factor of w'co2', the leading covariance of `Fc_wpl_pf`, true over measured, where the level has CO2; not applied (row 13) |

The factor includes the high-pass terms a and b of Eq. (11), trend
removal and block averaging, as well as the low-pass terms, so the
variables are named for the spectral correction, the term of the
Table 2 caption, rather than for the high-frequency end only.

A factor is NaN in a period where the stored flux it is named for
(`H_pf`, `LE_wpl_pf` or `Fc_wpl_pf`) is NaN, the engine testing the
stored value rather than the flags (engine.py:575-581); it is also NaN
where z/L is undefined, which covers every rotation- or Ts-flagged
period, and where U is not positive. A factor column that is NaN in
every period of a run is dropped from the product by
`FluxTable.trimmed`, which `FluxTables.store` applies to every table.

## Deviation register

| Paper | Code | Consequence | Label |
|---|---|---|---|
| 1. One height for f_x and z/L: "Here z denotes the eddy covariance measurement height, L is the Monin-Obukhov length" (p. 189); the scenarios set "the measurement height above the zero plane displacement (z or z-d)" (p. 191). | The run's zeta is z/L; the engine passes zeta * (z - d)/z, so both the stability and f_x use z - d. | None against the paper; the factor differs from one computed with z/L where d > 0. | `[DERIVED]` |
| 2. Table 1 gives sonic line averaging for scalar flux "(lw/8.4u)" and a separate "Line averaging scalar sensor", "lscalar/4.0u" (p. 188); no passage assigns the scalar-sensor row to sonic thermometry. | For H, `path_len_scalar = sonicPathLength`, so Ts is line-averaged a second time on the sonic path. | tau_e of H is sqrt(1/8.4^2 + 1/4^2) l/u = 0.277 l/u against 0.119 l/u with the sonic term alone, a factor 2.33 `[DERIVED]`; the H factor is larger than the paper's scenario would give. | `[ASSUMED]` |
| 3. Scalar sensor time constant: the closed-path CO2 value "is taken to be 0.1 s" (p. 191); Table 2 has the "Fast-response open path systems" row (p. 191). | `tau_sensor = 0` for every flux. | The open-path analyzer's own response is not represented; no slow-sensor term. | `[ASSUMED]` |
| 4. "For the present purposes anti-noise filters are not assumed to be part of the data acquisition system." (p. 189). | `anti_noise_filter` is off. | Agrees with the paper; any filter in the logger or analyzer is not represented. | `[CITED]` |
| 5. "the longitudinal separation distance llon is a function of wind direction." and "the present study assumes that the wind direction is basically parallel to the axis (mouth) of the sonic" (p. 194); "Consequently, a coordinate rotation may be required to estimate llon and llat for any given block averaging period." (p. 195). | `irgaSeparationLateral` and `irgaSeparationLongitudinal` are fixed numbers, the same for every period and wind direction. | The separation terms are wrong for winds off the configured axis. | `[ASSUMED]` |
| 6. The source has one sonic and one scalar sensor per scenario. | One geometry serves every sonic level. | A level whose instruments differ from the configured geometry gets a factor for the wrong geometry. | `[ASSUMED]` |
| 7. The Table 1 note: "The tube lag time, Lt/Ut, is usually removed from consideration by digital time shifting." (p. 188); "this study will assume that Lt/Ut=0 as far as the phase shift is concerned." (p. 189); "For any given estimate of the flux the maximum correlation method can be used to determine the total lag time" (p. 194). | The longitudinal term is kept whether or not `lagSearch` shifted the gas series. | With the lag search on, the separation may be corrected twice: once by the shift and once by the factor. | `[ASSUMED]` |
| 8. The factor multiplies the measured flux; the open-path correction "slightly less than 1.5, which is probably the maximum value most researchers would be comfortable in using" (p. 193). | Factors are stored in `flux_qc` and never multiplied into the stored fluxes. | The stored fluxes carry no spectral correction; applying it is the user's step. | `[HOUSE]` |
| 9. The open-path scenario: "the CO2 sensor path length is 0.20 m with an aspect ratio for volume averaging over a right circular cylinder (=[diameter]/[path length]) of 0.2 (Andreas, 1981)" (p. 191); Table 1 gives that process as "Volume averaging right circular cylinder 0.2<=d/lright<=2.0", "(0.2+0.4(d/lright))(lright/u)" (p. 188). | For LE and Fc the code uses the Table 1 row "Line averaging scalar sensor", "lscalar/4.0u" (p. 188), with `irgaPathLength` as the path; the configuration carries a path length and no sensor diameter. | The scalar-path constant is 0.25 l/u against 0.28 l/u at the paper's aspect ratio of 0.2 `[DERIVED]`; the LE and Fc factors are smaller than the volume-averaging row would give for the same path. | `[ASSUMED]` |
| 10. Table 2: "Stable atmospheric conditions (0<z/L<=2)" (p. 191); the method gave "reasonable estimates of the correction factors for 0<=z/L<=2" (p. 195). | `attenuation_stable` is used for any z/L > 0, with no upper bound. | For z/L > 2 the factor is evaluated outside the range the closed form was tested against; it is a diagnostic only. | `[ASSUMED]` |
| 11. The method was tested for "0.1 ms-1<=u<=20 ms-1" (p. 195). | Any positive U is accepted; the factor is NaN only where U is not positive. | For U below 0.1 m s-1 or above 20 m s-1 the factor is evaluated outside the range the closed form was tested against; it is a diagnostic only. | `[ASSUMED]` |
| 12. "for numerical purposes only, this study therefore assumes that the maximum horizontal separation distance is 0.30 m." (p. 189). | Any non-negative `irgaSeparationLateral` and `irgaSeparationLongitudinal` is accepted (run_config.py:198-208, engine.py:75-82). | For a separation above 0.30 m the factor is evaluated outside the range the closed form was tested against; it is a diagnostic only. | `[ASSUMED]` |
| 13. The factor is for one covariance, the true flux w'beta' of Eq. (1) (p. 186), with beta' a fluctuation of "either horizontal wind speed or scalar concentration" (p. 186). The paper does not treat the density (WPL) correction: a case-insensitive grep of both extractions for `webb`, `wpl` and `densit` finds no occurrence. | `scf_H` is the factor of w'Ts' with the sonic path (row 2), stored beside `H_pf`, which is rho c_p w'T' with w'T' = w'Ts' - 0.51 T w'q' and q' from the gas analyser where the level has H2O (engine.py:360-364, 551). `scf_LE` and `scf_Fc` are the factors of w'h2o' and w'co2' with the gas-analyser geometry, stored beside `LE_wpl_pf` and `Fc_wpl_pf`, the WPL composites of w'h2o' or w'co2' with w'T' and, for Fc, the WPL LE (engine.py:409-412, 533-535). | Each factor is that of the leading covariance of its flux only; the other covariances of the composite are not corrected with their own geometry, and multiplying the stored flux by the factor scales them with the leading covariance's factor. | `[ASSUMED]` |
| 14. "u the mean horizontal wind speed" (p. 189). | `U` is the period mean of `u_pf`, the streamwise component after the planar fit and the per-period yaw rotation (engine.py:622): a vector mean, at or below the scalar mean horizontal speed. | U cancels from the low-pass term p = 2 pi f_x tau_e, since f_x scales with u and every Table 1 low-pass constant used here with 1/u; it enters through a and b, and the measured-to-true ratio of Eq. (11) as implemented falls as a and b fall, so a smaller U gives a larger factor `[DERIVED]` from Table 1, n_x = f_x z/u and Eq. (11) with `tau_sensor = 0`. Where the wind direction varies within the period, the factor is larger than one computed with the scalar mean speed. | `[ASSUMED]` |
