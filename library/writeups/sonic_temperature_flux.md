# Sonic temperature, the temperature flux, and the WPL density terms

What the sonic anemometer measures as temperature, how the air-temperature
flux is recovered from it, and which flux the Webb–Pearman–Leuning density
corrections require. Written 2026-08-22 from fresh extractions of the four
PDFs (`pdftotext -layout`; Schotanus 1983 and Webb 1980 are OCR layers with
garbled equation typography, so equation forms below were read from the
surrounding prose, figure legends and the later papers that restate them —
Kaimal & Gaynor 1991 and Liu et al. 2001 are clean). Code links are by file
and symbol.

## Speed of sound and sonic temperature — Schotanus et al. (1983), eqs. 3, 5 [@Schotanus1983]

$$ c^2 = \gamma R\,T\,(1 + 0.51\,q), \qquad \bar T_s = \bar T\,(1 + 0.51\,\bar q) $$

(p. 82, γR = 403 m² s⁻² K⁻¹; q specific humidity). Kaimal & Gaynor (1991)
eqs. 1 and 3 (pp. 402–403) write the same as $T_s = T(1 + 0.32\,e/p)$; with
$q = 0.622\,e/p$ the coefficient is 0.51. Their footnote records why this
differs from the virtual temperature $T_v = T(1 + 0.61 q)$ (eq. 2b): 0.61
is $(1 - M_v/M_a)$, 0.51 is $(\gamma_v/\gamma_a - M_v/M_a)$.

Implemented by `SONIC_HUMIDITY_COEFF` and `air_temperature_from_sonic` in
[utespac/sonic_temperature.py](../../utespac/sonic_temperature.py); the
mean rescale `theta_son_air` in [utespac/flux/levels.py](../../utespac/flux/levels.py)
uses it. MATLAB `fluxes.m:575` carried 0.51 commented out and 0.61 active
("modified by Diane"); the 0.61 path went with MATLAB parity on 2026-08-22.

## Fluctuations and the temperature flux — Schotanus et al. (1983), eqs. 6, 8 [@Schotanus1983]

$$ T_s' = T' + 0.51\,\bar T\,q' + \frac{2\,\bar T\,\bar u\,u'}{c^2} $$

$$ \overline{w'T_s'} = \overline{w'T'} + 0.51\,\bar T\,\overline{w'q'} - \frac{2\,\bar T\,\bar u\,\overline{u'w'}}{c^2} $$

(pp. 83, 86–87; eq. 8 as restated in the Fig. 5 legend, p. 88). Measured
magnitudes at Cabauw, 3.5 m (Section 4, p. 86): the humidity term is ~10 %
of the flux in both unstable and near-neutral runs; the crosswind term is
small unstable and ~20 % near neutral. Kaimal & Gaynor (1991) eq. 7
(p. 404) give the humidity term as $0.1\,\overline{w'e'}$ (e in mb) and
eq. 4 / Appendix B show the ATI head removing the crosswind term in
real time; CSAT3 and IRGASON heads likewise output a crosswind-corrected
$T_s$, so only the humidity term remains. Liu et al. (2001) eq. 12 (p. 463)
generalises the crosswind term to three-path heads (factors A, B of their
Table I; CSAT3 A = 7/8, B = 7/8) and measures the humidity term at ~20 % of
$\overline{w'T'}$ unstable and ~5 % stable (Table III, p. 466).

Implemented by `air_temperature_perturbation` in
[utespac/sonic_temperature.py](../../utespac/sonic_temperature.py):
$T' = T_s' - 0.51\,\bar T\,q'$ sample by sample, with $q' = \rho_v'/\bar\rho$
from the high-frequency hygrometer (IRGASON/LI-7500/KH2O) at the level and
$\bar T$ the period-mean air temperature. `utespac/flux/engine.py` applies it inside the
H2O block so the `T_air'w'` / `T_air'wPF'` columns of `H` are
$\overline{w'T'}$; where no high-frequency humidity exists they fall back
to the mean-humidity rescale (`theta_son_air`), which corrects the mean but
not the covariance. Deviation: the crosswind term is not applied (heads
correct it internally); the rescale-only columns remain where no high-frequency humidity exists.

## WPL density corrections — Webb, Pearman & Leuning (1980), eqs. 14, 24, 25, 44 [@Webb1980]

With $\mu = m_a/m_v$ and $\sigma = \bar\rho_v/\bar\rho_a$ (p. 88, below
eq. 9a), the mean vertical velocity and the corrected fluxes of water
vapour (E) and a trace constituent (F) are

$$ \bar w = \mu\,\overline{w'\rho_v'}/\bar\rho_a + (1 + \mu\sigma)\,\overline{w'T'}/\bar T \quad (14) $$

$$ F = \overline{w'\rho_c'} + \mu\,\frac{\bar\rho_c}{\bar\rho_a}\,\overline{w'\rho_v'} + (1 + \mu\sigma)\,\frac{\bar\rho_c}{\bar T}\,\overline{w'T'} \quad (24) $$

$$ E = (1 + \mu\sigma)\left\{\overline{w'\rho_v'} + \frac{\bar\rho_v}{\bar T}\,\overline{w'T'}\right\} \quad (25) $$

$$ F = F_{raw} + \frac{\bar\rho_c}{\bar\rho_a}\,\frac{\mu}{1+\mu\sigma}\,E + \frac{\bar\rho_c}{\bar\rho_a}\,\frac{H}{c_p\,\bar\rho\,\bar T}\ \text{(flux form, eq. 44, p. 91)} $$

$T'$ here is the air temperature: the heat term is the thermal expansion of
dry air. Implemented in [utespac/flux/engine.py](../../utespac/flux/engine.py) by
`kin_sen_flux` (the `T_air'wPF'` column above) feeding the `LHflux` W/m²
columns, the KH2O O₂ correction and the `CO2flux` WPL column, and by
`rhov_ext` / `rhoc_ext` (sample-wise external fluctuations using `TairP`).
MATLAB `fluxes.m:1073` drove these with the buoyancy flux
$\overline{w'\theta_v'}$ (`Theta_v'wPF'`), which embeds the humidity term
of eq. 8 a second time; that path was retired with MATLAB parity on 2026-08-22.

## Field check

VAC001 2023 IOP vs EddyPro (which consumed the logger's humidity-corrected
`T_SONIC_corr`), GPF, block averaging, 712 periods: before this change the
UTESpac−EddyPro H difference regressed on LE with slope 0.060 (theory
$0.51\,\bar T c_p/L_v$ = 0.062) and the buoyancy-flux column had bias
+11.1 W/m², RMSE 15.7. With the correction in code the `T_air'wPF'` column
has bias −0.03 W/m², RMSE 0.75, slope 1.001; split by sign (the site is
advective, H < 0 in 63 % of periods): −0.16 / 0.52 for H < 0, +0.18 / 1.03
for H > 0. Driving the WPL terms with w′T′ moved the CO₂ flux slope from
0.940 to 0.975 and lowered LE by ~0.5 %. Record:
`tasks/archive/audit/2026-08-22_code-audit-and-python-gameplan.md`, "VAC001 test
dataset".
