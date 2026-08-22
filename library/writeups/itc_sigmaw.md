# Integral turbulence characteristic σw/u* — the ITC reference in the SSITC flags

The "developed turbulence" test compares the measured σw/u* of a period with
a similarity model and flags the period when the relative deviation exceeds
30 % / 100 % (Foken & Wichura 1996 eq. 16 and Section 5.3; Foken et al.
2012 eq. 4.41). This note records which model `utespac` uses on each side of
neutral and where each piece comes from. Written 2026-08-22 from fresh
`pdftotext` extractions. Code links by file and symbol.

## Unstable side — Foken (2008) Table 2.11 [@Foken2008], = Foken et al. (2004) Table 9.1 [@Foken2004], = Foken et al. (2012) Table 4.2 [@Foken2012]

$$ \frac{\sigma_w}{u_*} = c_1 \left(\frac{z}{L}\right)^{c_2}, \qquad
\begin{cases} c_1 = 1.3,\ c_2 = 0 & 0 > z/L > -0.032 \\ c_1 = 2.0,\ c_2 = 1/8 & z/L < -0.032 \end{cases} $$

(Foken 2008 p. 52; Handbook p. 192; Eddy Covariance p. 117; all three cite
Foken et al. 1991, 1997 and Thomas & Foken 2002.) The two branches meet at
|z/L| = 0.0319, which is the threshold the MATLAB port carried. Foken &
Wichura (1996) Table 4 (p. 95) is the older form of the same table: 1.41
near neutral, 2.00(−z/L)^{1/8} for −1 < z/L < −0.0625 and
2.00(−z/L)^{1/6} below −1; the 2004/2008/2012 tables supersede it and are
what `utespac` follows.

Implemented by `_above_canopy_sigmaw` in
[utespac/calc_ssitc_flags.py](../../utespac/calc_ssitc_flags.py) for
z/L < 0.

## Neutral-to-stable range — Thomas & Foken (2002) via Foken (2008) Table 2.12 [@Foken2008] (= Handbook Table 9.2, Eddy Covariance Table 4.3)

$$ \frac{\sigma_w}{u_*} = 0.21 \ln\!\left(\frac{z_+ f}{u_*}\right) + 3.1,
\qquad z_+ = 1\ \text{m},\quad -0.2 < z/L < 0.4 $$

with f = 2Ω sin φ the Coriolis parameter (the "external forcing assumed by
Johansson et al. 2001", Handbook p. 192). Foken et al. (2012) note that the
scalar characteristics take extremely high values near neutral and the test
fails there; the w-based test is the one `utespac` runs.

Implemented by `_above_canopy_sigmaw` (with `coriolis_parameter`) for
0 ≤ z/L ≤ 0.4 when `SiteInfo.latitude` is set (VAC001: 38.300056 N from
the EddyPro metadata). Deviation: the unstable part of the −0.2 < z/L < 0
overlap keeps Table 2.11 (1.3), as the same tables list for it.

## Stable side beyond z/L = 0.4, or without a latitude — Pahlow, Parlange & Porté-Agel (2001), eq. 14 [@Pahlow2001]

$$ \frac{\sigma_w}{u_*} = a + b\left(\frac{z}{L}\right)^{c}, \qquad a = 1.1,\ b = 0.9,\ c = 0.6 $$

(eq. 14 p. 232, coefficients p. 232–233, Figure 3 p. 235). Least-squares fit
on 446 runs from five surface-layer experiments (sonics at 1–4.3 m), from
near neutral to z/L = 32.9: σw/u* is constant at 1.1 for z/L ≲ 0.1, then
rises; the Dias et al. (1995) average of earlier constants is 1.32 (p. 233).
De Bruin et al. (1993) report a constant σw/u* ≈ 1.4 for stable La Crau
data (p. 239, eq. 17) [@DeBruin1993]; Nieuwstadt (1984) [@Nieuwstadt1984]
and Sorbjan (1986) [@Sorbjan1986] are the local-scaling antecedents Pahlow
cites (Nieuwstadt's PDF here is a scan without a text layer; Sorbjan's
constants were not extracted for this note).

Implemented by `_above_canopy_sigmaw` for z/L > 0.4, and for the whole
stable side when `SiteInfo.latitude` is unset. Deviation: at z/L = 0.4 the
Thomas & Foken form (~1.4 at typical f/u*) hands over to Pahlow (1.62), a
15 % step that follows from using Foken's own tables inside their stated
range and the published stable-ABL fit beyond it; recorded in
`tests/KNOWN_DIVERGENCES.md`. Nothing on the σw/u* reference is
`[ASSUMED]` any more.

## Flag thresholds — Foken et al. (2004) Table 9.4 [@Foken2004]

ITC deviation < 30 % → good, < 100 % → usable, else poor, combined with the
steady-state test; `_ssitc_flag` and `_ss_only_flag` in
[utespac/calc_ssitc_flags.py](../../utespac/calc_ssitc_flags.py). Inside
the canopy (`useCanopyITC`, z ≤ `canopyHeight`) the reference is the Rannik
et al. profile in `_canopy_sigmaw`, not covered by this note.
