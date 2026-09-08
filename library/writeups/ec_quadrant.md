# ec_coherent quadrant and octant analysis

Step 3 of the gameplan, second half. Written 2026-08-23 from fresh
PyMuPDF extractions of Wallace2016, Lu1973 (OCR layer, equations partly
garbled -- definitions read from the surrounding prose), Raupach1981,
Li2011, Li2019. Code in
[ec_coherent/quadrant.py](../../ec_coherent/quadrant.py).

## History and attribution -- Wallace (2016) [@Wallace2016]

Quadrant analysis was "conceived of and carried out" by Wallace,
Eckelmann & Brodkey 1972 [@Wallace1972] (on hand), classifying $u'w'$
products by sign: Q1 $(+u,+v)$ outward interaction, Q2 $(-u,+v)$
ejection, Q3 $(-u,-v)$ inward interaction, Q4 $(+u,-v)$ sweep (§2.1),
quantifying the ejections/sweeps seen by Corino & Brodkey 1969.

**The hole is Willmarth & Lu (1972)** [@Willmarth1972] (not on hand; DOI
Crossref-verified 2026-08-23), *not* Lu & Willmarth 1973 as the gameplan
recalled: they filtered to products "outside of what they called the
hole, deﬁned by the set of hyperbolas in each quadrant,
$|uv| \ge H\,|\overline{uv}|$, where $\overline{uv}$ is the mean value of
this product" (§3.1, p. 138). Lu & Willmarth 1973 applied it across the
boundary layer -- with a different normalization (below). The
RMS-normalized scalar-flux hole $|v\theta| = H_\theta v'\theta'$ is Perry
& Hoffmann 1976 (fig. 5 caption, p. 140). Octant analysis "beginning with
the investigation of Suzuki et al. (1988)" (§3.7); Volino & Simon 1994
[@Volino1994] (on hand, unread) is an early application.

## The hole, RMS form -- Lu & Willmarth (1973) [@Lu1973]

§4.3, p. 494: "The size of the 'hole' is decided by the curves
$|uv| =$ constant. Introduce the parameter $H$ and let $|uv| = Hu'v'$,
where $u'$ and $v'$ are the local root-mean-square values ... The
parameter $H$ is called the hole size." So the gameplan's recalled
$|u'w'| \ge H\sigma_u\sigma_w$ hole *is* Lu & Willmarth 1973's own
definition -- attribution confirmed, with the origin one paper earlier in
the flux-normalized form. Five regions: four quadrants outside the hole
plus the hole; the five fractional contributions sum to 1 at every $H$
(their eq. after 10a). At $H = 0$ across most of their boundary layer:
$\overline{uv_2}/\overline{uv} \approx 0.77$ (ejections),
$\overline{uv_4}/\overline{uv} \approx 0.55$ (sweeps), interactions
negative; at $H = 4.5$ ejections still carry 15-30 % while the other
quadrants have "almost no contributions" (pp. 495-496).

## Formal statistics, $\Delta S$, third moments -- Raupach (1981) [@Raupach1981]

Definitions we implement (his eqs. 1-4): the conditional contribution
$\overline{(u'w')}_{i,H}$ with indicator $I_{i,H} = 1$ when $(u',w')$ is
in quadrant $i$ and $|u'w'| \ge H\,|\overline{u'w'}|$; stress fraction
$S_{i,H} = \overline{(u'w')}_{i,H}/\overline{u'w'}$; time fraction
$T_{i,H} = \overline{I_{i,H}}$; and $S_{1,0}+S_{2,0}+S_{3,0}+S_{4,0}=1$.
His labels match Wallace1972 (outward $i=1$; ejection $i=2$; inward
$i=3$; sweep $i=4$). **Normalization note** (p. 365): his $H$ multiplies
$|\overline{u'w'}|$, and "another definition of $H$, differing from this
one by a factor $\rho$ (the correlation coefficient) is also in common
use, e.g. by Lu & Willmarth (1973)" -- i.e.
$H_{flux} = |\rho_{uw}|\,H_{rms}$. Both landed as `hole_norm`.

$\Delta S_H = S_{4,H} - S_{2,H}$ (eq. 7) measures sweep-vs-ejection
dominance; via a third-order Gram-Charlier expansion (eqs. 8-10, after
Nakagawa & Nezu 1977) it is set by the third moments, empirically
$\Delta S_0 = 0.37 M_{30}$ over all his surfaces (eq. 12, with
$M_{30} = -2.02M_{21} = 1.97M_{12} = -1.70M_{03}$, eq. 11). Sweeps
dominate the roughness sublayer ($\Delta S_0$ up to $>0.5$; within the
canopy $S_{4,0} = 0.93$ and sweeps at $H \ge 10$ carry 40 % of the
stress in 2.5 % of the time, Table 2); ejections dominate the outer
layer (90 % vs 40 % at $\eta \approx 0.7$). His figures sweep
$H = 0..20$ -- the source for our default hole grid. Event counting for
timescales (§6): negative crossings of level $-H|\overline{u'w'}|$ by
$u'(t)w'(t)$ within a quadrant; $T u_*/\delta$ roughness-invariant.

## ASL quadrants, efficiencies -- Li & Bou-Zeid (2011) [@Li2011]

Quadrant definitions "follow Shaw et al. (1983) and Katul et al.
(1997a,b) for momentum and scalar ﬂuxes" (neither on hand; the operative
definitions are their own eqs. 4-6): flux fraction
$S(i) = \overline{w'c'}_i/\overline{w'c'}$ with
$\overline{w'c'}_i = t_p^{-1}\int w'c' I_i\,dt$, duration fraction
$D(i) = t_p^{-1}\int I_i\,dt$, $c = u, q, T$, no hole. Transport
efficiency (eqs. 8-9):
$\eta = \overline{w'c'}/(\overline{w'c'}_{ej} + \overline{w'c'}_{sw})$,
total over downgradient -- exactly UTESpac's `find_eta`. $\Delta S =
S_{ej} - S_{sw}$, $\Delta D = D_{ej} - D_{sw}$ (eqs. 16-17) -- UTESpac's
`delta_flux`/`delta_time`, the A.3 cross-check. Findings:
transport similarity holds near neutral and breaks with instability
(scalars gain, momentum loses efficiency, MOST fits eqs. 12-13 and
Table 2); ejections' flux share grows with $-z/L$ while their duration
share shrinks for scalars.

## ASL octants and reference values -- Li & Bo (2019) [@Li2019]

QLOA sand-surface ASL, $Re_\tau \sim O(10^6)$, 11 heights. Momentum
quadrants numbered "as follows (Shaw, 1985)" -- the Wallace1972 layout
(their eq. 6); scalar quadrants on $(w',\theta')$ with Q1 $(+w,+\theta)$
ejection, Q2 $(-w,+\theta)$, Q3 $(-w,-\theta)$ sweep, Q4 $(+w,-\theta)$
(eq. 10, labels for upward heat flux; their fig. 10 shows the dominant
pair flipping to Q2/Q4 when $z/L > 0$ -- the sign-aware labeling our
module applies). Hole: "the hyperbolic hole size (H) method of Lu and
Willmarth (1973)", indicator
$|u'w'|_{Qi} \ge H\,(\overline{u'^2})^{1/2}(\overline{w'^2})^{1/2}$
(eq. 9) -- RMS normalization, $H$ to 15+ in their figs. 6-7.

**Octants** (eq. 11), triplet $(u', w', \theta')$: O1 $(+,+,+)$ hot
outward interaction, O2 $(-,+,+)$ hot ejection, O3 $(-,+,-)$ cold
ejection, O4 $(+,+,-)$ cold outward interaction, O5 $(+,-,+)$ hot sweep,
O6 $(-,-,+)$ hot wallward interaction, O7 $(-,-,-)$ cold wallward
interaction, O8 $(+,-,-)$ cold sweep. Octant fractions
$O_i = \sum (u'w')_{O_i} / \sum u'w'$ computed for each of the three
2-D components $u'w'$, $u'\theta'$, $w'\theta'$ (fig. 11); "the
contribution of Octant 2 and Octant 8 is dominant to the Reynolds
stress", growing with $|z/L|$. Every octant source they build on (Suzuki
1988; Volino & Simon 1994; Vincont 2000; Li 2010; Park 2012) uses this
velocity-pair-plus-one-scalar triplet -- the basis for our default.

Near-neutral reference values (QC targets, not identities): duration
fractions $D_{Q1}\approx0.19$, $D_{Q2}\approx0.296$, $D_{Q3}\approx0.2$,
$D_{Q4}\approx0.314$, height- and $u_*$-invariant below 30 m (their
§4.1.1, matching Katul et al. 1997's $\approx 0.29$); Q2/Q4 carry ~70 %
of both streamwise and vertical TKE. Their "intensity ratio" (e.g.
$Q2 = 2.86 \pm 0.27$ near neutral) is *not* the stress fraction $S_i$ --
the four values sum to ~2.3, consistent with flux fraction over time
fraction; not used as a target.

## What the code does

`quadrant.py`: `quadrant_stats` (per pair: $S_{i,H}$, $T_{i,H}$, counts
on the configured hole grid, both `hole_norm` conventions), derived
sign-aware ejection/sweep splits, $\Delta S_0$, $\Delta D_0$, $\eta$,
and exuberance; `octant_stats` (fractions of $u'w'$, $u'\theta'$,
$w'\theta'$ -- generally $x'z'$, $x'c'$, $z'c'$ of the configured
triplet -- over the 8 sign octants, plus counts); `run` (the
`/quadrant` and `/octant` groups). Invariants: quadrant $S_{i,0}$ sum to
1 exactly (Raupach eq. 4), octant fractions sum to 1 per component, and
at $H = 0$ the sign-aware $\Delta S$, $\Delta D$, $\eta$ reproduce
UTESpac's `find_delta_flux`, `find_delta_time`, `find_eta` (integration
notes A.3) -- asserted against those functions directly in
`tests/test_ec_quadrant.py`.

Sign convention: quadrant indices are fixed by the literature layouts
above; ejection/sweep *labels* are sign-aware -- downgradient samples are
those whose product shares the sign of the total flux (UTESpac
convention; the validation site's $H$ flips sign diurnally), ejection the downgradient
half with $w' > 0$. Exuberance is emitted as
$Ex = \overline{w'c'}_{counter}/\overline{w'c'}_{down} = \eta - 1$
`[DERIVED]` from Li2011 eq. 8; the *name* "exuberance" is common usage
whose coining paper is not on hand `[ASSUMED]` -- UTESpac's `find_eta`
docstring already flags the distinction.

## Deviation register

- **Hole normalization.** `hole_norm = "rms"` (default; Lu1973 §4.3,
  Li2019 eq. 9 -- the ASL convention) or `"flux"` (Willmarth1972 via
  Wallace2016; Raupach1981 eq. 2). Related by
  $H_{flux} = |\rho|\,H_{rms}$ (Raupach1981 p. 365), asserted in the
  tests. The choice is stored in the output attrs.
- **Hole grid.** Default $H = 0..20$ (denser below 5): the range is
  Raupach1981 figs. 6-7 `[CITED]`; the specific grid spacing `[ASSUMED]`.
- **Octant triplets.** Ruled in (user, 2026-08-23): the module runs a
  *list* of triplets, default $(u', w', Ts')$ `[CITED]` Li2019 eq. 11
  (every read octant source uses velocity pair + one scalar) plus the
  gameplan's $(w', Ts', q')$ scalar-dissimilarity triplet `[ASSUMED]` --
  it appears in no read source (Li2011 studies scalar dissimilarity with
  *quadrants*, not octants). Output variables are tagged per triplet
  (`flux_frac_u_w_ts_wts`, `flux_frac_w_ts_rho_h2o_wts`, ...); any signals are
  accepted ($v$, $rhoCO_2$ anticipated). Validation-site check: in the
  $(w,Ts,\rho_v)$ sign space the warm-moist-updraft (+++) and
  cool-dry-downdraft (---) octants carry the $w'Ts'$ flux, the moist
  analog of the hot-ejection/cold-sweep dominance.
- **Scalar quadrant plane.** Scalar pairs use the $(w', c')$ plane with
  Li2019 eq. 10 numbering; ejection/sweep labels attach sign-aware as
  above, since their eq.-10 names assume upward flux.
