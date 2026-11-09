# ec_coherent Reynolds-stress anisotropy

The anisotropy tensor, its eigenvalues, the barycentric map and the
Lumley invariants are written from the extractions
`library/extracted/Banerjee2007.txt` and
`library/extracted/Stiperski2018.txt`, read in the drafting session.
The Banerjee et al. Table 1 and eq. 19 and the Stiperski & Calaf eqs.
3-5 and Table 1 were checked against rendered pages. The code is in
[ec_coherent/anisotropy.py](../../ec_coherent/anisotropy.py) and the
tests in [tests/test_ec_anisotropy.py](../../tests/test_ec_anisotropy.py).
Each section names the symbol that implements it. The module holds pure
functions only: nothing in `run` or the CLI calls it.

The flow each source was written for:

- **Banerjee et al. (2007) [@Banerjee2007].** The method is built and
  shown on engineering flows. Its main example is the "direct numerical
  simulation of fully developed plane turbulent channel" flow at
  $Re_\tau = 180$ (Fig. 9 caption, p. 17). It adds pipe flow ("Similar
  conclusions can be drawn about pipe", p. 10) and "the return to
  isotropy experiments for plane strain of" Choi (p. 17). No
  atmospheric data and no sonic anemometer are involved.
- **Stiperski & Calaf (2018) [@Stiperski2018].** The source is one
  campaign, CASES-99. "The data consist of a month of measurements at a
  60 m tower with seven levels of sonic anemometers (5, 10, 20, 30, 40,
  50 and 55 m above ground)" (p. 643).
  - Instruments: "levels 5, 30 and 50 m were installed with a
    non-orthogonal CSAT3, whereas the other levels used orthogonal ATI-K
    sonic anemometers" (p. 644).
  - Processing: "The data were processed using double rotation and were
    detrended prior to block averaging." (p. 644).
  - Averaging time, chosen by multi-resolution decomposition: in stable
    conditions "the most appropriate averaging time was chosen to be 1
    min"; in unstable conditions the "most appropriate averaging period
    was found to be 30 min" (p. 644).

The instrument at hand, its sampling and its rotation are not those of
either source. The rows on input preparation in the deviation register
cover that gap.

## Anisotropy tensor -- Stiperski & Calaf (2018) eq. 3 [@Stiperski2018]; Banerjee et al. (2007) eq. 1 [@Banerjee2007]

Stiperski & Calaf p. 642 define the deviatoric part, "which, upon
normalization by 2e, results in a non-dimensional anisotropy tensor with
components"

$$ b_{ij} = \frac{\overline{u_i u_j}}{\overline{u_l u_l}} - \frac{1}{3}\delta_{ij} \qquad (3) $$

Banerjee et al. p. 2 credit Lumley "by the introduction of the
anisotropy tensor"

$$ a_{ij} = \frac{\overline{u_i u_j}}{2k} - \frac{\delta_{ij}}{3}, \qquad k = \frac{\overline{u_i u_i}}{2} \qquad (1) $$

Since $2k = \overline{u_l u_l}$, the two definitions are the same tensor.

Implemented by `anisotropy_tensor` `[CITED]`. It is vectorised over
leading axes and returns a trace-free result.

Two behaviours are not in either source and are `[ASSUMED]`:

- The result is NaN where the input holds a non-finite element.
- The result is NaN where the trace $\overline{u_l u_l}$ is not
  positive, since the normalisation is undefined there.

## Eigenvalues -- Banerjee et al. (2007) eqs. 14-16 [@Banerjee2007]

Banerjee et al. pp. 5-6 order the eigenvalues as follows:
$\lambda_1$ is the largest diagonal element in the principal coordinate
system (eq. 14) and $\lambda_3 = -\lambda_1 - \lambda_2$ (eq. 15).
"From notation (14), the following inequality holds for the eigenvalues
of" $a_{ij}$:

$$ \lambda_1 \ge \lambda_2 \ge \lambda_3 \qquad (16) $$

They go on: "Introducing the non-increasing order on diagonal elements
of the anisotropy tensor" (p. 6). Stiperski & Calaf write the same three
eigenvalues as $\lambda_I, \lambda_{II}, \lambda_{III}$ (p. 643).

Implemented by `eigenvalues` `[CITED]`. The method is
`numpy.linalg.eigvalsh` on the symmetric tensor, reversed to descending
order `[HOUSE]`. Rows with a non-finite input are NaN `[ASSUMED]`.

## Barycentric weights -- Banerjee et al. (2007) eqs. 31-34 [@Banerjee2007]

Banerjee et al. p. 9: "any anisotropy tensor" "can be expressed as a
convex combination (29) of the limiting states (one-component,
two-component, three-component) by setting"

$$ C_{1c} = \lambda_1 - \lambda_2 \quad (32), \qquad C_{2c} = 2(\lambda_2 - \lambda_3) \quad (33), \qquad C_{3c} = 3\lambda_3 + 1 \quad (34) $$

with $C_{1c} + C_{2c} + C_{3c} = 1$ (eq. 31). "The normalization is done
in such a way that all metrics {C1c, C2c, C3c} lie in the range [0, 1]."
(p. 9).

Stiperski & Calaf p. 643 give the same weights, "with the corresponding
weights (Cic) determined by the eigenvalues of the normalized Reynolds
stress anisotropy tensor", in the form
$C_{1c} = \lambda_I - \lambda_{II}$,
$C_{2c} = 2(\lambda_{II} - \lambda_{III})$ and
$C_{3c} = 3\lambda_{III} + 1$.

Implemented by `barycentric` (returns `C1c`, `C2c`, `C3c`) `[CITED]`.

Banerjee et al. Appendix B gives a second set of metrics,
$C_{1cr}, C_{2cr}, C_{3cr}$ (eqs. 117-119). These are built on the
Reynolds stress tensor normalised by $q^2$ and are "normalized by the
magnitude of the largest eigenvalue of the tensor." (p. 26). The module
does not implement them `[ASSUMED]`: Stiperski & Calaf use eqs. 32-34.

## Map coordinates -- Banerjee et al. (2007) eqs. 38-39 [@Banerjee2007]; Stiperski & Calaf (2018) eqs. 4-5 [@Stiperski2018]

Banerjee et al. p. 9 identify the basis matrices with the vertices of a
triangle. "To plot a point, we have to take a convex combination of the
limiting" states:

$$ x_{new} = C_{1c}x_{1c} + C_{2c}x_{2c} + C_{3c}x_{3c} \quad (38), \qquad y_{new} = C_{1c}y_{1c} + C_{2c}y_{2c} + C_{3c}y_{3c} \quad (39) $$

They place no vertices: the map "can be constructed by taking three
arbitrary basis points" (p. 10). They draw it as an equilateral triangle
because "an equilateral triangle does not introduce any visual bias of
the limiting states" (p. 10).

Stiperski & Calaf p. 643 fix the vertices: "where the limiting states
are placed at" $x_{1C} = (1, 0)$, $x_{2C} = (0, 0)$ and
$x_{3C} = (1/2, \sqrt{3}/2)$. With these vertices the coordinates are

$$ x_B = C_{1c} + \tfrac{1}{2} C_{3c} \quad (4), \qquad y_B = \tfrac{\sqrt{3}}{2} C_{3c} \quad (5) $$

They add that this map "weighs the different limiting states of
turbulence anisotropy equally" (p. 643).

Implemented by `barycentric` (returns `xB`, `yB`) with the module
constants `VERTEX_1C`, `VERTEX_2C`, `VERTEX_3C` `[CITED]`. The
coordinates are computed by the general sums (38)-(39), so other vertex
choices change only the constants. Adopting the Stiperski & Calaf
placement over any other is `[ASSUMED]`: Banerjee et al. leave it open.

## Lumley invariants -- Stiperski & Calaf (2018) p. 643 and Table 1 [@Stiperski2018]

Stiperski & Calaf p. 643: "the state of anisotropy can be described by
two invariants", defined as $6\eta^2 = b_{ij}b_{ji}$ and
$6\xi^3 = b_{ij}b_{jk}b_{ki}$ "(see Pope, 2000 for more details)".

- The first invariant "is positive-definite and measures the degree of
  anisotropy in the flow field".
- The second is signed: positive $\xi$ is one-component dominated,
  "while for negative values the flow is dominated by two-component
  axisymmetric" turbulence.

They "can be derived alternatively through an eigenvalue decomposition
of the anisotropy tensor", citing "(Spencer, 1971)":

$$ \eta^2 = \tfrac{1}{3}\left(\lambda_I^2 + \lambda_I\lambda_{II} + \lambda_{II}^2\right), \qquad \xi^3 = -\tfrac{1}{2}\lambda_I\lambda_{II}(\lambda_I + \lambda_{II}) $$

The equations are unnumbered in the source. Table 1 (p. 643) lists the
limiting states:

| State | Invariants | Eigenvalues |
|---|---|---|
| isotropic | $\eta = \xi = 0$ | $\lambda_I = \lambda_{II} = \lambda_{III} = 0$ |
| two-component axisymmetric | $\eta = 1/6$, $\xi = -1/6$ | $\lambda_I = \lambda_{II} = 1/6$ |
| one-component | $\eta = \xi = 1/3$ | $\lambda_I = 2/3$, $\lambda_{II} = \lambda_{III} = -1/3$ |

Implemented by `lumley_invariants` `[CITED]`, from the eigenvalue
forms. $\eta$ is the non-negative square root. $\xi$ is the real cube
root, so it keeps the sign of $\xi^3$. The three Table 1 rows are the
test oracles.

**Banerjee et al. disagree with these values.**

- Eq. 19 (p. 6, checked on the rendered page) reads
  $\zeta^3 = III/2$, $\eta^2 = -II/2$, with $II = a_{ij}a_{ji}$ from
  eq. 2. This differs from Stiperski & Calaf in normalisation ($II/2$
  and $III/2$ against $II/6$ and $III/6$, by eq. 21) and in the sign of
  $\eta^2$; Table 1's magnitudes $1/3$ and $1/6$ match the $/6$ forms,
  not eq. 19.
- Their Table 1 (p. 7, checked on the rendered page) gives the
  invariants of the one-component and two-component axisymmetric
  limits interchanged:
  - the one-component limit is given as $\zeta = -1/6$, $\eta = 1/6$;
  - the two-component axisymmetric limit is given as
    $\zeta = \eta = 1/3$, with $\lambda_1 = \lambda_3 = 1/6$ and
    $\lambda_2 = -1/3$, which contradicts their own eq. 16.
- Their Fig. 2 (right, p. 6) places 1C near $(1/3, 1/3)$ and 2C-axi
  near $(-1/6, 1/6)$. That agrees with Stiperski & Calaf, not with
  their Table 1.

The disagreement is recorded here, not resolved. The code follows
Stiperski & Calaf, whose values also follow from hand evaluation of the
eigenvalue forms at the Table 1 eigenvalues.

## Reynolds stress from series -- no source `[ASSUMED]`

`from_series(u, v, w, axis=-1)` returns
$R_{ij} = \mathrm{mean}(u_i' u_j')$ over one block:

- The block mean is removed.
- The products are averaged with population normalisation ($1/N$),
  which cancels in eq. 3.
- A NaN sample propagates to its component's row and column of $R$, and
  then to the whole block through `anisotropy_tensor`.

Neither source specifies this step. Stiperski & Calaf double-rotate and
detrend before block averaging (p. 644). Here the coordinate frame and
any detrending are left to the caller, and only the block mean is
removed.

## Outputs

These are function returns only. The module writes no netCDF group,
attribute or variable, and no pipeline stage calls it.

| Function | Returns | Shape |
|---|---|---|
| `anisotropy_tensor(R)` | `b` | `(..., 3, 3)` |
| `eigenvalues(b)` | `lam`, descending | `(..., 3)` |
| `barycentric(lam)` | `xB, yB, C1c, C2c, C3c` | `(...)` each |
| `lumley_invariants(lam)` | `xi, eta` | `(...)` each |
| `from_series(u, v, w, axis=-1)` | `R` | `(..., 3, 3)` |

## Deviation register

| Paper | Code | Why | Validation |
|---|---|---|---|
| Banerjee et al. Table 1 (p. 7) gives the 1c and 2c-axi limits as $(\zeta, \eta) = (-1/6, 1/6)$ and $(1/3, 1/3)$; eq. 19 (p. 6) normalises differently | Stiperski & Calaf p. 643 forms and Table 1: 1c at $(\xi, \eta) = (1/3, 1/3)$, 2c-axi at $(-1/6, 1/6)$ `[CITED]` | Banerjee's Table 1 contradicts its own Fig. 2 (right) and eq. 16; Stiperski's values follow from the eigenvalue forms | `test_lumley_limits` (three states plus NaN) |
| Banerjee et al. leave the vertices arbitrary (p. 10) | Stiperski & Calaf vertices (p. 643) `[ASSUMED]` choice | Stiperski & Calaf is the atmospheric reference for the map; any other placement needs only new constants | `test_barycentric_vertex` (three vertices plus NaN) |
| Stiperski & Calaf (CASES-99, p. 644): double rotation, detrending, 1 min (stable) or 30 min (unstable) blocks chosen by MRD, no spectral flux correction because it "would add asymmetry to the Reynolds stress tensor, causing unphysical anisotropy" | `from_series` removes the block mean only; frame, detrending, block length and the input's correction state are the caller's `[ASSUMED]` | The module is a pure function with no wiring; input preparation belongs to the caller | `test_from_series` against `numpy.cov(..., bias=True)` |
| Stiperski & Calaf criterion 4 (p. 645): data in transition regions of the map removed; "The limiting lines for each kite were chosen to cover 70% of the sides of the equilateral triangle." | No filtering; every block gets coordinates `[ASSUMED]` | The kite screen belongs to their scaling analysis, not to the map | none: not implemented |
| Neither source covers non-finite input or non-positive trace | NaN out `[ASSUMED]` | Normalisation undefined | `test_anisotropy_tensor` (off-diagonal NaN, zero trace); `nan` and `negative_trace` cases in `test_barycentric_vertex`, `test_lumley_limits` |
| Banerjee et al. Appendix B metrics $C_{1cr}$ to $C_{3cr}$ (eqs. 117-119) | Not implemented `[ASSUMED]` | Stiperski & Calaf use eqs. 32-34 | none |
