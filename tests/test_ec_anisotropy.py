"""Anisotropy module: limiting states on the barycentric map and the Lumley triangle.

Oracles (library/writeups/ec_anisotropy.md): the barycentric vertices
x1C = (1, 0), x2C = (0, 0), x3C = (1/2, sqrt(3)/2) and the invariants of the
isotropic, one-component and two-component axisymmetric states from
Stiperski & Calaf 2018 p. 643 and Table 1; the weights C1c, C2c, C3c from
Banerjee et al. 2007 eqs. 32-34. Frame invariance follows from the
eigenvalue construction (Banerjee et al. 2007 eqs. 12-13).
"""

import numpy as np
import pytest

from ec_coherent import anisotropy as an

# Closed-form values in double precision: rounding is O(1e-16); 1e-12 still
# fails any change of the result at the 1e-12 level and above.
TOL = 1e-12
SQ3 = np.sqrt(3.0)

# Reynolds stress tensors of the limiting states (scaled, rotated off the
# principal frame for the anisotropic ones so the eigen-decomposition matters).
_ROT = np.array([[np.cos(0.7), -np.sin(0.7), 0.0],
                 [np.sin(0.7), np.cos(0.7), 0.0],
                 [0.0, 0.0, 1.0]])
_STATES = {
    "isotropic": 2.5 * np.eye(3),
    "one_component": 1.3 * _ROT @ np.diag([1.0, 0.0, 0.0]) @ _ROT.T,
    "two_component_axisymmetric": 0.4 * _ROT @ np.diag([0.0, 1.0, 1.0]) @ _ROT.T,
    "nan": np.where(np.eye(3) == 1, np.nan, 0.0),
    "negative_trace": -np.eye(3),
}

# Off-diagonal tensor with trace 4: b = R / 4 - I / 3 worked by hand.
_R_OFFDIAG = np.array([[2.0, 0.5, 0.0],
                       [0.5, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])
_B_OFFDIAG = np.array([[1 / 6, 1 / 8, 0.0],
                       [1 / 8, -1 / 12, 0.0],
                       [0.0, 0.0, -1 / 12]])
_R_OFFDIAG_NAN = _R_OFFDIAG.copy()
_R_OFFDIAG_NAN[0, 1] = _R_OFFDIAG_NAN[1, 0] = np.nan
_R_OFFDIAG_INF = _R_OFFDIAG.copy()
_R_OFFDIAG_INF[0, 0] = np.inf
# Eigenvalues of _B_OFFDIAG by hand: the upper 2x2 block has mean 1/24 and
# radius sqrt((1/8)**2 + (1/8)**2) = sqrt(2)/8; the third is -1/12.
_LAM_OFFDIAG = np.array([1 / 24 + np.sqrt(2.0) / 8, -1 / 12,
                         1 / 24 - np.sqrt(2.0) / 8])
_B_OFFDIAG_NAN = _B_OFFDIAG.copy()
_B_OFFDIAG_NAN[0, 1] = _B_OFFDIAG_NAN[1, 0] = np.nan
# inf in the upper triangle, which eigvalsh does not read: only the
# finiteness guard in eigenvalues turns the row into NaN.
_B_OFFDIAG_INF = _B_OFFDIAG.copy()
_B_OFFDIAG_INF[0, 1] = np.inf


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("R, expected", [
    (_R_OFFDIAG, _B_OFFDIAG),
    (_R_OFFDIAG_NAN, np.full((3, 3), np.nan)),
    (_R_OFFDIAG_INF, np.full((3, 3), np.nan)),
    (np.zeros((3, 3)), np.full((3, 3), np.nan)),
])
def test_anisotropy_tensor(R, expected):
    np.testing.assert_allclose(an.anisotropy_tensor(R), expected, atol=TOL)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("b, expected", [
    (_B_OFFDIAG, _LAM_OFFDIAG),
    (_B_OFFDIAG_NAN, np.full(3, np.nan)),
    (_B_OFFDIAG_INF, np.full(3, np.nan)),
])
def test_eigenvalues(b, expected):
    np.testing.assert_allclose(an.eigenvalues(b), expected, atol=TOL)


@pytest.mark.parametrize("state, xy, weights", [
    ("isotropic", (0.5, SQ3 / 2), (0.0, 0.0, 1.0)),
    ("one_component", (1.0, 0.0), (1.0, 0.0, 0.0)),
    ("two_component_axisymmetric", (0.0, 0.0), (0.0, 1.0, 0.0)),
    ("nan", (np.nan, np.nan), (np.nan, np.nan, np.nan)),
    ("negative_trace", (np.nan, np.nan), (np.nan, np.nan, np.nan)),
])
def test_barycentric_vertex(state, xy, weights):
    lam = an.eigenvalues(an.anisotropy_tensor(_STATES[state]))
    xB, yB, C1c, C2c, C3c = an.barycentric(lam)
    np.testing.assert_allclose([xB, yB], xy, atol=TOL)
    np.testing.assert_allclose([C1c, C2c, C3c], weights, atol=TOL)


@pytest.mark.parametrize("state, xi, eta", [
    ("isotropic", 0.0, 0.0),
    ("one_component", 1 / 3, 1 / 3),
    ("two_component_axisymmetric", -1 / 6, 1 / 6),
    ("nan", np.nan, np.nan),
    ("negative_trace", np.nan, np.nan),
])
def test_lumley_limits(state, xi, eta):
    lam = an.eigenvalues(an.anisotropy_tensor(_STATES[state]))
    got_xi, got_eta = an.lumley_invariants(lam)
    np.testing.assert_allclose([got_xi, got_eta], [xi, eta], atol=TOL)


def test_rotation_invariance():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(6, 3, 3))
    R = A @ np.swapaxes(A, -1, -2) + 0.1 * np.eye(3)
    Q, _ = np.linalg.qr(rng.normal(size=(6, 3, 3)))
    R_rot = Q @ R @ np.swapaxes(Q, -1, -2)

    def chain(r):
        lam = an.eigenvalues(an.anisotropy_tensor(r))
        return np.concatenate([lam, np.stack(an.barycentric(lam)[:2] + an.lumley_invariants(lam), -1)], -1)

    # cross-check of the vectorised path against a loop over single tensors
    loop = np.stack([chain(r) for r in R])
    np.testing.assert_allclose(chain(R), loop, atol=TOL)
    np.testing.assert_allclose(chain(R_rot), chain(R), atol=TOL)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("bad", [None, np.nan, np.inf])
def test_from_series(bad):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(3, 200)) * [[2.0], [1.0], [0.5]]
    x[2] += 0.3 * x[0]
    x += [[4.0], [-1.0], [0.2]]   # non-zero means, removed by from_series
    if bad is not None:
        x[1, 17] = bad
    R = an.from_series(x[0], x[1], x[2])
    # NaN-propagating, population normalised; a non-finite sample acts as NaN
    expected = np.cov(np.where(np.isfinite(x), x, np.nan), bias=True)
    np.testing.assert_allclose(R, expected, atol=TOL)


def test_from_series_axis():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(3, 200, 2)) * [[[2.0]], [[1.0]], [[0.5]]]
    x[2] += 0.3 * x[0]
    R = an.from_series(x[0], x[1], x[2], axis=0)
    expected = np.stack([np.cov(x[..., k], bias=True) for k in range(2)])
    np.testing.assert_allclose(R, expected, atol=TOL)


@pytest.mark.parametrize("call, match", [
    (lambda: an.anisotropy_tensor(np.eye(2)), "trailing shape"),
    (lambda: an.anisotropy_tensor(np.ones((4, 3))), "trailing shape"),
    (lambda: an.eigenvalues(np.eye(2)), "trailing shape"),
    (lambda: an.eigenvalues(np.ones((4, 3))), "trailing shape"),
    (lambda: an.barycentric(np.zeros(2)), "trailing shape"),
    (lambda: an.barycentric(np.zeros(4)), "trailing shape"),
    (lambda: an.lumley_invariants(np.zeros(2)), "trailing shape"),
    (lambda: an.lumley_invariants(np.zeros(4)), "trailing shape"),
    (lambda: an.from_series(np.zeros(5), np.zeros(5), np.zeros(4)),
     "u, v and w must have the same shape"),
    (lambda: an.from_series(np.zeros(4), np.zeros(5), np.zeros(5)),
     "u, v and w must have the same shape"),
    (lambda: an.from_series(np.zeros(5), np.zeros(4), np.zeros(5)),
     "u, v and w must have the same shape"),
])
def test_shape_errors(call, match):
    with pytest.raises(ValueError, match=match):
        call()
