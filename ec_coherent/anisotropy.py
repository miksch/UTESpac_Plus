"""Reynolds-stress anisotropy: eigenvalues, barycentric map, Lumley invariants.

Science and sources: library/writeups/ec_anisotropy.md. The normalised
anisotropy tensor b_ij = <u_i u_j>/<u_l u_l> - delta_ij/3 (Stiperski & Calaf
2018 eq. 3; Banerjee et al. 2007 eq. 1), its eigenvalues in non-increasing
order (Banerjee et al. 2007 eqs. 14-16), the barycentric weights
C1c, C2c, C3c (Banerjee et al. 2007 eqs. 32-34) placed on the map with the
vertices of Stiperski & Calaf 2018 eqs. 4-5, and the Lumley invariants
(xi, eta) from the eigenvalues (Stiperski & Calaf 2018 p. 643). Pure
functions, vectorised over leading axes; NaN in any input element gives NaN
in every output derived from it.
"""

from typing import Tuple

import numpy as np

# Barycentric-map vertices (x, y): one-component, two-component and
# three-component (isotropic) limiting states, Stiperski & Calaf 2018 p. 643.
VERTEX_1C = (1.0, 0.0)
VERTEX_2C = (0.0, 0.0)
VERTEX_3C = (0.5, np.sqrt(3.0) / 2.0)


def anisotropy_tensor(R: np.ndarray) -> np.ndarray:
    """Normalised anisotropy tensor of a Reynolds stress tensor.

    Parameters
    ----------
    R : ndarray, shape (..., 3, 3)
        Symmetric Reynolds stress tensor <u_i' u_j'>.

    Returns
    -------
    b : ndarray, shape (..., 3, 3)
        ``b_ij = R_ij / R_ll - delta_ij / 3``, trace zero. NaN where ``R``
        holds a non-finite element or its trace is not positive.

    Raises
    ------
    ValueError
        If the trailing shape of ``R`` is not (3, 3).
    """
    R = np.asarray(R, dtype=float)
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"R must have trailing shape (3, 3), got {R.shape}")
    trace = np.trace(R, axis1=-2, axis2=-1)
    bad = ~np.isfinite(R).all(axis=(-2, -1)) | ~(trace > 0)
    safe = np.where(bad, 1.0, trace)
    b = R / safe[..., None, None] - np.eye(3) / 3.0
    return np.where(bad[..., None, None], np.nan, b)


def eigenvalues(b: np.ndarray) -> np.ndarray:
    """Eigenvalues of a symmetric anisotropy tensor, sorted descending.

    Parameters
    ----------
    b : ndarray, shape (..., 3, 3)
        Symmetric anisotropy tensor.

    Returns
    -------
    lam : ndarray, shape (..., 3)
        ``lam[..., 0] >= lam[..., 1] >= lam[..., 2]`` (lambda_1, lambda_2,
        lambda_3). NaN rows where ``b`` holds a non-finite element.

    Raises
    ------
    ValueError
        If the trailing shape of ``b`` is not (3, 3).
    """
    b = np.asarray(b, dtype=float)
    if b.shape[-2:] != (3, 3):
        raise ValueError(f"b must have trailing shape (3, 3), got {b.shape}")
    bad = ~np.isfinite(b).all(axis=(-2, -1))
    safe = np.where(bad[..., None, None], 0.0, b)
    lam = np.linalg.eigvalsh(safe)[..., ::-1]
    return np.where(bad[..., None], np.nan, lam)


def barycentric(lam: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Barycentric-map coordinates and limiting-state weights.

    Parameters
    ----------
    lam : ndarray, shape (..., 3)
        Anisotropy-tensor eigenvalues sorted descending, finite or NaN, as
        returned by :func:`eigenvalues`.

    Returns
    -------
    xB, yB : ndarray, shape (...)
        Position on the map, ``xB = C1c + C3c / 2``, ``yB = C3c * sqrt(3) / 2``.
    C1c, C2c, C3c : ndarray, shape (...)
        One-, two- and three-component weights, ``C1c = lambda_1 - lambda_2``,
        ``C2c = 2 (lambda_2 - lambda_3)``, ``C3c = 3 lambda_3 + 1``; they sum
        to 1 for a trace-free tensor.

    Raises
    ------
    ValueError
        If the trailing shape of ``lam`` is not (3,).
    """
    lam = np.asarray(lam, dtype=float)
    if lam.shape[-1:] != (3,):
        raise ValueError(f"lam must have trailing shape (3,), got {lam.shape}")
    l1, l2, l3 = lam[..., 0], lam[..., 1], lam[..., 2]
    C1c = l1 - l2
    C2c = 2.0 * (l2 - l3)
    C3c = 3.0 * l3 + 1.0
    xB = C1c * VERTEX_1C[0] + C2c * VERTEX_2C[0] + C3c * VERTEX_3C[0]
    yB = C1c * VERTEX_1C[1] + C2c * VERTEX_2C[1] + C3c * VERTEX_3C[1]
    return xB, yB, C1c, C2c, C3c


def lumley_invariants(lam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Lumley-triangle invariants from the anisotropy eigenvalues.

    Parameters
    ----------
    lam : ndarray, shape (..., 3)
        Anisotropy-tensor eigenvalues sorted descending, finite or NaN, as
        returned by :func:`eigenvalues`.

    Returns
    -------
    xi, eta : ndarray, shape (...)
        ``eta**2 = (lambda_1**2 + lambda_1 lambda_2 + lambda_2**2) / 3`` and
        ``xi**3 = -lambda_1 lambda_2 (lambda_1 + lambda_2) / 2``; ``eta >= 0``,
        ``xi`` carries the sign of its cube (positive towards the
        one-component state, negative towards the two-component
        axisymmetric state).

    Raises
    ------
    ValueError
        If the trailing shape of ``lam`` is not (3,).
    """
    lam = np.asarray(lam, dtype=float)
    if lam.shape[-1:] != (3,):
        raise ValueError(f"lam must have trailing shape (3,), got {lam.shape}")
    l1, l2 =lam[..., 0], lam[..., 1]
    eta = np.sqrt((l1 * l1 + l1 * l2 + l2 * l2) / 3.0)
    xi = np.cbrt(-0.5 * l1 * l2 * (l1 + l2))
    return xi, eta


def from_series(u: np.ndarray, v: np.ndarray, w: np.ndarray, axis: int = -1) -> np.ndarray:
    """Reynolds stress tensor from velocity component series.

    Parameters
    ----------
    u, v, w : ndarray
        Velocity components of one averaging block, same shape; the mean
        along ``axis`` is removed before the products are averaged.
    axis : int, optional
        Time axis. Default -1.

    Returns
    -------
    R : ndarray, shape (..., 3, 3)
        ``R_ij = mean(u_i' u_j')`` over ``axis`` (population normalisation,
        which cancels in :func:`anisotropy_tensor`). A non-finite sample in
        one component makes that component's row and column NaN, and
        :func:`anisotropy_tensor` then returns NaN for the whole block.

    Raises
    ------
    ValueError
        If ``u``, ``v`` and ``w`` differ in shape.
    """
    comps = [np.moveaxis(np.asarray(c, dtype=float), axis, -1) for c in (u, v, w)]
    if not (comps[0].shape == comps[1].shape == comps[2].shape):
        raise ValueError("u, v and w must have the same shape")
    x = np.stack(comps, axis=-2)
    x = np.where(np.isfinite(x), x, np.nan)
    xp = x - x.mean(axis=-1, keepdims=True)
    return np.einsum("...it,...jt->...ij", xp, xp) / xp.shape[-1]
