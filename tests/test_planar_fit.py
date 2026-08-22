"""Planar-fit mechanics: coefficient contract, offset removal, compat path."""

import numpy as np
import pytest

from utespac.pf_coefficients import pf_coefficients
from utespac.sonic_rotation import _apply_pf, _build_pf_matrix


def _synthetic_plane(b0=0.04, b1=-0.08, b2=0.03, n=500, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(1, 8, n)
    v = rng.uniform(-3, 3, n)
    w = b0 + b1 * u + b2 * v
    return u, v, w


def test_pf_coefficients_returns_b0_b1_b2_in_order():
    u, v, w = _synthetic_plane()
    coef = pf_coefficients(np.column_stack([u, v, w, np.full_like(u, 200.0)]), [])
    b = coef["degrees_0_to_0"]
    assert b == pytest.approx([0.04, -0.08, 0.03], abs=1e-9)


def test_apply_pf_with_correct_coefficients_zeroes_mean_w():
    u, v, w = _synthetic_plane()
    b = pf_coefficients(np.column_stack([u, v, w, np.zeros_like(u)]), [])["degrees_0_to_0"]
    P = _build_pf_matrix(b[1], b[2])
    rotated = _apply_pf(P, b[0], u, v, w)
    # every point lies on the fitted plane, so rotated w vanishes pointwise
    assert np.allclose(rotated[:, 2], 0.0, atol=1e-9)


def test_b0_not_removed_leaves_offset_in_w():
    u, v, w = _synthetic_plane(b0=0.04)
    b = pf_coefficients(np.column_stack([u, v, w, np.zeros_like(u)]), [])["degrees_0_to_0"]
    P = _build_pf_matrix(b[1], b[2])
    rotated = _apply_pf(P, 0.0, u, v, w)   # the legacy (MATLAB) path
    assert abs(rotated[:, 2].mean() - 0.04 * P[2, 2]) < 1e-9


def test_legacy_index_bug_applies_wrong_tilt():
    """MATLAB sonicRotation read [b0, b1] as (b1, b2); pitch is then wrong."""
    u, v, w = _synthetic_plane(b0=0.04, b1=-0.08, b2=0.03)
    b = pf_coefficients(np.column_stack([u, v, w, np.zeros_like(u)]), [])["degrees_0_to_0"]
    P_wrong = _build_pf_matrix(b[0], b[1])
    rotated = _apply_pf(P_wrong, 0.0, u, v, w)
    assert not np.allclose(rotated[:, 2], 0.0, atol=1e-3)


def test_plane_normal_maps_to_vertical():
    b1, b2 = -0.08, 0.03
    n = np.array([-b1, -b2, 1.0]) / np.sqrt(b1**2 + b2**2 + 1)
    P = _build_pf_matrix(b1, b2)
    assert np.allclose(P @ n, [0, 0, 1], atol=1e-12)
