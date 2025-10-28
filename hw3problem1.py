"""hw3problem1: Tapered beam 3x3 flexibility matrix using principle of virtual forces

This script computes the 3x3 flexibility (compliance) matrix for a straight,
planar beam element of length L, Young's modulus E, constant thickness t, and
width b(x) that varies along the axis (0..L). The cross-section is assumed
rectangular so I(x) = (1/12) * b(x) * t^3.

The flexibility matrix F relates the three nodal sectional actions (axial
displacement u, transverse displacement v, rotation theta at one node relative
to the other) through virtual work. We compute the influence fields for unit
generalized forces (axial force, shear, bending moment) and integrate strain
energy contributions 1/(E*A) * axial^2 + 1/(E*I) * curvature^2 along the beam.

The code supports symbolic `b(x)` (SymPy expression) and will attempt analytic
integration; if that fails it falls back to numerical quadrature.

Example usage at bottom runs a linear taper test and prints both symbolic
where possible and numeric evaluation.
"""

from __future__ import annotations

from typing import Callable, Union
import math
import sys
import numpy as np
import scipy.integrate as spi


Number = Union[float, int]

def assert_square(matrix):
    """Asserts that a NumPy array is a square matrix."""
    rows, cols = matrix.shape
    assert rows == cols, "Matrix is not square. It has shape {}.".format(matrix.shape)


def compute_flexibility_matrix(
    L: Number,
    E: Number,
    b: Number,
) -> np.Matrix:
    """Compute the 3x3 flexibility matrix for a tapered beam element.

    Args:
        L: Length of the beam element.
        E: Young's modulus of the beam material.
        b: Width of the beam cross-section (constant).
        h: Height of the beam cross-section as a function of x (can be symbolic).
    Returns:
        3x3 Matrix representing the flexibility matrix F.
    """
    d = np.matrix(np.zeros((3, 3)))
    # Axial flexibility F11
    F11 = spi.quad(lambda x: 1 / ((0.3 * (x - x**5 / L**4)) + 0.1 * L), 0, L)[0] * (1 / (E * b))
    d[0, 0] = F11

    # F22
    F22 = spi.quad(lambda x: 1 / ((0.3 * (x - x**5 / L**4)) + 0.1 * L), 0, L)[0] * (
        1 / (E * b)
    )
    d[1, 1] = F22

    # F33
    F33 = F22 # No adjustment needed since M is unchanged
    d[2, 2] = F33

    # F23 and F32 (symmetric so only compute once)
    F23 = spi.quad(lambda x: (L - x) / ((0.3 * (x - x**5 / L**4)) + 0.1 * L), 0, L)[
        0
    ] * (1 / (E * b))
    d[1, 2] = F23
    d[2, 1] = F23
    assert_square(d)
    return d


def flexibility_to_stiffness(d:np.Matrix, L: float) -> np.Matrix:
    """Convert flexibility matrix to stiffness matrix by inversion. Heavily references chapter 4 of textbook.

    Args:
        d: 3x3 flexibility matrix.
    Returns:
        3x3 stiffness matrix.
    """
    d_inv = np.linalg.inv(d)
    T_l = np.matrix(np.zeros((3, 3)))
    T_l[0, 0] = -1
    T_l[1, 1] = -1
    T_l[2, 2] = -1
    T_l[2, 1] = L
    T_r = np.identity(3)
    T = np.hstack((T_l, T_r))
    return T.T @ d_inv @ T


if __name__ == "__main__":
    L, E, b = 2000.0, 200e9, 300.0 # Making a generalized guess on E
    F = compute_flexibility_matrix(L, E, b)
    print("Flexibility matrix F:")
    for row in np.array(F):
        print(row)
    K = flexibility_to_stiffness(F, L)
    print("\nStiffness matrix K:")
    for row in np.array(K):
        print(row)
