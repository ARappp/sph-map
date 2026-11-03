"""Real, orthonormal tesseral spherical harmonics.

SciPy's complex harmonics use the Condon--Shortley phase. The real basis is

``R_lm = sqrt(2)(-1)^m Re(Y_lm)`` for ``m > 0``;
``R_lm = sqrt(2)(-1)^|m| Im(Y_l|m|)`` for ``m < 0``; and
``R_l0 = Re(Y_l0)``.

It satisfies ``integral |R_lm|^2 dOmega = 1``. Inputs use polar ``theta`` and
azimuth ``phi`` in radians; SciPy receives those in its current API order.
"""

from __future__ import annotations

import numpy as np
from scipy.special import sph_harm_y


def evaluate_complex_harmonic(
    order_m: int,
    degree_l: int,
    azimuth_phi_rad: np.ndarray,
    polar_theta_rad: np.ndarray,
) -> np.ndarray:
    """Evaluate SciPy's complex ``Y_lm`` with compatibility argument order."""
    return sph_harm_y(degree_l, order_m, polar_theta_rad, azimuth_phi_rad)


def evaluate_real_harmonic(
    order_m: int,
    degree_l: int,
    azimuth_phi_rad: np.ndarray,
    polar_theta_rad: np.ndarray,
) -> np.ndarray:
    """Evaluate the validated real harmonic in crystal/local angular axes."""
    if abs(order_m) > degree_l or degree_l < 0:
        raise ValueError("Harmonic indices require l >= 0 and -l <= m <= l.")
    if order_m > 0:
        return (
            np.sqrt(2.0)
            * (-1) ** order_m
            * evaluate_complex_harmonic(
                order_m, degree_l, azimuth_phi_rad, polar_theta_rad
            ).real
        )
    if order_m < 0:
        positive_order = -order_m
        return (
            np.sqrt(2.0)
            * (-1) ** positive_order
            * evaluate_complex_harmonic(
                positive_order, degree_l, azimuth_phi_rad, polar_theta_rad
            ).imag
        )
    return evaluate_complex_harmonic(
        0, degree_l, azimuth_phi_rad, polar_theta_rad
    ).real
