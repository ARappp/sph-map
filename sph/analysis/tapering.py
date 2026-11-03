"""Named radial taper functions in reciprocal radians per angstrom."""

from __future__ import annotations

import numpy as np


def cosine_taper(
    reciprocal_radius_rad_per_A: np.ndarray,
    maximum_radius_rad_per_A: float,
    eta: float = 0.9,
) -> np.ndarray:
    """Return the raised-cosine endpoint taper."""
    radii = np.asarray(reciprocal_radius_rad_per_A, dtype=np.float64)
    threshold = eta * maximum_radius_rad_per_A
    values = np.ones_like(radii)
    mask = (radii > threshold) & (radii <= maximum_radius_rad_per_A)
    values[mask] = 0.5 * (
        1.0
        + np.cos(
            np.pi
            * (radii[mask] - threshold)
            / (maximum_radius_rad_per_A - threshold)
        )
    )
    values[radii > maximum_radius_rad_per_A] = 0.0
    return values


def kaiser_taper(
    reciprocal_radius_rad_per_A: np.ndarray,
    maximum_radius_rad_per_A: float,
    beta: float = 8.0,
) -> np.ndarray:
    """Return the Kaiser taper, including its nonzero endpoint."""
    radii = np.asarray(reciprocal_radius_rad_per_A, dtype=np.float64)
    argument = np.clip(
        1.0 - (radii / maximum_radius_rad_per_A) ** 2, 0.0, 1.0
    )
    values = np.i0(beta * np.sqrt(argument)) / np.i0(beta)
    values[radii > maximum_radius_rad_per_A] = 0.0
    return values
