"""Reciprocal-space coordinates in the crystallographic Cartesian frame."""

from __future__ import annotations

from dataclasses import dataclass

import gemmi
import numpy as np


TWO_PI = 2.0 * np.pi


def reciprocal_radius_cycles_per_A_from_d(
    spacing_A: np.ndarray | float,
) -> np.ndarray:
    """Return crystallographic reciprocal radius ``q=1/d`` in cycles/A."""
    spacing = np.asarray(spacing_A, dtype=np.float64)
    if np.any(~np.isfinite(spacing)) or np.any(spacing <= 0.0):
        raise ValueError("Interplanar spacings must be finite and positive.")
    return 1.0 / spacing


def reciprocal_radius_rad_per_A_from_cycles(
    reciprocal_radius_cycles_per_A: np.ndarray | float,
) -> np.ndarray:
    """Convert crystallographic ``q`` (cycles/A) to ``k=2*pi*q`` (rad/A)."""
    radius = np.asarray(reciprocal_radius_cycles_per_A, dtype=np.float64)
    if np.any(~np.isfinite(radius)) or np.any(radius < 0.0):
        raise ValueError("Reciprocal radii must be finite and non-negative.")
    return TWO_PI * radius


def reciprocal_radius_cycles_per_A_from_rad(
    reciprocal_radius_rad_per_A: np.ndarray | float,
) -> np.ndarray:
    """Convert wavevector radius ``k`` (rad/A) to ``q=k/(2*pi)`` (cycles/A)."""
    radius = np.asarray(reciprocal_radius_rad_per_A, dtype=np.float64)
    if np.any(~np.isfinite(radius)) or np.any(radius < 0.0):
        raise ValueError("Reciprocal radii must be finite and non-negative.")
    return radius / TWO_PI


@dataclass(frozen=True)
class AngularCoordinates:
    """Directions and radii derived from row-vector Miller indices.

    ``reciprocal_vectors_rad_per_A = hkl @ B.T`` where
    ``B = 2*pi*A**(-T)``. If a local frame is supplied, only the unit
    directions used for angles are passively transformed; radii do not change.
    """

    polar_theta_rad: np.ndarray
    azimuth_phi_rad: np.ndarray
    reciprocal_radius_rad_per_A: np.ndarray
    reciprocal_vectors_rad_per_A: np.ndarray
    unit_directions: np.ndarray

    @property
    def reciprocal_radius_cycles_per_A(self) -> np.ndarray:
        """Return ``q=1/d`` in crystallographic cycles per angstrom."""
        return reciprocal_radius_cycles_per_A_from_rad(
            self.reciprocal_radius_rad_per_A
        )

    @property
    def reciprocal_vectors_cycles_per_A(self) -> np.ndarray:
        """Return reciprocal vectors in crystallographic cycles per angstrom."""
        return self.reciprocal_vectors_rad_per_A / TWO_PI


def reciprocal_matrix_cycles_per_A(cell: gemmi.UnitCell) -> np.ndarray:
    """Return columns ``a*``, ``b*``, ``c*`` in cycles per angstrom."""
    return np.asarray(cell.frac.mat, dtype=np.float64).T


def reciprocal_matrix_rad_per_A(cell: gemmi.UnitCell) -> np.ndarray:
    """Return columns ``a*``, ``b*``, ``c*`` in radians per angstrom.

    Gemmi's ``cell.frac.mat.T`` is the reciprocal matrix in cycles per
    angstrom. Multiplication by ``2*pi`` preserves the historical convention.
    """
    return TWO_PI * reciprocal_matrix_cycles_per_A(cell)


def hkl_to_angular_coordinates(
    hkl: np.ndarray,
    reciprocal_matrix_rad_per_A_value: np.ndarray,
    local_to_global_rotation: np.ndarray | None = None,
) -> AngularCoordinates:
    """Map Miller-index row vectors to polar angles and ``k=2*pi/d``.

    Parameters
    ----------
    hkl
        Integer or real array of shape ``(N, 3)`` interpreted as row vectors.
    reciprocal_matrix_rad_per_A_value
        Matrix ``B`` with reciprocal basis vectors as columns, in rad/A.
    local_to_global_rotation
        Optional right-handed matrix whose columns are local axes represented
        in the crystal-Cartesian frame. The passive row transform is
        ``unit_direction_local = unit_direction_global @ R``.
    """
    indices = np.asarray(hkl)
    matrix = np.asarray(reciprocal_matrix_rad_per_A_value, dtype=np.float64)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("HKL values must have shape (N, 3).")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Reciprocal matrix must be a finite 3 by 3 array.")

    reciprocal_vectors = indices @ matrix.T
    radii = np.linalg.norm(reciprocal_vectors, axis=1)
    directions = reciprocal_vectors / np.maximum(radii[:, None], 1.0e-16)
    if local_to_global_rotation is not None:
        rotation = np.asarray(local_to_global_rotation, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("Local-to-global rotation must be a finite 3 by 3 array.")
        directions = directions @ rotation

    theta = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2.0 * np.pi)
    return AngularCoordinates(theta, phi, radii, reciprocal_vectors, directions)
