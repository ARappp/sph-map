"""Pure P1 Fourier synthesis for directional-filter reflection channels."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import gemmi
import numpy as np
from scipy.fft import next_fast_len

from sph.errors import SPHUserError


class MapSynthesisError(SPHUserError):
    """Raised when a real, alias-safe directional map cannot be synthesized."""


@dataclass(frozen=True)
class DirectionalMapGrid:
    """Complete full-cell P1 grid convention."""

    shape: tuple[int, int, int]
    target_spacing_A: float
    axis_step_lengths_A: tuple[float, float, float]
    minimum_alias_safe_shape: tuple[int, int, int]
    rounding_rule: str
    axis_order: str = "array[u,v,w] = crystallographic[a,b,c]"
    fractional_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fourier_sign: str = "+2*pi*i*h_dot_r"
    fourier_normalization: str = "1/unit_cell_volume_A3"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible grid record."""
        return {
            "shape": list(self.shape),
            "target_spacing_A": self.target_spacing_A,
            "axis_step_lengths_A": list(self.axis_step_lengths_A),
            "minimum_alias_safe_shape": list(self.minimum_alias_safe_shape),
            "rounding_rule": self.rounding_rule,
            "axis_order": self.axis_order,
            "fractional_origin": list(self.fractional_origin),
            "fourier_sign": self.fourier_sign,
            "fourier_normalization": self.fourier_normalization,
            "space_group": "P 1",
            "periodic_full_cell": True,
            "missing_reflections": "zero_filled",
            "duplicate_hkls": "summed",
        }


@dataclass(frozen=True)
class DirectionalDensityMap:
    """One validated real directional density grid in float64 precision."""

    values: np.ndarray
    grid: DirectionalMapGrid
    imaginary_residual_max: float
    imaginary_tolerance: float
    input_observation_count: int
    unique_hkl_count: int

    def statistics(self) -> dict[str, float | int]:
        """Return finite full-cell statistics without rescaling values."""
        values = np.asarray(self.values, dtype=np.float64)
        return {
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values)),
            "rms": float(np.sqrt(np.mean(values * values))),
            "imaginary_residual_max": self.imaginary_residual_max,
            "imaginary_tolerance": self.imaginary_tolerance,
            "input_observation_count": self.input_observation_count,
            "unique_hkl_count": self.unique_hkl_count,
            "duplicate_observation_count": (
                self.input_observation_count - self.unique_hkl_count
            ),
        }


def _validated_hkl(hkl: np.ndarray) -> np.ndarray:
    indices = np.asarray(hkl)
    if indices.ndim != 2 or indices.shape[1] != 3 or len(indices) == 0:
        raise MapSynthesisError("Map synthesis requires a non-empty HKL array of shape (N, 3).")
    if not np.isfinite(indices).all() or not np.equal(indices, np.rint(indices)).all():
        raise MapSynthesisError("Map synthesis requires finite integer Miller indices.")
    return indices.astype(np.int64, copy=False)


def directional_map_grid(
    cell: gemmi.UnitCell,
    hkl: np.ndarray,
    *,
    d_min_A: float,
    samples_per_d_min: float = 3.0,
    exact_shape: Sequence[int] | None = None,
) -> DirectionalMapGrid:
    """Return the approved upward-rounded, alias-safe P1 grid."""
    indices = _validated_hkl(hkl)
    if not math.isfinite(d_min_A) or d_min_A <= 0.0:
        raise MapSynthesisError("Map d_min must be finite and positive in angstrom.")
    if not math.isfinite(samples_per_d_min) or samples_per_d_min <= 0.0:
        raise MapSynthesisError("Map samples-per-d_min must be finite and positive.")
    cell_lengths = np.asarray([cell.a, cell.b, cell.c], dtype=np.float64)
    if np.any(~np.isfinite(cell_lengths)) or np.any(cell_lengths <= 0.0) or cell.volume <= 0.0:
        raise MapSynthesisError("Map synthesis requires a finite positive-volume unit cell.")
    maximum_indices = np.max(np.abs(indices), axis=0)
    alias_safe = tuple(int(2 * value + 1) for value in maximum_indices)
    target_spacing = d_min_A / samples_per_d_min
    spacing_minimum = np.ceil(cell_lengths / target_spacing).astype(np.int64)
    minimum = np.maximum(spacing_minimum, np.asarray(alias_safe, dtype=np.int64))

    if exact_shape is None:
        shape = tuple(int(next_fast_len(int(value))) for value in minimum)
        rounding = (
            "max(ceil(axis_length/(d_min/samples_per_d_min)), "
            "2*max_abs_hkl+1), then scipy.fft.next_fast_len upward"
        )
    else:
        if len(exact_shape) != 3:
            raise MapSynthesisError("Exact map grid must contain three dimensions (nu,nv,nw).")
        shape = tuple(int(value) for value in exact_shape)
        if any(value <= 0 for value in shape):
            raise MapSynthesisError("Exact map-grid dimensions must be positive integers.")
        if any(actual < required for actual, required in zip(shape, alias_safe, strict=True)):
            raise MapSynthesisError(
                "Exact map grid aliases represented Miller indices; require at least "
                f"{alias_safe}, received {shape}."
            )
        rounding = "user-supplied exact dimensions; validated against 2*max_abs_hkl+1"

    return DirectionalMapGrid(
        shape=shape,
        target_spacing_A=float(target_spacing),
        axis_step_lengths_A=tuple(
            float(length / count) for length, count in zip(cell_lengths, shape, strict=True)
        ),
        minimum_alias_safe_shape=alias_safe,
        rounding_rule=rounding,
    )


def synthesize_filtered_density(
    hkl: np.ndarray,
    complex_values: np.ndarray,
    cell: gemmi.UnitCell,
    grid: DirectionalMapGrid,
) -> DirectionalDensityMap:
    """Synthesize one real full-cell map using positive-sign P1 Fourier synthesis.

    Missing reciprocal indices remain zero. Duplicate HKL rows are summed.
    A non-negligible imaginary result is rejected rather than discarded.
    """
    indices = _validated_hkl(hkl)
    coefficients = np.asarray(complex_values, dtype=np.complex128)
    if coefficients.ndim != 1 or len(coefficients) != len(indices):
        raise MapSynthesisError("HKL and complex map-coefficient arrays must have equal length.")
    if not np.isfinite(coefficients.real).all() or not np.isfinite(coefficients.imag).all():
        raise MapSynthesisError("Map coefficients must contain only finite complex values.")
    alias_safe = np.asarray(grid.minimum_alias_safe_shape)
    if np.any(np.asarray(grid.shape) < alias_safe):
        raise MapSynthesisError("Map grid is smaller than its declared alias-safe dimensions.")
    if not math.isfinite(cell.volume) or cell.volume <= 0.0:
        raise MapSynthesisError("Map synthesis requires a positive unit-cell volume.")

    reciprocal_grid = np.zeros(grid.shape, dtype=np.complex128)
    positions = np.mod(indices, np.asarray(grid.shape, dtype=np.int64))
    np.add.at(
        reciprocal_grid,
        (positions[:, 0], positions[:, 1], positions[:, 2]),
        coefficients,
    )
    point_count = int(np.prod(grid.shape, dtype=np.int64))
    density = np.fft.ifftn(reciprocal_grid) * (point_count / float(cell.volume))

    imaginary_max = float(np.max(np.abs(density.imag)))
    fft_depth = sum(math.log2(max(value, 2)) for value in grid.shape)
    roundoff_bound = (
        64.0
        * np.finfo(np.float64).eps
        * max(1.0, fft_depth)
        * max(1.0, float(np.sum(np.abs(coefficients))) / float(cell.volume))
    )
    if imaginary_max > roundoff_bound:
        raise MapSynthesisError(
            "Filtered coefficients do not synthesize a real map: maximum "
            f"imaginary value {imaginary_max:.6g} exceeds the complex128 "
            f"roundoff bound {roundoff_bound:.6g}. Review Friedel completion, "
            "the symmetry policy, Hermiticity, and complex-basis conversion."
        )

    real_values = np.asarray(density.real, dtype=np.float64)
    unique_count = int(len(np.unique(indices, axis=0)))
    return DirectionalDensityMap(
        values=real_values,
        grid=grid,
        imaginary_residual_max=imaginary_max,
        imaginary_tolerance=float(roundoff_bound),
        input_observation_count=len(indices),
        unique_hkl_count=unique_count,
    )


__all__ = [
    "DirectionalDensityMap",
    "DirectionalMapGrid",
    "MapSynthesisError",
    "directional_map_grid",
    "synthesize_filtered_density",
]
