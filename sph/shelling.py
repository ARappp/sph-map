"""Resolution-shell partitioning in reciprocal radians per angstrom."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .analysis.tapering import cosine_taper, kaiser_taper
from .errors import SPHUserError


ShellMode = Literal["linear", "log", "quantile"]
RadialWeightConvention = Literal["legacy_k_unscaled", "crystallographic_q"]
RADIAL_UNITS = "radians_per_angstrom"
LEGACY_K_TO_Q_VOLUME_SCALE = (2.0 * np.pi) ** 3


class ShellPartitionError(SPHUserError):
    """Raised when reciprocal radii cannot define a valid shell partition."""


@dataclass(frozen=True)
class ShellPartition:
    """Complete shell geometry and assignments for reciprocal observations."""

    edges_rad_per_A: np.ndarray
    centers_rad_per_A: np.ndarray
    widths_rad_per_A: np.ndarray
    assignments: np.ndarray
    populations: np.ndarray
    mode: ShellMode
    radial_units: str
    interval_convention: str
    warnings: tuple[str, ...]
    radial_weight_convention: RadialWeightConvention = "legacy_k_unscaled"

    @property
    def shell_count(self) -> int:
        """Return the actual number of shells after any linear-mode merging."""
        return len(self.edges_rad_per_A) - 1

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible scientific description of the partition."""
        record: dict[str, object] = {
            "mode": self.mode,
            "radial_units": self.radial_units,
            "reciprocal_radius_definition": "k = 2*pi/d",
            "interval_convention": self.interval_convention,
            "shell_count": self.shell_count,
            "edges_rad_per_A": self.edges_rad_per_A.tolist(),
            "centers_rad_per_A": self.centers_rad_per_A.tolist(),
            "widths_rad_per_A": self.widths_rad_per_A.tolist(),
            "populations": self.populations.astype(int).tolist(),
            "warnings": list(self.warnings),
        }
        if self.radial_weight_convention == "crystallographic_q":
            record.update({
                "radial_weight_convention": "crystallographic_q",
                "weight_radius_definition": "q = 1/d",
                "weight_radius_units": "cycles_per_angstrom",
                "edges_cycles_per_A": (
                    self.edges_rad_per_A / (2.0 * np.pi)
                ).tolist(),
                "centers_cycles_per_A": (
                    self.centers_rad_per_A / (2.0 * np.pi)
                ).tolist(),
                "widths_cycles_per_A": (
                    self.widths_rad_per_A / (2.0 * np.pi)
                ).tolist(),
            })
        return record


def _assign_to_edges(radii: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if np.any(radii < edges[0]) or np.any(radii > edges[-1]):
        below = int(np.count_nonzero(radii < edges[0]))
        above = int(np.count_nonzero(radii > edges[-1]))
        raise ShellPartitionError(
            "Reciprocal radii fall outside the declared shell range: "
            f"{below} below the first edge and {above} above the last edge. "
            "Shell assignment will not clip observations silently."
        )
    assignments = np.searchsorted(edges, radii, side="right") - 1
    assignments[radii == edges[-1]] = len(edges) - 2
    return assignments.astype(np.int64, copy=False)


def _linear_edges_with_forward_merge(
    radii: np.ndarray,
    shell_count: int,
    minimum_per_shell: int,
) -> np.ndarray:
    edges = np.linspace(0.0, float(np.max(radii)), shell_count + 1)
    assignments = _assign_to_edges(radii, edges)
    populations = np.bincount(assignments, minlength=shell_count).tolist()
    merged_edges = edges.tolist()
    index = 0
    while index < len(populations):
        if populations[index] < minimum_per_shell and index < len(populations) - 1:
            merged_edges.pop(index + 1)
            populations[index + 1] += populations[index]
            populations.pop(index)
        else:
            index += 1
    return np.asarray(merged_edges, dtype=np.float64)


def create_shell_partition(
    reciprocal_radius_rad_per_A: np.ndarray,
    mode: ShellMode = "log",
    shell_count: int = 80,
    minimum_per_shell: int = 50,
    radial_weight_convention: RadialWeightConvention = "legacy_k_unscaled",
) -> ShellPartition:
    """Partition finite reciprocal radii using explicit boundary semantics."""
    radii = np.asarray(reciprocal_radius_rad_per_A, dtype=np.float64)
    if radii.ndim != 1 or radii.size == 0:
        raise ShellPartitionError("Shell construction requires a non-empty 1-D radius array.")
    nonfinite_count = int(np.count_nonzero(~np.isfinite(radii)))
    if nonfinite_count:
        raise ShellPartitionError(
            f"Reciprocal radii contain {nonfinite_count} non-finite value(s). "
            "Check HKL values and the reciprocal-cell matrix."
        )
    if np.any(radii < 0.0):
        raise ShellPartitionError("Reciprocal radii must be non-negative.")
    if shell_count < 1 or minimum_per_shell < 1:
        raise ShellPartitionError("Shell count and minimum population must be positive.")
    if mode not in {"linear", "log", "quantile"}:
        raise ShellPartitionError(
            f"Unsupported shell mode {mode!r}; use linear, log, or quantile."
        )
    if radial_weight_convention not in {
        "legacy_k_unscaled", "crystallographic_q"
    }:
        raise ShellPartitionError(
            "Unsupported radial-weight convention; use legacy_k_unscaled or "
            "crystallographic_q."
        )

    minimum = float(np.min(radii))
    maximum = float(np.max(radii))
    if maximum <= minimum:
        raise ShellPartitionError(
            "Shell construction requires at least two distinct reciprocal radii."
        )

    if mode == "linear":
        edges = _linear_edges_with_forward_merge(
            radii, shell_count, minimum_per_shell
        )
    elif mode == "quantile":
        edges = np.quantile(radii, np.linspace(0.0, 1.0, shell_count + 1))
        edges = np.unique(np.round(edges, 12))
        if len(edges) < 2:
            raise ShellPartitionError(
                "Quantile shells collapsed to fewer than two distinct boundaries."
            )
        edges[0] = minimum
        edges[-1] = maximum
    else:
        if minimum <= 0.0:
            zero_count = int(np.count_nonzero(radii <= 0.0))
            raise ShellPartitionError(
                "Logarithmic shells require strictly positive reciprocal radii; "
                f"found {zero_count} zero-radius observation(s), for which angular "
                "direction is undefined. Remove the origin reflection."
            )
        edges = np.geomspace(minimum, maximum, shell_count + 1)
        edges[0] = minimum
        edges[-1] = maximum

    assignments = _assign_to_edges(radii, edges)
    populations = np.bincount(assignments, minlength=len(edges) - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    partition_warnings: list[str] = []
    empty = np.flatnonzero(populations == 0)
    if empty.size:
        partition_warnings.append(
            "Empty shell(s): " + ", ".join(str(int(index)) for index in empty)
        )
    low = np.flatnonzero((populations > 0) & (populations < minimum_per_shell))
    if low.size:
        partition_warnings.append(
            f"Shell(s) below minimum population {minimum_per_shell}: "
            + ", ".join(
                f"{int(index)} ({int(populations[index])})" for index in low
            )
        )

    return ShellPartition(
        edges_rad_per_A=edges,
        centers_rad_per_A=centers,
        widths_rad_per_A=widths,
        assignments=assignments,
        populations=populations,
        mode=mode,
        radial_units=RADIAL_UNITS,
        interval_convention="[lower, upper), final shell [lower, upper]",
        warnings=tuple(partition_warnings),
        radial_weight_convention=radial_weight_convention,
    )


def shell_volume_weights(
    reciprocal_radius_rad_per_A: np.ndarray,
    partition: ShellPartition,
    taper: Literal["none", "cosine", "kaiser"] = "cosine",
    eta: float = 0.9,
    beta: float = 8.0,
    radial_weight_convention: RadialWeightConvention | None = None,
) -> np.ndarray:
    """Return volume/count weights under an explicit radius convention.

    ``legacy_k_unscaled`` uses ``4*pi*k_s^2*dk_s/N_s`` with
    ``k=2*pi/d`` and preserves historical outputs. ``crystallographic_q``
    uses ``4*pi*q_s^2*dq_s/N_s`` with ``q=1/d``. The measures differ only by
    the global factor ``(2*pi)^3`` before their common taper.
    """
    radii = np.asarray(reciprocal_radius_rad_per_A, dtype=np.float64)
    if radii.shape != partition.assignments.shape:
        raise ShellPartitionError(
            "Radius and shell-assignment arrays have inconsistent dimensions."
        )
    maximum = float(partition.edges_rad_per_A[-1])
    if taper == "none":
        taper_values = np.ones_like(radii)
    elif taper == "cosine":
        taper_values = cosine_taper(radii, maximum, eta)
    elif taper == "kaiser":
        taper_values = kaiser_taper(radii, maximum, beta)
    else:
        raise ShellPartitionError(
            f"Unsupported taper {taper!r}; use none, cosine, or kaiser."
        )

    safe_populations = np.where(partition.populations == 0, 1, partition.populations)
    per_shell_legacy_k = (
        4.0
        * np.pi
        * partition.centers_rad_per_A**2
        * partition.widths_rad_per_A
        / safe_populations
    )
    convention = radial_weight_convention or partition.radial_weight_convention
    legacy_observation_weights = (
        per_shell_legacy_k[partition.assignments] * taper_values
    )
    if convention == "legacy_k_unscaled":
        return legacy_observation_weights
    elif convention == "crystallographic_q":
        return legacy_observation_weights / LEGACY_K_TO_Q_VOLUME_SCALE
    else:
        raise ShellPartitionError(
            "Unsupported radial-weight convention; use legacy_k_unscaled or "
            "crystallographic_q."
        )
