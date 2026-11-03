"""Pure numerical core for reciprocal-space directional filtering."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator
from typing import Literal

import numpy as np

from sph.crystallography.coordinates import (
    AngularCoordinates,
    hkl_to_angular_coordinates,
)
from sph.harmonics.indexing import HarmonicIndex, harmonic_indices
from sph.harmonics.real import evaluate_real_harmonic
from sph.shelling import (
    LEGACY_K_TO_Q_VOLUME_SCALE,
    ShellPartition,
    create_shell_partition,
    shell_volume_weights,
)

from .hermiticity import enforce_hermiticity
from .normalization import normalization_factors
from sph.crystallography.phase_conventions import coefficients_for_positive_synthesis


@dataclass(frozen=True)
class FilteringOptions:
    """Validated numerical choices consumed by the pure filtering core."""

    lmax: int
    shell_mode: Literal["linear", "log", "quantile"] = "log"
    shell_count: int = 80
    minimum_per_shell: int = 50
    taper: Literal["none", "cosine", "kaiser"] = "cosine"
    taper_eta: float = 0.9
    taper_beta: float = 8.0
    shell_weights: bool = True
    normalization: Literal["none", "per-l", "per-lm"] = "per-l"
    hermiticity: bool = False
    radial_weight_convention: Literal[
        "legacy_k_unscaled", "crystallographic_q"
    ] = "legacy_k_unscaled"
    phase_convention: Literal["historical", "ccp4"] = "historical"


@dataclass(frozen=True)
class HarmonicChannel:
    """Complex pointwise values for one canonical ``(l,m)`` channel."""

    index: HarmonicIndex
    complex_values: np.ndarray


@dataclass(frozen=True)
class CoherentDegreeSum:
    """Coordinate-dependent coherent sum over all ``m`` at one degree."""

    degree_l: int
    complex_values: np.ndarray


@dataclass(frozen=True)
class DirectionalFilterResult:
    """Complete numerical result with no file-format or CLI dependencies."""

    channels: tuple[HarmonicChannel, ...]
    coherent_degree_sums: tuple[CoherentDegreeSum, ...]
    shell_partition: ShellPartition
    shell_weights: np.ndarray
    global_coordinates: AngularCoordinates
    selected_coordinates: AngularCoordinates

    def channel(self, degree_l: int, order_m: int) -> np.ndarray:
        """Return a channel by scientific index or raise ``KeyError``."""
        for channel in self.channels:
            if channel.index == HarmonicIndex(degree_l, order_m):
                return channel.complex_values
        raise KeyError((degree_l, order_m))

    def coherent_sum(self, degree_l: int) -> np.ndarray:
        """Return the backward-compatible coherent degree sum."""
        for coherent in self.coherent_degree_sums:
            if coherent.degree_l == degree_l:
                return coherent.complex_values
        raise KeyError(degree_l)


@dataclass(frozen=True)
class DirectionalFilterContext:
    """Shared `O(N)` state for materialized or channel-streamed filtering."""

    hkl: np.ndarray
    complex_coefficients: np.ndarray
    options: FilteringOptions
    shell_partition: ShellPartition
    shell_weights: np.ndarray
    global_coordinates: AngularCoordinates
    selected_coordinates: AngularCoordinates
    inverse_normalization_per_lm: dict[tuple[int, int], float]
    inverse_normalization_per_l: dict[int, float]


def prepare_directional_filter(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    phases_deg: np.ndarray,
    reciprocal_matrix_rad_per_A: np.ndarray,
    options: FilteringOptions,
) -> DirectionalFilterContext:
    """Validate inputs and prepare shared coordinates, weights, and coefficients."""
    indices = np.asarray(hkl)
    amplitude_values = np.asarray(amplitudes, dtype=np.float64)
    phase_values_deg = np.asarray(phases_deg, dtype=np.float64)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("HKL values must have shape (N, 3).")
    if len(indices) != len(amplitude_values) or len(indices) != len(phase_values_deg):
        raise ValueError("HKL, amplitude, and phase arrays must have equal length.")
    if not np.isfinite(amplitude_values).all() or not np.isfinite(phase_values_deg).all():
        raise ValueError("Filtering inputs must contain only finite amplitudes and phases.")
    if not 0 <= options.lmax <= 1:
        raise ValueError("lmax must be 0 or 1.")
    if options.radial_weight_convention not in {
        "legacy_k_unscaled", "crystallographic_q"
    }:
        raise ValueError(
            "Radial-weight convention must be legacy_k_unscaled or crystallographic_q."
        )

    global_coordinates = hkl_to_angular_coordinates(
        indices, reciprocal_matrix_rad_per_A
    )
    selected_coordinates = global_coordinates
    partition = create_shell_partition(
        global_coordinates.reciprocal_radius_rad_per_A,
        mode=options.shell_mode,
        shell_count=options.shell_count,
        minimum_per_shell=options.minimum_per_shell,
        radial_weight_convention=options.radial_weight_convention,
    )
    legacy_k_weights = shell_volume_weights(
        global_coordinates.reciprocal_radius_rad_per_A,
        partition,
        taper=options.taper,
        eta=options.taper_eta,
        beta=options.taper_beta,
        radial_weight_convention="legacy_k_unscaled",
    )
    if options.shell_weights:
        weights = (
            legacy_k_weights
            if options.radial_weight_convention == "legacy_k_unscaled"
            else legacy_k_weights / LEGACY_K_TO_Q_VOLUME_SCALE
        )
        # The measure-scale conversion must not perturb the existing
        # dimensionless angular sampling normalization.
        normalization_weights = legacy_k_weights
    else:
        weights = np.ones_like(legacy_k_weights)
        normalization_weights = weights

    coefficients = coefficients_for_positive_synthesis(
        amplitude_values, phase_values_deg, convention=options.phase_convention
    )
    if options.hermiticity:
        coefficients = enforce_hermiticity(indices, coefficients)

    inverse_per_lm, inverse_per_l = normalization_factors(
        selected_coordinates.polar_theta_rad,
        selected_coordinates.azimuth_phi_rad,
        options.lmax,
        mode=options.normalization,
        weights=normalization_weights * normalization_weights,
    )
    return DirectionalFilterContext(
        indices,
        coefficients,
        options,
        partition,
        weights,
        global_coordinates,
        selected_coordinates,
        inverse_per_lm,
        inverse_per_l,
    )


def iter_directional_filter_outputs(
    context: DirectionalFilterContext,
) -> Iterator[HarmonicChannel | CoherentDegreeSum]:
    """Yield each channel then its degree sum in canonical order."""
    options = context.options
    coordinates = context.selected_coordinates
    for degree in range(options.lmax + 1):
        coherent = np.zeros(len(context.hkl), dtype=np.complex128)
        for index in harmonic_indices(degree):
            if index.degree_l != degree:
                continue
            harmonic = (1j**degree) * evaluate_real_harmonic(
                index.order_m,
                degree,
                coordinates.azimuth_phi_rad,
                coordinates.polar_theta_rad,
            )
            if options.normalization == "per-lm":
                harmonic *= context.inverse_normalization_per_lm.get(
                    (degree, index.order_m), 1.0
                )
            elif options.normalization == "per-l":
                harmonic *= context.inverse_normalization_per_l.get(degree, 1.0)
            values = context.complex_coefficients * harmonic * context.shell_weights
            coherent += values
            yield HarmonicChannel(index, values)
        yield CoherentDegreeSum(degree, coherent)


def iter_real_map_components(
    context: DirectionalFilterContext,
) -> Iterator[HarmonicChannel]:
    """Yield the real tesseral component channels used for real P1 maps."""
    for item in iter_directional_filter_outputs(context):
        if isinstance(item, HarmonicChannel):
            yield item


def filter_directional_reflections(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    phases_deg: np.ndarray,
    reciprocal_matrix_rad_per_A: np.ndarray,
    options: FilteringOptions,
) -> DirectionalFilterResult:
    """Apply the real directional filter pointwise.

    This function performs no fitting, file I/O, command-line parsing,
    printing, global mutation, or output naming. Amplitudes and phases must
    already be finite and symmetry/resolution processing must already be done.
    The filter is exactly ``F(h)i^l R_lm(khat)W(k)`` for l = 0 and 1.
    """
    context = prepare_directional_filter(
        hkl,
        amplitudes,
        phases_deg,
        reciprocal_matrix_rad_per_A,
        options,
    )
    return build_directional_channels(context)


def build_directional_channels(
    context: DirectionalFilterContext,
) -> DirectionalFilterResult:
    """Materialize all channels from a prepared directional-filter context.

    This collects pointwise filter outputs. It does not fit or project
    coefficients across reflections or shells.
    """
    outputs = tuple(iter_directional_filter_outputs(context))
    channels = tuple(item for item in outputs if isinstance(item, HarmonicChannel))
    coherent_sums = tuple(
        item for item in outputs if isinstance(item, CoherentDegreeSum)
    )

    return DirectionalFilterResult(
        channels,
        coherent_sums,
        context.shell_partition,
        context.shell_weights,
        context.global_coordinates,
        context.selected_coordinates,
    )


__all__ = [
    "CoherentDegreeSum",
    "DirectionalFilterContext",
    "DirectionalFilterResult",
    "FilteringOptions",
    "HarmonicChannel",
    "build_directional_channels",
    "filter_directional_reflections",
    "iter_directional_filter_outputs",
    "iter_real_map_components",
    "prepare_directional_filter",
]
