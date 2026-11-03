"""Self-describing run-manifest construction for SPH filtering."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import shlex
import sys

import numpy as np

from . import __version__
from .input_validation import ObservationCounts, ReflectionMetadata, sha256_file


def dependency_versions() -> dict[str, str]:
    """Return versions of dependencies that affect numerical or MTZ behavior."""
    versions: dict[str, str] = {}
    for package in (
        "numpy",
        "pandas",
        "scipy",
        "gemmi",
        "reciprocalspaceship",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def resolved_configuration(arguments) -> dict[str, object]:
    """Convert an argparse namespace to stable JSON-compatible values."""
    configuration: dict[str, object] = {}
    for key, value in vars(arguments).items():
        if isinstance(value, Path):
            configuration[key] = str(value)
        elif isinstance(value, tuple):
            configuration[key] = list(value)
        else:
            configuration[key] = value
    return configuration


def build_run_manifest(
    *,
    arguments,
    input_metadata: ReflectionMetadata,
    observation_counts: ObservationCounts,
    expanded_rows_before_resolution: int,
    rows_used: int,
    actual_resolution_range_A: dict[str, float],
    shell_partition: dict[str, object],
    reciprocal_matrix_rad_per_A: np.ndarray,
    output_path: Path,
    output_columns: list[str],
    runtime_seconds: float,
    warnings: list[str],
    symmetry_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the complete scientific and provenance record for one successful run."""
    coordinate_frame: dict[str, object] = {
        "name": "crystal_cartesian_frame",
        "global_frame": "crystal_cartesian_frame",
        "handedness": "right-handed",
        "hkl_vector_convention": "row vector",
        "reciprocal_vector_equation": "K = HKL @ B.T",
        "reciprocal_matrix_definition": "B = 2*pi*A^(-T)",
        "reciprocal_matrix_rad_per_A": np.asarray(
            reciprocal_matrix_rad_per_A, dtype=float
        ).tolist(),
        "angular_definition": "theta=acos(Kz/|K|), phi=atan2(Ky,Kx) mod 2*pi",
        "rotation_convention": "no rotation",
    }

    radial_weight_convention = getattr(
        arguments, "radial_weight_convention", "legacy_k_unscaled"
    )
    radial_weight_symbol = (
        "q" if radial_weight_convention == "crystallographic_q" else "k"
    )
    method = {
        "name": "directional filtering",
        "equation": (
            "G_lm(h) = F(h) * i^l * R_lm(k_hat) * "
            f"W({radial_weight_symbol})"
        ),
        "estimator": "pointwise filtering of each reflection",
        "harmonic_basis": "real orthonormal tesseral harmonics from SciPy",
        "harmonic_normalization": "integral |R_lm|^2 dOmega = 1",
        "l_values": list(range(arguments.lmax + 1)),
        "m_ordering": "ascending m=-l,...,+l",
        "coherent_per_l_sum": {
            "labels": "FYLM_Ll and PHYLM_Ll",
            "equation": "F_l(h) = sum_m G_lm(h)",
            "rotationally_invariant": False,
            "angular_power": False,
            "coordinate_dependent": True,
        },
    }

    return {
        "schema_version": 1,
        "software": {
            "name": "SPH directional filtering",
            "version": __version__,
            "python_version": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "dependency_versions": dependency_versions(),
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "resolved_configuration": resolved_configuration(arguments),
        "method": method,
        "input": {
            **input_metadata.to_dict(),
            "selected_amplitude_column": arguments.sf,
            "selected_phase_column": arguments.ph,
            "selected_amplitude_type": input_metadata.column_types[arguments.sf],
            "selected_phase_type": input_metadata.column_types[arguments.ph],
        },
        "observation_counts": {
            **observation_counts.to_dict(),
            "expanded_rows_before_resolution": expanded_rows_before_resolution,
            "rows_used": rows_used,
        },
        "symmetry": {
            "input_space_group": input_metadata.space_group_hm,
            "p1_expansion": input_metadata.space_group_number != 1,
            "friedel_policy": arguments.symmetry_policy,
            "output_space_group": "P 1",
            "input_contract": symmetry_metadata,
        },
        "resolution": {
            "requested_high_resolution_cutoff_A": arguments.resol,
            "actual_range_A": actual_resolution_range_A,
            "reciprocal_radius_definition": "k = 2*pi/d",
            "reciprocal_radius_units": "radians_per_angstrom",
        },
        "shells": shell_partition,
        "taper": {
            "name": arguments.taper,
            "eta": arguments.eta,
            "beta": arguments.beta,
            "shell_volume_count_weights": arguments.shell_weights,
        },
        "normalization": arguments.norm,
        "hermiticity_enforced": arguments.hermiticity,
        "coordinate_frame": coordinate_frame,
        "warnings": warnings,
        "runtime_seconds": runtime_seconds,
        "output": {
            "path": str(output_path.resolve()),
            "format": "MTZ reciprocal-space directional-filter channels",
            "real_space_map": False,
            "columns": output_columns,
            "sha256": sha256_file(output_path),
            "row_count": rows_used,
        },
    }
