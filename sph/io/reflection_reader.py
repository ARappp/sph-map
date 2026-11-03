"""Validated MTZ loading, expansion, and observation selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import reciprocalspaceship as rs

from sph.config import SPHConfig
from sph.crystallography.symmetry import (
    SymmetryApplication,
    apply_symmetry_policy,
    diagnose_symmetry_input,
)
from sph.errors import ReflectionInputError
from sph.input_validation import (
    ObservationCounts,
    ReflectionMetadata,
    load_validated_reflections,
)


@dataclass(frozen=True)
class ReflectionData:
    """Validated numerical observations and crystallographic metadata.

    Invalid selected amplitudes/phases have been handled before P1/Friedel
    expansion. Arrays represent the final resolution-filtered rows.
    """

    dataset: rs.DataSet
    input_hkl: np.ndarray
    hkl: np.ndarray
    amplitudes: np.ndarray
    phases_deg: np.ndarray
    observation_counts: ObservationCounts
    input_metadata: ReflectionMetadata
    expanded_rows_before_resolution: int
    actual_resolution_range_A: dict[str, float]
    symmetry_application: SymmetryApplication


def ensure_hkl_columns(dataset: rs.DataSet) -> rs.DataSet:
    """Expose integer ``H,K,L`` columns without changing row order."""
    if {"H", "K", "L"}.issubset(dataset.columns):
        dataset[["H", "K", "L"]] = dataset[["H", "K", "L"]].astype(np.int32)
        return dataset
    try:
        index_frame = dataset.index.to_frame(index=False)
        columns = list(index_frame.columns)
        if {"H", "K", "L"}.issubset(columns):
            for name in ("H", "K", "L"):
                dataset[name] = index_frame[name].to_numpy(np.int32)
        elif len(columns) >= 3:
            first_three = index_frame.iloc[:, :3].copy()
            first_three.columns = ["H", "K", "L"]
            for name in ("H", "K", "L"):
                dataset[name] = first_three[name].to_numpy(np.int32)
        else:
            raise KeyError("Index has fewer than 3 levels")
        return dataset
    except Exception:
        pass
    try:
        hkl = dataset.get_hkls().to_numpy(np.int32)
        dataset["H"], dataset["K"], dataset["L"] = hkl[:, 0], hkl[:, 1], hkl[:, 2]
        return dataset
    except Exception as error:
        raise ReflectionInputError(
            f"Could not locate integer HKL columns or index: {error}. "
            "Supply a reflection table with H, K, and L."
        ) from error


def read_reflection_data(
    config: SPHConfig,
    *,
    track_provenance: bool = False,
    stage_callback: Callable[..., None] | None = None,
) -> ReflectionData:
    """Read, validate, expand, and resolution-filter observations."""
    dataset, counts, metadata = load_validated_reflections(
        config.input_mtz,
        config.amplitude_column,
        config.phase_column,
        missing_policy=config.missing_policy,
        track_source_rows=track_provenance,
        stage_callback=stage_callback,
    )
    try:
        input_hkl = np.asarray(dataset.get_hkls(), dtype=np.int32)
    except Exception as error:
        raise ReflectionInputError(
            f"Could not extract input HKLs before symmetry expansion: {error}."
        ) from error
    diagnostics = diagnose_symmetry_input(dataset)
    if stage_callback is not None:
        stage_callback("symmetry_diagnostics", details=diagnostics.to_dict())
    symmetry_application = apply_symmetry_policy(
        dataset,
        config.symmetry_policy,
        diagnostics=diagnostics,
        track_provenance=track_provenance,
        source_dataset=str(config.input_mtz),
        source_file_sha256=metadata.sha256,
    )
    dataset = symmetry_application.dataset
    if stage_callback is not None:
        stage_callback(
            "p1_friedel_expansion",
            details={
                "input_rows": diagnostics.input_rows,
                "expanded_rows": len(dataset),
                "provenance_tracked": track_provenance,
            },
        )
    dataset = ensure_hkl_columns(dataset.reset_index())
    expanded_count = len(dataset)

    if config.resolution_cutoff_A > 0.0:
        indices = dataset[["H", "K", "L"]].to_numpy(np.int32)
        spacings = np.asarray(
            [dataset.cell.calculate_d(list(map(int, item))) for item in indices],
            dtype=np.float64,
        )
        keep = np.isfinite(spacings) & (spacings >= config.resolution_cutoff_A)
        dataset = dataset.iloc[keep].reset_index(drop=True)
        if symmetry_application.provenance is not None:
            symmetry_application = SymmetryApplication(
                **{
                    **{
                        field: getattr(symmetry_application, field)
                        for field in (
                            "dataset", "requested_policy", "applied_policy", "diagnostics",
                            "rows_after_space_group_expansion", "rows_after_friedel_handling",
                            "synthesized_friedel_rows", "preserved_input_rows", "warnings",
                        )
                    },
                    "dataset": dataset,
                    "provenance": symmetry_application.provenance.subset(keep),
                }
            )
    if len(dataset) == 0:
        raise ReflectionInputError(
            "No reflections remain after the resolution cutoff. Relax --resol "
            "or check the input unit cell and HKL values."
        )

    indices = dataset[["H", "K", "L"]].to_numpy(np.int32)
    amplitudes = dataset[config.amplitude_column].to_numpy(np.float64)
    phases = dataset[config.phase_column].to_numpy(np.float64)
    spacings = np.asarray(
        [dataset.cell.calculate_d(list(map(int, item))) for item in indices],
        dtype=np.float64,
    )
    finite_spacings = spacings[np.isfinite(spacings) & (spacings > 0.0)]
    if finite_spacings.size == 0:
        raise ReflectionInputError(
            "No finite positive resolution values can be calculated from the "
            "selected unit cell and HKL rows."
        )
    return ReflectionData(
        dataset=dataset,
        input_hkl=input_hkl,
        hkl=indices,
        amplitudes=amplitudes,
        phases_deg=phases,
        observation_counts=counts,
        input_metadata=metadata,
        expanded_rows_before_resolution=expanded_count,
        actual_resolution_range_A={
            "low": float(np.max(finite_spacings)),
            "high": float(np.min(finite_spacings)),
        },
        symmetry_application=symmetry_application,
    )
