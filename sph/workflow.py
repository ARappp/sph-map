"""I/O orchestration for one SPH directional-filtering run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import gemmi
import numpy as np

from sph.analysis.filtering import (
    FilteringOptions,
    build_directional_channels,
    iter_real_map_components,
    prepare_directional_filter,
)
from sph.analysis.map_synthesis import (
    directional_map_grid,
    synthesize_filtered_density,
)
from sph.config import SPHConfig
from sph.crystallography.coordinates import reciprocal_matrix_rad_per_A
from sph.io.reflection_reader import ReflectionData, read_reflection_data
from sph.io.map_writer import (
    DirectionalMapOutput,
    directional_component_map_path,
    requested_directional_map_paths,
    write_directional_map_bundle,
)
from sph.io.reflection_writer import (
    build_output_dataset,
    build_output_dataset_low_memory,
    infer_label_prefix,
)
from sph.metadata import build_run_manifest
from sph.crystallography.phase_conventions import phase_convention_metadata
from sph.output import (
    transactional_write_json,
    transactional_write_mtz,
    transactional_write_text,
    validate_output_paths,
)
from sph.reporting.logging import RunLogger
from sph.reporting.reports import render_run_readme


@dataclass(frozen=True)
class WorkflowResult:
    """Paths and counts returned by a successful complete workflow."""

    output_mtz: Path
    manifest_path: Path
    input_rows: int
    valid_rows: int
    expanded_rows: int
    output_rows: int
    warnings: tuple[str, ...]
    runtime_seconds: float
    directional_map_paths: tuple[Path, ...] = ()


def _requested_output_paths(config: SPHConfig) -> tuple[Path, ...]:
    paths = [config.output_mtz, config.resolved_manifest_path]
    if config.timings and config.timings_json:
        paths.append(config.timings_json)
    if config.write_readme:
        paths.append(config.write_readme)
    if config.directional_map_directory is not None:
        paths.extend(
            requested_directional_map_paths(
                config.output_mtz,
                config.directional_map_directory,
                config.lmax,
                conventional_control=config.map_conventional_control,
            )
        )
    return tuple(paths)


def _write_directional_maps(
    config: SPHConfig,
    context,
    cell: gemmi.UnitCell,
    *,
    d_min_A: float,
) -> tuple[dict[str, object], ...]:
    """Synthesize and transactionally write the requested real P1 map bundle."""
    if config.directional_map_directory is None:
        return ()
    grid = directional_map_grid(
        cell,
        context.hkl,
        d_min_A=d_min_A,
        samples_per_d_min=config.map_samples_per_d_min,
        exact_shape=config.map_grid_shape,
    )

    def outputs():
        for channel in iter_real_map_components(context):
            index = channel.index
            yield DirectionalMapOutput(
                name=f"SPH directional component L{index.degree_l} M{index.order_m}",
                output_path=directional_component_map_path(
                    config.output_mtz,
                    config.directional_map_directory,
                    index,
                ),
                density=synthesize_filtered_density(
                    context.hkl, channel.complex_values, cell, grid
                ),
                degree_l=index.degree_l,
                order_m=index.order_m,
            )
        if config.map_conventional_control:
            target = (
                config.directional_map_directory
                / f"{config.output_mtz.stem}.CONVENTIONAL.ccp4"
            )
            yield DirectionalMapOutput(
                name="SPH conventional F/PHI control",
                output_path=target,
                density=synthesize_filtered_density(
                    context.hkl, context.complex_coefficients, cell, grid
                ),
                kind="unfiltered_conventional_control",
            )

    records = write_directional_map_bundle(
        outputs(), cell, overwrite=config.overwrite
    )
    grid_metadata = grid.to_dict()
    return tuple(
        {
            **record,
            "grid": grid_metadata,
            "phase_convention": phase_convention_metadata(config.phase_convention),
            "basis": "real tesseral",
            "fourier_equation": (
                "rho(u,v,w)=(1/V)*sum_h G(h)*exp(+2*pi*i*(h*u+k*v+l*w))"
            ),
            "units_interpretation": (
                "relative_or_normalized_angularly_filtered_density"
                if config.shell_weights == "on"
                else "conventional Fourier scaling; absolute units not independently validated"
            ),
        }
        for record in records
    )


def _log_input(logger: RunLogger, config: SPHConfig, data: ReflectionData) -> None:
    counts = data.observation_counts
    logger.info(f"Input: {config.input_mtz}")
    logger.info(
        f"Columns: amplitude={config.amplitude_column}, phase={config.phase_column}"
    )
    logger.info(
        f"Observations: input={counts.input_rows}, valid={counts.valid_rows}, "
        f"expanded={data.expanded_rows_before_resolution}, used={len(data.hkl)}"
    )
    logger.info(
        "Resolution (A): "
        f"{data.actual_resolution_range_A['low']:.6g} to "
        f"{data.actual_resolution_range_A['high']:.6g}"
    )


def run_workflow(
    config: SPHConfig,
    *,
    logger: RunLogger | None = None,
) -> WorkflowResult:
    """Run validated reflection I/O, pure filtering, and transactional output."""
    started = time.perf_counter()
    config.validate()
    logger = logger or RunLogger(verbose=config.verbose)
    validate_output_paths(_requested_output_paths(config), overwrite=config.overwrite)
    if config.directional_map_directory is not None:
        logger.warning(
            "SPH-W015",
            (
                "Shell-weighted directional maps are relative or normalized "
                "angularly filtered density; absolute e/A^3 units have not "
                "been independently validated."
                if config.shell_weights == "on"
                else "Directional maps use conventional 1/V Fourier scaling, "
                "but absolute e/A^3 units have not been independently validated."
            ),
        )

    reflection_data = read_reflection_data(config)
    _log_input(logger, config, reflection_data)
    for code, message in reflection_data.symmetry_application.warnings:
        logger.warning(code, message)
    counts = reflection_data.observation_counts
    if counts.invalid_combined_rows:
        logger.warning(
            "SPH-W001",
            f"Dropped {counts.invalid_combined_rows} input row(s) with non-finite "
            "selected amplitude or phase values before expansion.",
        )

    reciprocal_matrix = reciprocal_matrix_rad_per_A(reflection_data.dataset.cell)
    logger.info(
        f"Harmonics: l=0..{config.lmax}, real basis; "
        f"shells={config.shell_mode}"
    )
    logger.info("Coordinate frame: crystal_cartesian_frame")
    logger.detail(
        "Reciprocal matrix (rad/A): "
        + np.array2string(reciprocal_matrix, precision=8, suppress_small=False)
    )

    filtering_started = time.perf_counter()
    filtering_options = FilteringOptions(
        lmax=config.lmax,
        phase_convention=config.phase_convention,
        shell_mode=config.shell_mode,
        shell_count=config.shell_count,
        minimum_per_shell=config.minimum_per_shell,
        taper=config.taper,
        taper_eta=config.taper_eta,
        taper_beta=config.taper_beta,
        shell_weights=config.shell_weights == "on",
        normalization=config.normalization,
        hermiticity=config.hermiticity,
        radial_weight_convention=config.radial_weight_convention,
    )
    prefix = infer_label_prefix(config.amplitude_column, config.phase_column)
    context = prepare_directional_filter(
        reflection_data.hkl,
        reflection_data.amplitudes,
        reflection_data.phases_deg,
        reciprocal_matrix,
        filtering_options,
    )
    calculation = None
    if config.output_strategy == "low-memory":
        output_dataset = build_output_dataset_low_memory(
            reflection_data.dataset, context, prefix
        )
        shell_partition = context.shell_partition
        shell_weights = context.shell_weights
    else:
        calculation = build_directional_channels(context)
        output_dataset = build_output_dataset(
            reflection_data.dataset, calculation, prefix
        )
        shell_partition = calculation.shell_partition
        shell_weights = calculation.shell_weights
    filtering_seconds = time.perf_counter() - filtering_started
    for warning in shell_partition.warnings:
        code = "SPH-W002" if warning.startswith("Empty shell") else "SPH-W003"
        logger.warning(code, warning)
    if config.verbose:
        logger.detail(
            "Shell populations: "
            + np.array2string(shell_partition.populations)
        )
        logger.detail(
            "Shell weights: min="
            f"{float(np.min(shell_weights)):.8g}, max="
            f"{float(np.max(shell_weights)):.8g}"
        )

    # Capture the small values needed after serialization. When maps are not
    # requested, context is still retained only through the MTZ write and is
    # released immediately afterward.
    input_metadata = reflection_data.input_metadata
    expanded_rows_before_resolution = reflection_data.expanded_rows_before_resolution
    rows_used = len(reflection_data.hkl)
    actual_resolution_range_A = reflection_data.actual_resolution_range_A
    symmetry_metadata = reflection_data.symmetry_application.to_dict()
    shell_metadata = shell_partition.to_dict()
    map_cell = reflection_data.dataset.cell
    if config.directional_map_directory is None:
        # Release expanded inputs before Gemmi allocates MTZ writer buffers.
        del context, calculation, shell_weights, shell_partition, reflection_data
    transactional_write_mtz(
        output_dataset, config.output_mtz, overwrite=config.overwrite
    )
    logger.info(f"Output: {config.output_mtz}")
    del output_dataset

    map_started = time.perf_counter()
    if config.directional_map_directory is None:
        map_records = ()
    else:
        map_records = _write_directional_maps(
            config,
            context,
            map_cell,
            d_min_A=actual_resolution_range_A["high"],
        )
        del context, calculation, shell_weights, shell_partition, reflection_data
    map_seconds = time.perf_counter() - map_started if map_records else 0.0
    if map_records:
        logger.info(
            f"Directional maps: {len(map_records)} CCP4 file(s) in "
            f"{config.directional_map_directory}"
        )

    timings = {
        f"filter_{prefix}": filtering_seconds,
        "directional_map_synthesis_and_serialization": map_seconds,
        "total_run": time.perf_counter() - started,
    }
    if config.timings:
        logger.info(
            f"Timing: filter={filtering_seconds:.3f}s, total={timings['total_run']:.3f}s"
        )
        if config.timings_json:
            transactional_write_json(
                timings, config.timings_json, overwrite=config.overwrite
            )
    if config.write_readme:
        transactional_write_text(
            render_run_readme(config, logger.warnings),
            config.write_readme,
            overwrite=config.overwrite,
        )

    written_mtz = gemmi.read_mtz_file(str(config.output_mtz))
    manifest = build_run_manifest(
        arguments=config.to_legacy_namespace(),
        input_metadata=input_metadata,
        observation_counts=counts,
        expanded_rows_before_resolution=expanded_rows_before_resolution,
        rows_used=rows_used,
        actual_resolution_range_A=actual_resolution_range_A,
        shell_partition=shell_metadata,
        reciprocal_matrix_rad_per_A=reciprocal_matrix,
        output_path=config.output_mtz,
        output_columns=[column.label for column in written_mtz.columns],
        runtime_seconds=time.perf_counter() - started,
        warnings=logger.warnings,
        symmetry_metadata=symmetry_metadata,
    )
    manifest["architecture"] = {
        "name": "separated-workflow",
        "configuration_api": "sph.config.SPHConfig",
        "numerical_api": "sph.analysis.filtering.filter_directional_reflections",
        "output_strategy": config.output_strategy,
    }
    manifest["resolved_configuration_descriptive"] = config.to_manifest_dict()
    manifest["radial_weighting_convention"] = {
        "selected": config.radial_weight_convention,
        "crystallographic_radius": "q = 1/d in cycles_per_angstrom",
        "wavevector_radius": "k = 2*pi*q in radians_per_angstrom",
        "legacy_to_q_global_scale": float((2.0 * np.pi) ** 3),
        "legacy_definition": "4*pi*k_s^2*delta_k_s/N_s without (2*pi)^-3",
        "crystallographic_q_definition": "4*pi*q_s^2*delta_q_s/N_s",
        "shell_weights_enabled": config.shell_weights == "on",
        "map_units_claim": "relative_or_normalized_when_shell_weighted",
    }
    manifest["phase_convention"] = phase_convention_metadata(config.phase_convention)
    manifest["directional_maps"] = {
        "enabled": bool(map_records),
        "count": len(map_records),
        "format": "CCP4 full-cell P1 float32" if map_records else None,
        "component_count": sum(
            record["kind"] == "directional_component" for record in map_records
        ),
        "conventional_control": config.map_conventional_control,
        "records": list(map_records),
        "runtime_seconds": map_seconds,
        "absolute_e_per_A3_validated": False,
    }
    manifest["output"]["real_space_map"] = bool(map_records)
    manifest["output"]["directional_map_count"] = len(map_records)
    transactional_write_json(
        manifest, config.resolved_manifest_path, overwrite=config.overwrite
    )

    total_seconds = time.perf_counter() - started
    return WorkflowResult(
        config.output_mtz,
        config.resolved_manifest_path,
        counts.input_rows,
        counts.valid_rows,
        expanded_rows_before_resolution,
        rows_used,
        tuple(logger.warnings),
        total_seconds,
        tuple(Path(record["path"]) for record in map_records),
    )
