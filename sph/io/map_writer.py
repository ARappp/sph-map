"""Transactional CCP4 writing for validated real directional-density maps."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Iterable

import gemmi
import numpy as np

from sph.analysis.map_synthesis import DirectionalDensityMap
from sph.errors import OutputError
from sph.harmonics.indexing import HarmonicIndex, harmonic_indices
from sph.input_validation import sha256_file
from sph.output import commit_temporary_output, validate_output_paths


@dataclass(frozen=True)
class DirectionalMapOutput:
    """One named map and its requested final path."""

    name: str
    output_path: Path
    density: DirectionalDensityMap
    degree_l: int | None = None
    order_m: int | None = None
    kind: str = "directional_component"


def directional_component_map_path(
    output_mtz: str | Path,
    map_directory: str | Path,
    index: HarmonicIndex,
) -> Path:
    """Return the deterministic CCP4 filename for one component channel."""
    stem = Path(output_mtz).stem
    return Path(map_directory).expanduser() / (
        f"{stem}.L{index.degree_l}M{index.order_m:+d}.ccp4"
    )


def requested_directional_map_paths(
    output_mtz: str | Path,
    map_directory: str | Path,
    lmax: int,
    *,
    conventional_control: bool,
) -> tuple[Path, ...]:
    """Return every requested map path in canonical component order."""
    paths = [
        directional_component_map_path(output_mtz, map_directory, index)
        for index in harmonic_indices(lmax)
    ]
    if conventional_control:
        paths.append(
            Path(map_directory).expanduser()
            / f"{Path(output_mtz).stem}.CONVENTIONAL.ccp4"
        )
    return tuple(paths)


def build_ccp4_map(
    density: DirectionalDensityMap,
    cell: gemmi.UnitCell,
    *,
    label: str,
) -> gemmi.Ccp4Map:
    """Build a P1 CCP4 map with crystallographic a/b/c axis order."""
    values = np.asarray(density.values, dtype=np.float64)
    if values.shape != density.grid.shape or not np.isfinite(values).all():
        raise OutputError("Directional map values must be finite and match the declared grid.")
    float32_values = np.asfortranarray(values.astype(np.float32))
    grid = gemmi.FloatGrid(
        float32_values,
        cell,
        gemmi.find_spacegroup_by_number(1),
    )
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header(2, True)
    # CCP4/MRC header words use one-based numbering in Gemmi.  Make the
    # documented full-cell origin and sampling contract explicit instead of
    # relying on writer defaults.
    for header_word in (5, 6, 7):
        ccp4.set_header_i32(header_word, 0)
    for header_word, sample_count in zip((8, 9, 10), density.grid.shape):
        ccp4.set_header_i32(header_word, sample_count)
    ccp4.set_header_i32(17, 1)
    ccp4.set_header_i32(18, 2)
    ccp4.set_header_i32(19, 3)
    for header_word in (50, 51, 52):
        ccp4.set_header_float(header_word, 0.0)
    ccp4.set_header_i32(56, 1)
    ccp4.set_header_str(57, label[:80])
    return ccp4


def _validate_written_map(
    path: Path,
    expected: DirectionalDensityMap,
    cell: gemmi.UnitCell,
) -> dict[str, object]:
    written = gemmi.read_ccp4_map(str(path))
    written.setup(0.0)
    shape = (written.grid.nu, written.grid.nv, written.grid.nw)
    if shape != expected.grid.shape:
        raise OutputError(
            f"CCP4 round trip changed grid shape from {expected.grid.shape} to {shape}."
        )
    if written.axis_positions() != [0, 1, 2]:
        raise OutputError(
            "CCP4 round trip did not retain MAPC/MAPR/MAPS = 1/2/3."
        )
    start_indices = [written.header_i32(word) for word in (5, 6, 7)]
    if start_indices != [0, 0, 0]:
        raise OutputError("CCP4 round trip did not retain zero start indices.")
    sampling_counts = [written.header_i32(word) for word in (8, 9, 10)]
    if sampling_counts != list(expected.grid.shape):
        raise OutputError(
            "CCP4 round trip did not retain full-cell sampling counts."
        )
    cartesian_origin_A = [written.header_float(word) for word in (50, 51, 52)]
    if cartesian_origin_A != [0.0, 0.0, 0.0]:
        raise OutputError("CCP4 round trip did not retain the Cartesian origin.")
    actual_cell = written.grid.unit_cell
    expected_cell = np.asarray(cell.parameters, dtype=np.float64)
    actual_parameters = np.asarray(actual_cell.parameters, dtype=np.float64)
    if not np.allclose(actual_parameters, expected_cell, rtol=0.0, atol=2.0e-5):
        raise OutputError("CCP4 round trip changed unit-cell metadata.")
    stored = np.asarray(written.grid, dtype=np.float32)
    expected_float32 = np.asarray(expected.values, dtype=np.float32)
    if not np.array_equal(stored, expected_float32):
        raise OutputError("CCP4 round trip changed float32 map values.")
    return {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "shape": list(shape),
        "axis_positions_zero_based": written.axis_positions(),
        "mapc_mapr_maps": [1, 2, 3],
        "start_indices": start_indices,
        "sampling_counts": sampling_counts,
        "cartesian_origin_A": cartesian_origin_A,
        "space_group": "P 1",
        "storage_dtype": "float32",
    }


def write_directional_map_bundle(
    outputs: Iterable[DirectionalMapOutput],
    cell: gemmi.UnitCell,
    *,
    overwrite: bool,
) -> tuple[dict[str, object], ...]:
    """Write and validate all maps before installing any requested final path."""
    pending: list[tuple[Path, DirectionalMapOutput, dict[str, object]]] = []
    temporary_paths: list[Path] = []
    final_paths: list[Path] = []
    try:
        for output in outputs:
            target = Path(output.output_path).expanduser()
            final_paths.append(target)
            validate_output_paths(final_paths, overwrite=overwrite)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp.ccp4", dir=target.parent
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            temporary_paths.append(temporary)
            ccp4 = build_ccp4_map(output.density, cell, label=output.name)
            ccp4.write_ccp4_map(str(temporary))
            round_trip = _validate_written_map(temporary, output.density, cell)
            pending.append((temporary, output, round_trip))

        records: list[dict[str, object]] = []
        for temporary, output, round_trip in pending:
            commit_temporary_output(
                temporary, output.output_path, overwrite=overwrite
            )
            final_path = Path(output.output_path).expanduser()
            records.append({
                "name": output.name,
                "kind": output.kind,
                "degree_l": output.degree_l,
                "order_m": output.order_m,
                "path": str(final_path.resolve()),
                **round_trip,
                "sha256": sha256_file(final_path),
                "statistics_float64_before_storage": output.density.statistics(),
            })
        return tuple(records)
    except OutputError:
        raise
    except Exception as error:
        raise OutputError(
            f"Failed while writing directional CCP4 map bundle: {error}. "
            "No uncommitted temporary map is retained."
        ) from error
    finally:
        for temporary in temporary_paths:
            if temporary.exists():
                temporary.unlink()


__all__ = [
    "DirectionalMapOutput",
    "build_ccp4_map",
    "directional_component_map_path",
    "requested_directional_map_paths",
    "write_directional_map_bundle",
]
