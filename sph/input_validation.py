"""Input validation and missing-observation handling for SPH filtering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Callable, Literal

import gemmi
import numpy as np
import reciprocalspaceship as rs

from .errors import ReflectionInputError, SPHUserError


MissingPolicy = Literal["drop", "error"]


@dataclass(frozen=True)
class ObservationCounts:
    """Counts recorded while validating selected amplitude and phase values."""

    input_rows: int
    invalid_amplitude_rows: int
    invalid_phase_rows: int
    invalid_combined_rows: int
    valid_rows: int
    missing_policy: MissingPolicy

    def to_dict(self) -> dict[str, int | str]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class ReflectionMetadata:
    """Crystallographic metadata read directly from the input MTZ."""

    path: str
    sha256: str
    columns: tuple[str, ...]
    column_types: dict[str, str]
    unit_cell_A_deg: tuple[float, float, float, float, float, float]
    space_group_hm: str
    space_group_number: int
    wavelength_A: float | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        value = asdict(self)
        value["columns"] = list(self.columns)
        value["unit_cell_A_deg"] = list(self.unit_cell_A_deg)
        return value


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_unit_cell(cell: gemmi.UnitCell) -> None:
    parameters = np.asarray(cell.parameters, dtype=np.float64)
    lengths = parameters[:3]
    angles = parameters[3:]
    if not np.isfinite(parameters).all():
        raise ReflectionInputError(
            "The MTZ unit cell contains non-finite values. Supply an MTZ with "
            "complete crystallographic unit-cell metadata."
        )
    if np.any(lengths <= 0.0) or np.any(angles <= 0.0) or np.any(angles >= 180.0):
        raise ReflectionInputError(
            "The MTZ unit cell is invalid: lengths must be positive and angles "
            "must lie strictly between 0 and 180 degrees."
        )
    if not np.isfinite(cell.volume) or cell.volume <= 0.0:
        raise ReflectionInputError(
            "The MTZ unit cell has zero or invalid volume. Check the cell axes "
            "and angles in the input file."
        )


def inspect_mtz(
    input_path: str | Path,
    amplitude_column: str,
    phase_column: str,
) -> tuple[gemmi.Mtz, ReflectionMetadata]:
    """Read and validate MTZ structure, required columns, cell, and space group."""
    path = Path(input_path).expanduser()
    if not path.exists():
        raise ReflectionInputError(
            f"Input MTZ does not exist: {path}. Check --mtz and supply an "
            "existing reflection file."
        )
    if not path.is_file():
        raise ReflectionInputError(
            f"Input MTZ path is not a regular file: {path}. Supply an MTZ file, "
            "not a directory."
        )
    try:
        mtz = gemmi.read_mtz_file(str(path))
    except Exception as error:
        raise ReflectionInputError(
            f"Could not read input MTZ {path}: {error}. Verify that the file is "
            "a complete, uncorrupted MTZ."
        ) from error

    columns = tuple(column.label for column in mtz.columns)
    missing = [
        label
        for label in ("H", "K", "L", amplitude_column, phase_column)
        if label not in columns
    ]
    if missing:
        raise ReflectionInputError(
            "Input MTZ is missing required column(s): "
            f"{', '.join(missing)}. Available columns: {', '.join(columns)}. "
            "Select existing amplitude and phase labels with --sf and --ph."
        )

    column_types = {column.label: column.type for column in mtz.columns}
    amplitude_type = column_types[amplitude_column]
    phase_type = column_types[phase_column]
    if amplitude_type not in {"D", "E", "F", "G"}:
        raise ReflectionInputError(
            f"Column {amplitude_column!r} has MTZ type {amplitude_type!r}, not "
            "an amplitude-like type (D, E, F, or G). Select an amplitude column."
        )
    if phase_type != "P":
        raise ReflectionInputError(
            f"Column {phase_column!r} has MTZ type {phase_type!r}, not phase "
            "type 'P'. Select a phase column in degrees."
        )

    _validate_unit_cell(mtz.cell)
    if mtz.spacegroup is None or mtz.spacegroup.number <= 0:
        raise ReflectionInputError(
            "The MTZ has no valid space group. Supply crystallographic symmetry "
            "metadata before directional filtering."
        )

    wavelength = float(mtz.dataset(0).wavelength) if mtz.datasets else None
    if wavelength is not None and (not np.isfinite(wavelength) or wavelength <= 0.0):
        wavelength = None
    metadata = ReflectionMetadata(
        path=str(path.resolve()),
        sha256=sha256_file(path),
        columns=columns,
        column_types=column_types,
        unit_cell_A_deg=tuple(float(value) for value in mtz.cell.parameters),
        space_group_hm=mtz.spacegroup.hm,
        space_group_number=int(mtz.spacegroup.number),
        wavelength_A=wavelength,
    )
    return mtz, metadata


def load_validated_reflections(
    input_path: str | Path,
    amplitude_column: str,
    phase_column: str,
    missing_policy: MissingPolicy = "drop",
    *,
    track_source_rows: bool = False,
    stage_callback: Callable[..., None] | None = None,
) -> tuple[rs.DataSet, ObservationCounts, ReflectionMetadata]:
    """Load an MTZ and apply the selected missing-data policy before expansion."""
    if missing_policy not in {"drop", "error"}:
        raise ReflectionInputError(
            f"Unsupported missing-data policy {missing_policy!r}; use 'drop' or 'error'."
        )
    mtz, metadata = inspect_mtz(input_path, amplitude_column, phase_column)
    if stage_callback is not None:
        stage_callback(
            "mtz_loading",
            details={"gemmi_rows": mtz.nreflections, "columns": len(mtz.columns)},
        )
    try:
        dataset = rs.read_mtz(metadata.path)
    except Exception as error:
        raise ReflectionInputError(
            f"Reciprocalspaceship could not load {metadata.path}: {error}. "
            "Check MTZ column integrity and integer HKL values."
        ) from error

    if len(dataset) != mtz.nreflections:
        raise ReflectionInputError(
            "MTZ readers disagree on the input row count "
            f"({mtz.nreflections} versus {len(dataset)}). The file may be corrupt."
        )

    amplitude = dataset[amplitude_column].to_numpy(np.float64)
    phase = dataset[phase_column].to_numpy(np.float64)
    invalid_amplitude = ~np.isfinite(amplitude)
    invalid_phase = ~np.isfinite(phase)
    invalid_combined = invalid_amplitude | invalid_phase
    if track_source_rows:
        dataset["_SPH_INPUT_ROW_INDEX"] = np.arange(len(dataset), dtype=np.int64)
    counts = ObservationCounts(
        input_rows=len(dataset),
        invalid_amplitude_rows=int(invalid_amplitude.sum()),
        invalid_phase_rows=int(invalid_phase.sum()),
        invalid_combined_rows=int(invalid_combined.sum()),
        valid_rows=int((~invalid_combined).sum()),
        missing_policy=missing_policy,
    )

    if counts.invalid_combined_rows and missing_policy == "error":
        raise ReflectionInputError(
            "Selected observations contain non-finite values: "
            f"{counts.invalid_amplitude_rows} amplitude row(s), "
            f"{counts.invalid_phase_rows} phase row(s), "
            f"{counts.invalid_combined_rows} combined invalid row(s). "
            "Use --missing-policy drop to exclude them before symmetry expansion, "
            "or repair the selected MTZ columns."
        )
    if counts.valid_rows == 0:
        raise ReflectionInputError(
            "No valid observations remain after checking the selected amplitude "
            f"({amplitude_column}) and phase ({phase_column}) columns."
        )
    if counts.invalid_combined_rows:
        dataset = dataset.iloc[~invalid_combined].copy()

    if stage_callback is not None:
        stage_callback(
            "valid_observation_filtering",
            arrays={"amplitude": amplitude, "phase_deg": phase},
            details=counts.to_dict(),
        )

    return dataset, counts, metadata
