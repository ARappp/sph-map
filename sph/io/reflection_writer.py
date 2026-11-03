"""MTZ construction for pure directional-filtering results."""

from __future__ import annotations

import gemmi
import numpy as np
import reciprocalspaceship as rs

from sph.analysis.filtering import (
    CoherentDegreeSum,
    DirectionalFilterContext,
    DirectionalFilterResult,
    HarmonicChannel,
    iter_directional_filter_outputs,
)
from sph.harmonics.indexing import channel_labels, coherent_labels


def infer_label_prefix(amplitude_column: str, phase_column: str) -> str:
    """Infer the label prefix from selected input labels."""
    amplitude = amplitude_column.upper()
    _phase = phase_column.upper()
    if amplitude in ("FWT", "2FOFCWT", "F2FOFC"):
        return "DENS_"
    if amplitude in ("DELFWT", "FOFCWT", "FOFC"):
        return "DIFF_"
    if amplitude in ("DANO", "DIFANO", "FANO"):
        return "ANOM_"
    return amplitude + "_"


def _base_output_dataset(source_dataset: rs.DataSet) -> rs.DataSet:
    output = rs.DataSet()
    output["H"] = source_dataset["H"].astype("H")
    output["K"] = source_dataset["K"].astype("K")
    output["L"] = source_dataset["L"].astype("L")
    output = output.set_index(["H", "K", "L"])
    output.cell = source_dataset.cell
    output.spacegroup = gemmi.find_spacegroup_by_number(1)
    if hasattr(source_dataset, "wavelength"):
        output.wavelength = getattr(source_dataset, "wavelength")
    return output


def _add_complex_columns(
    output: rs.DataSet,
    amplitude_label: str,
    phase_label: str,
    values: np.ndarray,
) -> None:
    """Add one complex vector using the exact amplitude/phase conversion."""
    output[amplitude_label] = np.abs(values).astype(np.float64)
    output[phase_label] = np.rad2deg(np.angle(values)).astype(np.float64)
    output[[amplitude_label]] = output[[amplitude_label]].astype("SFAmplitude")
    output[[phase_label]] = output[[phase_label]].astype("Phase")


def build_output_dataset(
    source_dataset: rs.DataSet,
    calculation: DirectionalFilterResult,
    label_prefix: str,
) -> rs.DataSet:
    """Construct the materialized P1 MTZ table without writing it."""
    output = _base_output_dataset(source_dataset)

    for channel in calculation.channels:
        amplitude_label, phase_label = channel_labels(label_prefix, channel.index)
        _add_complex_columns(
            output, amplitude_label, phase_label, channel.complex_values
        )

        if channel.index.order_m == channel.index.degree_l:
            coherent_values = calculation.coherent_sum(channel.index.degree_l)
            coherent_amplitude, coherent_phase = coherent_labels(
                label_prefix, channel.index.degree_l
            )
            _add_complex_columns(
                output, coherent_amplitude, coherent_phase, coherent_values
            )
    return output


def build_output_dataset_low_memory(
    source_dataset: rs.DataSet,
    context: DirectionalFilterContext,
    label_prefix: str,
) -> rs.DataSet:
    """Build the same MTZ table while retaining only one complex output at a time."""
    output = _base_output_dataset(source_dataset)
    for item in iter_directional_filter_outputs(context):
        if isinstance(item, HarmonicChannel):
            amplitude_label, phase_label = channel_labels(label_prefix, item.index)
        elif isinstance(item, CoherentDegreeSum):
            amplitude_label, phase_label = coherent_labels(
                label_prefix, item.degree_l
            )
        else:  # pragma: no cover - closed internal event union
            raise TypeError(f"Unexpected directional-filter output: {type(item)!r}")
        _add_complex_columns(
            output, amplitude_label, phase_label, item.complex_values
        )
    return output
