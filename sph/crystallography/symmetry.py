"""Explicit symmetry and Friedel-input contracts for ordinary data.

This module does not interpret anomalous differences.  Every policy either
preserves rows literally or applies the ordinary non-anomalous relation
``F(-h) = conj(F(h))`` through reciprocalspaceship's phase-aware expansion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import gemmi
import numpy as np
import reciprocalspaceship as rs

from sph.errors import SymmetryInputError
from sph.crystallography.provenance import (
    ObservationProvenance,
    ProvenanceFlag,
    canonical_friedel_hkl,
    compact_group_ids,
)


SymmetryPolicy = Literal[
    "legacy_expand", "asu_to_p1", "p1_preserve",
    "p1_complete_friedel", "reject_ambiguous",
]


def _hkl_tuples(dataset: rs.DataSet) -> list[tuple[int, int, int]]:
    values = np.asarray(dataset.get_hkls(), dtype=np.int32)
    return [tuple(map(int, row)) for row in values]


def canonical_friedel_pair(hkl: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return a deterministic identifier shared by ``h`` and ``-h``."""
    mate = tuple(-value for value in hkl)
    return min(hkl, mate)


@dataclass(frozen=True)
class SymmetryDiagnostics:
    """Row-identity and crystallographic-symmetry diagnostics before expansion."""

    input_rows: int
    unique_hkls: int
    unique_friedel_pairs: int
    duplicate_hkl_rows: int
    friedel_pairs_with_both_members: int
    rows_in_complete_friedel_pairs: int
    centric_rows: int
    acentric_rows: int
    space_group_hm: str
    space_group_number: int
    input_is_p1: bool
    merged_flag: bool | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible diagnostic record."""
        return asdict(self)


@dataclass(frozen=True)
class SymmetryApplication:
    """Result and audit trail from applying one explicit input policy."""

    dataset: rs.DataSet
    requested_policy: SymmetryPolicy
    applied_policy: str
    diagnostics: SymmetryDiagnostics
    rows_after_space_group_expansion: int
    rows_after_friedel_handling: int
    synthesized_friedel_rows: int
    preserved_input_rows: int
    warnings: tuple[tuple[str, str], ...]
    provenance: ObservationProvenance | None = None

    def to_dict(self) -> dict[str, object]:
        """Return manifest metadata without embedding the reflection table."""
        record: dict[str, object] = {
            "requested_policy": self.requested_policy,
            "applied_policy": self.applied_policy,
            "input_diagnostics": self.diagnostics.to_dict(),
            "rows_after_space_group_expansion": self.rows_after_space_group_expansion,
            "rows_after_friedel_handling": self.rows_after_friedel_handling,
            "synthesized_friedel_rows": self.synthesized_friedel_rows,
            "preserved_input_rows": self.preserved_input_rows,
            "friedel_relation_assumed": (
                "F(-h)=conj(F(h))" if self.synthesized_friedel_rows else None
            ),
        }
        # Preserve the stable directional-filter manifest schema when
        # provenance tracking is not requested.
        if self.provenance is not None:
            record["provenance"] = self.provenance.summary()
        return record


_PROVENANCE_COLUMNS = (
    "_SPH_INPUT_ROW_INDEX",
    "_SPH_SOURCE_ROW",
    "_SPH_PRE_FRIEDEL_ROW",
)


@dataclass(frozen=True)
class _SourceProvenanceLookup:
    source_row_index: np.ndarray
    source_hkl: np.ndarray
    duplicate_hkl: np.ndarray
    complete_friedel: np.ndarray
    merged: np.ndarray


def _attach_source_provenance(
    dataset: rs.DataSet,
) -> tuple[rs.DataSet, _SourceProvenanceLookup]:
    """Attach one propagated source ID and retain compact source lookups."""
    output = dataset.copy()
    hkl = np.asarray(output.get_hkls(), dtype=np.int32)
    _, inverse, counts = np.unique(hkl, axis=0, return_inverse=True, return_counts=True)
    hkl_set = {tuple(map(int, row)) for row in np.unique(hkl, axis=0)}
    complete_friedel = np.array(
        [
            np.any(row != 0) and tuple(map(int, -row)) in hkl_set
            for row in hkl
        ],
        dtype=np.int8,
    )
    source_rows = (
        output["_SPH_INPUT_ROW_INDEX"].to_numpy(np.int64)
        if "_SPH_INPUT_ROW_INDEX" in output.columns
        else np.arange(len(output), dtype=np.int64)
    )
    output["_SPH_SOURCE_ROW"] = source_rows
    if "_SPH_INPUT_ROW_INDEX" in output.columns:
        output = output.drop(columns=["_SPH_INPUT_ROW_INDEX"])
    return output, _SourceProvenanceLookup(
        source_row_index=source_rows,
        source_hkl=hkl,
        duplicate_hkl=(counts[inverse] > 1),
        complete_friedel=complete_friedel.astype(bool),
        merged=np.full(len(output), bool(getattr(dataset, "merged", False))),
    )


def _mark_pre_friedel(dataset: rs.DataSet) -> tuple[rs.DataSet, np.ndarray]:
    output = dataset.copy()
    pre_friedel_hkl = np.asarray(output.get_hkls(), dtype=np.int32)
    output["_SPH_PRE_FRIEDEL_ROW"] = np.arange(len(output), dtype=np.int64)
    return output, pre_friedel_hkl


def _extract_provenance(
    dataset: rs.DataSet,
    *,
    source_lookup: _SourceProvenanceLookup,
    pre_friedel_hkl_lookup: np.ndarray,
    source_dataset: str,
    source_file_sha256: str,
) -> ObservationProvenance:
    final_hkl = np.asarray(dataset.get_hkls(), dtype=np.int32)
    source_rows = dataset["_SPH_SOURCE_ROW"].to_numpy(np.int64)
    source_positions = np.searchsorted(source_lookup.source_row_index, source_rows)
    if not np.array_equal(source_lookup.source_row_index[source_positions], source_rows):
        raise RuntimeError("Expanded provenance contains an unknown source row ID.")
    source_hkl = source_lookup.source_hkl[source_positions]
    pre_positions = dataset["_SPH_PRE_FRIEDEL_ROW"].to_numpy(np.int64)
    pre_friedel_hkl = pre_friedel_hkl_lookup[pre_positions]
    symmetry_generated = np.any(pre_friedel_hkl != source_hkl, axis=1)
    friedel_generated = np.any(final_hkl != pre_friedel_hkl, axis=1)
    input_complete = source_lookup.complete_friedel[source_positions]
    flags = np.zeros(len(dataset), dtype=np.uint8)
    flags[~symmetry_generated & ~friedel_generated] |= int(ProvenanceFlag.ORIGINAL_INDEX)
    flags[symmetry_generated] |= int(ProvenanceFlag.SYMMETRY_GENERATED)
    flags[friedel_generated] |= int(ProvenanceFlag.FRIEDEL_GENERATED)
    flags[input_complete & ~friedel_generated] |= int(ProvenanceFlag.PRESERVED_FRIEDEL)
    flags[source_lookup.duplicate_hkl[source_positions]] |= int(
        ProvenanceFlag.DUPLICATE_INPUT_HKL
    )
    flags[input_complete] |= int(ProvenanceFlag.DUPLICATE_INPUT_FRIEDEL_PAIR)
    flags[source_lookup.merged[source_positions]] |= int(
        ProvenanceFlag.MERGED_INPUT
    )
    return ObservationProvenance(
        source_row_index=source_rows,
        source_hkl=source_hkl,
        pre_friedel_hkl=pre_friedel_hkl,
        final_hkl=final_hkl,
        source_group_id=compact_group_ids(source_rows[:, None]),
        exact_hkl_group_id=compact_group_ids(final_hkl),
        friedel_group_id=compact_group_ids(canonical_friedel_hkl(final_hkl)),
        flags=flags,
        source_dataset=source_dataset,
        source_file_sha256=source_file_sha256,
    )


def diagnose_symmetry_input(dataset: rs.DataSet) -> SymmetryDiagnostics:
    """Inspect exact HKL duplicates, Friedel coverage, and centricity."""
    hkls = np.asarray(dataset.get_hkls(), dtype=np.int32)
    unique_hkls = np.unique(hkls, axis=0)

    def canonical_rows(rows: np.ndarray) -> np.ndarray:
        canonical = rows.copy()
        nonzero = canonical != 0
        has_nonzero = np.any(nonzero, axis=1)
        first_positions = np.argmax(nonzero, axis=1)
        first_values = canonical[np.arange(len(canonical)), first_positions]
        flip = has_nonzero & (first_values > 0)
        canonical[flip] *= -1
        return canonical

    canonical_unique = canonical_rows(unique_hkls)
    pair_values, unique_member_counts = np.unique(
        canonical_unique, axis=0, return_counts=True
    )
    canonical_all = canonical_rows(hkls)
    all_pair_values, input_pair_counts = np.unique(
        canonical_all, axis=0, return_counts=True
    )
    if not np.array_equal(pair_values, all_pair_values):  # pragma: no cover
        raise RuntimeError("Internal Friedel diagnostic indexing mismatch")
    non_origin = np.any(pair_values != 0, axis=1)
    complete_mask = (unique_member_counts == 2) & non_origin
    complete_pairs = int(np.count_nonzero(complete_mask))
    rows_in_complete = int(np.sum(input_pair_counts[complete_mask]))
    spacegroup = dataset.spacegroup
    operations = spacegroup.operations()
    centric = int(np.count_nonzero(operations.centric_flag_array(hkls)))
    merged = getattr(dataset, "merged", None)
    return SymmetryDiagnostics(
        input_rows=len(hkls),
        unique_hkls=len(unique_hkls),
        unique_friedel_pairs=len(pair_values),
        duplicate_hkl_rows=len(hkls) - len(unique_hkls),
        friedel_pairs_with_both_members=complete_pairs,
        rows_in_complete_friedel_pairs=rows_in_complete,
        centric_rows=centric,
        acentric_rows=len(hkls) - centric,
        space_group_hm=spacegroup.hm,
        space_group_number=int(spacegroup.number),
        input_is_p1=int(spacegroup.number) == 1,
        merged_flag=bool(merged) if merged is not None else None,
    )


def _reject_duplicates(diagnostics: SymmetryDiagnostics, policy: str) -> None:
    if diagnostics.duplicate_hkl_rows:
        raise SymmetryInputError(
            "[SPH-E004] The input contains "
            f"{diagnostics.duplicate_hkl_rows} duplicate HKL row(s), which is "
            f"ambiguous under --symmetry-policy {policy}. Merge or deduplicate "
            "the observations, or use p1_preserve only when literal row "
            "preservation is scientifically intended."
        )


def _complete_p1_friedel(dataset: rs.DataSet) -> tuple[rs.DataSet, int]:
    expanded = dataset.expand_anomalous()
    expanded_hkls = _hkl_tuples(expanded)
    keep = np.empty(len(expanded_hkls), dtype=bool)
    seen: set[tuple[int, int, int]] = set()
    for index, hkl in enumerate(expanded_hkls):
        keep[index] = hkl not in seen
        seen.add(hkl)
    completed = expanded.iloc[keep].copy()
    return completed, len(completed) - len(dataset)


def apply_symmetry_policy(
    dataset: rs.DataSet,
    policy: SymmetryPolicy = "legacy_expand",
    *,
    diagnostics: SymmetryDiagnostics | None = None,
    track_provenance: bool = False,
    source_dataset: str = "unrecorded",
    source_file_sha256: str = "unrecorded",
) -> SymmetryApplication:
    """Validate and apply a symmetry/Friedel policy without changing defaults."""
    diagnostics = diagnostics or diagnose_symmetry_input(dataset)
    source_lookup = None
    pre_friedel_hkl_lookup = None
    if track_provenance:
        dataset, source_lookup = _attach_source_provenance(dataset)
    warnings: list[tuple[str, str]] = []
    requested = policy

    if policy == "reject_ambiguous":
        _reject_duplicates(diagnostics, policy)
        if diagnostics.friedel_pairs_with_both_members:
            raise SymmetryInputError(
                "[SPH-E004] Both members of one or more Friedel pairs are "
                "present, so ordinary and anomalous interpretations cannot be "
                "distinguished safely. Select p1_preserve explicitly for literal "
                "P1 rows, or provide merged ordinary ASU data."
            )
        policy = "p1_complete_friedel" if diagnostics.input_is_p1 else "asu_to_p1"

    if policy == "asu_to_p1":
        if diagnostics.input_is_p1:
            raise SymmetryInputError(
                "[SPH-E004] --symmetry-policy asu_to_p1 requires a non-P1 "
                "merged ASU input. For P1 use p1_preserve or p1_complete_friedel."
            )
        _reject_duplicates(diagnostics, policy)
        if diagnostics.friedel_pairs_with_both_members:
            raise SymmetryInputError(
                "[SPH-E004] The non-P1 input contains both members of Friedel "
                "pairs. This may be anomalous or already-expanded data and is "
                "outside the ordinary merged-ASU contract."
            )
        p1 = dataset.expand_to_p1()
        after_space_group = len(p1)
        if track_provenance:
            p1, pre_friedel_hkl_lookup = _mark_pre_friedel(p1)
        result = p1.expand_anomalous()
        synthesized = len(result) - after_space_group
        applied = "asu_to_p1"
    elif policy == "p1_preserve":
        if not diagnostics.input_is_p1:
            raise SymmetryInputError(
                "[SPH-E004] --symmetry-policy p1_preserve requires input space "
                "group P 1; it cannot discard nontrivial symmetry metadata."
            )
        if track_provenance:
            result, pre_friedel_hkl_lookup = _mark_pre_friedel(dataset)
        else:
            result = dataset.copy()
        after_space_group = len(result)
        synthesized = 0
        applied = "p1_preserve"
        if diagnostics.duplicate_hkl_rows:
            warnings.append((
                "SPH-W006",
                f"Preserving {diagnostics.duplicate_hkl_rows} duplicate HKL row(s) literally.",
            ))
    elif policy == "p1_complete_friedel":
        if not diagnostics.input_is_p1:
            raise SymmetryInputError(
                "[SPH-E004] --symmetry-policy p1_complete_friedel requires "
                "input space group P 1."
            )
        _reject_duplicates(diagnostics, policy)
        after_space_group = len(dataset)
        if track_provenance:
            dataset, pre_friedel_hkl_lookup = _mark_pre_friedel(dataset)
        result, synthesized = _complete_p1_friedel(dataset)
        applied = "p1_complete_friedel"
        if synthesized:
            warnings.append((
                "SPH-W005",
                f"Synthesized {synthesized} missing Friedel mate row(s) using F(-h)=conj(F(h)).",
            ))
    elif policy == "legacy_expand":
        p1 = dataset.expand_to_p1() if not diagnostics.input_is_p1 else dataset
        after_space_group = len(p1)
        if track_provenance:
            p1, pre_friedel_hkl_lookup = _mark_pre_friedel(p1)
        result = p1.expand_anomalous()
        synthesized = len(result) - after_space_group
        applied = "legacy_expand"
        warnings.append((
            "SPH-W004",
            "Applied backward-compatible expansion: optional ASU-to-P1 followed by expand_anomalous().",
        ))
        if diagnostics.friedel_pairs_with_both_members or diagnostics.duplicate_hkl_rows:
            warnings.append((
                "SPH-W006",
                "Legacy expansion received duplicate HKLs or existing Friedel pairs; output rows may be duplicated. Select an explicit symmetry policy after reviewing the input contract.",
            ))
    else:
        raise SymmetryInputError(f"[SPH-E004] Unsupported symmetry policy: {policy!r}.")

    provenance = (
        _extract_provenance(
            result,
            source_lookup=source_lookup,
            pre_friedel_hkl_lookup=pre_friedel_hkl_lookup,
            source_dataset=source_dataset,
            source_file_sha256=source_file_sha256,
        )
        if track_provenance
        else None
    )
    if track_provenance:
        # Provenance is now represented by compact NumPy arrays. Keeping the
        # propagation markers in the expanded DataSet duplicates the same
        # information and materially increases peak memory for large inputs.
        result = result.drop(
            columns=[name for name in _PROVENANCE_COLUMNS if name in result.columns]
        )
    return SymmetryApplication(
        dataset=result,
        requested_policy=requested,
        applied_policy=applied,
        diagnostics=diagnostics,
        rows_after_space_group_expansion=after_space_group,
        rows_after_friedel_handling=len(result),
        synthesized_friedel_rows=synthesized,
        preserved_input_rows=len(dataset),
        warnings=tuple(warnings),
        provenance=provenance,
    )
