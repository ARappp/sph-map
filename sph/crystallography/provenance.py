"""Compact row provenance for symmetry-expanded observations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag

import numpy as np


class ProvenanceFlag(IntFlag):
    """Composable transformation and input-status bits for one final row."""

    ORIGINAL_INDEX = 1
    SYMMETRY_GENERATED = 2
    FRIEDEL_GENERATED = 4
    PRESERVED_FRIEDEL = 8
    DUPLICATE_INPUT_HKL = 16
    DUPLICATE_INPUT_FRIEDEL_PAIR = 32
    MERGED_INPUT = 64


@dataclass(frozen=True)
class ObservationProvenance:
    """Compact source/final identities and derivation groups.

    All arrays are row aligned with the final reflection table. Source-file
    identity is stored once rather than repeated per row. ``source_group_id``
    is the original valid input-row index and is the conservative dependence
    group for deterministic symmetry/Friedel derivatives.
    """

    source_row_index: np.ndarray
    source_hkl: np.ndarray
    pre_friedel_hkl: np.ndarray
    final_hkl: np.ndarray
    source_group_id: np.ndarray
    exact_hkl_group_id: np.ndarray
    friedel_group_id: np.ndarray
    flags: np.ndarray
    source_dataset: str
    source_file_sha256: str
    shell_assignment: np.ndarray | None = None
    repeated_direction_group_id: np.ndarray | None = None

    def __post_init__(self) -> None:
        row_count = len(self.source_row_index)
        one_dimensional = (
            self.source_group_id,
            self.exact_hkl_group_id,
            self.friedel_group_id,
            self.flags,
        )
        if any(np.asarray(value).shape != (row_count,) for value in one_dimensional):
            raise ValueError("Provenance group and flag arrays must be row aligned.")
        if any(
            np.asarray(value).shape != (row_count, 3)
            for value in (self.source_hkl, self.pre_friedel_hkl, self.final_hkl)
        ):
            raise ValueError("Provenance HKL arrays must have shape (N,3).")
        if self.shell_assignment is not None and np.asarray(self.shell_assignment).shape != (row_count,):
            raise ValueError("Provenance shell assignments must be row aligned.")
        if self.repeated_direction_group_id is not None and np.asarray(
            self.repeated_direction_group_id
        ).shape != (row_count,):
            raise ValueError("Provenance direction groups must be row aligned.")

    @property
    def row_count(self) -> int:
        """Return final observation count."""
        return len(self.source_row_index)

    def has_flag(self, flag: ProvenanceFlag) -> np.ndarray:
        """Return the row mask containing *flag*."""
        return (self.flags.astype(np.uint8) & int(flag)) != 0

    def subset(self, keep: np.ndarray) -> "ObservationProvenance":
        """Return provenance rows selected by a Boolean mask or integer index."""
        selection = np.asarray(keep)
        return ObservationProvenance(
            source_row_index=self.source_row_index[selection],
            source_hkl=self.source_hkl[selection],
            pre_friedel_hkl=self.pre_friedel_hkl[selection],
            final_hkl=self.final_hkl[selection],
            source_group_id=compact_group_ids(
                self.source_row_index[selection, None]
            ),
            exact_hkl_group_id=compact_group_ids(self.final_hkl[selection]),
            friedel_group_id=compact_group_ids(
                canonical_friedel_hkl(self.final_hkl[selection])
            ),
            flags=self.flags[selection],
            source_dataset=self.source_dataset,
            source_file_sha256=self.source_file_sha256,
            shell_assignment=(
                None if self.shell_assignment is None else self.shell_assignment[selection]
            ),
            repeated_direction_group_id=(
                None
                if self.repeated_direction_group_id is None
                else compact_group_ids(
                    self.repeated_direction_group_id[selection, None]
                )
            ),
        )

    def summary(self) -> dict[str, object]:
        """Return compact provenance and dependence diagnostics."""
        flags = self.flags.astype(np.uint8)
        exact_counts = np.bincount(self.exact_hkl_group_id)
        friedel_counts = np.bincount(self.friedel_group_id)
        source_counts = np.bincount(self.source_group_id)
        summary: dict[str, object] = {
            "total_observations": self.row_count,
            "source_observation_groups": int(len(source_counts)),
            "deterministically_derived_observations": int(
                np.count_nonzero(
                    flags
                    & int(ProvenanceFlag.SYMMETRY_GENERATED | ProvenanceFlag.FRIEDEL_GENERATED)
                )
            ),
            "original_index_observations": int(
                np.count_nonzero(flags & int(ProvenanceFlag.ORIGINAL_INDEX))
            ),
            "symmetry_generated_observations": int(
                np.count_nonzero(flags & int(ProvenanceFlag.SYMMETRY_GENERATED))
            ),
            "friedel_generated_observations": int(
                np.count_nonzero(flags & int(ProvenanceFlag.FRIEDEL_GENERATED))
            ),
            "preserved_friedel_observations": int(
                np.count_nonzero(flags & int(ProvenanceFlag.PRESERVED_FRIEDEL))
            ),
            "rows_from_duplicate_input_hkl": int(
                np.count_nonzero(flags & int(ProvenanceFlag.DUPLICATE_INPUT_HKL))
            ),
            "rows_from_complete_input_friedel_pairs": int(
                np.count_nonzero(flags & int(ProvenanceFlag.DUPLICATE_INPUT_FRIEDEL_PAIR))
            ),
            "rows_from_merged_input": int(
                np.count_nonzero(flags & int(ProvenanceFlag.MERGED_INPUT))
            ),
            "exact_hkl_duplicate_groups": int(np.count_nonzero(exact_counts > 1)),
            "friedel_groups": int(len(friedel_counts)),
            "largest_exact_hkl_group": int(np.max(exact_counts, initial=0)),
            "largest_source_derivation_group": int(np.max(source_counts, initial=0)),
            "source_dataset": self.source_dataset,
            "source_file_sha256": self.source_file_sha256,
            "independence_contract": {
                "source_rows": "unknown or potentially independent; MTZ merging provenance is incomplete",
                "same_source_group": "deterministically related, not independent",
                "duplicate_input_hkl": "potentially correlated or repeated; independence unknown",
                "complete_input_friedel_pair": "ordinary versus anomalous independence unknown",
                "merged_input": "underlying measurement covariance unavailable",
            },
        }
        if self.repeated_direction_group_id is not None:
            direction_counts = np.bincount(self.repeated_direction_group_id)
            summary.update(
                {
                    "repeated_direction_groups": int(np.count_nonzero(direction_counts > 1)),
                    "largest_repeated_direction_group": int(
                        np.max(direction_counts, initial=0)
                    ),
                }
            )
        if self.shell_assignment is not None:
            shell_summaries = []
            for shell_id in range(int(np.max(self.shell_assignment, initial=-1)) + 1):
                mask = self.shell_assignment == shell_id
                shell_summaries.append(
                    {
                        "shell_id": shell_id,
                        "observations": int(np.count_nonzero(mask)),
                        "source_groups": int(np.unique(self.source_group_id[mask]).size),
                        "generated_observations": int(
                            np.count_nonzero(
                                mask
                                & (
                                    flags
                                    & int(
                                        ProvenanceFlag.SYMMETRY_GENERATED
                                        | ProvenanceFlag.FRIEDEL_GENERATED
                                    )
                                    != 0
                                )
                            )
                        ),
                        "exact_hkl_groups": int(np.unique(self.exact_hkl_group_id[mask]).size),
                        "friedel_groups": int(np.unique(self.friedel_group_id[mask]).size),
                        "direction_groups": (
                            None
                            if self.repeated_direction_group_id is None
                            else int(np.unique(self.repeated_direction_group_id[mask]).size)
                        ),
                    }
                )
            summary["shells"] = shell_summaries
        return summary


def canonical_friedel_hkl(hkl: np.ndarray) -> np.ndarray:
    """Return row-wise canonical representatives shared by ``h`` and ``-h``."""
    values = np.asarray(hkl, dtype=np.int32)
    canonical = values.copy()
    nonzero = canonical != 0
    has_nonzero = np.any(nonzero, axis=1)
    first_positions = np.argmax(nonzero, axis=1)
    first_values = canonical[np.arange(len(canonical)), first_positions]
    canonical[has_nonzero & (first_values > 0)] *= -1
    return canonical


def compact_group_ids(values: np.ndarray) -> np.ndarray:
    """Return deterministic int32 group IDs for equal rows."""
    _, inverse = np.unique(np.asarray(values), axis=0, return_inverse=True)
    return inverse.astype(np.int32)


__all__ = [
    "ObservationProvenance",
    "ProvenanceFlag",
    "canonical_friedel_hkl",
    "compact_group_ids",
]
