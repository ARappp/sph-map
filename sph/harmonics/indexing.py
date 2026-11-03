"""Canonical ordering and output labels for harmonic channels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class HarmonicIndex:
    """A real spherical-harmonic degree and order pair."""

    degree_l: int
    order_m: int

    def __post_init__(self) -> None:
        if self.degree_l < 0 or abs(self.order_m) > self.degree_l:
            raise ValueError("Harmonic indices require l >= 0 and -l <= m <= l.")


def harmonic_indices(lmax: int) -> tuple[HarmonicIndex, ...]:
    """Return ``(l,m)`` in degree-major, ascending-``m`` order."""
    if lmax < 0:
        raise ValueError("lmax must be non-negative.")
    return tuple(
        HarmonicIndex(degree, order)
        for degree in range(lmax + 1)
        for order in range(-degree, degree + 1)
    )


def channel_labels(prefix: str, index: HarmonicIndex) -> tuple[str, str]:
    """Return the amplitude and phase labels for one channel."""
    suffix = f"L{index.degree_l}M{index.order_m}"
    return f"{prefix}AYLM_{suffix}", f"{prefix}PYLM_{suffix}"


def coherent_labels(prefix: str, degree_l: int) -> tuple[str, str]:
    """Return backward-compatible labels for the coordinate-dependent sum."""
    if degree_l < 0:
        raise ValueError("Harmonic degree must be non-negative.")
    return f"{prefix}FYLM_L{degree_l}", f"{prefix}PHYLM_L{degree_l}"


def output_column_labels(prefix: str, lmax: int) -> tuple[str, ...]:
    """Return serialized MTZ labels in their exact order."""
    labels = ["H", "K", "L"]
    for degree in range(lmax + 1):
        for order in range(-degree, degree + 1):
            labels.extend(channel_labels(prefix, HarmonicIndex(degree, order)))
        labels.extend(coherent_labels(prefix, degree))
    # reciprocalspaceship appends the MTZ symmetry-mate bookkeeping column.
    labels.append("M/ISYM")
    return tuple(labels)
