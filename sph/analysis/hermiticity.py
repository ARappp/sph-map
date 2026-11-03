"""Friedel-pair Hermiticity enforcement used by directional filtering."""

from __future__ import annotations

import numpy as np


def enforce_hermiticity(hkl: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Average available Friedel mates using the pair convention."""
    keys = [tuple(map(int, reflection)) for reflection in hkl]
    lookup = {reflection: index for index, reflection in enumerate(keys)}
    result = coefficients.copy()
    marked = np.zeros(len(hkl), dtype=bool)
    for index, reflection in enumerate(keys):
        if marked[index]:
            continue
        friedel = (-reflection[0], -reflection[1], -reflection[2])
        mate_index = lookup.get(friedel)
        if mate_index is None:
            continue
        value = 0.5 * (coefficients[index] + np.conj(coefficients[mate_index]))
        result[index] = value
        result[mate_index] = np.conj(value)
        marked[index] = marked[mate_index] = True
    return result
