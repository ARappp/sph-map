"""Sampling normalization for orthonormal real harmonic multiplets."""

from __future__ import annotations

from typing import Literal

import numpy as np

from sph.harmonics.real import evaluate_real_harmonic


def normalization_factors(
    polar_theta_rad: np.ndarray,
    azimuth_phi_rad: np.ndarray,
    lmax: int,
    mode: Literal["none", "per-l", "per-lm"] = "per-l",
    weights: np.ndarray | None = None,
) -> tuple[dict[tuple[int, int], float], dict[int, float]]:
    """Return inverse RMS factors of the real harmonics.

    Independently scaled ``per-lm`` components are rejected by configuration
    validation.
    """
    count = len(polar_theta_rad)
    if mode == "none":
        return {}, {}
    inverse_per_lm: dict[tuple[int, int], float] = {}
    inverse_per_l: dict[int, float] = {}
    means_per_lm: dict[tuple[int, int], float] = {}
    means_per_l: dict[int, float] = {}
    for degree in range(lmax + 1):
        multiplet_power = np.zeros(count, dtype=np.float64)
        for order in range(-degree, degree + 1):
            harmonic = evaluate_real_harmonic(
                order, degree, azimuth_phi_rad, polar_theta_rad
            )
            power = np.abs(harmonic) ** 2
            if mode == "per-lm":
                mean = (
                    np.average(power, weights=weights)
                    if weights is not None
                    else float(power.mean())
                )
                means_per_lm[(degree, order)] = np.sqrt(mean)
            multiplet_power += power
        if mode == "per-l":
            mean = (
                np.average(multiplet_power, weights=weights)
                if weights is not None
                else float(multiplet_power.mean())
            )
            means_per_l[degree] = np.sqrt(mean)
    inverse_per_lm.update(
        {key: 1.0 / max(value, 1.0e-12) for key, value in means_per_lm.items()}
    )
    inverse_per_l.update(
        {key: 1.0 / max(value, 1.0e-12) for key, value in means_per_l.items()}
    )
    return inverse_per_lm, inverse_per_l
