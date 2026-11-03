"""Explicit amplitude/phase registration for positive Fourier synthesis.

This boundary accepts real amplitudes and phases in degrees, not arbitrary
preconstructed complex coefficients. Apply it after crystallographic symmetry
expansion, whose phase translations must retain their input convention.
"""

from typing import Literal

import numpy as np

PhaseConvention = Literal["historical", "ccp4"]


def phase_convention_metadata(convention: PhaseConvention) -> dict[str, object]:
    """Describe input coefficient construction and the unchanged synthesis sign.

    Raises ValueError for undeclared conventions. Coordinates are fractional
    crystal coordinates; synthesis uses 1/V with V in cubic angstroms.
    """
    if convention not in {"historical", "ccp4"}:
        raise ValueError("phase convention must be historical or ccp4")
    return {
        "selected": convention,
        "input_phase_units": "degrees",
        "input_phase_declaration": (
            "CCP4 amplitude/phase columns" if convention == "ccp4"
            else "historical positive-phase coefficient construction"
        ),
        "coefficient_equation": (
            "A*exp(-i*phi_CCP4)" if convention == "ccp4" else "A*exp(+i*phi)"
        ),
        "conjugation_of_input_amplitude_phase_coefficient": convention == "ccp4",
        "fourier_exponent_sign": "positive",
        "fourier_kernel": "exp(+2*pi*i*HKL.dot(fractional_position))/V",
        "output_coefficient_convention": "positive_synthesis_ready_do_not_conjugate_again",
        "automatic_filename_inference": False,
    }


def coefficients_for_positive_synthesis(
    amplitudes: np.ndarray,
    phases_deg: np.ndarray,
    *,
    convention: PhaseConvention,
) -> np.ndarray:
    """Construct complex128 coefficients using an explicitly declared convention.

    Parameters are real, finite, equally shaped arrays: amplitude in the input
    column's units and phase in degrees. No coordinate transformation occurs.
    The CCP4 bridge negates the phase before positive-exponent synthesis.
    Historical arithmetic is retained verbatim. Raises ValueError for invalid
    shapes, non-finite values, complex inputs or an unknown convention.
    """
    phase_convention_metadata(convention)
    if np.iscomplexobj(amplitudes) or np.iscomplexobj(phases_deg):
        raise ValueError("Supply real amplitude/phase arrays, not complex coefficients.")
    amplitude_values = np.asarray(amplitudes, dtype=np.float64)
    phase_values_deg = np.asarray(phases_deg, dtype=np.float64)
    if amplitude_values.shape != phase_values_deg.shape:
        raise ValueError("Amplitude and phase arrays must have identical shapes.")
    if not (np.isfinite(amplitude_values).all() and np.isfinite(phase_values_deg).all()):
        raise ValueError("Amplitude and phase arrays must be finite before registration.")
    if convention == "ccp4":
        return amplitude_values * np.exp(-1j * np.deg2rad(phase_values_deg))
    return amplitude_values * np.exp(1j * np.deg2rad(phase_values_deg))
