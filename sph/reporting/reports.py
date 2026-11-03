"""Small text reports produced after successful filtering."""

from __future__ import annotations

from sph.config import SPHConfig


def render_run_readme(config: SPHConfig, warnings: list[str]) -> str:
    """Render a concise, scientifically explicit run README."""
    map_statement = (
        "It also writes validated full-cell P1 CCP4 directional-component maps; "
        "their shell-weighted values are relative/normalized angularly filtered "
        "density, not validated absolute e/A^3."
        if config.directional_map_directory is not None
        else "No real-space CCP4 map was requested for this run."
    )
    lines = [
        "# SPH reciprocal-space directional-filter run\n",
        "\nThis run applies pointwise directional filters. It does not fit shell ",
        f"coefficients. {map_statement}\n",
        f"\n**MTZ:** `{config.input_mtz}`  \n",
        f"**SF/PH:** `{config.amplitude_column}` / `{config.phase_column}`  \n",
        f"**lmax:** {config.lmax}  \n",
        "**basis:** real  \n",
        f"**input phase convention:** {config.phase_convention}; positive Fourier exponent; "
        f"input phase negation={config.phase_convention == 'ccp4'}  \n",
        f"**hermiticity:** {config.hermiticity}  \n",
        "**shells:** "
        f"mode={config.shell_mode}, n={config.shell_count}, "
        f"taper={config.taper} (eta={config.taper_eta}, "
        f"beta={config.taper_beta}), weights={config.shell_weights}  \n",
        f"**radial-weight convention:** {config.radial_weight_convention}  \n",
        f"**normalization:** {config.normalization}  \n",
        "**coordinate frame:** crystal_cartesian_frame  \n",
    ]
    if warnings:
        lines.append("\n## Warnings\n")
        lines.extend(f"- {warning}\n" for warning in warnings)
    return "".join(lines)
