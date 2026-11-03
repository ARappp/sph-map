"""Validated, immutable configuration for SPH directional filtering."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Literal

from .errors import ConfigurationError


MissingPolicy = Literal["drop", "error"]
SymmetryPolicy = Literal[
    "legacy_expand", "asu_to_p1", "p1_preserve",
    "p1_complete_friedel", "reject_ambiguous",
]
ShellMode = Literal["linear", "log", "quantile"]
TaperName = Literal["none", "cosine", "kaiser"]
NormalizationMode = Literal["none", "per-l", "per-lm"]
OutputStrategy = Literal["low-memory", "materialized"]
RadialWeightConvention = Literal["legacy_k_unscaled", "crystallographic_q"]


@dataclass(frozen=True)
class SPHConfig:
    """Complete immutable configuration for one filtering run.

    Path fields use :class:`pathlib.Path`; field names state physical units
    where applicable.
    """

    input_mtz: Path
    amplitude_column: str
    phase_column: str
    lmax: int
    output_mtz: Path
    resolution_cutoff_A: float = 0.0
    manifest_path: Path | None = None
    overwrite: bool = False
    missing_policy: MissingPolicy = "drop"
    symmetry_policy: SymmetryPolicy = "legacy_expand"
    hermiticity: bool = False
    shell_mode: ShellMode = "log"
    shell_count: int = 80
    minimum_per_shell: int = 50
    taper: TaperName = "cosine"
    taper_eta: float = 0.9
    taper_beta: float = 8.0
    shell_weights: str = "on"
    normalization: NormalizationMode = "per-l"
    write_readme: Path | None = None
    timings: bool = False
    timings_json: Path | None = None
    verbose: bool = False
    output_strategy: OutputStrategy = "low-memory"
    radial_weight_convention: RadialWeightConvention = "legacy_k_unscaled"
    directional_map_directory: Path | None = None
    map_samples_per_d_min: float = 3.0
    map_grid_shape: tuple[int, int, int] | None = None
    map_conventional_control: bool = False
    phase_convention: str = "historical"

    @property
    def resolved_manifest_path(self) -> Path:
        """Return the explicit or conventional run-manifest path."""
        return self.manifest_path or Path(f"{self.output_mtz}.manifest.json")

    @property
    def l_values(self) -> tuple[int, ...]:
        """Return harmonic degrees in output order."""
        return tuple(range(self.lmax + 1))

    def validate(self) -> "SPHConfig":
        """Validate scientific constraints and return self."""
        if self.phase_convention not in {"historical", "ccp4"}:
            raise ConfigurationError("phase convention must be historical or ccp4")
        if not self.amplitude_column.strip() or not self.phase_column.strip():
            raise ConfigurationError("Amplitude and phase column names cannot be empty.")
        if self.lmax not in (0, 1):
            raise ConfigurationError("lmax must be 0 or 1")
        if not math.isfinite(self.resolution_cutoff_A) or self.resolution_cutoff_A < 0:
            raise ConfigurationError("Resolution cutoff must be finite and >= 0")
        if self.shell_count < 1 or self.minimum_per_shell < 1:
            raise ConfigurationError("Shell counts must be positive")
        if not math.isfinite(self.taper_eta) or not 0 < self.taper_eta < 1:
            raise ConfigurationError("Cosine taper eta must be between 0 and 1")
        if not math.isfinite(self.taper_beta) or self.taper_beta < 0:
            raise ConfigurationError("Kaiser beta must be finite and nonnegative")
        if self.normalization == "per-lm":
            raise ConfigurationError("Per-m normalization breaks orthogonal multiplet rotation; use none or per-l")
        if self.symmetry_policy not in {
            "legacy_expand", "asu_to_p1", "p1_preserve",
            "p1_complete_friedel", "reject_ambiguous",
        }:
            raise ConfigurationError(
                "Unsupported symmetry policy; use legacy_expand, asu_to_p1, "
                "p1_preserve, p1_complete_friedel, or reject_ambiguous."
            )
        if self.output_strategy not in {"low-memory", "materialized"}:
            raise ConfigurationError(
                "Output strategy must be 'low-memory' or 'materialized'."
            )
        if self.radial_weight_convention not in {
            "legacy_k_unscaled", "crystallographic_q"
        }:
            raise ConfigurationError(
                "Radial-weight convention must be legacy_k_unscaled or "
                "crystallographic_q."
            )
        if (
            not math.isfinite(self.map_samples_per_d_min)
            or self.map_samples_per_d_min <= 0.0
        ):
            raise ConfigurationError(
                "Map samples-per-d_min must be finite and greater than zero."
            )
        if self.map_grid_shape is not None and (
            len(self.map_grid_shape) != 3
            or any(value <= 0 for value in self.map_grid_shape)
        ):
            raise ConfigurationError(
                "--map-grid requires three positive dimensions: NU NV NW."
            )
        if self.map_conventional_control and self.directional_map_directory is None:
            raise ConfigurationError(
                "--map-conventional-control requires --map-dir."
            )
        if self.map_grid_shape is not None and self.directional_map_directory is None:
            raise ConfigurationError("--map-grid requires --map-dir.")
        if self.input_mtz.expanduser().resolve() == self.output_mtz.expanduser().resolve():
            raise ConfigurationError("Input MTZ and output MTZ must be different files.")
        return self

    def to_manifest_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation using descriptive keys."""
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }

    def to_legacy_namespace(self) -> Namespace:
        """Return the command-line argument names used in the run manifest."""
        return Namespace(
            mtz=str(self.input_mtz), sf=self.amplitude_column,
            ph=self.phase_column, lmax=self.lmax,
            resol=self.resolution_cutoff_A, out=str(self.output_mtz),
            manifest=str(self.resolved_manifest_path), overwrite=self.overwrite,
            missing_policy=self.missing_policy,
            symmetry_policy=self.symmetry_policy,
            hermiticity=self.hermiticity,
            phase_convention=self.phase_convention,
            shell_mode=self.shell_mode, n_shell=self.shell_count,
            min_per_shell=self.minimum_per_shell, taper=self.taper,
            eta=self.taper_eta, beta=self.taper_beta,
            shell_weights=self.shell_weights, norm=self.normalization,
            write_readme=str(self.write_readme) if self.write_readme else None,
            timings=self.timings,
            timings_json=str(self.timings_json) if self.timings_json else None,
            verbose=self.verbose,
            output_strategy=self.output_strategy,
            radial_weight_convention=self.radial_weight_convention,
            map_dir=(
                str(self.directional_map_directory)
                if self.directional_map_directory else None
            ),
            map_samples_per_dmin=self.map_samples_per_d_min,
            map_grid=self.map_grid_shape,
            map_conventional_control=self.map_conventional_control,
        )

    @classmethod
    def from_namespace(cls, arguments: Namespace) -> "SPHConfig":
        """Create and validate a configuration from parsed CLI arguments."""
        def optional_path(value: str | None) -> Path | None:
            return Path(value).expanduser() if value else None

        return cls(
            input_mtz=Path(arguments.mtz).expanduser(),
            amplitude_column=arguments.sf, phase_column=arguments.ph,
            lmax=arguments.lmax,
            output_mtz=Path(arguments.out).expanduser(),
            resolution_cutoff_A=arguments.resol,
            manifest_path=optional_path(arguments.manifest),
            overwrite=arguments.overwrite, missing_policy=arguments.missing_policy,
            symmetry_policy=arguments.symmetry_policy,
            phase_convention=arguments.phase_convention,
            hermiticity=arguments.hermiticity, shell_mode=arguments.shell_mode,
            shell_count=arguments.n_shell,
            minimum_per_shell=arguments.min_per_shell, taper=arguments.taper,
            taper_eta=arguments.eta, taper_beta=arguments.beta,
            shell_weights=arguments.shell_weights, normalization=arguments.norm,
            write_readme=optional_path(arguments.write_readme),
            timings=arguments.timings,
            timings_json=optional_path(arguments.timings_json),
            verbose=arguments.verbose,
            output_strategy=arguments.output_strategy,
            radial_weight_convention=arguments.radial_weight_convention,
            directional_map_directory=optional_path(arguments.map_dir),
            map_samples_per_d_min=arguments.map_samples_per_dmin,
            map_grid_shape=(
                tuple(arguments.map_grid)
                if arguments.map_grid is not None
                else None
            ),
            map_conventional_control=arguments.map_conventional_control,
        ).validate()
