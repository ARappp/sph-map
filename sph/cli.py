"""Command-line parsing for SPH directional filtering.

This module intentionally performs no reflection I/O or numerical work.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys

from .config import SPHConfig
from .errors import ConfigurationError
from .errors import SPHUserError
from .version import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Apply discrete reciprocal-space spherical-harmonic directional "
            "filters (l = 0, 1) and write Fourier-synthesis-ready P1 MTZ channels."
        )
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument("-m", "--mtz", required=True, help="Input MTZ with |F| and PHI")
    parser.add_argument("-s", "--sf", required=True, help="Amplitude label (e.g. FWT)")
    parser.add_argument("-p", "--ph", required=True, help="Phase label in degrees (e.g. PHWT)")
    parser.add_argument("-lmax", "--lmax", required=True, type=int, choices=range(2), help="Maximum degree l (0 or 1)")
    parser.add_argument("-r", "--resol", type=float, default=0.0, help="Optional high-resolution cutoff (A)")
    parser.add_argument("-o", "--out", required=True, help="Output directional-filter MTZ")
    parser.add_argument("--manifest", help="Run-manifest JSON path (default: <output>.manifest.json)")
    parser.add_argument(
        "--phase-convention", choices=["historical", "ccp4"], default="historical",
        help="Input phase declaration: historical retains A*exp(+i*phi) (default); "
        "ccp4 uses A*exp(-i*phi_CCP4) to register positive-synthesis maps to PDB coordinates. "
        "Never inferred from filenames.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing output and sidecar files")
    parser.add_argument("--missing-policy", choices=["drop", "error"], default="drop", help="Policy for non-finite selected amplitude/phase rows before symmetry expansion (default: drop)")
    parser.add_argument(
        "--symmetry-policy",
        choices=[
            "legacy_expand", "asu_to_p1", "p1_preserve",
            "p1_complete_friedel", "reject_ambiguous",
        ],
        default="legacy_expand",
        help=(
            "Ordinary-data symmetry/Friedel contract: legacy_expand preserves "
            "historical behavior; asu_to_p1 requires unique merged non-P1 ASU; "
            "p1_preserve keeps literal P1 rows; p1_complete_friedel appends only "
            "missing ordinary mates; reject_ambiguous refuses uncertain input. "
            "Anomalous meaning cannot be inferred from F/PHI metadata."
        ),
    )
    parser.add_argument("--hermiticity", action="store_true", help="Enforce z(h)=conj(z(-h)) symmetry")
    parser.add_argument("--shell-mode", choices=["linear", "quantile", "log"], default="log")
    parser.add_argument("--n-shell", type=int, default=80)
    parser.add_argument("--min-per-shell", type=int, default=50)
    parser.add_argument("--taper", choices=["none", "cosine", "kaiser"], default="cosine")
    parser.add_argument("--eta", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=8.0)
    parser.add_argument("--shell-weights", choices=["on", "off"], default="on")
    parser.add_argument(
        "--radial-weight-convention",
        choices=["legacy_k_unscaled", "crystallographic_q"],
        default="legacy_k_unscaled",
        help=(
            "Shell-volume measure: historical unscaled k=2*pi/d or consistent "
            "crystallographic q=1/d (default preserves historical outputs)"
        ),
    )
    parser.add_argument("--norm", choices=["none", "per-l"], default="per-l")
    parser.add_argument("--write-readme", nargs="?", const="YLM_RUN_README.md")
    parser.add_argument("--timings", action="store_true")
    parser.add_argument("--timings-json")
    parser.add_argument(
        "--output-strategy", choices=["low-memory", "materialized"],
        default="low-memory",
        help=(
            "Internal output construction: low-memory streams one complex "
            "channel at a time; materialized keeps all channels in memory (identical values)."
        ),
    )
    parser.add_argument(
        "--map-dir",
        help=(
            "Optional existing directory for validated full-cell P1 CCP4 "
            "directional-component maps; MTZ output remains mandatory"
        ),
    )
    parser.add_argument(
        "--map-samples-per-dmin", type=float, default=3.0,
        help=(
            "Map sampling density along each cell edge before upward FFT-friendly "
            "rounding (default: 3 samples per selected d_min)"
        ),
    )
    parser.add_argument(
        "--map-grid", type=int, nargs=3, metavar=("NU", "NV", "NW"),
        help=(
            "Exact P1 grid dimensions in array[a,b,c] order; rejected if any "
            "dimension aliases represented Miller indices"
        ),
    )
    parser.add_argument(
        "--map-conventional-control", action="store_true",
        help=(
            "Also synthesize the unfiltered F/PHI conventional-map control on "
            "the identical P1 grid (requires --map-dir)"
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed shell and timing diagnostics")
    return parser


def parse_configuration(arguments: Sequence[str] | None = None) -> SPHConfig:
    """Parse CLI arguments and return a validated immutable configuration."""
    parser = build_parser()
    namespace = parser.parse_args(arguments)
    try:
        return SPHConfig.from_namespace(namespace)
    except ConfigurationError as error:
        parser.error(str(error))


def main(arguments: Sequence[str] | None = None) -> int:
    """Parse arguments, invoke the workflow, and return a stable exit code."""
    config = parse_configuration(arguments)
    try:
        from .workflow import run_workflow

        run_workflow(config)
    except SPHUserError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0
