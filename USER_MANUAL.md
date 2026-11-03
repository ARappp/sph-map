# sph-map User Manual

## Overview

`sph-map` multiplies each phased crystallographic Fourier coefficient by a real spherical harmonic of the
reciprocal-lattice direction and writes the filtered coefficients to a P1 MTZ file and, optionally, as
full-cell P1 CCP4 maps. For degree l and order m the filtered coefficient is

\[
G_{lm}(h)=F(h)\,i^l\,R_{lm}(\widehat{k}_h)\,W(k_h),
\]

where `R_lm` are real orthonormal (tesseral) spherical harmonics and `W` is an optional radial weight. This
release provides l = 0 and l = 1. The l = 0 channel is an isotropically filtered density; the three l = 1
channels form a vector field that is the gradient of a smoothed density (Panjikar, Weiss & Jayatilaka,
*Acta Cryst. A*, manuscript IB5132).

The operation is pointwise filtering of each reflection followed by conventional Fourier synthesis; it does not
fit shell-wise coefficients.

## Installation

```bash
conda env create -f environment-paper.yml
conda activate sph-map-ib5132
pip install -e .
```

`environment-paper.yml` contains the versions used for the paper (Python 3.10, NumPy 2.2.6, SciPy 1.15.3,
Gemmi 0.7.1, pandas 2.3.2, Reciprocalspaceship 1.0.8); with it the scripts in `paper_ib5132/` reproduce the
paper on Linux and macOS. The program was developed under Linux (WSL2) with Python 3.13 (`environment-tested.yml`).
The installed command is `sph-map`; `python -m sph` is equivalent. Other dependency versions can change MTZ
bytes in the last digits but not the scientific results.

## Input requirements

The MTZ file must contain:

- `H`, `K` and `L` indices;
- a selected amplitude-like column of MTZ type `D`, `E`, `F` or `G`;
- a selected phase column of MTZ type `P`, in degrees;
- a finite, positive-volume unit cell and a valid space group.

By default rows whose selected amplitude or phase is non-finite are dropped before symmetry or Friedel
expansion; `--missing-policy error` rejects such input instead. Missing values are never converted to zero.
Anomalous data are not supported.

## Phase convention

`--phase-convention historical` (the default) uses coefficients `A exp(+i phi)` with the synthesis
`exp(+2 pi i h.r)/V`. Phases from CCP4, REFMAC or Phenix require `--phase-convention ccp4`, which uses
`A exp(-i phi_CCP4)` and registers the maps to the coordinates of the model. With the wrong convention the map
is inverted through the origin and the l = 1 channels change sign. The convention is never inferred from file
names. The filtered MTZ coefficients are ready for positive-exponent synthesis and must not be conjugated again.

## Symmetry policy

`--symmetry-policy` states how the input is organised:

- `asu_to_p1`: unique merged non-P1 asymmetric-unit data, expanded to P1 with the symmetry phase shifts;
- `p1_complete_friedel`: unique P1 data (or symmetry-expanded data written in P1); only missing Friedel mates
  are added;
- `p1_preserve`: P1 rows kept literally;
- `reject_ambiguous`: stop when the organisation of the input cannot be established;
- `legacy_expand` (default): historical behaviour, expansion to P1 followed by Friedel expansion; an already
  expanded P1 input containing both Friedel mates is expanded again.

MTZ metadata alone cannot distinguish complete ordinary P1 pairs from independently measured anomalous pairs;
select the policy deliberately.

## Quick start (settings of the paper)

```bash
mkdir maps
sph-map -m input.mtz -s F -p PHI -lmax 1 -o out.mtz \
  --phase-convention ccp4 --symmetry-policy p1_complete_friedel \
  --norm none --taper none --radial-weight-convention crystallographic_q \
  --shell-weights off --map-dir maps --map-samples-per-dmin 6
```

This writes `out.mtz` (filtered coefficients), `out.mtz.manifest.json` (complete run record) and the maps
`maps/out.L0M+0.ccp4`, `maps/out.L1M-1.ccp4`, `maps/out.L1M+0.ccp4` and `maps/out.L1M+1.ccp4`. The l = 1 maps
are the Cartesian components (`m = -1, 0, +1` correspond to y, z, x; Condon–Shortley phase) in the orthogonal
frame of the model with x parallel to a. For proteins 3 samples per d_min are sufficient.

Checks before interpreting a map: the l = 0 map peaks on the atoms; the l = 1 field points from each atom
towards its bonded neighbours; site symmetry is respected.

## Command-line reference

| Option | Type/default | Meaning |
|---|---|---|
| `-m`, `--mtz` | path, required | Input MTZ. |
| `-s`, `--sf` | string, required | Amplitude column label. |
| `-p`, `--ph` | string, required | Phase column label (degrees). |
| `-lmax`, `--lmax` | 0 or 1, required | Maximum degree. |
| `-r`, `--resol` | angstrom, default 0 | High-resolution cutoff; 0 disables it. |
| `-o`, `--out` | path, required | Output MTZ. |
| `--manifest` | path | Manifest path; default `<output>.manifest.json`. |
| `--phase-convention` | `historical`/`ccp4`, default `historical` | Input phase convention (see above). |
| `--overwrite` | flag | Replace existing output files. |
| `--missing-policy` | `drop`/`error`, default `drop` | Treatment of non-finite selected values. |
| `--symmetry-policy` | default `legacy_expand` | `legacy_expand`, `asu_to_p1`, `p1_preserve`, `p1_complete_friedel` or `reject_ambiguous`. |
| `--hermiticity` | flag | Enforce `F(h)=F(-h)*` before filtering. |
| `--shell-mode` | `linear`/`quantile`/`log`, default `log` | Radial partition used for shell weights. |
| `--n-shell` | integer, default 80 | Requested number of shells. |
| `--min-per-shell` | integer, default 50 | Linear merge threshold; warning threshold for other modes. |
| `--taper` | `none`/`cosine`/`kaiser`, default `cosine` | Radial outer window (applied with shell weights). |
| `--eta` | float, default 0.9 | Cosine taper onset as a fraction of the maximum radius. |
| `--beta` | float, default 8 | Kaiser shape parameter. |
| `--shell-weights` | `on`/`off`, default `on` | Shell-volume/count weighting. |
| `--radial-weight-convention` | default `legacy_k_unscaled` | Radial measure of the shell weights: historical `k = 2 pi/d` or `crystallographic_q` (`q = 1/d`). Weighted results differ by `(2 pi)^3`; no effect with weights off. |
| `--norm` | `none`/`per-l`, default `per-l` | Per-degree sampling normalisation. |
| `--write-readme [PATH]` | optional path | Write a run summary (default `YLM_RUN_README.md`). |
| `--timings`, `--timings-json` | flag, path | Print (and write) timings. |
| `--output-strategy` | `low-memory`/`materialized`, default `low-memory` | Internal construction; identical results. |
| `--map-dir` | existing directory | Write one full-cell P1 CCP4 map per channel. |
| `--map-samples-per-dmin` | float, default 3 | Grid samples per d_min (rounded up to FFT-friendly, alias-free sizes). |
| `--map-grid NU NV NW` | three integers | Exact grid; rejected if it aliases any Miller index. |
| `--map-conventional-control` | flag | Also write the unfiltered F/PHI map on the same grid. |
| `-v`, `--verbose` | flag | Detailed shell and timing diagnostics. |

## Shells and radial weights

The Cartesian wavevector is `K = HKL @ B.T` with `B = 2 pi A^-T`, so `k = |K| = 2 pi/d` in radians per angstrom.
The crystallographic radius is `q = k/(2 pi) = 1/d` in cycles per angstrom. Logarithmic shell edges span the
finite positive minimum and maximum; intervals are lower-closed and upper-open, and only the final upper boundary
is included. Empty and sparsely populated shells are retained and reported. With `--shell-weights off` the shell
partition, the taper and the radial-weight convention have no effect on the result. Shell-weighted maps are
relative (normalised) angularly filtered densities, not absolute e/angstrom^3.

## Output

For every `(l, m)` the MTZ contains amplitude and phase columns `AYLM_LlMm` and `PYLM_LlMm` (for example
`AYLM_L1M-1`, preceded by a prefix derived from the input labels). `FYLM_Ll` and `PHYLM_Ll` store the coherent sum over m; this sum depends on the coordinate axes
and is neither an angular power nor a rotational invariant.

Map synthesis uses `rho = (1/V) sum_h G(h) exp(+2 pi i h.r)` on the grid `[u, v, w] = [a, b, c]` with origin at
fractional (0, 0, 0); missing reciprocal indices are zero. Values are computed in double precision and stored as
float32 CCP4 maps in P1.

Outputs are written through temporary files and installed only after successful serialisation; existing files
cause an error unless `--overwrite` is given. Keep the manifest with every MTZ: it records input hashes,
software and dependency versions, all options, coordinate matrices, shell populations, warnings and the output
hash.

## Programming interface

`sph.analysis.filtering.filter_directional_reflections` performs the filtering on arrays (HKL, amplitudes,
phases, reciprocal matrix and `FilteringOptions`) without file input or output. `sph.workflow.run_workflow`
runs the complete file workflow from an `SPHConfig`.

## Common errors

- "Input MTZ does not exist": correct `--mtz`.
- "missing required column": check the MTZ labels and `--sf`/`--ph`.
- "non-finite values": use the default drop policy or repair the input.
- "Output already exists": choose another output or add `--overwrite`.
- "Empty shell": a warning; review the shell count.
- `SPH-E004`: the symmetry policy conflicts with the P1/ASU, duplicate or Friedel content of the input; check
  the organisation of the input and select the matching policy.

## Limitations

Degrees l = 0 and 1 only, in the crystal Cartesian frame. Anomalous decomposition, unmerged or partial
observations and sigma weighting are not supported.

## Licence and citation

BSD 3-Clause (`LICENSE`). Cite Panjikar, Weiss & Jayatilaka, *Acta Cryst. A* (manuscript IB5132), and the
software as given in `CITATION.cff`; record the software version and the run manifest of each calculation.
