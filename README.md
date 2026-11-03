# sph-map: directional analysis of crystallographic electron-density maps

`sph-map` multiplies each phased Fourier coefficient by i^ℓ R_ℓm(k̂), a real spherical harmonic of the
reciprocal-lattice direction, and writes the filtered coefficients and maps. The ℓ = 1 channels are the
gradient of a smoothed density. The method and its controls are described in:

> Panjikar, S., Weiss, M. S. & Jayatilaka, D. (2026). Directional analysis of electron-density maps by
> spherical-harmonic Fourier filtering. *Acta Cryst. A* (submitted, manuscript IB5132).

This repository contains the program (version 11.8.0, the version used in the paper, for l = 0 and 1) and the scripts that
reproduce every number, table and figure of the paper (`paper_ib5132/`). The input data are archived on Zenodo
together with the same scripts: **https://doi.org/10.5281/zenodo.23608339**. Results are not archived; the scripts
regenerate them, and each step checks its results against stored reference values.

## Install

```bash
conda env create -f environment-paper.yml     # exact versions used for the paper (Python 3.10, also on macOS)
conda activate sph-map-ib5132
pip install -e .
```

`environment-tested.yml` is the environment in which the program was developed (Linux, Python 3.13).

## Settings used in the paper

The program defaults are **not** the paper settings. Always state the phase convention and symmetry
policy explicitly:

```bash
sph-map -m input.mtz -s F -p PHI -lmax 1 -o out.mtz --overwrite \
  --phase-convention ccp4 \
  --symmetry-policy p1_complete_friedel \
  --norm none --taper none \
  --radial-weight-convention crystallographic_q \
  --shell-weights off \
  --map-dir maps --map-samples-per-dmin 6      # 3 for proteins, with --output-strategy low-memory
```

Conventions to keep in mind:

| Item | Setting | Why |
|---|---|---|
| Phase sign | `--phase-convention ccp4` | CCP4/REFMAC/Phenix phases; the default (`historical`) reverses the ℓ = 1 channels |
| Symmetry | `--symmetry-policy p1_complete_friedel` | expands to P1 with phase shifts and Friedel mates; do not simply relabel data as P1 |
| Cartesian frame | model frame, x ∥ a (gemmi `cell.orth`) | directions are compared with the atomic model in this frame |
| Channels | (y, z, x) = m(−1, 0, +1), Condon–Shortley | check that the ℓ = 1 field points towards the atoms |
| Normalisation | `--norm none` | separate standardisation of channels rotates the vectors |
| Taper | none | with `--shell-weights off` the program's taper has no effect; taper the amplitudes beforehand if needed |
| F(000) | omitted | maps have zero mean |
| Phenix maps | `2FOFCWT_no_fill`, not `2FOFCWT` | filled coefficients add model bias |

Three checks before trusting a map: the ℓ = 0 map peaks on the atoms; the ℓ = 1 field points from each
atom towards its bonded neighbours; site symmetry is respected (in urea, C=O lies along the twofold axis).

## Reproducing the paper

Unpack the Zenodo record (it contains `01_input_data/` and the scripts of `paper_ib5132/`) and run the steps in
order (Fig. 3 needs PyMOL):

```bash
conda activate sph-map-ib5132
cd ib5132_zenodo
for s in 02_prepare 03_filter 04_small_molecules 05_hydrogen_tests 06_6mu9 07_pdbredo 08_figures; do
  sh $s/run.sh > $s.log 2>&1 && python3 tools/check_step.py $s | tail -1      # each line ends with "-> OK"
done
```

To run from this repository instead, copy or link the record's `01_input_data/` into `paper_ib5132/`. Results
are written to `paper_ib5132/work/` (set `IB5132_WORK` to write elsewhere). See `paper_ib5132/README.md` for which script
produces which number, table and figure.

## Documentation

`USER_MANUAL.md` (all options and conventions).

## Licence

BSD 3-Clause (see `LICENSE`). In the Zenodo record the input data are licensed CC BY 4.0 and the scripts BSD 3-Clause.

## How to cite

See `CITATION.cff`, or cite the paper above and the software DOI **[software DOI to be added]**.
