"""Atoms of the crystal (all symmetry mates) with IT92 form factors and anisotropic ADPs, read from a small-molecule
CIF or a PDB model, and the analytic structure factors of selected atoms (used for F_H, the hydrogen contribution).
Free-atom densities: 
ρ⁰_A(r) = analytic real-space transform of the IT92 form factor times the atom's anisotropic Debye–Waller factor:
f(q) = Σ_i a_i exp(-b_i |q|²/4) + c   (q in cycles/Å, s = |q|/2),   T(q) = exp(-2π² qᵀ U q)
⇒ ρ⁰(r) = occ · Σ_terms a π^{3/2} det(M)^{-1/2} exp(-π² rᵀ M⁻¹ r),  M = (b/4) I + 2π² U  (c term: b = 0).
Not resolution-truncated (ρ⁰ > 0 everywhere, so the weights are defined). ∫ρ⁰ = occ·f(0) ≈ occ·Z.
Taken from the method-comparison code of the authors (methods/C_atom_partition/promolecule.py, round3_h2.sf)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import gemmi


@dataclass
class Atom:
    label: str
    element: str
    cart: np.ndarray       # Å, model frame
    occ: float
    U: np.ndarray          # 3×3 Cartesian, Å²


class FreeAtom:
    """Precomputed Gaussian terms of one atom's thermally smeared IT92 density."""
    def __init__(self, element: str, U: np.ndarray, occ: float = 1.0):
        co = gemmi.Element(element).it92.get_coefs()
        a, b, c = list(co[:4]) + [co[8]], list(co[4:8]) + [0.0], None
        U = np.asarray(U, float)
        if np.linalg.eigvalsh(U).min() <= 0:
            raise ValueError(f"{element}: ADP not positive definite")
        self.terms = []
        for ai, bi in zip(a, b):
            M = (bi / 4.0) * np.eye(3) + 2 * np.pi ** 2 * U
            self.terms.append((occ * ai * np.pi ** 1.5 / np.sqrt(np.linalg.det(M)), np.pi ** 2 * np.linalg.inv(M)))
        self.electrons = occ * float(sum(a))

    def density(self, r: np.ndarray) -> np.ndarray:
        """r: (..., 3) displacement from the nucleus (Å)."""
        r = np.asarray(r, float); out = np.zeros(r.shape[:-1])
        for amp, Q in self.terms:
            out += amp * np.exp(-np.einsum("...i,ij,...j->...", r, Q, r))
        return out


class Crystal:
    """Unit-cell atoms (all symmetry mates) + periodic images on demand."""
    def __init__(self, cell: gemmi.UnitCell, atoms: list[Atom]):
        self.cell = cell; self.atoms = atoms
        self.A = np.array(cell.orth.mat, float); self.Ainv = np.linalg.inv(self.A)
        self.free = [FreeAtom(a.element, a.U, a.occ) for a in atoms]
        self.cart = np.array([a.cart for a in atoms])

    def images_near(self, centre: np.ndarray, cutoff: float):
        """All (atom index, image position) with |image − centre| <= cutoff."""
        f = (self.cart - centre) @ self.Ainv.T            # fractional offsets
        f -= np.round(f)
        n = int(np.ceil(cutoff / min(self.cell.parameters[:3]))) + 1
        rng = np.arange(-n, n + 1)
        shifts = np.array(np.meshgrid(rng, rng, rng, indexing="ij")).reshape(3, -1).T
        idx, pos = [], []
        for s in shifts:
            p = centre + (f + s) @ self.A.T
            ok = np.linalg.norm(p - centre, axis=1) <= cutoff
            idx.extend(np.nonzero(ok)[0]); pos.extend(p[ok])
        return np.array(idx, int), np.array(pos)

    def total_electrons(self) -> float:
        return float(sum(fa.electrons for fa in self.free))


def _u_cif_to_cart(cell: gemmi.UnitCell, u6) -> np.ndarray:
    """CIF U_ij (U11,U22,U33,U12,U13,U23; axes a*,b*,c*-normalised) → Cartesian U = A N U N Aᵀ."""
    U11, U22, U33, U12, U13, U23 = u6
    Ucif = np.array([[U11, U12, U13], [U12, U22, U23], [U13, U23, U33]])
    rc = cell.reciprocal(); N = np.diag([rc.a, rc.b, rc.c]); A = np.array(cell.orth.mat)
    return A @ N @ Ucif @ N @ A.T


def _num(s: str) -> float:
    return float(s.split("(")[0])


def _element_from_label(label: str) -> str:
    sym = "".join(ch for ch in label if ch.isalpha())
    return sym[:2].capitalize() if sym[:2].capitalize() in ("Cl", "Br", "Na", "Mg", "Zn", "Fe", "Ca") else sym[:1].upper()


def crystal_from_cif(path) -> Crystal:
    """Small-molecule CIF: label, fract xyz, occupancy, U_aniso (or U_iso); symmetry ops expand the cell."""
    b = gemmi.cif.read(str(path)).sole_block()
    cell = gemmi.UnitCell(*[_num(b.find_value(t)) for t in ("_cell_length_a", "_cell_length_b", "_cell_length_c",
                                                             "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma")])
    ops = [gemmi.Op(o.strip("'").replace(" ", "")) for o in b.find_loop("_symmetry_equiv_pos_as_xyz")] or \
          [gemmi.Op(o.strip("'").replace(" ", "")) for o in b.find_loop("_space_group_symop_operation_xyz")]
    t = b.find(["_atom_site_label", "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z", "_atom_site_U_iso_or_equiv"])
    occs = [_num(r[0]) for r in b.find(["_atom_site_occupancy"])] or [1.0] * len(t)
    an = {r[0]: [_num(r[i]) for i in range(1, 7)] for r in b.find(["_atom_site_aniso_label", "_atom_site_aniso_U_11",
          "_atom_site_aniso_U_22", "_atom_site_aniso_U_33", "_atom_site_aniso_U_12", "_atom_site_aniso_U_13", "_atom_site_aniso_U_23"])}
    A = np.array(cell.orth.mat); Ainv = np.linalg.inv(A); atoms = []; subs = []
    for row, occ in zip(t, occs):
        lab = row[0]; x = np.array([_num(row[i]) for i in (1, 2, 3)])
        Uc = _u_cif_to_cart(cell, an[lab]) if lab in an else np.eye(3) * _num(row[4])
        if np.linalg.eigvalsh(Uc).min() <= 0:          # non-positive-definite U_aniso (e.g. HAR H): use U_eq
            subs.append(dict(label=lab, eigenvalues=np.linalg.eigvalsh(Uc).round(5).tolist(), U_iso_used=_num(row[4])))
            Uc = np.eye(3) * _num(row[4])
        seen = []
        for op in ops:
            xs = np.array(op.apply_to_xyz(list(x))) % 1.0
            if any(np.allclose(((xs - p) + 0.5) % 1 - 0.5, 0, atol=1e-4) for p in seen):
                continue
            seen.append(xs)
            R = A @ np.array(op.rot, float) / gemmi.Op.DEN @ Ainv          # Cartesian rotation of this op
            atoms.append(Atom(f"{lab}#{len(seen)}", _element_from_label(lab), A @ xs, occ, R @ Uc @ R.T))
    cr = Crystal(cell, atoms); cr.adp_substitutions = subs
    return cr


def crystal_from_pdb(path) -> Crystal:
    """Macromolecular model: every atom incl. altlocs, occupancy-weighted, ANISOU if present (else B_iso), riding H removed
    only if requested elsewhere. All space-group mates are generated (fractional ops, Cartesian rotation of U); atoms on
    special positions are kept once."""
    st = gemmi.read_structure(str(path)); st.setup_entities()
    sg = st.find_spacegroup() or gemmi.find_spacegroup_by_name(st.spacegroup_hm)
    cell = st.cell; A = np.array(cell.orth.mat); Ainv = np.linalg.inv(A)
    ops = list(sg.operations()) if sg is not None else [gemmi.Op("x,y,z")]
    rots = [A @ (np.array(op.rot, float) / gemmi.Op.DEN) @ Ainv for op in ops]
    atoms = []
    for ch in st[0]:
        for res in ch:
            for a in res:
                U = (np.array(a.aniso.as_mat33().tolist()) if a.aniso.nonzero() else np.eye(3) * a.b_iso / (8 * np.pi ** 2))
                lab = f"{ch.name}{res.seqid.num}{res.name}:{a.name}{a.altloc if a.altloc != chr(0) else ''}"
                f0 = Ainv @ np.array(a.pos.tolist()); seen = []
                for k, (op, R) in enumerate(zip(ops, rots)):
                    f = np.array(op.apply_to_xyz(list(f0)))
                    if any(np.allclose(((f - g) + 0.5) % 1 - 0.5, 0, atol=1e-4) for g in seen):
                        continue
                    seen.append(f)
                    atoms.append(Atom(lab if k == 0 else f"{lab}#op{k}", a.element.name, A @ f, a.occ, R @ U @ R.T))
    cr = Crystal(cell, atoms); cr.n_asu = sum(1 for a in atoms if "#op" not in a.label); cr.n_ops = len(ops)
    return cr


def sf(cr, atoms_idx, q):
    """analytic transform in the internal convention: C = sum_atoms f(q) T(q) exp(-2 pi i q.r)"""
    F = np.zeros(len(q), complex)
    for i in atoms_idx:
        a, fa = cr.atoms[i], cr.free[i]; amp_q = np.zeros(len(q))
        for amp_, Q in fa.terms:
            M = np.pi ** 2 * np.linalg.inv(Q); a_ = amp_ * np.sqrt(np.linalg.det(M)) / np.pi ** 1.5
            amp_q += a_ * np.exp(-np.einsum("ni,ij,nj->n", q, M, q))
        F += amp_q * np.exp(-2j * np.pi * (q @ a.cart))
    return F


