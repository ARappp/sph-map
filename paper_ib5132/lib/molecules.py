"""Atoms, cell, symmetry and analysed bonds of urea and Gly-Ala, read from the HAR CIFs in 01_input_data."""
import gemmi, numpy as np
from config import INPUT
CIF = {"urea": INPUT / "urea" / "urea.archive.cif", "glyala": INPUT / "gly-ala" / "gly-ala.cif"}
BONDS = {"urea": [("C", "O"), ("C", "N1"), ("N1", "H1"), ("N1", "H3")],
         "glyala": [("C1", "O1"), ("C1", "O2"), ("C4", "O3"), ("N1", "H1N1"), ("N2", "H1N2"), ("N2", "H2N2"), ("N2", "H3N2"),
                    ("C2", "H2"), ("C3", "H3A"), ("C3", "H3B"), ("C3", "H3C"), ("C5", "H5A"), ("C5", "H5B"), ("C4", "N1"), ("C1", "C2")]}
# bonds excluded from the summaries (field nearly perpendicular to the bond; main text Section 4.1)
EXCLUDED = {("urea", "C-N1"), ("glyala", "C1-C2")}
F000 = {"urea": 64.0, "glyala": 312.0}
def _num(s): return float(s.split("(")[0])
def load(mol):
    """Return dict(atoms={label: fractional xyz}, cell=array(a,b,c), ops=[symop strings], bonds=[(A,B)])."""
    b = gemmi.cif.read(str(CIF[mol])).sole_block()
    at = b.find(["_atom_site_label", "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"])
    ops = [o.strip("'") for o in b.find_loop("_symmetry_equiv_pos_as_xyz")]
    return dict(atoms={r[0]: tuple(_num(r[i]) for i in (1, 2, 3)) for r in at},
                cell=np.array([_num(b.find_value("_cell_length_" + x)) for x in "abc"]), ops=ops, bonds=BONDS[mol])
