"""Direct (exact) evaluation of the filtered maps, FFT maps on the sph-map grid, and reading of phased coefficients.
Convention: C_h = |F_h| exp(-i phi_CCP4), rho(x) = V^-1 sum_h C_h exp(+2 pi i q_h.x), q_h = A^-T h (cycles/A);
multipliers G = i^l R_lm(k^) (Eq. 2 of the paper, w = 1). Real orthonormal harmonics, Condon-Shortley phase:
R_{1,(-1,0,+1)} = sqrt(3/4pi) (y, z, x). Taken from the method-comparison code of the authors (common/, methods/A_sph_map)."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np
import gemmi
from scipy.special import sph_harm_y
from scipy.fft import ifftn, next_fast_len

# ---- reading coefficients


@dataclass
class Coefficients:
    cell: gemmi.UnitCell
    hkl: np.ndarray            # (N,3) int, full P1 set incl. both Friedel mates (and (0,0,0) only if F000 given)
    C: np.ndarray              # (N,) complex128, internal +exponent convention
    info: dict = field(default_factory=dict)

    @property
    def V(self) -> float:
        return self.cell.volume

    @property
    def q(self) -> np.ndarray:
        """Cartesian reciprocal vectors, cycles/A (model frame, x || a)."""
        return self.hkl.astype(np.float64) @ np.linalg.inv(np.array(self.cell.orth.mat, dtype=np.float64))

    @property
    def k(self) -> np.ndarray:
        """Cartesian reciprocal vectors, rad/A (k = 2 pi q)."""
        return 2 * np.pi * self.q

    @property
    def dmin(self) -> float:
        qn = np.linalg.norm(self.q, axis=1)
        return float(1.0 / qn[qn > 0].max())


def complete_friedel(hkl: np.ndarray, C: np.ndarray, tol: float = 0.005):
    """Same rules as radial_angular_mtz.complete_friedel, vectorised: average consistent duplicates,
    generate missing mates as conj, reject pairs whose relative disagreement exceeds tol."""
    lookup: dict[tuple, complex] = {}
    dup = 0
    for h, c in zip(map(tuple, hkl.tolist()), C.tolist()):
        if h in lookup:
            old = lookup[h]
            if abs(old - c) > tol * max(1.0, abs(old), abs(c)):
                raise ValueError(f"conflicting duplicate P1 coefficient for {h}")
            lookup[h] = (old + c) / 2; dup += 1
        else:
            lookup[h] = c
    added = 0; worst = 0.0
    for h in list(lookup):
        mh = (-h[0], -h[1], -h[2])
        if mh not in lookup:
            lookup[mh] = np.conj(lookup[h]); added += 1
        elif mh != h:
            c, d = lookup[h], lookup[mh]
            disc = abs(c - np.conj(d)) / max(1.0, abs(c), abs(d)); worst = max(worst, disc)
            if disc > tol:
                raise ValueError(f"inconsistent Friedel pair {h}/{mh} (relative {disc:.3g})")
            avg = (c + np.conj(d)) / 2; lookup[h], lookup[mh] = avg, np.conj(avg)
    keys = np.array(list(lookup), dtype=np.int32)
    vals = np.array(list(lookup.values()), dtype=np.complex128)
    return keys, vals, dict(friedel_added=added, duplicates_averaged=dup, friedel_max_rel_discrepancy=worst)


def load(path, amplitude: str, phase: str, *, f000: float = 0.0, dmin: float | None = None) -> Coefficients:
    mtz = gemmi.read_mtz_file(str(path))
    labels = mtz.column_labels()
    for lab, typ in ((amplitude, "F"), (phase, "P")):
        if lab not in labels:
            raise ValueError(f"{path}: column {lab} missing ({labels})")
        if mtz.column_with_label(lab).type != typ:
            raise ValueError(f"{path}: {lab} has type {mtz.column_with_label(lab).type}, expected {typ}")
    sg_in = mtz.spacegroup.hm
    rows_in = mtz.nreflections
    mtz.expand_to_p1()
    hkl = mtz.make_miller_array().astype(np.int32)
    amp = np.asarray(mtz.column_with_label(amplitude).array, dtype=np.float64)
    ph = np.asarray(mtz.column_with_label(phase).array, dtype=np.float64)
    ok = np.isfinite(amp) & np.isfinite(ph) & (amp >= 0)
    hkl, amp, ph = hkl[ok], amp[ok], ph[ok]
    zero = np.all(hkl == 0, axis=1)
    had_000 = bool(zero.any())
    hkl, amp, ph = hkl[~zero], amp[~zero], ph[~zero]       # F(000) handled only via the f000 argument
    C = amp * np.exp(-1j * np.deg2rad(ph))
    hkl, C, fstats = complete_friedel(hkl, C)
    cell = mtz.cell
    q = hkl @ np.linalg.inv(np.array(cell.orth.mat))
    if dmin is not None:
        keep = np.linalg.norm(q, axis=1) <= (1 / dmin) * (1 + 1e-8)
        hkl, C = hkl[keep], C[keep]
    if f000:
        hkl = np.vstack([hkl, np.zeros((1, 3), np.int32)]); C = np.append(C, complex(f000, 0.0))
    info = dict(path=str(path), amplitude=amplitude, phase=phase, space_group_in=sg_in, rows_in=rows_in,
                rows_finite=int(ok.sum()), input_had_000_row=had_000, n_p1=int(len(hkl)), f000=float(f000),
                dmin_cut=dmin, cell=list(cell.parameters), **fstats)
    co = Coefficients(cell, hkl, C, info)
    info["dmin"] = co.dmin
    return co

# ---- real spherical harmonics


def _real(ell: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    if m == 0:
        return np.asarray(sph_harm_y(ell, 0, theta, phi).real)
    h = sph_harm_y(ell, abs(m), theta, phi)
    sign = (-1) ** abs(m)
    return math.sqrt(2) * sign * np.asarray(h.real if m > 0 else h.imag)


def real_ylm(vectors: np.ndarray, lmax: int) -> list[np.ndarray]:
    """Return [Y_0, ..., Y_lmax]; Y_l has shape (N, 2l+1), columns m = -l..l, evaluated at vectors/|vectors|.
    For zero-length vectors the l >= 1 values are set to 0 (direction undefined)."""
    v = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
    r = np.linalg.norm(v, axis=1)
    nz = r > 0
    theta = np.zeros(len(v)); phi = np.zeros(len(v))
    theta[nz] = np.arccos(np.clip(v[nz, 2] / r[nz], -1, 1))
    phi[nz] = np.mod(np.arctan2(v[nz, 1], v[nz, 0]), 2 * np.pi)
    out = []
    for ell in range(lmax + 1):
        y = np.stack([_real(ell, m, theta, phi) for m in range(-ell, ell + 1)], axis=1)
        if ell:
            y[~nz] = 0.0
        out.append(y)
    return out


def l1_to_xyz(c: np.ndarray) -> np.ndarray:
    """Map an l=1 multiplet (..., 3) in m order (-1,0,+1) to (x, y, z) = (c_{+1}, c_{-1}, c_0)."""
    c = np.asarray(c)
    return np.stack([c[..., 2], c[..., 0], c[..., 1]], axis=-1)

# ---- exact sums and FFT maps


def exact_sum(co: Coefficients, G: np.ndarray, points_cart: np.ndarray, max_elements: int = 8_000_000) -> np.ndarray:
    """Real map values V^-1 Re sum_h G_h C_h exp(2 pi i q_h.x) at Cartesian points.
    G: (N,) or (N,K) multipliers. Returns (P,) or (P,K)."""
    """Memory: points are processed in blocks of max_elements // N_reflections (≈128 MB of complex128 per block)."""
    G = np.asarray(G); single = G.ndim == 1
    if single:
        G = G[:, None]
    X = np.atleast_2d(np.asarray(points_cart, dtype=np.float64))
    q = co.q
    W = G * co.C[:, None]                                  # (N,K)
    out = np.zeros((len(X), G.shape[1]))
    chunk = max(1, max_elements // max(1, len(q)))
    for s in range(0, len(X), chunk):
        E = np.exp(2j * np.pi * (X[s:s + chunk] @ q.T))      # (p,N)
        out[s:s + chunk] = (E @ W).real
    out /= co.V
    return out[:, 0] if single else out


def sphmap_grid_shape(co: Coefficients, samples_per_dmin: float) -> tuple[int, int, int]:
    """sph-map 11.8.0 rule (DIRECTIONAL_MAP_SPECIFICATION.md): n_i = max(ceil(a_i/(d_min/s)), 2 h_i,max + 1),
    rounded up with next_fast_len; d_min from the rows actually used."""
    hk = co.hkl[np.any(co.hkl != 0, axis=1)]
    spacing = co.dmin / samples_per_dmin
    abc = co.cell.parameters[:3]
    return tuple(int(next_fast_len(max(math.ceil(a / spacing), 2 * int(np.abs(hk[:, i]).max()) + 1)))
                 for i, a in enumerate(abc))


def fft_map(co: Coefficients, G: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Whole-cell real map of V^-1 sum G C exp(+2 pi i h.r) on grid `shape` (array axes u,v,w = a,b,c)."""
    hk = co.hkl
    need = [2 * int(np.abs(hk[:, i]).max()) + 1 for i in range(3)]
    if any(n < r for n, r in zip(shape, need)):
        raise ValueError(f"grid {shape} aliases; need >= {need}")
    g = np.zeros(shape, dtype=np.complex128)
    idx = hk % np.asarray(shape)
    np.add.at(g, (idx[:, 0], idx[:, 1], idx[:, 2]), np.asarray(G) * co.C)
    m = ifftn(g) * (math.prod(shape) / co.V)
    leak = float(np.abs(m.imag).max())
    if leak > 1e-8 * max(1.0, float(np.abs(m.real).max())):
        raise ValueError(f"imaginary leakage {leak:.3g}")
    return m.real


def read_ccp4(path) -> gemmi.FloatGrid:
    m = gemmi.read_ccp4_map(str(path)); m.setup(0.0)
    return m.grid


def grid_from_array(arr: np.ndarray, cell: gemmi.UnitCell) -> gemmi.FloatGrid:
    g = gemmi.FloatGrid(np.ascontiguousarray(arr, dtype=np.float32))
    g.unit_cell = cell; g.spacegroup = gemmi.SpaceGroup("P 1")
    return g


def write_ccp4(path, arr: np.ndarray, cell: gemmi.UnitCell) -> None:
    c = gemmi.Ccp4Map(); c.grid = grid_from_array(arr, cell); c.update_ccp4_header(); c.write_ccp4_map(str(path))


def trilinear(grid: gemmi.FloatGrid, frac_points: np.ndarray) -> np.ndarray:
    """Paper path: gemmi trilinear interpolation at fractional points (one call per point, as in the paper scripts)."""
    return np.array([grid.interpolate_value(gemmi.Fractional(*map(float, p))) for p in np.atleast_2d(frac_points)])


def trilinear_np(arr: np.ndarray, frac_points: np.ndarray) -> np.ndarray:
    """Vectorised periodic trilinear interpolation of a whole-cell grid at fractional points; same definition as
    gemmi interpolate_value (nodes at i/n). Used where millions of points are needed (6MU9 spheres for D)."""
    n = np.array(arr.shape); f = np.atleast_2d(frac_points) * n
    i0 = np.floor(f).astype(int); t = f - i0
    out = np.zeros(len(f))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (t[:, 0] if dx else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1]) * (t[:, 2] if dz else 1 - t[:, 2])
                out += w * arr[(i0[:, 0] + dx) % n[0], (i0[:, 1] + dy) % n[1], (i0[:, 2] + dz) % n[2]]
    return out

# ---- multipliers of Eq. (2)
def multipliers(co: Coefficients, lmax: int = 1, frame=None, radial=None) -> np.ndarray:
    """G_A,lm = i^l R_lm(k^) [× radial(l, k)]: columns [l=0 | l=1 m=-1,0,+1 | ...].
    `frame`: 3×3, columns = local axes in the model frame. `radial`: optional callable (l, k[rad/Å]) -> factor;
    None = paper settings (factor 1). Used for bridge 3 (Gaussian radial factor reproduces method D)."""
    q = co.q if frame is None else co.q @ np.asarray(frame)
    Y = real_ylm(q, lmax); k = 2 * np.pi * np.linalg.norm(q, axis=1)
    cols = []
    for l in range(lmax + 1):
        f = (1j ** l) * (np.ones_like(k) if radial is None else radial(l, k))
        cols.append(f[:, None] * Y[l])
    return np.concatenate(cols, axis=1)


def exact(co: Coefficients, cart_points, lmax: int = 1, frame=None) -> np.ndarray:
    return exact_sum(co, multipliers(co, lmax, frame), cart_points)




# ---- coefficient sets, phase controls and sampling spheres
def align(co_ref, co):
    idx = {tuple(h): i for i, h in enumerate(co.hkl.tolist())}
    j = np.array([idx[tuple(h)] for h in co_ref.hkl.tolist()]); return co.C[j]


def with_f000(co, C, f0, tag):
    keep = ~np.all(co.hkl == 0, axis=1)
    return Coefficients(co.cell, np.vstack([co.hkl[keep], [[0, 0, 0]]]).astype(np.int32), np.append(C[keep], f0), dict(source=tag, f000=f0))


def random_phases(hkl, seed):
    """uniform psi per Friedel pair, psi(-h) = -psi(h); (0,0,0) gets 0."""
    rng = np.random.default_rng(seed); key = {tuple(h): i for i, h in enumerate(hkl.tolist())}; psi = np.zeros(len(hkl)); done = np.zeros(len(hkl), bool)
    r = rng.uniform(0, 2 * np.pi, len(hkl))
    for i, h in enumerate(hkl.tolist()):
        if done[i]: continue
        j = key.get((-h[0], -h[1], -h[2]), i)
        if j == i: psi[i] = 0.0 if not any(h) else (0.0 if r[i] < np.pi else np.pi)   # self-mate (000): real
        else: psi[i], psi[j] = r[i], -r[i]; done[j] = True
        done[i] = True
    return np.exp(1j * psi)


def sphere_offsets(r0: float, step: float = 0.1) -> np.ndarray:
    """Sphere rule of the paper: the 0.1 A cubic-lattice points with |p| <= r0 (surface points included, exact integer test)."""
    from lattice import ball
    return ball(r0, step)
