# Fig. 4 (6MU9) v12, 27 Oct 2026: as v11 (fig4_render.py) but
#  - arrows from exact per-site fields (fig4_exact.json, written by fig4_exact_data.py), deposited coordinates;
#  - mFo-DFc maps computed with gemmi from the MTZ files (fofc_dep/b/c.ccp4);
#  - ray-traced (headless);
#  - new panel D: O299 against the re-refined models (b) riding H and (c) (b) + H on O299, same view as C.
# Run: python3 fig4_render_v12.py   (pymol-open-source 3.x as a Python module)
import os, json, numpy as np
import pymol; pymol.finish_launching(['pymol', '-cq'])
from pymol import cmd
X = json.load(open('fig4_exact.json'))
BLUE = (0.16, 0.47, 0.84); GREY = (0.30, 0.30, 0.30); SIG = 2.8; W, Hh = 2400, 1800
def arrow_cgo(start, direction, length, color, rad, head_len=0.45, head_rad=None):
    head_rad = head_rad or 2.3 * rad
    s = np.array(start, float); u = np.array(direction, float); u /= np.linalg.norm(u)
    end = s + u * length; neck = end - u * head_len; c = list(color)
    return [9.0, *s, *neck, rad, *c, *c] + [27.0, *neck, *end, head_rad, 0.0, *c, *c, 1.0, 1.0]
def base(pdb='6MU9_deposited.pdb', mapf='fofc_dep.ccp4', keepH=False):
    cmd.reinitialize()
    cmd.set_color('diffpos', [0.05, 0.62, 0.05])
    cmd.set('auto_zoom', 0); cmd.set('normalize_ccp4_maps', 0)
    cmd.load(pdb, 'prot'); cmd.load(mapf, 'mapfofc')
    if keepH: cmd.remove('prot and hydro and not resn HOX')
    else: cmd.remove('prot and hydro')
    cmd.remove("prot and not (alt '' or alt A)"); cmd.alter('prot', 'alt=""')
    cmd.hide('everything')
    cmd.select('bb', 'prot and resi 296-304 and name N+CA+C+O+CB and not resn HOH')
    cmd.select('w767', 'prot and resn HOH and resi 767')
    cmd.show('sticks', 'bb'); cmd.show('spheres', 'w767')
    cmd.set('sphere_scale', 0.18); cmd.set('stick_radius', 0.10); cmd.set('stick_transparency', 0.35, 'prot')
    cmd.color('grey85', 'bb and elem C'); cmd.color('palecyan', 'bb and resi 299 and elem C')
    cmd.color('blue', 'bb and elem N'); cmd.color('red', 'bb and elem O'); cmd.color('red', 'w767')
    for k, v in dict(bg_rgb='white', valence=0, ray_trace_mode=1, ray_trace_gain=0.06, antialias=2, ray_shadows=0,
                     specular=0.1, ambient=0.5, depth_cue=0, ray_opaque_background=1, float_labels=1, dash_color='black',
                     dash_gap=0.3, dash_radius=0.035, mesh_radius=0.018, orthoscopic=1, ray_trace_color='black').items():
        cmd.set(k, v)
    cmd.distance('hb1', 'prot and resi 299 and name O', 'w767'); cmd.distance('hb2', 'prot and resi 299 and name O', 'prot and resi 303 and name N')
    cmd.hide('labels', 'hb1'); cmd.hide('labels', 'hb2')
def mesh(tag, resis, rms, lev=None):
    for r in resis:
        pk = X[tag]['peaks'][str(r)]; l = lev or (2.0 if (tag == 'dep' and r == 298) else SIG)
        cmd.pseudoatom(f'pk{r}', pos=pk['pos']); cmd.hide('everything', f'pk{r}')
        cmd.isomesh(f'fofc_p{r}', 'mapfofc', l * rms, f'pk{r}', carve=0.9); cmd.color('diffpos', f'fofc_p{r}')
def arrows(tag, resis=None, length=1.7, rad=0.12):
    for resi, start, ud, um, a in X[tag]['arrows']:
        if resis and resi not in resis: continue
        cmd.load_cgo(arrow_cgo(start, um, length, GREY, rad * 0.8), f'ar_model_{resi}')
        cmd.load_cgo(arrow_cgo(start, ud, length, BLUE, rad), f'ar_data_{resi}')
def plane_view(center_sel, extent_sel, buffer):
    a = [x for x in X['dep']['arrows'] if x[0] == 299][0]
    n = np.cross(a[2], a[3]); n /= np.linalg.norm(n)
    m = cmd.get_model('prot and resi 297+303 and name CA').atom
    ax = np.array(m[1].coord) - np.array(m[0].coord); ax -= ax.dot(n) * n; ax /= np.linalg.norm(ax); by = np.cross(n, ax)
    c = np.array(cmd.get_coords(center_sel).mean(0))
    for i, (u, L) in enumerate(((ax, 6.0), (by, 3.0), (n, 0.5))):
        for sgn in (1, -1): cmd.pseudoatom('vp', pos=list(c + sgn * L * u), name=f'P{i}{sgn}')
    cmd.orient('vp'); cmd.delete('vp'); cmd.zoom(extent_sel, buffer)
def screen_xy(xyz):
    v = cmd.get_view(); R = np.array(v[0:9]).reshape(3, 3); o = np.array(v[12:15]); cam = np.array(v[9:12])
    fov = float(cmd.get('field_of_view')); hgt = 2.0 * abs(v[11]) * np.tan(np.radians(fov) / 2.0); wid = hgt * W / Hh
    out = {}
    for tag, M in (('rowvec', R), ('colvec', R.T)):
        c = (np.asarray(xyz) - o) @ M + cam; out[tag] = [(c[0] / wid + 0.5) * W, (0.5 - c[1] / hgt) * Hh]
    return out
def render(name, labels):
    cmd.hide('labels'); cmd.ray(W, Hh); cmd.png(name, dpi=300)
    lab = [dict(text=t, **screen_xy(cmd.get_coords(sel)[0])) for t, sel in labels]
    lab.append(dict(text='W767', **screen_xy(cmd.get_coords('w767')[0])))
    v = cmd.get_view(); fov = float(cmd.get('field_of_view')); hgt = 2.0 * abs(v[11]) * np.tan(np.radians(fov) / 2.0)
    json.dump(dict(labels=lab, size=[W, Hh], A_per_px=hgt / Hh), open(name.replace('.png', '.json'), 'w'), indent=1); print('wrote', name)
SEG = [('Ala298', 'prot and resi 298 and name CA'), ('Lys299', 'prot and resi 299 and name CA'), ('Val302', 'prot and resi 302 and name CA')]
# A: segment with difference density and arrows together (old A and B merged)
base(); rms = X['dep']['fofc_rms']; mesh('dep', (298, 299, 302), rms); arrows('dep')
plane_view('prot and resi 297-303 and name CA', 'bb or w767', 0.6); render('fig3_A_v13.png', SEG)
def closeup(tag, pdb, mapf, keepH, out, rms):
    base(pdb, mapf, keepH)
    cmd.hide('sticks', 'bb and not resi 299+303'); cmd.show('sticks', 'prot and resi 300 and name N')
    if keepH:
        cmd.bond('prot and resi 299 and name O', 'prot and resn HOX')
        cmd.show('sticks', 'prot and resn HOX'); cmd.show('spheres', 'prot and resn HOX')
        cmd.color('grey50', 'prot and resn HOX'); cmd.set('sphere_scale', 0.13, 'prot and resn HOX')
    mesh(tag, (299,), rms); arrows(tag, resis=(299,), length=1.5, rad=0.1)
    global VIEWC
    if VIEWC is None:
        plane_view('(prot and resi 299 and name C+O) or w767 or (prot and resi 303 and name N)',
                   '(prot and resi 299 and name CA+C+O) or w767 or (prot and resi 303 and name N+CA)', 2.2); VIEWC = cmd.get_view()
    cmd.set_view(VIEWC)
    render(out, [('Lys303 N', 'prot and resi 303 and name N'), ('Lys299', 'prot and resi 299 and name CA')])
VIEWC = None
closeup('b', 'rerefined_b_ridingH.pdb', 'fofc_b.ccp4', False, 'fig3_Db_v13.png', X['b']['fofc_rms'])
closeup('c', 'rerefined_c_ridingH_plus_H_O299.pdb', 'fofc_c.ccp4', True, 'fig3_Dc_v13.png', X['c']['fofc_rms'])
