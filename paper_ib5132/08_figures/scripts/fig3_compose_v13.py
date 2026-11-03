import os
# Fig. 3 v13 (31 Oct 2026, single-column): A = segment 296-304 with mFo-DFc mesh and observed/IAM arrows (old A+B merged);
# B = O299 close-up against re-refined model (b) riding H; C = same view, model (c) with the added H on O299. 1-A scale bars.
import json
from PIL import Image, ImageDraw, ImageFont, ImageChops
X = json.load(open('fig4_exact.json')); DPI = 600; WFIN = int(3.46 * DPI)
def font(sz, bold=True):
    return ImageFont.truetype(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'), sz)   # DejaVu fonts shipped with the scripts
def redness(im, xy):
    x, y = int(xy[0]), int(xy[1])
    if not (0 <= x < im.width and 0 <= y < im.height): return -1e9
    r, g, b = im.getpixel((x, y))[:3]; return r - (g + b) / 2
def load(k):
    im = Image.open(f'fig3_{k}_v13.png').convert('RGB'); J = json.load(open(f'fig3_{k}_v13.json'))
    sx = im.width / J['size'][0]; w = [l for l in J['labels'] if l['text'] == 'W767'][0]
    conv = max(('rowvec', 'colvec'), key=lambda c: redness(im, (w[c][0] * sx, w[c][1] * sx)))
    return im, J, conv, sx
def bbox(im):
    bg = Image.new('RGB', im.size, (255, 255, 255)); return ImageChops.difference(im, bg).convert('L').point(lambda v: 255 if v > 12 else 0).getbbox()
def union(*bs): return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))
def pad(b, im, m): return (max(0, b[0]-m), max(0, b[1]-m), min(im.width, b[2]+m), min(im.height, b[3]+m))
def panel(k, crop, width, fsz, offs):
    im, J, conv, sx = load(k); c = im.crop(crop); s = width / c.width; c = c.resize((width, round(c.height * s)), Image.LANCZOS)
    d = ImageDraw.Draw(c); f = font(fsz)
    for l in J['labels']:
        x = (l[conv][0] * sx - crop[0]) * s; y = (l[conv][1] * sx - crop[1]) * s; dx, dy = offs.get(l['text'], (0, -1.1))
        x += dx * fsz; y += dy * fsz; bb = d.textbbox((0, 0), l['text'], font=f); tw = bb[2] - bb[0]
        d.text((x - tw / 2, y - (bb[3] - bb[1]) / 2), l['text'], fill='black', font=f, stroke_width=max(2, fsz // 12), stroke_fill='white')
    px_per_A = 1 / (J['A_per_px'] / sx) * s   # rendered px per A, after crop+resize
    return c, px_per_A
def scalebar(c, ppa, fsz):
    d = ImageDraw.Draw(c); L = int(round(ppa)); x0 = int(0.05 * c.width); y0 = int(c.height - 0.07 * c.height)
    d.rectangle([x0, y0, x0 + L, y0 + max(4, fsz // 6)], fill='black'); f = font(int(fsz * 0.8), bold=False)
    d.text((x0 + L / 2, y0 - 0.1 * fsz), '1 Å', fill='black', font=f, anchor='mb')
FS = int(0.105 * DPI)             # ~2.7 mm label height at final size
imA = load('A')[0]; cA = pad(bbox(imA), imA, int(0.02 * imA.width))
A, ppaA = panel('A', cA, WFIN, FS, {'W767': (0, -1.2)}); scalebar(A, ppaA, FS)
imB = load('Db')[0]; imC = load('Dc')[0]; u = pad(union(bbox(imB), bbox(imC)), imB, int(0.03 * imB.width))
gap = int(0.03 * WFIN); wBC = (WFIN - gap) // 2
B, ppaB = panel('Db', u, wBC, int(FS * 0.9), {'W767': (0, -1.2), 'Lys303 N': (1.6, -0.9)}); C, _ = panel('Dc', u, wBC, int(FS * 0.9), {'W767': (0, -1.2), 'Lys303 N': (1.6, -0.9)})
scalebar(B, ppaB, int(FS * 0.9)); scalebar(C, ppaB, int(FS * 0.9))
tH = int(FS * 1.6)
canvas = Image.new('RGB', (WFIN, A.height + gap + tH + B.height), 'white'); d = ImageDraw.Draw(canvas)
canvas.paste(A, (0, 0)); canvas.paste(B, (0, A.height + gap + tH)); canvas.paste(C, (wBC + gap, A.height + gap + tH))
fl = font(int(FS * 1.25)); ft = font(int(FS * 0.95), bold=False)
d.text((int(0.01 * WFIN), int(0.005 * WFIN)), 'A', fill='black', font=fl)
ang = {k: [a for a in X[k]['arrows'] if a[0] == 299][0][4] for k in ('b', 'c')}
for i, (lab, t, x) in enumerate((('B', f'(b) riding H: {ang["b"]:.1f}°', 0), ('C', f'(c) + H on O299: {ang["c"]:.1f}°', wBC + gap))):
    y = A.height + gap; d.text((x + int(0.01 * WFIN), y), lab, fill='black', font=fl)
    bb = d.textbbox((0, 0), t, font=ft); d.text((x + (wBC - (bb[2] - bb[0])) / 2 + FS * 0.5, y + int(0.15 * FS)), t, fill='black', font=ft)
canvas.save('Fig3_6mu9_O299.png', dpi=(DPI, DPI)); print(canvas.size, ang)
