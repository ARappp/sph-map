"""Step 08b: draw all figures of the paper from WORK/08_figures/data (written by a_figure_data.py).
The figure scripts in scripts/ are those used for the manuscript; each is run with the data folder as working folder.
Fig. 3 needs PyMOL (pymol-open-source, as a Python module) and Pillow. Output: WORK/08_figures/*.pdf, *.png
  Fig. 1  Fig1_smallmol.pdf            Fig. S1  FigS2_channels.pdf (file name from an earlier numbering)
  Fig. 2  Fig2_descriptors.pdf         Fig. S2  FigS_fieldlines.pdf
  Fig. 3  Fig3_6mu9_O299.png           Fig. S3  FigS3_ratio_profiles.pdf
  Fig. 4  Fig4_6mu9_resolution.pdf     Fig. S4  FigS4_bond_angles.pdf
  Fig. 5  Fig5_aromatics.pdf           Fig. S5  FigS5_O299_inclination.pdf
                                       Fig. S6  FigS6_aromatics.pdf"""
import _paths, os, sys, subprocess, shutil
from pathlib import Path
from _paths import step
OUT = step("08_figures"); DATA = OUT / "data"; SCR = Path(__file__).resolve().parent / "scripts"
RUN = ["make_fig1_v5.py", "make_figS3_ratio_profiles_v4.py", "make_fig2_v6.py", "fig3_render_v13.py", "fig3_compose_v13.py",
       "make_fig4_v5.py", "make_fig5_v6.py", "make_figS6_v4.py", "figS2_channels_v3.py", "make_figS_fieldlines_v2.py"]
FIGS = ["Fig1_smallmol", "Fig2_descriptors", "Fig3_6mu9_O299", "Fig4_6mu9_resolution", "Fig5_aromatics", "FigS2_channels", "FigS_fieldlines",
        "FigS3_ratio_profiles", "FigS4_bond_angles", "FigS5_O299_inclination", "FigS6_aromatics"]
env = dict(os.environ, PYTHONPATH=str(SCR) + os.pathsep + os.environ.get("PYTHONPATH", ""), MPLBACKEND="Agg")
only = sys.argv[1:]
for s in RUN:
    if only and not any(o in s for o in only): continue
    print("==", s, flush=True); subprocess.run([sys.executable, str(SCR / s)], cwd=DATA, env=env, check=True)
for f in FIGS:
    for ext in (".pdf", ".png"):
        if (DATA / (f + ext)).exists(): shutil.move(str(DATA / (f + ext)), OUT / (f + ext))
print("figures in", OUT); [print("  ", p.name) for p in sorted(OUT.glob("*.p*"))]
