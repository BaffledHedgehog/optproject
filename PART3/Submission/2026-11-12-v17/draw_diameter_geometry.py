"""Draw the two proved triangle constructions; no numerical optimization."""
from pathlib import Path
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                     "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(6.1, 2.35))
specs = [(0.0, 0.6, r"$r$"), (-0.3, 0.5, r"$g/P$")]
for index, (ax, (left, right, base_label)) in enumerate(zip(axes, specs)):
    length = right - left
    top = ((left + right) / 2, math.sqrt(1 - length * length / 4))
    ax.add_patch(Polygon([(left, 0), (right, 0), top], closed=True,
                         facecolor="#e7edf3", edgecolor="#33495b", linewidth=1.25))
    ax.plot([0, right], [0, 0], "o", color="#1c2833", markersize=4)
    ax.text(0, -0.09, r"$x^*$", ha="center", va="top")
    ax.text(right + 0.04, -0.02, r"$x$", ha="left", va="top")
    ax.text(left + (top[0] - left) * .52 - .065, top[1] * .52, r"$D$", ha="right")
    ax.text(right + (top[0] - right) * .52 + .065, top[1] * .52, r"$D$", ha="left")
    ax.annotate("", xy=(left, -.23), xytext=(right, -.23),
                arrowprops={"arrowstyle": "|-|", "linewidth": .8, "color": "#586875"})
    ax.text((left + right) / 2, -.25, base_label, ha="center", va="top")
    delta = (.18, .1125) if index == 0 else (.24, 0)
    ax.annotate("", xy=(right + delta[0], delta[1]), xytext=(right, 0),
                arrowprops={"arrowstyle": "->", "linewidth": 1.3, "color": "#9c4326"})
    ax.text(right + delta[0] * .6, delta[1] + .08, r"$\nabla f(x)$",
            ha="left", va="bottom", color="#9c4326")
    ax.set_xlim(left - .17, right + .47)
    ax.set_ylim(-.43, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
fig.subplots_adjust(left=.015, right=.99, bottom=.03, top=.99, wspace=.13)
fig.savefig(OUT / "diameter_geometry.pdf", bbox_inches="tight", pad_inches=.025)
fig.savefig(OUT / "diameter_geometry.png", dpi=180, bbox_inches="tight", pad_inches=.025)
print("Saved diameter construction diagram.")
