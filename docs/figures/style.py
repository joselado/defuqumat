"""Shared matplotlib style of the user-guide figures.

Every docs/figures/<id>/plot.py uses it the same way:

    import numpy as np
    import style

    d = np.load("data.npz")

    def draw():
        fig, ax = style.figure()           # 1, 2 or 3 panels
        ax.plot(d["x"], d["y"], label="defuqumat")       # series colors in turn
        ax.plot(d["x"], d["exact"], **style.reference(), label="exact")
        ax.legend()
        return fig

    style.main(draw)    # calls draw() once per theme, writes both PNG files

plot.py is started by docs/figures/plot.py inside its own folder, so the
figure id is the name of that folder. The PNG files go to docs/guide/img/.
The plot is transparent, so the page color shows through, and the ink
colors switch with the theme. style.ink(), style.ink2() and
style.color(k) give the colors of the theme being drawn. This module never
imports defuqumat.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(os.path.dirname(HERE), "guide", "img")

INK = {"light": "#0b0b0b", "dark": "#f0efec"}
INK2 = {"light": "#52514e", "dark": "#c3c2b7"}

# blue, orange, aqua, yellow, magenta, green, violet, red
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500",
             "#d55181", "#008300", "#9085e9", "#e66767"],
}

LW = 1.6          # computed data, lines
MS = 5            # computed data, markers
LW_REF = 1.2      # reference curves (experiment, an analytic curve)
CMAP = "magma"    # heat maps, both themes
SIZES = {1: (6.4, 4.0), 2: (9.6, 3.8), 3: (11.0, 3.6)}
DPI = 160


THEME = "light"     # the theme being drawn, set by apply()


def ink():
    """The ink color of the theme being drawn."""
    return INK[THEME]


def ink2():
    """The secondary ink color of the theme being drawn."""
    return INK2[THEME]


def color(k):
    """Series color k of the theme being drawn."""
    return SERIES[THEME][k]


def reference():
    """Line keywords of a reference curve: ink, dashed, 1.2 pt."""
    return {"color": INK[THEME], "ls": "--", "lw": LW_REF}


def zero_line(ax, vertical=False):
    """A thin line at zero, in the secondary ink."""
    (ax.axvline if vertical else ax.axhline)(0.0, color=INK2[THEME], lw=0.8)


def heatmap_line():
    """Line keywords of a curve drawn on a heat map: white, dashed."""
    return {"color": "white", "ls": "--", "lw": LW_REF}


def apply(theme):
    """Set the rcParams of a theme. Called by main()."""
    global THEME
    THEME = theme
    ink, ink2 = INK[theme], INK2[theme]
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": ink,
        "axes.labelcolor": ink,
        "axes.facecolor": "none",
        "figure.facecolor": "none",
        "savefig.facecolor": "none",
        "text.color": ink,
        "xtick.color": ink,
        "ytick.color": ink,
        "xtick.labelcolor": ink2,
        "ytick.labelcolor": ink2,
        "legend.labelcolor": ink,
        "axes.prop_cycle": plt.cycler(color=SERIES[theme]),
        "image.cmap": CMAP,
        "lines.linewidth": LW,
        "lines.markersize": MS,
    })


def figure(npanels=1, **kwargs):
    """A figure with 1, 2 or 3 panels side by side, in the plan's sizes."""
    fig, axs = plt.subplots(1, npanels, figsize=SIZES[npanels], **kwargs)
    return fig, axs


def colorbar(fig, mappable, ax, label):
    """A colorbar with its outline, ticks and label in the ink color."""
    cb = fig.colorbar(mappable, ax=ax)
    cb.set_label(label, color=INK[THEME])
    cb.outline.set_edgecolor(INK[THEME])
    cb.ax.tick_params(colors=INK[THEME], labelcolor=INK2[THEME])
    return cb


def save(fig, figid, theme):
    """Write docs/guide/img/<id>.png (light) or <id>-dark.png (dark)."""
    os.makedirs(IMG, exist_ok=True)
    name = figid + ("-dark" if theme == "dark" else "") + ".png"
    path = os.path.join(IMG, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("wrote", path)
    return path


def main(draw, figid=None):
    """Call draw() once per theme and save. The id is the current folder.

    THEMES (set by docs/figures/plot.py --theme) limits it to one theme.
    """
    figid = figid or os.path.basename(os.getcwd())
    for theme in THEMES:
        apply(theme)
        save(draw(), figid, theme)


THEMES = ["light", "dark"]
