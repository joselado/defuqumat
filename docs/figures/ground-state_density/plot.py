"""Plot figure ground-state_density from data.npz.

Writes docs/guide/img/ground-state_density.png and -dark.png.
    python3 plot.py [--theme light|dark]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common
import style

params, d = common.load(os.path.join(HERE, "data.npz"))


def draw(theme):
    fig, ax = style.figure(1)
    mesh = ax.pcolormesh(d["u"], d["v"], d["density"].T, shading="gouraud")
    ax.scatter(d["atoms"][:, 0], d["atoms"][:, 1], s=40, facecolors="none",
               edgecolors="white", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlabel("along [110] (bohr)")
    ax.set_ylabel("along [001] (bohr)")
    style.colorbar(fig, mesh, ax, r"$n(\mathbf{r})$ (electrons/bohr$^3$)", theme)
    return fig


style.main(draw, "ground-state_density")
