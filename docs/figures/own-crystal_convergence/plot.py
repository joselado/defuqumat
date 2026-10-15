"""Plot figure own-crystal_convergence from data.npz.

Writes docs/guide/img/own-crystal_convergence.png and -dark.png.
    python3 plot.py [--theme light|dark]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common
import style

params, d = common.load(os.path.join(HERE, "data.npz"))


def draw(theme):
    c = style.colors(theme)
    fig, (a, b) = style.figure(2)
    for ax, x, e, label in ((a, d["cutoffs"], d["e_cutoff"], "ecutwfc (Ry)"),
                            (b, d["meshes"], d["e_mesh"], r"k-mesh $n\times n\times n$")):
        ax.semilogy(x[:-1], np.abs(e[:-1] - e[-1]), "o-", color=c["series"][0])
        ax.axhline(1e-4, **style.reference(theme), label=r"$10^{-4}$ Ry")
        ax.set_xlabel(label)
        ax.legend(loc="upper right")
    a.set_ylabel(r"$|E - E_{\rm last}|$ (Ry)")
    b.set_xticks(d["meshes"][:-1])
    return fig


style.main(draw, "own-crystal_convergence")
