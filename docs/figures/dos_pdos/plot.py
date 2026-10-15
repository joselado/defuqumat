"""Plot figure dos_pdos from data.npz.

Writes docs/guide/img/dos_pdos.png and -dark.png.
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
    c = style.colors(theme)
    fig, ax = style.figure(1)
    e = d["energies"]
    ax.fill_between(e, d["total"], color=c["ink2"], alpha=0.18, lw=0, label="total")
    ax.plot(e, d["s"], color=c["series"][0], label=r"$s$, both atoms")
    ax.plot(e, d["p"], color=c["series"][1], label=r"$p$, both atoms")
    ax.axvline(0.0, color=c["ink2"], lw=0.8)
    ax.set_xlim(-13.0, 6.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$E - E_{\rm VBM}$ (eV)")
    ax.set_ylabel("PDOS (states/eV per cell)")
    ax.legend(loc="upper left")
    return fig


style.main(draw, "dos_pdos")
