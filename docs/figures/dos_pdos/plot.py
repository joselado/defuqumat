"""The projected DOS of silicon: s and p of both atoms over the total."""
import numpy as np
import style

d = np.load("data.npz")


def draw():
    fig, ax = style.figure()
    e = d["energies"]
    ax.fill_between(e, d["total"], color=style.ink2(), alpha=0.18, lw=0, label="total")
    ax.plot(e, d["s"], label=r"$s$, both atoms")
    ax.plot(e, d["p"], label=r"$p$, both atoms")
    style.zero_line(ax, vertical=True)
    ax.set_xlim(-13.0, 6.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$E - E_{\rm VBM}$ (eV)")
    ax.set_ylabel("PDOS (states/eV per cell)")
    ax.legend(loc="upper left")
    return fig


style.main(draw)
