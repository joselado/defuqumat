"""The Hofstadter butterfly of the honeycomb model: every level against the flux per cell."""
import numpy as np
import style

d = np.load("data.npz")


def draw():
    fig, ax = style.figure(1)
    ax.plot(d["flux"], d["energy"], ",", color=style.ink())
    ax.set_xlabel(r"flux per cell $p/q$ (flux quanta)")
    ax.set_ylabel(r"$E$ (eV)")
    ax.set_title(r"honeycomb, $t = -2.7$ eV, $q \leq 50$")
    ax.set_xlim(0.0, 1.0)
    return fig


style.main(draw)
