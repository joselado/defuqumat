"""PBE alone, the D2 pair sum, and their sum, against the interlayer distance."""
import numpy as np
import style

d = np.load("data.npz")
x = d["separation"]
mev = lambda e: (e - e[-1]) * 13605.693 / 4       # meV per atom, from 10 bohr


def draw():
    fig, ax = style.figure()
    ax.plot(x, mev(d["total"] - d["dispersion"]), "o-", label="PBE alone")
    ax.plot(x, mev(d["dispersion"]), "s-", label="D2 pair sum")
    ax.plot(x, mev(d["total"]), "o-", lw=2.2, label="PBE + D2")
    ax.axvline(3.35 / 0.529177, **style.reference(), label="experiment, 3.35 Å")
    style.zero_line(ax)
    ax.set_xlabel("distance between the layers (bohr)")
    ax.set_ylabel("energy from 10 bohr (meV/atom)")
    ax.legend()
    return fig


style.main(draw)
