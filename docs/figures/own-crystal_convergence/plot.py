"""How far the energy is from its last value, against the cutoff and the k-mesh."""
import numpy as np
import style

d = np.load("data.npz")


def draw():
    fig, (a, b) = style.figure(2)
    for ax, x, e, label in ((a, d["cutoffs"], d["e_cutoff"], "ecutwfc (Ry)"),
                            (b, d["meshes"], d["e_mesh"], r"k-mesh $n\times n\times n$")):
        ax.semilogy(x[:-1], np.abs(e[:-1] - e[-1]), "o-")
        ax.axhline(1e-4, **style.reference(), label=r"$10^{-4}$ Ry")
        ax.set_xlabel(label)
        ax.legend(loc="upper right")
    a.set_ylabel(r"$|E - E_{\rm last}|$ (Ry)")
    b.set_xticks(d["meshes"][:-1])
    return fig


style.main(draw)
