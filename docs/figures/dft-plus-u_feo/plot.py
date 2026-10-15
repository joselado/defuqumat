"""FeO: the total DOS, and the eigenvalues of the occupation matrices."""
import numpy as np
import style

d = np.load("data.npz")


def draw():
    fig, (a, b) = style.figure(2)
    for k, (name, label) in enumerate((("noU", "U close to 0"), ("U", "U = 4.3 eV"))):
        a.plot(d[name + "_e"], d[name + "_dos"], color=style.color(k), label=label)
        b.plot(d[name + "_occ"], np.full(d[name + "_occ"].size, k), "o",
               color=style.color(k), alpha=0.7, ms=8)
    a.axvline(0.0, **style.reference())
    a.set_xlim(-8.0, 5.0)
    a.set_ylim(bottom=0.0)
    a.set_xlabel(r"$E - E_F$ (eV)")
    a.set_ylabel("DOS (states/eV per cell)")
    a.legend(loc="upper left")
    b.set_yticks([0, 1], ["U close to 0", "U = 4.3 eV"])
    b.set_ylim(-0.6, 1.6)
    b.set_xlim(-0.05, 1.05)
    b.set_xlabel("eigenvalues of the occupation matrix of Fe 3d")
    return fig


style.main(draw)
