"""Graphene at 10 T: E_n against sgn(n) sqrt(|n|), with the Dirac line and the zero modes."""
import numpy as np
import style

d = np.load("data.npz")


def draw():
    fig, ax = style.figure(1)
    nmax = 10
    x = np.sign(np.arange(-nmax, nmax + 1)) * np.sqrt(np.abs(np.arange(-nmax, nmax + 1)))
    ax.plot(x, d["dirac"] * 1e3, **style.reference(), label="Dirac formula")
    n = d["n"]
    ax.plot(np.sign(n) * np.sqrt(np.abs(n)), d["energy"] * 1e3, "o", color=style.color(0),
            label="Peierls on the $p_z$ Wannier model")
    ax.plot([0.0], [float(d["zero_level"]) * 1e3], "s", color=style.color(1), ms=8,
            label=f"zero modes ({int(d['zero_multiplicity'])})")
    style.zero_line(ax)
    ax.set_xlabel(r"${\rm sgn}(n)\sqrt{|n|}$")
    ax.set_ylabel(r"$E_n - E_D$ (meV)")
    ax.set_title(f"graphene, B = {float(d['field']):.2f} T")
    ax.legend(loc="upper left")
    return fig


style.main(draw)
