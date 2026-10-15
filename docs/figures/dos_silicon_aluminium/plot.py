"""The DOS of silicon (tetrahedra and Gaussian) and of aluminium."""
import numpy as np
import style

d = np.load("data.npz")


def draw():
    fig, (a, b) = style.figure(2)
    a.axvspan(0.0, d["si_gap"], color=style.ink2(), alpha=0.15, lw=0)
    a.plot(d["si_gauss_e"], d["si_gauss"], color=style.color(1),
           label="Gaussian, degauss = 0.02 Ry")
    a.plot(d["si_e"], d["si_tetra"], color=style.color(0), label="tetrahedra")
    a.set_xlim(-13.0, 8.0)
    a.set_xlabel(r"$E - E_{\rm VBM}$ (eV)")
    a.set_ylabel("DOS (states/eV per cell)")
    a.set_title("silicon")

    b.plot(d["al_e"], d["al_dos"], label="tetrahedra")
    b.plot(d["al_e"], d["al_free"], **style.reference(),
           label=r"free electrons, $\sqrt{E-E_0}$")
    style.zero_line(b, vertical=True)
    b.set_xlim(-12.0, 4.0)
    b.set_xlabel(r"$E - E_F$ (eV)")
    b.set_title("aluminium")
    for ax in (a, b):
        ax.set_ylim(bottom=0.0)
        ax.legend(loc="upper left")
    return fig


style.main(draw)
