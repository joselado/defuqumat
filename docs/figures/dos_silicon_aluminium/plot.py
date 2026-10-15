"""Plot figure dos_silicon_aluminium from data.npz.

Writes docs/guide/img/dos_silicon_aluminium.png and -dark.png.
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
    fig, (a, b) = style.figure(2)
    a.axvspan(0.0, d["si_gap"], color=c["ink2"], alpha=0.15, lw=0)
    a.plot(d["si_gauss_e"], d["si_gauss"], color=c["series"][1],
           label="Gaussian, degauss = %.2f Ry" % params["degauss"])
    a.plot(d["si_e"], d["si_tetra"], color=c["series"][0], label="tetrahedra")
    a.set_xlim(-13.0, 8.0)
    a.set_ylim(bottom=0.0)
    a.set_xlabel(r"$E - E_{\rm VBM}$ (eV)")
    a.set_ylabel("DOS (states/eV per cell)")
    a.set_title("silicon", color=c["ink"])
    a.legend(loc="upper left")

    b.plot(d["al_e"], d["al_dos"], color=c["series"][0], label="tetrahedra")
    b.plot(d["al_e"], d["al_free"], **style.reference(theme),
           label=r"free electrons, $\sqrt{E-E_0}$")
    b.axvline(0.0, color=c["ink2"], lw=0.8)
    b.set_xlim(-12.0, 4.0)
    b.set_ylim(bottom=0.0)
    b.set_xlabel(r"$E - E_F$ (eV)")
    b.set_title("aluminium", color=c["ink"])
    b.legend(loc="upper left")
    return fig


style.main(draw, "dos_silicon_aluminium")
