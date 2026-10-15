"""The second-harmonic susceptibility of AlAs, and of silicon as the control."""
import numpy as np
import style

d = np.load("data.npz")


def draw():
    fig, ax = style.figure()
    xyz = d["xyz"]
    ax.plot(d["ev"], xyz.real, color=style.color(0), label=r"Re $\chi^{xyz}$")
    ax.plot(d["ev"], xyz.imag, color=style.color(1), label=r"Im $\chi^{xyz}$")
    ax.plot(d["ev"], np.abs(xyz), color=style.ink(), lw=0.9, label=r"$|\chi^{xyz}|$")
    ax.plot(d["si_ev"], d["si_xyz"].real, color=style.color(2), ls=":",
            label=r"silicon, Re $\chi^{xyz}$")
    style.zero_line(ax)
    top = 1.15 * np.abs(xyz).max()
    for edge, name in ((d["gap"] / 2, "two-photon edge"), (d["gap"], "one-photon edge")):
        ax.axvline(edge, color=style.ink2(), ls="--", lw=0.9)
        ax.text(edge + 0.05, 0.92 * top, name, color=style.ink2(), fontsize=9)
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(-top, top)
    ax.set_xlabel(r"fundamental photon energy $\hbar\omega$ (eV)")
    ax.set_ylabel(r"$\chi^{(2)}$ (pm/V)")
    ax.legend(loc="lower left")
    return fig


style.main(draw)
