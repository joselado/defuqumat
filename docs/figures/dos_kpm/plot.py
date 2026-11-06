"""The KPM DOS of silicon against 12 bands, and over the whole plane-wave basis."""
import numpy as np
import style

d = np.load("data.npz")


def draw():
    fig, (a, b) = style.figure(2)
    a.fill_between(d["e"], d["kpm"] - 2 * d["kpm_error"], d["kpm"] + 2 * d["kpm_error"],
                   color=style.color(0), alpha=0.3, lw=0, label="KPM, two standard errors")
    a.plot(d["e"], d["kpm"], color=style.color(0))
    a.plot(d["bands_e"], d["bands"], **style.reference(), label="from 12 bands")
    a.set_xlabel(r"$E - E_{\rm VBM}$ (eV)")
    a.set_ylabel("DOS (states/eV per cell)")
    a.set_title("silicon, near the gap")

    b.plot(d["whole_e"], d["whole"], color=style.color(1), label="KPM, whole basis")
    b.plot(d["whole_e"], d["free"], **style.reference(), label="free electrons below the cutoff")
    b.set_xlabel(r"$E$ (eV)")
    b.set_title("silicon, the whole basis")
    for ax in (a, b):
        ax.set_ylim(bottom=0.0)
        ax.legend(loc="upper left")
    return fig


style.main(draw)
