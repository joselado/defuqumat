"""Bismuthene: the bands without and with spin-orbit coupling."""
import numpy as np
import style

d = np.load("data.npz")


def draw():
    fig, axes = style.figure(2, sharey=True)
    for ax, tag, title in zip(axes, ("nosoc", "soc"),
                              ("without spin-orbit coupling", "with spin-orbit coupling")):
        ax.plot(d["x"], d[tag], color=style.color(0), lw=1.2)
        ax.axhline(0.0, **style.reference())
        for t in d["ticks"][1:-1]:
            ax.axvline(t, color=style.ink2(), lw=0.6)
        ax.set_xticks(d["ticks"], [r"$\Gamma$", "M", "K", r"$\Gamma$"])
        ax.set_xlim(d["x"][0], d["x"][-1])
        ax.set_ylim(-3.0, 3.0)
        ax.set_title(title)
        ax.text(0.97, 0.04, "direct gap %.2f eV" % d[tag + "_gap"],
                transform=ax.transAxes, ha="right")
    axes[0].set_ylabel(r"$E - E_F$ (eV)")
    return fig


style.main(draw)
