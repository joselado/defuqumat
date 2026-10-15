"""The spin-resolved DOS of FeO, without and with U."""
import numpy as np
import style

d = np.load("data.npz")


def draw():
    fig, axes = style.figure(2, sharey=True)
    for ax, name, title in zip(axes, ("noU", "U"), ("U close to 0", "U = 4.3 eV")):
        e, dos = d[name + "_e"], d[name + "_dos"]
        ax.plot(e, dos[0], color=style.color(0), label="majority")
        ax.plot(e, -dos[1], color=style.color(1), label="minority")
        style.zero_line(ax)
        ax.axvline(0.0, **style.reference())
        ax.set_xlim(-8.0, 5.0)
        ax.set_xlabel(r"$E - E_F$ (eV)")
        ax.set_title(title)
    axes[0].set_ylabel("DOS (states/eV per cell)")
    axes[0].legend(loc="upper left")
    return fig


style.main(draw)
