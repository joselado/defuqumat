"""The six phonon branches of silicon, one point per computed q."""
import numpy as np
import style

d = np.load("data.npz")
labels = [r"$\Gamma$", "X", "K", r"$\Gamma$", "L"]


def draw():
    fig, ax = style.figure()
    for t in d["ticks"]:
        ax.axvline(t, color=style.ink2(), lw=0.6)
    for n in range(d["frequencies"].shape[1]):
        ax.plot(d["x"], d["frequencies"][:, n], "-o", color=style.color(0 if n < 3 else 1),
                lw=1.0, ms=4)
    ax.plot([], [], "-o", color=style.color(0), lw=1.0, ms=4, label="acoustic")
    ax.plot([], [], "-o", color=style.color(1), lw=1.0, ms=4, label="optical")
    ax.set_xticks(d["ticks"], labels)
    ax.set_xlim(d["x"][0], d["x"][-1])
    ax.set_ylim(0.0, 560.0)
    ax.set_ylabel(r"frequency (cm$^{-1}$)")
    ax.legend(loc="lower right")
    return fig


style.main(draw)
