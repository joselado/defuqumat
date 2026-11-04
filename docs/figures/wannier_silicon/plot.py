"""Wannier-interpolated bands of silicon against the bands computed directly."""
import numpy as np
import style

d = np.load("data.npz")


def draw():
    fig, ax = style.figure(1)
    x = d["x"] / d["x"][-1]
    for b in range(d["direct"].shape[1]):
        ax.plot(x, d["direct"][:, b], color="0.6", lw=2.4, label="direct" if b == 0 else None)
    for k, n in enumerate((4, 6)):
        w = d[f"wannier{n}"]
        for b in range(w.shape[1]):
            ax.plot(x, w[:, b], color=style.color(k), lw=1.2, ls="--" if n == 4 else "-",
                    label=f"Wannier, {n}x{n}x{n}" if b == 0 else None)
    ax.axhline(float(d["frozen"]), **style.reference())
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-7.0, 17.0)
    for t in d["ticks"]:
        ax.axvline(t / d["x"][-1], color=style.ink2(), lw=0.6)
    names = {"G": r"$\Gamma$"}
    ax.set_xticks(d["ticks"] / d["x"][-1], [names.get(str(p), str(p)) for p in d["labels"]])
    ax.set_ylabel("energy (eV)")
    ax.legend(frameon=False, loc="lower right")
    return fig


style.main(draw)
