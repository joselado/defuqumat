"""The band structure of silicon: valence bands blue, conduction bands orange."""
import numpy as np
import style

d = np.load("data.npz")
names = {"G": r"$\Gamma$", "U": "U|K", "K": ""}


def draw():
    fig, ax = style.figure()
    e = d["energies"]
    for n in range(e.shape[1]):
        ax.plot(d["x"], e[:, n], color=style.color(0 if n < 4 else 1))
    ax.axhline(0.0, **style.reference())
    for t in d["ticks"]:
        ax.axvline(t, color=style.ink2(), lw=0.6)
    labels = [names.get(str(p), str(p)) for p in d["labels"]]
    keep = [i for i, p in enumerate(labels) if p]
    ax.set_xticks(d["ticks"][keep], [labels[i] for i in keep])
    ax.set_xlim(d["x"][0], d["x"][-1])
    ax.set_ylim(-13.0, 6.0)
    ax.set_ylabel(r"$E - E_{\rm VBM}$ (eV)")
    ax.text(0.98, 0.04, "indirect gap %.2f eV" % d["gap"], transform=ax.transAxes,
            ha="right")
    return fig


style.main(draw)
