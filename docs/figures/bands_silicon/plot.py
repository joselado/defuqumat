"""Plot figure bands_silicon from data.npz.

Writes docs/guide/img/bands_silicon.png and -dark.png.
    python3 plot.py [--theme light|dark]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common
import style

params, d = common.load(os.path.join(HERE, "data.npz"))
x, e = d["x"], d["energies"]
names = {"G": r"$\Gamma$", "U": "U|K", "K": None}


def draw(theme):
    c = style.colors(theme)
    fig, ax = style.figure(1)
    for n in range(e.shape[1]):
        ax.plot(x, e[:, n], color=c["series"][0 if n < 4 else 1])
    ax.axhline(0.0, **style.reference(theme))
    ticks, labels = [], []
    for t, name in zip(d["ticks"], d["labels"]):
        label = names.get(str(name), str(name))
        if label is not None:
            ticks.append(t)
            labels.append(label)
        ax.axvline(t, color=c["ink2"], lw=0.6)
    ax.set_xticks(ticks, labels)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-13.0, 6.0)
    ax.set_ylabel(r"$E - E_{\rm VBM}$ (eV)")
    ax.text(0.98, 0.04, "indirect gap %.2f eV" % d["gap"], transform=ax.transAxes,
            ha="right", color=c["ink"])
    return fig


style.main(draw, "bands_silicon")
