"""The spin density wave of silicon, and the tip current above it."""
import numpy as np
import style

d = np.load("data.npz")


def draw():
    fig, (a, b) = style.figure(2)
    scale = np.abs(d["moments"]).max() / np.abs(d["field"]).max()
    a.bar(d["cells"], d["moments"], width=0.7, color=style.color(0),
          label="moment of each cell")
    x = np.linspace(0, 8, 200)
    a.plot(x, scale * 0.02 * np.cos(2 * np.pi * x / 8), **style.reference(),
           label="applied field (scaled)")
    style.zero_line(a)
    a.set_xlim(0, 8)
    a.set_xlabel("position along the long cell (unit cells)")
    a.set_ylabel(r"moment ($\mu_B$)")
    a.legend(loc="upper center")

    for k, (name, label) in enumerate((("up", "tip polarized up"),
                                       ("down", "tip polarized down"),
                                       ("charge", "unpolarized tip"))):
        cell = d[name].mean(axis=1).reshape(8, -1).mean(axis=1)
        b.plot(d["cells"], cell / cell.mean(), "o-", color=style.color(k),
               label=label)
    b.axhline(1.0, color=style.ink2(), lw=0.8)
    b.set_xlim(0, 8)
    b.set_ylim(0.5, 1.9)
    b.set_xlabel("position along the long cell (unit cells)")
    b.set_ylabel("tip current of each cell / its mean")
    b.legend(loc="upper center", ncol=1)
    return fig


style.main(draw)
