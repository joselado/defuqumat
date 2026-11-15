"""Graphene seen by an s tip and a pxy tip, and the contrast against the height."""
import numpy as np
import style

d = np.load("data.npz")
xy = d["xy"]
n = xy.shape[0]
a1, a2 = n * (xy[1, 0] - xy[0, 0]), n * (xy[0, 1] - xy[0, 0])
f = np.arange(n + 1) / n                      # one cell, both edges included
x = f[:, None] * a1[0] + f[None, :] * a2[0]
y = f[:, None] * a1[1] + f[None, :] * a2[1]
atoms = {"A": (0, 0), "B": (1 / 3, 2 / 3)}
hollow = (2 / 3, 1 / 3)


def draw():
    fig, axes = style.figure(3, layout="constrained")
    for ax, key, title in zip(axes[:2], ["s", "pxy"], ["s tip", "pxy tip (CO-like)"]):
        field = np.pad(d[key] / d[key].max(), ((0, 1), (0, 1)), mode="wrap")
        for i in range(-1, 3):
            for j in range(-1, 3):
                shift = i * a1 + j * a2
                m = ax.pcolormesh(x + shift[0], y + shift[1], field, shading="gouraud",
                                  cmap=style.CMAP, vmin=0.0, vmax=1.0)
        for name, (f1, f2) in atoms.items():
            for i in range(-1, 3):
                for j in range(-1, 3):
                    p = (f1 + i) * a1 + (f2 + j) * a2
                    ax.plot(*p, "o", mfc="none", mec=style.color(2), ms=6, mew=1.1)
        p = hollow[0] * a1 + hollow[1] * a2
        ax.plot(*p, "x", color=style.color(2), ms=7, mew=1.4)
        ax.set_xlim(-2.0, 7.0)
        ax.set_ylim(-1.0, 6.5)
        ax.set_aspect("equal")
        ax.set_xlabel("x (bohr)")
        ax.set_title(title)
    axes[0].set_ylabel("y (bohr)")
    style.colorbar(fig, m, list(axes[:2]), "map / its maximum")

    c = axes[2]
    c.plot(d["heights"], d["continued"], "o-", color=style.color(0),
           label="continued from 3 bohr")
    c.plot(d["heights"], np.clip(d["raw"], 0.0, 3.0), "s--", color=style.color(1),
           label="raw plane-wave tail")
    c.axhline(2.0, color=style.color(2), lw=0.8)
    c.set_ylim(0.0, 3.0)
    c.set_xlabel("tip height above the sheet (bohr)")
    c.set_ylabel("pxy map: hollow / atom")
    c.set_title("contrast against height")
    c.legend(loc="lower right", fontsize=9)
    return fig


style.main(draw)
