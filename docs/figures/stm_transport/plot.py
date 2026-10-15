"""Bilayer graphene: the STM image against the vertical transport map."""
import numpy as np
import style

d = np.load("data.npz")
xy = d["xy"]
n = xy.shape[0]
a1, a2 = n * (xy[1, 0] - xy[0, 0]), n * (xy[0, 1] - xy[0, 0])
f = np.arange(n + 1) / n                      # one cell, both edges included
x = f[:, None] * a1[0] + f[None, :] * a2[0]
y = f[:, None] * a1[1] + f[None, :] * a2[1]
atoms = {"A": (0, 0), "B": (2 / 3, 1 / 3)}


def draw():
    fig, axes = style.figure(2)
    for ax, key, title in zip(axes, ["stm", "transport"],
                              ["STM image, at the tip", "transport, tip to substrate"]):
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
                    ax.plot(*p, "o", mfc="none", mec=style.color(2), ms=7, mew=1.2)
            p = f1 * a1 + f2 * a2
            ax.annotate(name, p, color=style.color(2), xytext=(6, 4),
                        textcoords="offset points", fontsize=11)
        ax.set_xlim(-2.0, 7.0)
        ax.set_ylim(-1.0, 6.5)
        ax.set_aspect("equal")
        ax.set_xlabel("x (bohr)")
        ax.set_title(title)
    axes[0].set_ylabel("y (bohr)")
    style.colorbar(fig, m, list(axes), "map / its maximum")
    return fig


style.main(draw)
