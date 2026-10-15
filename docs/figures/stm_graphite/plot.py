"""The STM image of graphite at constant height, and the decay into the vacuum."""
import numpy as np
import style

d = np.load("data.npz")
xy = d["xy"]
n = xy.shape[0]
a1, a2 = n * (xy[1, 0] - xy[0, 0]), n * (xy[0, 1] - xy[0, 0])
f = np.arange(n + 1) / n                      # one cell, both edges included
x = f[:, None] * a1[0] + f[None, :] * a2[0]
y = f[:, None] * a1[1] + f[None, :] * a2[1]
image = np.pad(d["image"], ((0, 1), (0, 1)), mode="wrap")
atoms = {"A": (0, 0), "B": (2 / 3, 1 / 3)}


def draw():
    fig, (a, b) = style.figure(2, layout="constrained",
                              gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.15})
    for i in range(-1, 3):
        for j in range(-1, 3):
            shift = i * a1 + j * a2
            m = a.pcolormesh(x + shift[0], y + shift[1], image, shading="gouraud",
                             cmap=style.CMAP, vmin=0.0, vmax=image.max())
    for name, (f1, f2) in atoms.items():
        for i in range(-1, 3):
            for j in range(-1, 3):
                p = (f1 + i) * a1 + (f2 + j) * a2
                a.plot(*p, "o", mfc="none", mec=style.color(2), ms=8, mew=1.3)
        a.annotate(name, f1 * a1 + f2 * a2, color=style.color(2), xytext=(6, 4),
                   textcoords="offset points", fontsize=11)
    a.set_xlim(-2.0, 7.0)
    a.set_ylim(-1.0, 6.5)
    a.set_aspect("equal")
    a.set_xlabel("x (bohr)")
    a.set_ylabel("y (bohr)")
    a.set_title("3.5 bohr above the top layer")
    style.colorbar(fig, m, a, r"$\rho_{\rm STM}$ (1/(bohr$^3$ Ry))")

    for k, name, c in [(0, "above A", 0), (2, "above B", 1), (1, "above the hexagon centre", 3)]:
        b.semilogy(d["z"], d["decay"][k], color=style.color(c), label=name)
    b.axvline(3.5, ymin=0.3, **style.reference())       # the plane of the image
    b.set_xlim(0.5, d["z"][-1])
    b.set_ylim(1.0e-12, 1.0)
    b.set_xlabel("height above the top layer (bohr)")
    b.set_ylabel(r"$\rho_{\rm STM}$ (same units)")
    b.legend(loc="lower left")
    return fig


style.main(draw)
