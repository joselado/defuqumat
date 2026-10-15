"""The valence density of silicon in the (1-10) plane, with the atoms as circles."""
import numpy as np
import style

d = np.load("data.npz")


def draw():
    fig, ax = style.figure()
    mesh = ax.pcolormesh(d["u"], d["v"], d["density"].T, shading="gouraud")
    ax.scatter(d["atoms"][:, 0], d["atoms"][:, 1], s=40, facecolors="none",
               edgecolors="white", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlabel("along [110] (bohr)")
    ax.set_ylabel("along [001] (bohr)")
    style.colorbar(fig, mesh, ax, r"$n(\mathbf{r})$ (electrons/bohr$^3$)")
    return fig


style.main(draw)
