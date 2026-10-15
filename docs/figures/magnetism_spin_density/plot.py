"""The spin density of nickel on the (001) face of the cubic cell, and two cuts through it."""
import numpy as np
import style

d = np.load("data.npz")
s, m = d["s"], d["spin_density"]


def draw():
    fig, (a, b) = style.figure(2, gridspec_kw={"wspace": 0.45})
    mesh = a.pcolormesh(s, s, m.T, shading="gouraud", vmin=0.0)
    a.contour(s, s, m.T, levels=[0.0], colors="white", linestyles="--", linewidths=1.0)
    a.set_aspect("equal")
    a.set_xlabel("along [100] (bohr)")
    a.set_ylabel("along [010] (bohr)")
    style.colorbar(fig, mesh, a, r"$\mu_B$/bohr$^3$")

    # from the atom at the corner to the next atom, along [110] and along [100]
    n = len(s)
    b.plot(np.sqrt(2) * s[: n // 2 + 1], np.diag(m)[: n // 2 + 1], color=style.color(0),
           label="along [110], to the nearest neighbour")
    b.plot(s, m[:, 0], color=style.color(1), label="along [100], to the second neighbour")
    style.zero_line(b)
    b.set_yscale("symlog", linthresh=1e-4)
    b.set_ylim(-3e-3, 30.0)
    b.set_xlabel("distance from the atom (bohr)")
    b.set_ylabel(r"$n_\uparrow - n_\downarrow$ ($\mu_B$/bohr$^3$)")
    b.legend(loc="upper center")
    return fig


style.main(draw)
