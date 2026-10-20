"""Hydrogen sheet QPI: log |delta rho(q)| of the two readers."""
import numpy as np
import style

d = np.load("data.npz")
bg = d["bg"]
half = 0.5 * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]])
zone = half @ bg                               # the first Brillouin zone
reach = 1.5


def draw():
    fig, axes = style.figure(2)
    for ax, key, title in zip(axes, ["ldos", "landauer"],
                              ["LDOS at the tip", "tip to exit plane (Landauer)"]):
        values = d[key].copy()
        values[tuple(s // 2 for s in values.shape)] = 0.0     # the mean, q = 0
        logs = np.log10(values / values.max() + 1e-6)
        mesh = ax.pcolormesh(d["qx"], d["qy"], logs, shading="gouraud",
                             cmap=style.CMAP, vmin=-3.0, vmax=0.0)
        ax.plot(zone[:, 0], zone[:, 1], color=style.ink2(), lw=0.8)
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_aspect("equal")
        ax.set_xlabel("$q_x$ (1/bohr)")
        ax.set_title(title)
    axes[0].set_ylabel("$q_y$ (1/bohr)")
    style.colorbar(fig, mesh, list(axes), r"$\log_{10}$ of $|\delta\rho(q)|$ / its maximum")
    return fig


style.main(draw)
