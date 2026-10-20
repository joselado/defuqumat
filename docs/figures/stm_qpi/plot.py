"""Graphene QPI: |delta rho(q)| of the two readers, in the first zones."""
import numpy as np
import style

d = np.load("data.npz")
bg = d["bg"]                                   # b1, b2 in 1/bohr, rows
n, m = d["divisions"], d["shape"]
p1 = np.fft.fftshift(np.fft.fftfreq(n[0] * m[0], d=1.0 / (n[0] * m[0]))) / n[0]
p2 = np.fft.fftshift(np.fft.fftfreq(n[1] * m[1], d=1.0 / (n[1] * m[1]))) / n[1]
f1, f2 = np.meshgrid(p1, p2, indexing="ij")
qx = f1 * bg[0, 0] + f2 * bg[1, 0]
qy = f1 * bg[0, 1] + f2 * bg[1, 1]

# the corners of the first zone, the K and K' points
corners = np.array([(2, 1), (1, 2), (-1, 1), (-2, -1), (-1, -2), (1, -1)]) / 3.0
corners = corners @ bg
corners = corners[np.argsort(np.arctan2(corners[:, 1], corners[:, 0]))]
hexagon = np.vstack([corners, corners[:1]])
reach = 1.9 * np.linalg.norm(corners[0])


def draw():
    fig, axes = style.figure(2)
    for ax, key, title in zip(axes, ["ldos", "landauer"],
                              ["LDOS at the tip", "tip to exit plane (Landauer)"]):
        values = d[key].copy()
        values[tuple(s // 2 for s in values.shape)] = 0.0     # the mean, q = 0
        mesh = ax.pcolormesh(qx, qy, values / values.max(), shading="gouraud",
                             cmap=style.CMAP, vmin=0.0, vmax=1.0)
        ax.plot(hexagon[:, 0], hexagon[:, 1], color=style.ink2(), lw=0.8)
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_aspect("equal")
        ax.set_xlabel("$q_x$ (1/bohr)")
        ax.set_title(title)
    axes[0].set_ylabel("$q_y$ (1/bohr)")
    style.colorbar(fig, mesh, list(axes), r"$|\delta\rho(\mathbf q)|$ / its maximum")
    return fig


style.main(draw)
