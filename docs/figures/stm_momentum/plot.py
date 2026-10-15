"""The Fermi surface of a hydrogen sheet, and the part of it that tunnels."""
import numpy as np
import style

d = np.load("data.npz")


def zone(name):
    """A column on the grid, with Gamma in the middle and both zone edges."""
    field = np.fft.fftshift(d[name][:, :, 0])
    return np.pad(field, ((0, 1), (0, 1)), mode="wrap")


bare, weight = zone("bare"), zone("weight")
n = bare.shape[0] - 1
k = (np.arange(n + 1) - n // 2) / n
# the current per state, on the k-points of the Fermi surface
kxy = d["k"][:, :2] - np.round(d["k"][:, :2])
size = np.linalg.norm(kxy, axis=1)
b, w = d["bare"].ravel(), d["weight"].ravel()
on = b > 0.2 * b.max()


def draw():
    fig, axes = style.figure(3, layout="constrained")
    for ax, field, title in zip(axes, [bare, weight], ["Fermi surface", "tunnelling current"]):
        m = ax.pcolormesh(k, k, (field / field.max()).T, shading="gouraud",
                          cmap=style.CMAP, vmin=0.0, vmax=1.0)
        ax.set_aspect("equal")
        ax.set_xticks([-0.4, 0.0, 0.4])
        ax.set_yticks([-0.4, 0.0, 0.4])
        ax.set_xlabel(r"$k_x$ ($2\pi/a$)")
        ax.set_title(title)
    axes[0].set_ylabel(r"$k_y$ ($2\pi/a$)")
    style.colorbar(fig, m, list(axes[:2]), "map / its maximum")

    ratio = w[on] / b[on]
    c = axes[2]
    c.plot(size[on], ratio / ratio.max(), "o", color=style.color(0), ms=3)
    c.set_ylim(0.0, 1.05)
    c.set_xlabel(r"$|\mathbf{k}_\parallel|$ ($2\pi/a$)")
    c.set_ylabel("current / Fermi surface, scaled")
    c.set_title("per state of the Fermi surface")
    return fig


style.main(draw)
