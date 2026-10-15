"""The Berry curvature of AlAs and of silicon, on one color scale."""
import numpy as np
import style

d = np.load("data.npz")


def centred(omega):
    """Roll the mesh so that Gamma is in the middle, k from -1/2 to 1/2."""
    n = omega.shape[0]
    return np.roll(omega, (n // 2, n // 2), axis=(0, 1)), (np.arange(n + 1) - n // 2 - 0.5) / n


def draw():
    fig, axs = style.figure(2, sharey=True)
    limit = np.abs(d["alas"]).max()
    for ax, name, title in zip(axs, ("alas", "si"), ("AlAs", "Si")):
        omega, edges = centred(d[name])
        m = ax.pcolormesh(edges, edges, omega.T, cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_aspect("equal")
        ax.set_title(title, color=style.ink())
        ax.set_xlabel(r"$k_1$ (units of $\mathbf{b}_1$)")
    axs[0].set_ylabel(r"$k_2$ (units of $\mathbf{b}_2$)")
    style.colorbar(fig, m, axs, r"$\Omega(\mathbf{k})$ (bohr$^2$)")
    return fig


style.main(draw)
