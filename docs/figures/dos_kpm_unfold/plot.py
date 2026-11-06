"""Unfolded spectral functions of a perfect and a vacancy Si8 supercell, with the fcc bands."""
import numpy as np
import style

d = np.load("data.npz")


def draw():
    fig, axes = style.figure(2)
    x = np.arange(d["bands"].shape[0])
    for ax, name, title in zip(axes, ("perfect", "vacancy"), ("perfect Si8", "Si8 with a vacancy")):
        a = d[name]
        ax.pcolormesh(x, d[f"{name}_e"], a.T, shading="auto",
                      vmin=0.0, vmax=np.percentile(a, 99))
        ax.plot(x, d["bands"], **style.heatmap_line())
        ax.set_ylim(-13.0, 5.0)
        ax.set_xticks([0, 20, x[-1]], ["L", r"$\Gamma$", "X"])
        ax.set_title(title)
    axes[0].set_ylabel(r"$E - E_F$ (eV)")
    return fig


style.main(draw)
