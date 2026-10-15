"""The Wannier charge centres of the Bi(111) bilayer, over half the zone."""
import numpy as np
import style

d = np.load("data.npz")


def draw():
    fig, ax = style.figure()
    for n in range(d["centers"].shape[1]):
        ax.plot(d["pump"], d["centers"][:, n], "o", ms=3, color=style.color(0),
                label="charge centres" if n == 0 else None)
    gap = (d["gap"] / (2 * np.pi)) % 1.0
    ax.plot(d["pump"], gap, "x", ms=6, mew=1.5, color=style.color(1), label="largest gap")
    ax.set_xlim(-0.02, 0.52)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"pumping $k_2$ (units of $\mathbf{b}_2$)")
    ax.set_ylabel("Wannier charge centre (units of $a_1$)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2)
    return fig


style.main(draw)
