"""The trace of the quantum metric along L-G-X for diamond, silicon and AlAs."""
import numpy as np
import style

d = np.load("data.npz")
NAMES = (("C", "diamond"), ("Si", "Si"), ("AlAs", "AlAs"))


def draw():
    fig, ax = style.figure(1)
    for k, (name, label) in enumerate(NAMES):
        x = d[name + "_x"] / d[name + "_x"][-1]
        ax.plot(x, d[name + "_trace"], color=style.color(k), lw=1.8,
                label=f"{label}, " + r"$\langle x^2\rangle_c$" + f" = {float(d[name + '_x2']):.2f} bohr$^2$")
    # L-G is sqrt(3)/2 and G-X is 1 in units of 2 pi / a, so G sits at 0.464.
    gamma = (3 ** 0.5 / 2) / (3 ** 0.5 / 2 + 1.0)
    ax.set_xticks([0.0, gamma, 1.0])
    ax.set_xticklabels(["L", r"$\Gamma$", "X"])
    ax.axvline(gamma, **style.reference())
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel(r"${\rm Tr}\, g(\mathbf{k})$ (bohr$^2$)")
    ax.legend(frameon=False)
    return fig


style.main(draw)
