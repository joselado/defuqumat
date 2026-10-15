"""The change of the dipole of AlAs when one sublattice moves, and the Born charges."""
import numpy as np
import style

d = np.load("data.npz")
u, dipole, slopes = d["u"], d["dipole"], d["slopes"]


def draw():
    fig, ax = style.figure()
    for atom, name in ((0, "Al"), (1, "As")):
        change = dipole[atom, -1] - dipole[atom, -1][len(u) // 2]     # 8x8 strings
        ax.plot(u, change, "o", color=style.color(atom),
                label=f"{name} moved, $Z^* = {slopes[atom, -1]:+.3f}$")
        ax.plot(u, slopes[atom, -1] * u, color=style.color(atom), lw=1.0)
    style.zero_line(ax)
    ax.set_xlabel("displacement of one sublattice along x (bohr)")
    ax.set_ylabel(r"change of $\Omega P_x$ (e bohr)")
    ax.set_xticks([-0.08, -0.04, 0.0, 0.04, 0.08])
    ax.legend(loc="upper center")
    return fig


style.main(draw)
