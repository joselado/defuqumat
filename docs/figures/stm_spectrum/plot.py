"""dI/dV above the two surface atoms of graphite, and a zoom near zero bias."""
import numpy as np
import style

d = np.load("data.npz")


def draw():
    fig, axes = style.figure(2)
    for ax, (lo, hi) in zip(axes, [(d["bias"][0], d["bias"][-1]), (-1.0, 1.0)]):
        ax.plot(d["bias"], d["didv"][:, 0], color=style.color(0), label="above A, over an atom")
        ax.plot(d["bias"], d["didv"][:, 1], color=style.color(1), label="above B, over a hollow")
        style.zero_line(ax, vertical=True)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("sample bias $V$ (V)")
    near = np.abs(d["bias"]) <= 1.0
    axes[1].set_ylim(0.0, 1.1 * d["didv"][near].max())
    axes[0].set_ylim(bottom=0.0)
    axes[0].set_ylabel(r"d$I$/d$V$ $\propto \rho(\mathbf{r}, E_F + eV)$ (1/(bohr$^3$ eV))")
    axes[0].legend(loc="upper left")
    axes[1].set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    axes[1].set_title("near the Fermi level")
    return fig


style.main(draw)
