"""E(q) of the chain two ways, and the size of the moment along the curve."""
import numpy as np
import style

d = np.load("data.npz")


def draw():
    fig, (a, b) = style.figure(2)
    a.plot(d["q"], d["direct"], "o", label="E(q), one SCF per point")
    a.plot(d["q"], d["integrated"], label=r"integral of $dE/dq$")
    style.zero_line(a)
    a.set_ylabel(r"$E(q) - E(0)$ (mRy)")
    a.legend()
    b.plot(d["q"], d["moment"], "o-")
    b.set_ylabel(r"moment per cell ($\mu_B$)")
    for ax in (a, b):
        ax.set_xlabel(r"$q_3$ (units of $b_3$)")
    return fig


style.main(draw)
