"""The energy and the largest force at each step of the relaxation."""
import numpy as np
import style

d = np.load("data.npz")


def draw():
    fig, (a, b) = style.figure(2)
    a.plot(d["step"], (d["energy"] - d["energy"][-1]) * 1e3, "o-")
    a.set_ylabel(r"$E - E_{\rm final}$ (mRy)")
    b.semilogy(d["step"], d["max_force"], "o-")
    b.axhline(1e-3, **style.reference(), label="forc_conv_thr = $10^{-3}$")
    b.set_ylabel("largest force (Ry/bohr)")
    b.legend()
    for ax in (a, b):
        ax.set_xlabel("ionic step")
        ax.set_xticks(d["step"])
    return fig


style.main(draw)
