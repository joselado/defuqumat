"""N(q) along the hydrogen chain, and N(q) of aluminium against |q|."""
import numpy as np
import style

d = np.load("data.npz")


def draw():
    fig, (a, b) = style.figure(2)
    a.plot(d["q3"], d["chain"], "o-", ms=3, color=style.color(0))
    a.axvline(0.5, **style.reference())
    a.text(0.52, 0.5, r"$q = 2k_F$", color=style.ink())
    a.set_xlabel(r"$q_3$ (crystal)")
    a.set_ylabel(r"$N(q)\,/\,N(0)$")
    a.set_ylim(0, 1.08)
    a.set_title("hydrogen chain", color=style.ink(), fontsize=10)

    b.plot(d["q"][1:], d["metal"][1:], ".", ms=4, color=style.color(2))
    b.axhline(float(d["chain_ratio"]), **style.reference())
    b.text(0.05, float(d["chain_ratio"]) - 0.08, "peak of the chain", color=style.ink2())
    b.set_xlabel(r"$|\mathbf{q}|$ (1/bohr)")
    b.set_ylabel(r"$N(\mathbf{q})\,/\,N(0)$")
    b.set_ylim(0, 1.08)
    b.set_title("fcc aluminium", color=style.ink(), fontsize=10)
    fig.tight_layout()
    return fig


style.main(draw)
