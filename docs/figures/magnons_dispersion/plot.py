"""The magnon energy along Gamma-X, and the spectral function at the last q."""
import numpy as np
import style

d = np.load("data.npz")


def draw():
    fig, (a, b) = style.figure(2)
    a.plot(d["q"], d["magnon"], "o-")
    a.set_xlabel(r"$|q|$ along $\Gamma$-X (1/Å)")
    a.set_ylabel("magnon energy (meV)")
    i = len(d["q"]) - 1
    b.fill_between(d["energies"], d["stoner"][i], color=style.ink2(), alpha=0.25, lw=0,
                   label="independent spin flips")
    b.plot(d["energies"], d["spectral"][i], color=style.color(1),
           label="with the exchange kernel")
    b.set_xlabel("energy (meV)")
    b.set_ylabel("spectral weight at X")
    b.set_ylim(bottom=0.0)
    b.legend()
    return fig


style.main(draw)
