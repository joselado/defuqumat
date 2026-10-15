"""E(V) of silicon, and the pressure from the stress and from -dE/dV."""
import numpy as np
import style

d = np.load("data.npz")
v, e = d["volume"], d["energy"]
ry_bohr3_to_kbar = 147105.08


def draw():
    fig, (a, b) = style.figure(2)
    a.plot(v, e, "o-")
    a.set_ylabel("total energy (Ry)")
    b.plot(v, d["pressure"], "o-", label="from the stress, one run per volume")
    b.plot(v, -np.gradient(e, v) * ry_bohr3_to_kbar, **style.reference(),
           label=r"$-dE/dV$ along the curve")
    style.zero_line(b)
    b.set_ylabel("pressure (kbar)")
    b.legend()
    for ax in (a, b):
        ax.set_xlabel(r"volume (bohr$^3$)")
    return fig


style.main(draw)
