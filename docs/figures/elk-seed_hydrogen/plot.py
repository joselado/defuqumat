"""The Elk density of hydrogen and the converged pseudo density, through the atom."""
import numpy as np
import style

d = np.load("data.npz")


def draw():
    fig, (a, b) = style.figure(2, sharey=True)
    for ax, key, title in [(a, "x", "along x"), (b, "d", "along the diagonal")]:
        half = d[key] <= d[key].max() / 2 + 1e-9
        r = d[key][half]
        ax.plot(r, d["seed_" + key][half], "o-", color=style.color(0),
                label="Elk, all-electron")
        ax.plot(r, d["scf_" + key][half], "s--", color=style.color(1),
                label="converged, pseudopotential")
        ax.set_xlabel("distance from the atom (bohr)")
        ax.set_title(title, color=style.ink())
        ax.set_xlim(-0.04 * r.max(), 1.04 * r.max())
    a.set_ylabel(r"$\rho$ (e/bohr$^3$)")
    a.set_ylim(0.0, None)
    a.legend()
    return fig


style.main(draw)
