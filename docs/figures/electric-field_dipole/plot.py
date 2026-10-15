"""The dipole of CO against the field, and the change of the density along z."""
import numpy as np
import style

d = np.load("data.npz")
e, p = d["fields"], d["dipole"]
slope, offset = np.polyfit(e, p, 1)
z = d["z"]
z = np.where(z > z[-1] / 2, z - z[-1] - z[1], z)        # the molecule in the middle
order = np.argsort(z)
change = (d["density"][-1] - d["density"][2]) * 1e3      # E = 0.004 minus E = 0


def draw():
    fig, (a, b) = style.figure(2)
    a.plot(e * 1e3, p, "o", label="defuqumat")
    a.plot(e * 1e3, slope * e + offset, **style.reference(),
           label=r"slope $\alpha_{zz}$ = %.1f bohr$^3$" % slope)
    a.set_xlabel(r"field (10$^{-3}$ Ha a.u.)")
    a.set_ylabel(r"dipole (e bohr)")
    a.legend()
    b.plot(z[order], change[order])
    style.zero_line(b)
    for atom, pos in (("O", -1.1), ("C", 1.1)):
        b.axvline(pos, color=style.ink2(), ls=":", lw=1)
        b.text(pos, 0.95, " " + atom, transform=b.get_xaxis_transform())
    b.set_xlabel("z (bohr)")
    b.set_ylabel(r"$\Delta\bar n(z)$ (10$^{-3}$ e/bohr$^3$)")
    return fig


style.main(draw)
