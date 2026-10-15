"""Nickel: sigma_xx, sigma_xy and the Kerr rotation on three k-meshes."""
import numpy as np
import style

d = np.load("data.npz")
e = d["energy"]
meshes = [(4, r"$4^3$"), (6, r"$6^3$"), (8, r"$8^3$")]


def draw():
    fig, (a, b, c) = style.figure(3)
    for k, (n, label) in enumerate(meshes):
        a.plot(e, d[f"sxx{n}"].real / 1e3, color=style.color(k), label=label)
        b.plot(e, d[f"sxy{n}"].real, color=style.color(k), label=label)
        c.plot(e[1:], d[f"kerr{n}"].real[1:], color=style.color(k), label=label)
    a.set_ylabel(r"Re $\sigma_{xx}$ ($10^3$ S/cm)")
    a.set_ylim(0.0, None)
    b.set_ylabel(r"Re $\sigma_{xy}$ (S/cm)")
    c.set_ylabel(r"Kerr rotation $\theta_K$ (degrees)")
    for ax in (a, b, c):
        ax.set_xlim(0.0, 6.0)
        ax.set_xlabel("photon energy (eV)")
    for ax in (b, c):
        style.zero_line(ax)
    c.set_ylim(-2.0, 1.5)
    c.axvspan(1.5, 3.5, color=style.ink2(), alpha=0.12, lw=0)
    a.legend(loc="upper right", title="k-mesh")
    fig.tight_layout()
    return fig


style.main(draw)
