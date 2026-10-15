"""The phonons of aluminium and their linewidths, along L-Gamma-X."""
import numpy as np
import style

d = np.load("data.npz")
SIGMA = 4                       # the broadening shown, d["sigmas"][SIGMA] in Ry
modes = {0: "transverse (two modes)", 2: "longitudinal"}


def draw():
    fig, (a, b) = style.figure(2)
    for k, (mode, name) in enumerate(modes.items()):
        a.plot(d["x"][:4], d["frequencies"][:4, mode], "-o", color=style.color(k), ms=4)
        a.plot(d["x"][4:], d["frequencies"][4:, mode], "-o", color=style.color(k), ms=4,
               label=name)
        for s, ls in ((1, ":"), (SIGMA, "-")):
            g = d["gamma_ghz"][:, s, mode]
            b.plot(d["x"][:4], g[:4], ls, marker="o", color=style.color(k), ms=4)
            b.plot(d["x"][4:], g[4:], ls, marker="o", color=style.color(k), ms=4)
    b.plot([], [], "-", color=style.ink2(), label=r"$\sigma$ = %.2f Ry" % d["sigmas"][SIGMA])
    b.plot([], [], ":", color=style.ink2(), label=r"$\sigma$ = %.2f Ry" % d["sigmas"][1])
    for ax in (a, b):
        style.zero_line(ax, vertical=True)
        ax.set_xticks([d["x"][0], 0.0, d["x"][-1]], ["L", r"$\Gamma$", "X"])
        ax.set_xlim(d["x"][0], d["x"][-1])
        ax.set_ylim(bottom=0.0)
    a.set_ylabel(r"frequency (cm$^{-1}$)")
    b.set_ylabel(r"linewidth $\gamma$ (GHz)")
    a.legend(loc="upper left")
    b.legend(loc="upper left")
    return fig


style.main(draw)
