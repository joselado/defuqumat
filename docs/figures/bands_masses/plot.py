"""The conduction valley of silicon along and across Gamma-X, with the two mass parabolas."""
import numpy as np
import style

d = np.load("data.npz")
t, inverse = d["t"], d["inverse_mass"]
ry_to_ev = 13.605693122994


def draw():
    fig, axs = style.figure(2, sharey=True)
    cuts = [(d["longitudinal"], inverse[0, 0], r"along $\Gamma$-X:  $m_l$"),
            (d["transverse"], inverse[1, 1], r"across $\Gamma$-X:  $m_t$")]
    for ax, (bands, curvature, name) in zip(axs, cuts):
        ax.plot(t, bands[:, 1:], color=style.ink2(), lw=1.0)
        ax.plot(t, bands[:, 0], "o", ms=3.5, color=style.color(0),
                label="lowest conduction band")
        ax.plot(t, curvature * t**2 * ry_to_ev, color=style.color(1),
                label=r"$k^2/m^*$ from the mass tensor")
        ax.set_title(name + f" = {1 / curvature:.3f} $m_e$", fontsize=10)
        ax.set_xlabel(r"$k - k_0$ (1/bohr)")
        ax.set_xticks([-0.08, -0.04, 0.0, 0.04, 0.08])
        ax.set_ylim(-0.03, 0.8)
    axs[0].set_ylabel("energy above the minimum (eV)")
    axs[0].legend(loc="center left", bbox_to_anchor=(0.0, 0.38))
    return fig


style.main(draw)
