"""The high-harmonic spectra of silicon and AlAs under the same pulse."""
import numpy as np
import style

d = np.load("data.npz")


def draw():
    fig, axes = style.figure(2, sharey=True)
    for k, (ax, name, title) in enumerate(zip(axes, ("si", "alas"), ("silicon", "AlAs"))):
        for n in range(1, 31):
            ax.axvline(n, color=style.ink2(), lw=0.6 if n % 2 else 0.3,
                       alpha=0.6 if n % 2 else 0.3, zorder=0)
        ax.semilogy(d[name + "_orders"], d[name + "_total"], color=style.color(k), lw=1.0)
        ax.set_xlim(0, 30)
        ax.set_ylim(1e-9, 1e2)
        ax.set_xlabel("harmonic order (1.55 eV each)")
        ax.set_title(title)
    axes[0].set_ylabel(r"$S(\omega)\,/\,S(\omega_0)$")
    return fig


style.main(draw)
