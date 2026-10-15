"""The piezoelectric tensor of AlAs and silicon, and e_14 of AlAs against the k-mesh."""
import numpy as np
import style

d = np.load("data.npz")
strains = ["xx", "yy", "zz", "yz", "xz", "xy"]


def table(ax, e, title, scale):
    m = ax.imshow(e, cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
    for i in range(3):
        for j in range(6):
            dark = abs(e[i, j]) > 0.5 * scale
            ax.text(j, i, f"{e[i, j] + 0.0:+.2f}", ha="center", va="center", fontsize=8,
                    color="white" if dark else "#0b0b0b")
    ax.set_xticks(range(6), strains)
    ax.set_yticks(range(3), ["x", "y", "z"])
    ax.set_xlabel("strain")
    ax.set_title(title, fontsize=10, color=style.ink())
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    return m


def draw():
    fig, (a, b, c) = style.figure(3, gridspec_kw={"width_ratios": [1.0, 1.0, 1.1]})
    scale = np.abs(d["alas"]).max()
    table(a, d["alas"], "AlAs, $4\\times4\\times4$", scale)
    a.set_ylabel("polarization along")
    m = table(b, d["silicon"], "Si, $4\\times4\\times4$", scale)
    b.set_yticks(range(3), [])
    style.colorbar(fig, m, b, r"$e_{iJ}$ (C/m$^2$)")

    c.plot(d["meshes"], d["e14"], "o-", color=style.color(0))
    for n, nk, e in zip(d["meshes"], d["nk"], d["e14"]):
        c.annotate(f"{nk} k", (n, e), textcoords="offset points", xytext=(6, -12),
                   fontsize=8, color=style.ink2())
    c.set_xticks(d["meshes"], [f"${n}^3$" for n in d["meshes"]])
    c.set_xlabel("k-mesh")
    c.set_ylabel(r"$e_{14}$ of AlAs (C/m$^2$)")
    c.margins(x=0.15, y=0.15)
    fig.tight_layout()
    return fig


style.main(draw)
