"""The moments of the helix along the chain, and seen along the chain."""
import numpy as np
import style

d = np.load("data.npz")
m = np.vstack([d["moments"], d["moments"][:1]])          # close the turn
z = np.append(d["z"], d["z"][0] + d["length"])


def draw():
    fig, (a, b) = style.figure(2)
    a.plot(z, m[:, 0], "o-", label="$m_x$")
    a.plot(z, m[:, 1], "s-", label="$m_y$")
    a.plot(z, m[:, 2], "^-", label="$m_z$")
    style.zero_line(a)
    a.set_xlabel("position along the chain (bohr)")
    a.set_ylabel(r"moment on the site ($\mu_B$)")
    a.legend()
    for site, (mx, my, _) in enumerate(d["moments"]):
        b.annotate("", xy=(mx, my), xytext=(0, 0),
                   arrowprops=dict(width=1.4, headwidth=7, color=style.color(0)))
        b.text(1.25 * mx, 1.25 * my, str(site + 1), ha="center", va="center")
    b.set_xlim(-0.65, 0.65)
    b.set_ylim(-0.65, 0.65)
    b.set_aspect("equal")
    b.set_xlabel(r"$m_x$ ($\mu_B$)")
    b.set_ylabel(r"$m_y$ ($\mu_B$)")
    return fig


style.main(draw)
