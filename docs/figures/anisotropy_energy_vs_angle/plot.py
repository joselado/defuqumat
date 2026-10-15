"""The free energy against the angle of the moment, with and without spin-orbit coupling."""
import numpy as np
import style

d = np.load("data.npz")
theta = np.radians(d["angles"])
s2 = np.sin(theta) ** 2
k1, k2 = np.linalg.lstsq(np.stack([s2, s2 ** 2], axis=1), d["soc"], rcond=None)[0]
grid = np.radians(np.linspace(0, 90, 181))


def draw():
    fig, ax = style.figure()
    ax.plot(d["angles"], d["soc"], "o", label="with spin-orbit coupling")
    ax.plot(np.degrees(grid), k1 * np.sin(grid) ** 2 + k2 * np.sin(grid) ** 4,
            **style.reference(),
            label=r"$K_1\sin^2\theta + K_2\sin^4\theta$, $K_1$ = %.3f meV" % k1)
    ax.plot(d["angles"], d["off"], "s-", label="coupling switched off")
    ax.set_xlabel(r"angle $\theta$ of the moment from the $c$ axis (degrees)")
    ax.set_ylabel("free energy from the $c$ axis (meV)")
    ax.set_xticks(d["angles"])
    ax.legend()
    return fig


style.main(draw)
