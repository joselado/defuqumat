"""The bands and the DOS of nickel: majority and minority channels."""
import numpy as np
import style

d = np.load("data.npz")
labels = ["W", "L", r"$\Gamma$", "X", "W", "K"]


def draw():
    fig, (a, b) = style.figure(2, gridspec_kw={"width_ratios": [2, 1]}, sharey=True)
    for spin, name in enumerate(["majority", "minority"]):
        for n in range(d["bands"].shape[2]):
            a.plot(d["x"], d["bands"][spin, :, n], color=style.color(spin),
                   label=name if n == 0 else None)
        sign = 1 if spin == 0 else -1
        b.plot(sign * d["dos"][spin], d["energies"], color=style.color(spin))
    for t in d["ticks"]:
        a.axvline(t, color=style.ink2(), lw=0.6)
    a.set_xticks(d["ticks"], labels)
    a.set_xlim(d["x"][0], d["x"][-1])
    a.set_ylim(-10.0, 4.0)
    a.set_ylabel(r"$E - E_F$ (eV)")
    a.legend(loc="lower left")
    for ax in (a, b):
        ax.axhline(0.0, **style.reference())
    style.zero_line(b, vertical=True)
    b.set_xlabel("DOS (states/eV)")
    b.text(0.03, 0.97, "minority", transform=b.transAxes, va="top", color=style.color(1))
    b.text(0.97, 0.97, "majority", transform=b.transAxes, va="top", ha="right",
           color=style.color(0))
    return fig


style.main(draw)
