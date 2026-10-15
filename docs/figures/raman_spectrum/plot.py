"""Raman and infrared spectra of silicon and AlAs, each mode a Lorentzian."""
import numpy as np
import style

d = np.load("data.npz")
w = np.linspace(0.0, 650.0, 1400)
WIDTH = 8.0                      # half-width of each line, in cm^-1


def curve(freq, activity):
    """A sum of Lorentzians, one per optical mode (the acoustic ones, below 20 cm^-1, are left out)."""
    keep = freq > 20.0
    return sum(a * WIDTH**2 / ((w - f) ** 2 + WIDTH**2)
               for f, a in zip(freq[keep], activity[keep]))


def draw():
    fig, (a, b) = style.figure(2, sharey=True)
    for k, (name, key) in enumerate([("silicon", "si"), ("AlAs", "alas")]):
        raman = curve(d[key + "_freq"], d[key + "_raman"])
        a.plot(w, raman / raman.max(), color=style.color(k), label=name)
    a.set_xlabel(r"Raman shift (cm$^{-1}$)")
    a.set_ylabel("intensity (each to its peak)")
    a.set_title("Raman", color=style.ink())
    a.legend(loc="upper left")

    ir_alas = curve(d["alas_freq"], d["alas_ir"])
    ir_si = curve(d["si_freq"], d["si_ir"])
    top = ir_alas.max()
    b.plot(w, ir_si / top, color=style.color(0), label="silicon", zorder=3)
    b.plot(w, ir_alas / top, color=style.color(1), label="AlAs")
    to, lo = d["alas_freq"].max(), d["lo_freq"].max()
    b.axvline(lo, **style.reference())
    b.text(to - 12, 0.9, "TO %.0f" % to, ha="right", color=style.ink2())
    b.text(lo + 8, 0.9, "LO %.0f" % lo, ha="left", color=style.ink2())
    b.set_xlabel(r"frequency (cm$^{-1}$)")
    b.set_title("infrared", color=style.ink())
    b.legend(loc="upper left")
    for ax in (a, b):
        ax.set_xlim(w[0], w[-1])
        ax.set_ylim(0.0, 1.08)
    return fig


style.main(draw)
