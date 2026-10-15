"""Silicon: Im eps_M with three kernels, and what each kernel changes."""
import numpy as np
import style

d = np.load("data.npz")
e = d["energy"]
kernels = [("rpa", "RPA"), ("alda", "ALDA"), ("bootstrap", "bootstrap")]


def draw():
    fig, (a, b) = style.figure(2, gridspec_kw={"width_ratios": [1.6, 1]})
    a.plot(e, d["rpa_no_local_fields"], **style.reference(),
           label="RPA, no local fields")
    for k, (name, label) in enumerate(kernels):
        a.plot(e, d[name], color=style.color(k), label=label)
        if name != "rpa":
            b.plot(e, d[name] - d["rpa"], color=style.color(k), label=label)
    a.set_xlim(1.5, 8.0)
    a.set_ylim(bottom=0.0)
    a.set_xlabel("photon energy (eV)")
    a.set_ylabel(r"Im $\epsilon_M$")
    a.legend(loc="upper right")
    style.zero_line(b)
    b.set_xlim(1.5, 8.0)
    b.set_xlabel("photon energy (eV)")
    b.set_ylabel(r"Im $\epsilon_M$ minus RPA")
    b.legend(loc="upper right")
    return fig


style.main(draw)
