"""Silicon's structure factors against |H|, and X-ray against neutron on a chain."""
import numpy as np
import style

d = np.load("data.npz")
FLOOR = 1e-4          # reflections that are zero by symmetry are drawn here


def label(hkl):
    return "(" + "".join(str(abs(int(v))) for v in sorted(np.abs(hkl), reverse=True)) + ")"


def draw():
    fig, (a, b) = style.figure(2, gridspec_kw={"width_ratios": [3, 2]})
    zero = d["F"] < 1e-8
    bond = np.array([label(h) == "(222)" for h in d["hkl"]])
    keep = (d["length"] > 0) & ~bond
    a.semilogy(d["length"][keep & ~zero], d["F"][keep & ~zero], "o", color=style.color(0),
               label="allowed")
    a.semilogy(d["length"][keep & zero], np.full((keep & zero).sum(), FLOOR), "x",
               color=style.ink2(), label="zero by symmetry")
    drawn = set()
    for i in np.flatnonzero(keep | bond):
        name = label(d["hkl"][i])
        if name in drawn:
            continue
        drawn.add(name)
        if name in ("(111)", "(220)", "(311)", "(400)", "(200)", "(420)"):
            y = d["F"][i] if not zero[i] else FLOOR
            dx = -30 if name == "(311)" else 4
            a.annotate(name, (d["length"][i], y), xytext=(dx, 4), textcoords="offset points",
                       color=style.ink2(), fontsize=8)
        if name == "(222)":
            a.semilogy(d["length"][i], d["F"][i], "o", ms=6, color=style.color(1), zorder=1)
            a.annotate("(222): bond charge", (d["length"][i], d["F"][i]),
                       xytext=(2.6, 1.3), color=style.color(1),
                       arrowprops=dict(arrowstyle="-", color=style.color(1), lw=0.8))
    a.set_xlabel(r"$|\mathbf{H}|$ (1/bohr)")
    a.set_ylabel(r"$|F(\mathbf{H})|$ (electrons)")
    a.set_ylim(5e-5, 4.0)
    a.legend(loc="upper right")

    w = 0.36
    b.bar(d["l"] - w / 2, d["xray"], w, color=style.color(0), label="charge (X-rays)")
    b.bar(d["l"] + w / 2, d["neutron"], w, color=style.color(1), label=r"magnetization (neutrons)")
    b.set_xticks(d["l"])
    b.set_xlabel(r"reflection $(00l)$ along the chain")
    b.set_ylabel(r"$|F|$ (electrons, $\mu_B$)")
    b.legend(loc="upper right")
    fig.tight_layout()
    return fig


style.main(draw)
