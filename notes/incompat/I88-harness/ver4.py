"""I88, third verification: an admitted metal (top band below 1e-12) whose top
band is still inside the strain projector's cutoff and nearly degenerate with
the first band not computed.

The guard compares the top band's *occupation* with 1e-12, while the cutoff
(``Smearing.cutoff``) places ``xmax`` where the Gaussian *delta function* falls
to 1e-12, for every smearing but Fermi-Dirac: 5.20 degauss. A Gaussian top band
between 4.97 and 5.20 degauss above e_F, or a Marzari-Vanderbilt one between
4.47 and 5.20, is admitted and still solved for.

Usage (from a checkout root, under slot.sh; r3.py is imported from beside it):
  ver4.py bands <smearing>:<degauss>:<nbnd>[:<celldm>:<grid>]
      SCF only; per band n, min over k of x = (e_n - e_F)/degauss, the gap to
      n+1 at that k, and sum_k occ_n(k)/gap_{n,n+1}(k) scaled to max occ 1.
  ver4.py tune <smearing>:<nbnd>:<x_target>:<degauss0>[:<celldm>:<grid>]
      move degauss until band nbnd's min x is x_target, report the guard, the
      cutoff and the gap to band nbnd+1 (from an nbnd+2 run), then the identity.
  ver4.py curve <smearing>:<degauss>:<nbnd>[:<celldm>:<grid>]
      the identity of r3.py on (00, rand).
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import r3  # noqa: E402
from defuqumat.response.strain import (  # noqa: E402
    STRAIN_SMEARING_SMALL, STRAIN_TOP_BAND_OCCUPATION)
from defuqumat.response.sternheimer import Smearing  # noqa: E402
from defuqumat.scf.occupations import smearing_order, wgauss  # noqa: E402

say = r3.say


def parse(spec, n):
    parts = spec.split(":")
    head, tail = parts[:n], parts[n:]
    celldm, grid = (tail + ["7.50", "4x4x2"][len(tail):])[:2]
    return head, celldm, grid


def table(state):
    system, _, calculation, result, eigenvalues, _, _ = state
    e3 = jnp.asarray(eigenvalues)
    e3 = e3 if e3.ndim == 3 else e3[None]
    _, lv = calculation.occupations(e3)
    ef = float(lv.get("fermi_energy"))
    deg = float(system.degauss)
    order = smearing_order(system.smearing)
    e = np.asarray(e3)[0]
    occ = np.abs(np.asarray(wgauss((ef - e) / deg, order)))
    cut = Smearing(ef=ef, degauss=deg, ngauss=order, small=STRAIN_SMEARING_SMALL).cutoff
    say(f"  ef {ef:.10f} degauss {deg} cutoff x {(cut - ef) / deg:.4f}")
    for n in range(e.shape[1]):
        x = (e[:, n] - ef) / deg
        k = int(np.argmin(x))
        line = f"  band {n + 1:2d} xmin {x[k]:8.4f} at k {k:2d} maxocc {occ[:, n].max():.3e}"
        if n + 1 < e.shape[1]:
            gap = e[:, n + 1] - e[:, n]
            ratio = occ[:, n] / occ[:, n].max() / np.maximum(gap, 1e-16)
            kg = int(np.argmin(gap))
            line += (f" gap@min {gap[k]:.3e} mingap {gap[kg]:.3e} (k {kg}, x {x[kg]:.3f})"
                     f" score {ratio.sum():.3e} worst-k {int(np.argmax(ratio))}")
        say(line)
    return e, ef, deg, order, cut


def bands(spec):
    (smearing, degauss, nbnd), celldm, grid = parse(spec, 3)
    state = r3.ground(r3.write_input(smearing, degauss, nbnd, celldm, grid))
    table(state)


def tune(spec):
    (smearing, nbnd, xt, d0), celldm, grid = parse(spec, 4)
    nbnd, xt, d = int(nbnd), float(xt), float(d0)
    for it in range(6):
        state = r3.ground(r3.write_input(smearing, f"{d:.8f}", str(nbnd), celldm, grid))
        e, ef, deg, order, cut = table(state)
        xmin = float(((e[:, -1] - ef) / deg).min())
        say(f"  tune {it}: degauss {d:.8f} xmin {xmin:.5f} target {xt}")
        if abs(xmin - xt) < 0.01:
            break
        d = d * xmin / xt
        jax.clear_caches()
    top = state[-1]
    solved = bool((e[:, -1] < cut).any())
    say(f"  TOP {top:.3e} limit {STRAIN_TOP_BAND_OCCUPATION:.0e} admitted {top <= STRAIN_TOP_BAND_OCCUPATION}"
        f" top band inside the cutoff at {int((e[:, -1] < cut).sum())} k ({solved})")
    above = r3.ground(r3.write_input(smearing, f"{d:.8f}", str(nbnd + 2), celldm, grid))
    ea, efa, _, _, _ = table(above)
    gap = ea[:, nbnd] - ea[:, nbnd - 1]
    occ = np.abs(np.asarray(wgauss((efa - ea[:, nbnd - 1]) / deg, order)))
    k = int(np.argmax(occ / np.maximum(gap, 1e-16)))
    say(f"  GAP to the first band not computed: min {gap.min():.3e} Ry; at the worst k {k}: "
        f"gap {gap[k]:.3e}, occ {occ[k]:.3e}, occ/gap {occ[k] / gap[k]:.3e}; "
        f"sum occ/gap {(occ / np.maximum(gap, 1e-16)).sum():.3e}")
    jax.clear_caches()
    if top <= STRAIN_TOP_BAND_OCCUPATION:
        r3.identity(state, names=("00", "rand"))
    else:
        say("  not admitted, identity skipped")


def curve(spec):
    (smearing, degauss, nbnd), celldm, grid = parse(spec, 3)
    state = r3.ground(r3.write_input(smearing, degauss, nbnd, celldm, grid))
    table(state)
    r3.identity(state, names=("00", "rand"))


def main():
    mode, specs = sys.argv[1], sys.argv[2:]
    r3.OUT.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        {"bands": bands, "tune": tune, "curve": curve}[mode](spec)
        jax.clear_caches()


if __name__ == "__main__":
    main()
    say("done")
