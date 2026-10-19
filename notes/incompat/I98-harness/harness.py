"""I98's check: the local-field shift of eps from a kick against the Sternheimer static shift.

    python3 harness.py <worktree> <out-dir> <case> <mesh> <ecut> <dual> <runs>

``runs`` is a comma list of ``potential:dt[:variant]``, e.g. ``frozen:0.2,hxc:0.2,hxc:0.1``;
``variant`` is ``augfrozen`` (the augmentation charge of rho(t) held at the ground
state's becsum, D(t) still rebuilt) for the A/B. For each run the current is saved to
``<out>/<case>-<mesh>-<potential>-<dt>[-variant].npz`` and eps(i eta) at several eta is
printed, with and without the static part of J removed. The static shift is
get_dielectric_tensor(screening='full') - (screening='none'); for PAW 'none' still
carries PAW_dpotential, so it is taken with paw_response patched to zero, which makes
it the independent-particle response the frozen propagation computes.
"""
import json
import re
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

root, out = Path(sys.argv[1]), Path(sys.argv[2])
case, mesh = sys.argv[3], sys.argv[4]
ecut, dual = float(sys.argv[5]), float(sys.argv[6])
runs = [r.split(":") for r in sys.argv[7].split(",")]
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root))

from defuqumat import Calculator  # noqa: E402
from defuqumat.realtime.pulse import Kick  # noqa: E402
from defuqumat.realtime.spectra import conductivity_from_kick  # noqa: E402
from defuqumat.realtime.pulse import EV_TO_HA  # noqa: E402

text = (root / "tests/data/qe" / f"{case}.in").read_text()
text = re.sub(r"ecutwfc\s*=\s*[0-9.dD+-]+", f"ecutwfc = {ecut}", text)
text = re.sub(r"ecutrho\s*=\s*[0-9.dD+-]+", f"ecutrho = {dual * ecut}", text)
text = re.sub(r"K_POINTS.*\n\s*[0-9 ]+\n", f"K_POINTS (automatic)\n {mesh}\n", text)
tag = f"{case}-{mesh.replace(' ', '')}-{ecut:g}"
path = out / f"{tag}.in"
path.write_text(text)
calc = Calculator.from_file(path, pseudo_dir=root / "tests/data/pseudo", announce=False,
                            conv_thr=1e-12)
t0 = time.time()
scf = calc.get_scf()
print(f"# {tag}: scf {time.time() - t0:.0f} s, nk {calc.system.kpoints.nk}", flush=True)

summary = {"tag": tag, "static": {}, "runs": {}}
import defuqumat.response.efield as efield  # noqa: E402
original = efield.paw_response
for screening in ("none", "full"):
    if screening == "none" and calc.calculation.is_paw:
        efield.paw_response = lambda c, d, b: jnp.zeros_like(original(c, d, b))
    try:
        t0 = time.time()
        eps = float(calc.get_dielectric_tensor(screening=screening, born_charges=False,
                                               threshold=1e-14).epsilon[0, 0])
    finally:
        efield.paw_response = original
    summary["static"][screening] = eps
    print(f"# static {screening}: {eps:.8f} ({time.time() - t0:.0f} s)", flush=True)
reference = summary["static"]["full"] - summary["static"]["none"]
print(f"# static shift full - none: {reference:.8f}", flush=True)

kappa = 1e-3
etas_ev = (0.2, 0.4, 0.8)
duration = 22.0 / (min(etas_ev) * EV_TO_HA)
import os  # noqa: E402
duration = float(os.environ.get("I98_DURATION", duration))
from defuqumat.scf.driver import Calculation  # noqa: E402
augmented_original = Calculation.augmented

for spec in runs:
    potential, dt = spec[0], float(spec[1])
    variant = spec[2] if len(spec) > 2 else ""
    name = f"{tag}-{potential}-{dt:g}" + (f"-{variant}" if variant else "")
    if variant == "augfrozen":
        frozen_becsum = scf.becsum

        def held(self, rho, becsum_, _frozen=frozen_becsum):
            return augmented_original(self, rho, _frozen)
        Calculation.augmented = held
    t0 = time.time()
    try:
        result = calc.get_realtime(Kick(strength=kappa, direction=(1.0, 0.0, 0.0)), dt=dt,
                                   start=0.0, duration=duration, potential=potential,
                                   block_steps=500)
    finally:
        Calculation.augmented = augmented_original
    seconds = time.time() - t0
    np.savez(out / f"{name}.npz", times=result.times, current=result.current,
             energy=result.energy, energy_times=result.energy_times)
    row = {"seconds": seconds, "steps": len(result.times) - 1,
           "energy_spread": float(np.ptp(result.energy)),
           "norm_drift": result.norm_drift,
           "potential_change": result.extras.get("potential_change", 0.0), "eps": {}}
    for eta in etas_ev:
        for static in (False, True):
            k = conductivity_from_kick(result.times, result.current, kappa, (1.0, 0.0, 0.0),
                                       [0.0], eta * EV_TO_HA, subtract_static=static)
            row["eps"][f"{eta}-{'sub' if static else 'raw'}"] = complex(k.epsilon[0, 0])
    summary["runs"][name] = row
    print(f"# {name}: {seconds:.0f} s, {row['steps']} steps, energy spread "
          f"{row['energy_spread']:.2e}, norm {row['norm_drift']:.2e}, dv "
          f"{row['potential_change']:.2e}", flush=True)
    for key, value in row["eps"].items():
        print(f"    eps({key}) = {value.real:.8f} {value.imag:+.2e}i", flush=True)

# the shifts, hxc - frozen at the same dt
for name, row in summary["runs"].items():
    if "-hxc-" not in name:
        continue
    partner = name.replace("-hxc-", "-frozen-").replace("-augfrozen", "")
    if partner not in summary["runs"]:
        continue
    for key in row["eps"]:
        shift = row["eps"][key] - summary["runs"][partner]["eps"][key]
        print(f"# shift {name} {key}: {shift.real:.8f} {shift.imag:+.2e}i, against static "
              f"{reference:.8f}: {shift.real - reference:+.3e} "
              f"({(shift.real - reference) / reference:+.2e} rel)", flush=True)
(out / f"{tag}-summary.json").write_text(json.dumps(
    summary, indent=1, default=lambda z: [z.real, z.imag] if isinstance(z, complex) else str(z)))
