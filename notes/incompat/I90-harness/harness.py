"""I90: the piezoelectric tensor of a *polar* crystal, two routes.

The response route (``clamped_ion_piezoelectric``/``zstar_eu``) contracts the
field response against the strain derivative of ``H`` at frozen plane-wave
coefficients. The claim being measured is that this contraction is
Vanderbilt's **proper** tensor, ``(1/Omega) sum_g a_g d(phi_g)/d(eps)``, with
no ``delta P`` correction to add, so the refusal of a polar crystal guards a
term that is not missing.

The cell is ``alas-raman.in`` (norm-conserving LDA AlAs, 10 Ry, nosym) with As
moved from ``a(1/4,1/4,1/4)`` to ``a(1/4,1/4,0.29)``: point group ``mm2``,
polar along ``z``. The other route is Elk's: one ground state per strain, three
Berry-phase polarizations each, differenced. From the same phases it builds

* the proper tensor, wrapped phase differences contracted with the
  *unstrained* lattice vectors and volume, and
* the improper tensor, ``dP_k/d eps_ij`` of the Cartesian ``P`` with the
  strained vectors and volume, on one branch,

so which one the response route reproduces is read off the data, and the size
of the difference (``P_z`` in the components that pair labels) is printed
beside it.

Every geometry runs in a spawned child (``piezo_measure.py``'s reason: the
process's mapping count, not its memory). Run:

    python3 harness.py OUTDIR
"""

from __future__ import annotations

import json
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
PSEUDO = ROOT / "tests" / "data" / "pseudo"
BASE = ROOT / "tests" / "data" / "qe" / "alas-raman.in"
DISPLACED_Z = 0.29

#: The Berry-phase side's own sampling, the calibration cell's in
#: ``piezo_measure.py`` (there agreeing with every denser string mesh tried to
#: 0.1 per cent), and the SCF mesh under it.
NPPSTR = 11
TRANSVERSE = (6, 6)
BERRY_SCF_MESH = 6
CONV_THR = 1.0e-10

#: The response route's convergence parameter, the k-mesh.
RESPONSE_MESHES = (4, 6, 8)

#: (label, (i, j), steps): the strain tensors differenced.
STRAINS = (
    ("xx", (0, 0), (0.005, 0.0025)),
    ("zz", (2, 2), (0.005,)),
    ("xz", (0, 2), (0.005, 0.0025)),
    ("yz", (1, 2), (0.005,)),
)


def _input_text(z: float) -> str:
    text = BASE.read_text()
    old = " As 0.25 0.25 0.25"
    assert old in text
    return text.replace(old, f" As 0.25 0.25 {z:.4f}")


def _calculator(payload):
    from defuqumat.calculator import Calculator
    from defuqumat.system.kpoints import KPoints

    path = Path(payload["outdir"]) / f"cell-{payload['z']:.4f}.in"
    if not path.exists():
        path.write_text(_input_text(payload["z"]))
    calculator = Calculator.from_file(path, pseudo_dir=str(PSEUDO),
                                      announce=False)
    grid = (int(payload["mesh"]),) * 3
    calculator = calculator.with_kpoints(
        KPoints.automatic(grid, (0, 0, 0), calculator.system.cell))
    return calculator


def _berry_child(payload, queue):
    calculator = _calculator(payload)
    at = np.asarray(calculator.system.cell.at)
    strain = np.asarray(payload["strain"])
    strained = calculator.with_cell(at @ (np.eye(3) + strain).T)
    start = time.time()
    scf = strained.get_scf(conv_thr=CONV_THR)
    cell = strained.system.cell
    phases = []
    quantum = None
    for gdir in range(3):
        pol = strained.get_polarization(gdir=gdir, nppstr=NPPSTR,
                                        transverse=TRANSVERSE)
        phases.append(float(pol.total_phase))
        quantum = float(pol.quantum)
    queue.put({"phases": phases, "quantum": quantum,
               "vectors": np.asarray(cell.at).tolist(),
               "volume": float(cell.volume),
               "energy": float(scf.total_energy),
               "iterations": int(scf.iterations),
               "seconds": time.time() - start})


def _response_child(payload, queue):
    import defuqumat.response.piezo as piezo

    # The refusal being measured; nothing else is lifted.
    if hasattr(piezo, "require_a_nonpolar_crystal"):
        piezo.require_a_nonpolar_crystal = lambda *a, **k: None
    calculator = _calculator(payload)
    start = time.time()
    scf = calculator.get_scf(conv_thr=CONV_THR)
    tensor = calculator.get_piezoelectric_tensor(method=payload["method"],
                                                 kmesh_warning=False)
    field = tensor.dielectric
    queue.put({"e": np.asarray(tensor.e).tolist(),
               "converged": bool(tensor.converged),
               "nk": int(calculator.system.kpoints.nk),
               "energy": float(scf.total_energy),
               "epsilon": np.asarray(field.epsilon).tolist(),
               "seconds": time.time() - start})


def _spawn(target, payload):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    child = context.Process(target=target, args=(payload, queue))
    child.start()
    result = queue.get()
    child.join()
    if child.exitcode != 0:
        raise RuntimeError(f"child exited {child.exitcode}")
    return result


def wrapped(x: float, quantum: float = 1.0) -> float:
    return float(x - quantum * np.rint(x / quantum))


def main() -> None:
    from defuqumat.units import E_BOHR2_TO_C_M2 as C

    outdir = Path(sys.argv[1]).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    record = {"displaced_z": DISPLACED_Z, "nppstr": NPPSTR,
              "transverse": TRANSVERSE, "berry_scf_mesh": BERRY_SCF_MESH,
              "conv_thr": CONV_THR}
    dump = lambda: (outdir / "result.json").write_text(json.dumps(record, indent=1))

    def berry(z, strain):
        return _spawn(_berry_child, {"outdir": str(outdir), "z": z,
                                     "mesh": BERRY_SCF_MESH,
                                     "strain": np.asarray(strain).tolist()})

    zero = np.zeros((3, 3))
    # The spontaneous polarization, relative to the zincblende reference on a
    # continuous branch: the P a delta-P correction would be built from.
    ref = berry(0.25, zero)
    pol = berry(DISPLACED_Z, zero)
    record["reference"], record["polar"] = ref, pol
    a0 = np.asarray(pol["vectors"])
    omega0 = pol["volume"]
    dphi = np.array([wrapped(pol["phases"][g] - ref["phases"][g], pol["quantum"])
                     for g in range(3)])
    p_spont = (dphi @ a0) / omega0
    p_raw = (np.asarray(pol["phases"]) @ a0) / omega0
    record["P_spontaneous_C_m2"] = (p_spont * C).tolist()
    record["P_raw_branch_C_m2"] = (p_raw * C).tolist()
    record["quantum_C_m2"] = (pol["quantum"] * np.linalg.norm(a0, axis=1)
                              / omega0 * C).tolist()
    print(f"P_s = {p_spont * C} C/m^2 (raw branch {p_raw * C}), "
          f"{ref['seconds']:.0f}+{pol['seconds']:.0f} s", flush=True)
    dump()

    record["fd"] = []
    for label, (i, j), steps in STRAINS:
        for s in steps:
            strain = np.zeros((3, 3))
            strain[i, j] = strain[j, i] = s
            plus, minus = berry(DISPLACED_Z, strain), berry(DISPLACED_Z, -strain)
            factor = 2.0 * s * (1.0 if i == j else 2.0)
            d = np.array([wrapped(plus["phases"][g] - minus["phases"][g],
                                  pol["quantum"]) for g in range(3)])
            proper = (d @ a0) / omega0 / factor
            # Improper: Cartesian P on one branch, strained vectors and volume.
            phi_minus = np.asarray(minus["phases"])
            phi_plus = phi_minus + d
            p_plus = phi_plus @ np.asarray(plus["vectors"]) / plus["volume"]
            p_minus = phi_minus @ np.asarray(minus["vectors"]) / minus["volume"]
            improper_raw = (p_plus - p_minus) / factor
            one = {"label": label, "ij": [i, j], "step": s,
                   "proper_C_m2": (proper * C).tolist(),
                   "improper_raw_branch_C_m2": (improper_raw * C).tolist(),
                   "plus": plus, "minus": minus}
            record["fd"].append(one)
            print(f"eps_{label} = +-{s}: proper {proper * C}, improper(raw "
                  f"branch) {improper_raw * C}, "
                  f"{plus['seconds']:.0f}+{minus['seconds']:.0f} s", flush=True)
            dump()

    record["response"] = []
    for mesh in RESPONSE_MESHES:
        out = _spawn(_response_child, {"outdir": str(outdir), "z": DISPLACED_Z,
                                       "mesh": mesh, "method": "zstar_eu"})
        e = np.asarray(out["e"]) * C
        out["mesh"] = mesh
        record["response"].append(out)
        comps = {lab: e[:, i, j].tolist() for lab, (i, j), _ in STRAINS}
        print(f"response {mesh}^3 ({out['nk']} k, converged {out['converged']}, "
              f"{out['seconds']:.0f} s): {comps}", flush=True)
        dump()
    print("done", flush=True)


if __name__ == "__main__":
    main()
