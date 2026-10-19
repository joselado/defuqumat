"""I106 check: force / stress under a field against a finite difference of E + E_field.

Usage: harness.py <arm> <outfile.json>
arms: nc (H2, norm-conserving, force), us (O2 ultrasoft, force),
      fe (bcc Fe ultrasoft, uniform field, stress), fsm (bcc Fe, 'fsm', stress)
"""
import json
import sys
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import defuqumat  # noqa: F401  (x64)
from defuqumat.forces import compute_forces
from defuqumat.forces.energy import energy_at, state_from_result
from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf import Calculation, run_scf
from defuqumat.stress import compute_stress
from defuqumat.system.builder import build_system

PSEUDO = Path("tests/data/pseudo")
CONV = 1e-12

H2 = """
 &control
    calculation = 'scf'
 /
 &system
    ibrav = 1, celldm(1) = 10.0, nat = 2, ntyp = 1,
    ecutwfc = 20.0,
    nspin = 2, starting_magnetization(1) = 0.3,
    occupations = 'smearing', smearing = 'gaussian', degauss = 0.05,
    nosym = .true.
    B_field(3) = 0.05
 /
 &electrons
 /
ATOMIC_SPECIES
 H  1.008  H.pz-vbc.UPF
ATOMIC_POSITIONS (bohr)
 H 0.0 0.0 0.0
 H 0.0 0.0 3.5
K_POINTS (tpiba)
 1
 0.0 0.0 0.0 1.0
"""

O2 = """
 &control
    calculation = 'scf'
 /
 &system
    ibrav = 1, celldm(1) = 9.0, nat = 2, ntyp = 1,
    ecutwfc = 25.0, ecutrho = 150.0,
    nspin = 2, starting_magnetization(1) = 0.5,
    occupations = 'smearing', smearing = 'gaussian', degauss = 0.02,
    nosym = .true.
    B_field(3) = 0.02
 /
 &electrons
 /
ATOMIC_SPECIES
 O  15.999  O.pz-rrkjus.UPF
ATOMIC_POSITIONS (bohr)
 O 0.0 0.0 0.0
 O 0.0 0.0 2.4
K_POINTS (tpiba)
 1
 0.0 0.0 0.0 1.0
"""

FE = """
 &control
    calculation = 'scf'
 /
 &system
    ibrav = 3, celldm(1) = 5.217, nat = 1, ntyp = 1,
    ecutwfc = 25.0, ecutrho = 200.0,
    occupations = 'smearing', smearing = 'marzari-vanderbilt', degauss = 0.05
    nspin = 2
    nosym = .true.
    starting_magnetization(1) = 0.5
    {constraint}
 /
 &electrons
    mixing_beta = 0.3
 /
ATOMIC_SPECIES
 Fe 55.847 Fe.pz-nd-rrkjus.UPF
ATOMIC_POSITIONS (alat)
 Fe 0.0 0.0 0.0
K_POINTS
 1
   0.1875000  0.1875000  0.4375000   1.00
"""
FE_FIELD = "B_field(3) = 0.02"
FE_FSM = ("constrained_magnetization = 'fsm'\n    lambda = 0.02\n"
          "    fixed_magnetization(3) = 2.0")


def setup(text):
    system = build_system(parse_pw_input(text))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return system, pseudos, Calculation(system, pseudos)


def scf(calc, pseudos, start=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = run_scf(calc.system, pseudos, calculation=calc, conv_thr=CONV,
                    max_iterations=600, starting_from=start)
    assert r.converged, "SCF did not converge"
    return r


def lagrangian(r):
    """E + E_field + E_constraint, and for a feedback field + B . m0."""
    L = r.total_energy + (r.field_energy or 0.0) + (r.constraint_energy or 0.0)
    f = r.magnetic_field
    if f is not None and f.constraint == "fsm":
        L += float(np.dot(np.asarray(f.uniform), np.asarray(f.targets)))
    return L


def force_arm(text, atom, axis, steps):
    system, pseudos, calc = setup(text)
    t0 = time.time()
    res = scf(calc, pseudos)
    f = compute_forces(calc, res)
    nat = system.structure.nat
    raw = np.asarray(f.forces) + np.asarray(f.total_before_correction) / nat
    state = state_from_result(res)
    pos0 = jnp.asarray(system.structure.positions)
    slope = np.asarray(jax.grad(
        lambda p: energy_at(calc.at_positions(p), state, terms=True)["field"])(pos0))
    out = dict(force=float(raw[atom, axis]), field_term_force=float(-slope[atom, axis]),
               force_without_term=float(raw[atom, axis] + slope[atom, axis]),
               moment=res.magnetization, field_energy=res.field_energy,
               energy=res.total_energy, points=[])
    for h in steps:
        L, E = [], []
        for s in (1, -1):
            pos = np.array(pos0)
            pos[atom, axis] += s * h
            moved = calc.at_positions(jnp.asarray(pos))
            r = scf(moved, pseudos)
            L.append(lagrangian(r))
            E.append(r.total_energy)
        fd_L = -(L[0] - L[1]) / (2 * h)
        fd_E = -(E[0] - E[1]) / (2 * h)
        out["points"].append(dict(step=h, fd_total_plus_field=fd_L, fd_reported_total=fd_E,
                                  diff_with_term=out["force"] - fd_L,
                                  diff_without_term=out["force_without_term"] - fd_L,
                                  diff_against_reported_total=out["force"] - fd_E))
        print(json.dumps(out["points"][-1]), flush=True)
    out["seconds"] = time.time() - t0
    return out


def stress_arm(text, steps):
    system, pseudos, calc = setup(text)
    t0 = time.time()
    res = scf(calc, pseudos)
    st = compute_stress(calc, res, terms=True)
    volume = float(system.cell.volume)
    # sigma = -(1/V) dE/d eps, so an isotropic eps gives dE/d eps = -V tr sigma.
    predicted = -volume * float(np.trace(st.tensor))
    term = -volume * float(np.trace(st.terms["field"])) if "field" in st.terms else 0.0
    out = dict(dE_deps=predicted, field_term=term, dE_deps_without_term=predicted - term,
               moment=res.magnetization, field_energy=res.field_energy,
               energy=res.total_energy, field=None if res.magnetic_field is None else np.asarray(res.magnetic_field.uniform).tolist(),
               points=[])
    at0 = jnp.asarray(system.cell.at)
    for h in steps:
        L, E = [], []
        for s in (1, -1):
            moved = calc.at_cell(at0 * (1.0 + s * h))
            r = scf(moved, pseudos)
            L.append(lagrangian(r))
            E.append(r.total_energy)
        fd_L = (L[0] - L[1]) / (2 * h)
        fd_E = (E[0] - E[1]) / (2 * h)
        out["points"].append(dict(step=h, fd_total_plus_field=fd_L, fd_reported_total=fd_E,
                                  diff_with_term=predicted - fd_L,
                                  diff_without_term=predicted - term - fd_L,
                                  diff_against_reported_total=predicted - fd_E))
        print(json.dumps(out["points"][-1]), flush=True)
    out["seconds"] = time.time() - t0
    return out


def diag_arm(text, steps=(1e-3, 5e-4)):
    """The frozen derivative two ways: autodiff through at_strain, and a
    central difference of the same frozen functional on at_cell (the
    constructor's tables, which is what the SCF difference runs on)."""
    from defuqumat.stress.energy import strained_energy
    system, pseudos, calc = setup(text)
    res = scf(calc, pseudos)
    state = state_from_result(res)
    frozen = float(energy_at(calc, state))
    expected = res.total_energy + (res.field_energy or 0.0) + (res.constraint_energy or 0.0)
    eye = jnp.eye(3)
    auto_total = float(jax.grad(lambda e: strained_energy(calc, e * eye, state))(0.0))
    auto_field = float(jax.grad(lambda e: energy_at(calc.at_strain(e * eye), state,
                                                    terms=True)["field"])(0.0))
    at0 = jnp.asarray(system.cell.at)
    out = dict(identity=frozen - expected, frozen=frozen, expected=expected,
               moment=res.magnetization, auto_total=auto_total, auto_field=auto_field,
               points=[])
    for h in steps:
        tp = energy_at(calc.at_cell(at0 * (1 + h)), state, terms=True)
        tm = energy_at(calc.at_cell(at0 * (1 - h)), state, terms=True)
        sp = energy_at(calc.at_strain((h) * eye), state, terms=True)
        sm = energy_at(calc.at_strain((-h) * eye), state, terms=True)
        point = dict(step=h)
        for key in tp:
            point["cell_" + key] = float(tp[key] - tm[key]) / (2 * h)
            point["strain_" + key] = float(sp[key] - sm[key]) / (2 * h)
        point["cell_total"] = sum(point["cell_" + k] for k in tp)
        point["strain_total"] = sum(point["strain_" + k] for k in tp)
        out["points"].append(point)
        print(json.dumps(dict(step=h, cell_total=point["cell_total"],
                              strain_total=point["strain_total"],
                              cell_field=point["cell_field"],
                              strain_field=point["strain_field"])), flush=True)
    return out


def ident_arm(text):
    system, pseudos, calc = setup(text)
    res = scf(calc, pseudos)
    state = state_from_result(res)
    terms = energy_at(calc, state, terms=True)
    frozen = float(sum(terms.values()))
    return dict(identity=frozen - (res.total_energy + res.field_energy + res.constraint_energy),
                field_term=float(terms["field"]), field_energy=res.field_energy,
                moment=res.magnetization, field_scale=res.field_scale)


if __name__ == "__main__":
    arm, target = sys.argv[1], sys.argv[2]
    if arm == "nc":
        out = force_arm(H2, 1, 2, (0.04, 0.02, 0.01))
    elif arm == "us":
        out = force_arm(O2, 1, 2, (0.04, 0.02, 0.01))
    elif arm == "fe":
        out = stress_arm(FE.format(constraint=FE_FIELD), (0.004, 0.002, 0.001))
    elif arm == "fsm":
        out = stress_arm(FE.format(constraint=FE_FSM), (0.004, 0.002, 0.001))
    elif arm == "fe0":
        # The control: the same cell and the same frozen-basis difference
        # with no field, so the floor the field arms reach can be told from
        # one that belongs to the difference itself.
        out = stress_arm(FE.format(constraint=""), (0.004, 0.002, 0.001, 0.0005))
    elif arm == "us2":
        out = force_arm(O2, 1, 2, (0.03, 0.015, 0.0075, 0.005))
    elif arm == "fe2":
        out = stress_arm(FE.format(constraint=FE_FIELD), (0.003, 0.0015, 0.0005))
    elif arm == "diag":
        out = diag_arm(FE.format(constraint=FE_FIELD))
    elif arm == "diag0":
        out = diag_arm(FE.format(constraint=""))
    elif arm == "diagfsm":
        out = diag_arm(FE.format(constraint=FE_FSM))
    elif arm == "nc2":
        out = force_arm(H2, 1, 2, (0.04, 0.02, 0.01))
        out["ident"] = ident_arm(H2)
    elif arm == "fe01":
        out = stress_arm(FE.format(constraint="B_field(3) = 0.01"), (0.002, 0.001))
    else:
        raise SystemExit(arm)
    print(json.dumps(out, indent=1, default=float))
    Path(target).write_text(json.dumps(out, indent=1, default=float))
