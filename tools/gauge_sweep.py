"""The gauge sweep over ``Calculator``'s front door (``notes/DEGENERACY-NEXT.md`` item 7, P151).

Every ``get_*`` a cell admits is computed in four arms on one converged ground state: unrotated
twice (the method's own repeat floor) and under two independent random gauges. A gauge is a
unitary inside every degenerate multiplet of every set of eigenvectors a consumer receives: the
converged states in the calculator's cache and the output of every eigensolve made while the
arm runs. A leaf of the result that moves under both gauges and not under the repeat is gauge
dependent; the plan and the decisions are ``plans/gauge-sweep-p151.md``.

Usage::

    python -u tools/gauge_sweep.py <cell> [--only get_x,get_y] [--out DIR]

with ``<cell>`` one of :data:`CELLS`. One line per (method, leaf) is printed, and a JSON of every
number goes to ``DIR/<cell>.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "qe"
PSEUDO = ROOT / "tests" / "data" / "pseudo"

#: How many solves went through the rotating wrapper, and how many times the cached states were
#: read, in the arm now running: zero on both is "not reached", never "invariant".
REACH = {"solves": 0, "states_read": 0}


# ----------------------------------------------------------------------------------------------
# the cells
# ----------------------------------------------------------------------------------------------

def _si_gamma() -> str:
    text = (DATA / "si2-lda-gap.in").read_text()
    assert "\n 6 6 6 0 0 0" in text
    return text.replace("\n 6 6 6 0 0 0", "\n 2 2 2 0 0 0")


def _si_gamma_nosym() -> str:
    """The same cell on the whole grid, for the consumers that refuse a wedge."""
    text = _si_gamma()
    assert "\n &system\n" in text
    return text.replace("\n &system\n", "\n &system\n    nosym = .true., noinv = .true.,\n", 1)


def _si_gamma_nosym_tight() -> str:
    """The whole-grid cell at ``conv_thr = 1e-14``: with the point group off, the density is
    symmetric only to the SCF's accuracy, so the occupied multiplets are split by 3e-7 to 5e-7 Ry
    at ``conv_thr = 1e-9``, and a rotation inside them is then not exactly a gauge."""
    text = _si_gamma_nosym()
    assert "    conv_thr = 1.0d-9" in text
    return text.replace("    conv_thr = 1.0d-9", "    conv_thr = 1.0d-14")


def _alas_soc() -> str:
    text = (DATA / "alas-epsilon-us-soc.in").read_text()
    assert "\n 4 4 4 1 1 1" in text
    return text.replace("\n 4 4 4 1 1 1", "\n 2 2 2 0 0 0")


def _o_smear() -> str:
    text = (DATA / "o-atom-fixed-lsda.in").read_text()
    old = "    occupations = 'fixed',"
    assert old in text and "    tot_magnetization = 2,\n" in text
    text = text.replace(old, "    occupations = 'smearing', smearing = 'gaussian', degauss = 0.01,")
    # one shared Fermi level: tot_magnetization with a smearing has two, and the Sternheimer
    # consumers refuse that by name
    return text.replace("    tot_magnetization = 2,\n", "")


CELLS = {"si-gamma": _si_gamma, "si-gamma-nosym": _si_gamma_nosym, "si-gamma-nosym-tight": _si_gamma_nosym_tight, "alas-soc": _alas_soc, "o-smear": _o_smear}


# ----------------------------------------------------------------------------------------------
# the gauge
# ----------------------------------------------------------------------------------------------

def tolerance(ethr, empty_floor=False):
    """The code's own degeneracy rule for states of this provenance (P149)."""
    from defuqumat.solvers.davidson import resolve_degeneracy_tol

    if ethr is not None:
        ethr = float(jnp.max(jnp.asarray(ethr)))
    return resolve_degeneracy_tol(0.0, ethr, empty_floor=empty_floor)


class Gauge:
    """A random unitary inside every multiplet, ``U = expm(i A o M)`` per (channel, k).

    ``A`` is a Hermitian matrix drawn from ``seed`` and the band count; ``M_ij`` is one where
    bands ``i`` and ``j`` carry the same multiplet label, the label being
    ``cumsum(diff(e) >= tol)`` on the sorted eigenvalues, so ``U`` is block diagonal by
    construction (a mask ``|e_i - e_j| < tol`` is not transitive and would mix along a chain).
    The diagonal of ``A`` puts a random phase on every singlet as well (fable, 2026-10-15).
    """

    def __init__(self, seed: int):
        self.seed = seed
        self._matrices = {}
        self.log = []

    def matrix(self, nbnd: int, count: int):
        key = (nbnd, count)
        if key not in self._matrices:
            rng = np.random.default_rng([self.seed, nbnd, count])
            a = rng.normal(size=(count, nbnd, nbnd)) + 1j * rng.normal(size=(count, nbnd, nbnd))
            self._matrices[key] = jnp.asarray(0.5 * np.pi * (a + np.conj(np.swapaxes(a, -1, -2))))
        return self._matrices[key]

    def unitary(self, eigenvalues, tol):
        """``(..., nbnd, nbnd)`` for eigenvalues ``(..., nbnd)``, any leading shape."""
        e = jax.lax.stop_gradient(jnp.asarray(eigenvalues))
        shape = e.shape
        flat = e.reshape((-1, shape[-1]))
        steps = jnp.diff(flat, axis=-1) >= tol
        labels = jnp.concatenate([jnp.zeros((flat.shape[0], 1), int),
                                  jnp.cumsum(steps, axis=-1)], axis=-1)
        mask = labels[:, :, None] == labels[:, None, :]
        generator = jnp.where(mask, self.matrix(shape[-1], flat.shape[0]), 0.0)
        if not isinstance(e, jax.core.Tracer):
            gaps = np.diff(np.asarray(flat), axis=-1)
            inside = gaps[~np.asarray(steps)]
            outside = gaps[np.asarray(steps)]
            self.log.append((float(tol), float(inside.max()) if inside.size else 0.0,
                             float(outside.min()) if outside.size else np.inf, int(inside.size)))
        u = jax.vmap(jax.scipy.linalg.expm)(1j * generator)
        return u.reshape(shape + (shape[-1],))

    def rotate(self, eigenvalues, psi, tol):
        """``psi (..., nbnd, ndim)`` with every multiplet rotated: ``psi'_n = sum_m U_mn psi_m``."""
        u = self.unitary(eigenvalues, tol).astype(psi.dtype)
        return jnp.einsum("...mn,...md->...nd", u, psi)

    def solver(self, registered):
        """``registered`` with its eigenvectors rotated after every solve, at that solve's ``ethr``."""
        def rotated(hamiltonian, nbnd, psi0=None, ethr=None, *args, **kwargs):
            out = registered(hamiltonian, nbnd, psi0, ethr, *args, **kwargs)
            REACH["solves"] += 1
            return (out[0], self.rotate(out[0], out[1], tolerance(ethr))) + tuple(out[2:])
        return rotated

    def summary(self):
        if not self.log:
            return "no rotation"
        return (f"{len(self.log)} rotations at tol {min(x[0] for x in self.log):.0e} to "
                f"{max(x[0] for x in self.log):.0e}, {sum(x[3] for x in self.log)} degenerate "
                f"neighbours; largest gap inside {max(x[1] for x in self.log):.1e}, smallest "
                f"outside {min(x[2] for x in self.log):.1e} Ry")


@contextlib.contextmanager
def gauged(gauge: Gauge | None):
    """Every eigensolve made inside goes through the rotating wrapper.

    The registry entries are replaced, which ``Calculation.__init__`` reads through
    ``get_eigensolver`` at construction; a ``Calculation`` built before the patch keeps the
    plain solver, which is why each arm builds its own calculator.
    """
    if gauge is None:
        yield
        return
    from defuqumat import solvers

    saved = dict(solvers.EIGENSOLVERS)
    for name, registered in saved.items():
        solvers.EIGENSOLVERS[name] = gauge.solver(registered)
    try:
        yield
    finally:
        solvers.EIGENSOLVERS.update(saved)


def tracked(result):
    """``result`` as a subclass that counts reads of its ``wavefunctions``."""
    import copy

    from defuqumat.scf.driver import SCFResult

    class Tracked(SCFResult):
        def __getattribute__(self, name):
            if name == "wavefunctions":
                REACH["states_read"] += 1
            return object.__getattribute__(self, name)

    out = copy.copy(result)
    out.__class__ = Tracked
    return out


def rotated_result(result, gauge: Gauge):
    """The ``SCFResult`` with its converged states rotated, everything else shared."""
    e = jnp.asarray(result.eigenvalues_by_spin)
    tol = tolerance(result.ethr, empty_floor=True)
    return dataclasses.replace(result, wavefunctions=gauge.rotate(e, result.wavefunctions, tol))


# ----------------------------------------------------------------------------------------------
# reading a result
# ----------------------------------------------------------------------------------------------

def leaves(obj, prefix="", depth=0, seen=None):
    """Every numeric array or scalar in ``obj``, by path."""
    seen = set() if seen is None else seen
    if depth > 6 or id(obj) in seen:
        return {}
    if isinstance(obj, (bool, str, bytes, type(None))) or callable(obj) and not hasattr(obj, "__dict__"):
        return {}
    if isinstance(obj, (int, float, complex, np.number)):
        return {prefix or "value": np.asarray(obj)}
    if isinstance(obj, (np.ndarray, jax.Array)):
        a = np.asarray(obj)
        return {prefix or "value": a} if np.issubdtype(a.dtype, np.number) else {}
    seen.add(id(obj))
    out = {}
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, (list, tuple)):
        items = enumerate(obj)
    elif dataclasses.is_dataclass(obj):
        items = ((f.name, getattr(obj, f.name, None)) for f in dataclasses.fields(obj))
    elif hasattr(obj, "__dict__"):
        items = vars(obj).items()
    else:
        return {}
    for key, value in items:
        if isinstance(key, str) and key.startswith("_"):
            continue
        out.update(leaves(value, f"{prefix}.{key}" if prefix else str(key), depth + 1, seen))
    return out


def distance(a, b):
    """``max|a - b| / max|a|``, or the absolute figure where ``a`` is zero; ``nan`` on a shape change."""
    if a.shape != b.shape:
        return float("nan")
    if a.size == 0:
        return 0.0
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.all() and not np.array_equal(np.isfinite(a), np.isfinite(b)):
        return float("inf")
    d = float(np.max(np.abs(a[finite] - b[finite]), initial=0.0))
    scale = float(np.max(np.abs(a[finite]), initial=0.0))
    return d / scale if scale > 0 else d


# ----------------------------------------------------------------------------------------------
# the arms
# ----------------------------------------------------------------------------------------------

def run_arm(text, parent, call, gauge):
    """``call(calc)`` on a fresh calculator holding ``parent``'s ground state, rotated by ``gauge``.

    Returns the leaves and the reach of the arm. The fresh calculator matters twice: its
    ``Calculation`` is built inside the patch, and ``_strain_response`` and ``_ultracell`` are
    caches too, so a response built in one arm and reused in another would be a null test.
    """
    from defuqumat import Calculator

    REACH.update(solves=0, states_read=0)
    with gauged(gauge):
        calc = Calculator.from_text(text, pseudo_dir=PSEUDO, announce=False)
        result = parent.scf_result if gauge is None else rotated_result(parent.scf_result, gauge)
        calc._scf, calc._scf_options = tracked(result), parent._scf_options
        calc.calculation
        out = leaves(call(calc))
    return out, dict(REACH)


def control(text, parent):
    """Two O(1) movers, so that a null result can be told from a pass.

    The cached states: the first member of every multiplet, ``|psi|^2`` summed over G, against
    the unrotated one. A re-diagonalising consumer: ``fixed_density_states`` at the converged
    density in a gauged calculator, the same reading.
    """
    from defuqumat import Calculator
    from defuqumat.workflows.nscf import fixed_density_states

    scf = parent.scf_result
    e = np.asarray(scf.eigenvalues_by_spin)

    def first_members(e, tol):
        same = np.diff(e, axis=-1) < tol
        start = np.zeros(e.shape, bool)
        start[..., :-1] = same
        start[..., 1:] &= ~same
        return start

    def moved(a, b, where):
        if not where.any():
            return 0.0
        a, b = np.abs(np.asarray(a)) ** 2, np.abs(np.asarray(b)) ** 2
        return float(np.abs(a[where] - b[where]).max() / a[where].max())

    gauge = Gauge(101)
    tol = tolerance(scf.ethr, empty_floor=True)
    cached = moved(scf.wavefunctions, rotated_result(scf, gauge).wavefunctions,
                   first_members(e, tol))
    print(f"control, cached states: tol {tol:.0e}, {int(first_members(e, tol).sum())} multiplets, "
          f"first member moved by {cached:.3f} of its largest coefficient; {gauge.summary()}",
          flush=True)

    states = {}
    for label, g in (("plain", None), ("rot", Gauge(102))):
        with gauged(g):
            calc = Calculator.from_text(text, pseudo_dir=PSEUDO, announce=False)
            calc._scf, calc._scf_options = scf, parent._scf_options
            _, _, values, psi = fixed_density_states(
                calc.system, calc.pseudos, scf.density, calculation=calc.calculation,
                becsum=tuple(scf.becsum or ()))
            states[label] = (np.asarray(values), np.asarray(psi), g)
    values, psi, _ = states["plain"]
    values = values.reshape(e.shape) if values.size == e.size else values
    g = states["rot"][2]
    tol = min(x[0] for x in g.log)
    solved = moved(psi, states["rot"][1], first_members(np.asarray(values), tol))
    print(f"control, a fixed-density solve: tol {tol:.0e}, first member moved by {solved:.3f}; "
          f"eigenvalues moved by {np.abs(values - states['rot'][0]).max():.1e}; {g.summary()}",
          flush=True)
    return {"cached": cached, "solved": solved}


def sweep(cell, calls, out=None):
    from defuqumat import Calculator

    text = CELLS[cell]()
    parent = Calculator.from_text(text, pseudo_dir=PSEUDO, announce=False)
    started = time.perf_counter()
    scf = parent.get_scf()
    e = np.asarray(scf.eigenvalues_by_spin)
    print(f"cell={cell} converged={scf.converged} E={scf.total_energy:.12f} ethr={scf.ethr:.2e} "
          f"shape={e.shape} in {time.perf_counter() - started:.1f}s", flush=True)
    if scf.fermi_energy is not None:
        near = np.abs(e - scf.fermi_energy) < 0.05
        print(f"bands within 0.05 Ry of e_F: {np.sort(e[near])}", flush=True)
    record = {"cell": cell, "control": control(text, parent), "methods": {}}

    for name, call in calls.items():
        t0 = time.perf_counter()
        try:
            arms, reach = {}, {}
            for label, gauge in (("plain-1", None), ("plain-2", None),
                                 ("rot-A", Gauge(1)), ("rot-B", Gauge(2))):
                arms[label], reach[label] = run_arm(text, parent, call, gauge)
                if gauge is not None:
                    reach[label]["rotations"] = gauge.summary()
        except Exception as error:  # a refusal, or a harness defect: recorded whole either way
            first_line = str(error).split("\n")[0][:240]
            print(f"{name}: {type(error).__name__}: {first_line}", flush=True)
            record["methods"][name] = {"error": f"{type(error).__name__}: {first_line}",
                                       "traceback": traceback.format_exc()[-3000:]}
            continue
        rows = {}
        for path, ref in arms["plain-1"].items():
            row = {label: distance(ref, arms[label].get(path, np.full(ref.shape, np.nan)))
                   for label in ("plain-2", "rot-A", "rot-B")}
            row["size"] = float(np.max(np.abs(ref), initial=0.0)) if ref.size else 0.0
            rows[path] = row
        moving = {p for p, r in rows.items()
                  if min(r["rot-A"], r["rot-B"]) > 10 * max(r["plain-2"], 1e-12)}
        a = reach["rot-A"]
        print(f"{name}: {len(rows)} leaves, {len(moving)} move under both gauges; "
              f"{time.perf_counter() - t0:.1f}s; reach: {a['solves']} solves, {a['states_read']} "
              f"reads of the cached states; {a['rotations']}", flush=True)
        for path, r in sorted(rows.items(), key=lambda pr: -min(pr[1]["rot-A"], pr[1]["rot-B"])):
            flag = "MOVES" if path in moving else "     "
            print(f"  {flag} {path:<56s} plain {r['plain-2']:.1e}  A {r['rot-A']:.1e}  "
                  f"B {r['rot-B']:.1e}  |O| {r['size']:.2e}", flush=True)
        record["methods"][name] = {"leaves": rows, "seconds": time.perf_counter() - t0,
                                   "reach": reach}
    if out is not None:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / f"{cell}.json").write_text(json.dumps(record, indent=1, default=str))
    print("DONE", flush=True)
    return record


def main():
    from tools.gauge_calls import CALLS

    parser = argparse.ArgumentParser()
    parser.add_argument("cell", choices=sorted(CELLS))
    parser.add_argument("--only", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--followups", action="store_true")
    args = parser.parse_args()
    calls = CALLS(args.cell, args.followups)
    if args.only:
        wanted = args.only.split(",")
        calls = {k: v for k, v in calls.items() if k in wanted}
    sweep(args.cell, calls, args.out)


if __name__ == "__main__":
    main()
