"""PAW under TB09: the conductivity and SHG hand the one-centre terms their ``c``.

A potential-only meta-GGA's PAW spheres need the cell-averaged ``c`` of the
plane-wave potential, and :meth:`~defuqumat.scf.driver.Calculation.onecenter`
refuses without it. ``run_conductivity`` and ``run_shg`` rebuilt ``ddd_paw``
with ``onecenter(becsum)`` after the band solve had passed ``c`` (in
``fixed_density_hamiltonians``), so a PAW TB09 ground state raised only after
the whole fixed-density solve (audit 2026-11-20c #6). The Kubo quantum metric
and the frozen real-time routes (``_solved_states``) had the same call, and the
last tests below are theirs (#6, the siblings).

No SCF is run. The calculation is a real PAW TB09 one, the density, ``tau`` and
``becsum`` are the starting ones, and the band solve and the assembly are
sentinels: what is tested is the plumbing between them.
"""

import numpy as np
import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo.upf import read_upf
from defuqumat.scf.driver import Calculation
from defuqumat.system.builder import build_system
from defuqumat.workflows import conductivity, geometry, realtime, shg

pytestmark = [
    pytest.mark.unit,
    # TB09 on a PBE dataset, as the meta-GGA tests do; the warning says so.
    pytest.mark.filterwarnings("ignore:input_dft asks for TB09"),
]

PSEUDO = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "pseudo"

_SILICON_PAW = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1,
  ecutwfc = 15.0, ecutrho = 120.0, nosym = .true., noinv = .true.
  {extra}
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pbe-n-kjpaw_psl.0.1.UPF
ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""

NBND = 8


def _state(extra):
    system = build_system(parse_pw_input(_SILICON_PAW.format(extra=extra)))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    density = calculation.starting_density()
    tau = None
    if calculation.functional.is_meta:
        # A Thomas-Fermi kinetic energy density of the starting density: any
        # positive field will do, the point is the path, not the number.
        rho = np.maximum(np.asarray(density), 1e-10)
        tau = 0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0) * rho ** (5.0 / 3.0)
    return system, pseudos, calculation, density, tau, calculation.starting_becsum()


@pytest.fixture(scope="module")
def tb09():
    return _state("input_dft = 'tb09'")


@pytest.fixture(scope="module")
def pbe():
    return _state("")


def _patch(monkeypatch, module, assembly, seen):
    """Replace the band solve and the assembly; record the ``ddd_paw`` handed over."""

    def solve(system, pseudos, density, *, nbnd, calculation, **_):
        nk = int(np.asarray(system.kpoints.weights).shape[0])
        eigenvalues = np.tile(np.linspace(-0.5, 1.0, nbnd), (nk, 1))
        return calculation, system, eigenvalues, np.zeros((nk, nbnd, 1))

    def assemble(*args, ddd_paw=None, **_):
        seen["ddd_paw"] = ddd_paw
        return "reached"

    monkeypatch.setattr(module, "fixed_density_states", solve)
    monkeypatch.setattr(module, assembly, assemble)


def _call(module, state):
    system, pseudos, calculation, density, tau, becsum = state
    entry = module.run_conductivity if module is conductivity else module.run_shg
    return entry(system, pseudos, density, nbnd=NBND, becsum=becsum, tau=tau,
                 calculation=calculation)


def _expected(state):
    """The one-centre D of the potential the band solve was built on."""
    _, _, calculation, density, tau, becsum = state
    meta_c = None
    if tau is not None:
        meta_c = calculation.potential(density, 1.0, None, tau=tau).meta_c
    return np.asarray(calculation.onecenter(becsum, meta_c)[1])


@pytest.mark.parametrize("module, assembly", [
    (conductivity, "optical_conductivity"),
    (shg, "second_harmonic"),
])
def test_a_paw_tb09_state_reaches_the_assembly_with_its_c(monkeypatch, tb09, module, assembly):
    _, _, calculation, *_ = tb09
    assert calculation.is_paw and calculation.functional.is_meta
    seen = {}
    _patch(monkeypatch, module, assembly, seen)
    assert _call(module, tb09) == "reached"
    assert np.array_equal(np.asarray(seen["ddd_paw"]), _expected(tb09))


@pytest.mark.parametrize("module, assembly", [
    (conductivity, "optical_conductivity"),
    (shg, "second_harmonic"),
])
def test_a_paw_pbe_state_hands_over_the_same_d_as_before(monkeypatch, pbe, module, assembly):
    """Without ``tau`` nothing changes: ``onecenter(becsum)`` exactly."""
    _, _, calculation, _, tau, becsum = pbe
    assert calculation.is_paw and not calculation.functional.is_meta and tau is None
    seen = {}
    _patch(monkeypatch, module, assembly, seen)
    assert _call(module, pbe) == "reached"
    assert np.array_equal(np.asarray(seen["ddd_paw"]),
                          np.asarray(calculation.onecenter(becsum)[1]))


# --------------------------------------------------------------------------
# the siblings: the Kubo quantum metric and the frozen real-time states
# --------------------------------------------------------------------------


def _quantum_metric_ddd(monkeypatch, state):
    """``run_quantum_metric(method='kubo')`` to its assembly; the ``ddd_paw`` handed over."""
    system, pseudos, calculation, density, tau, becsum = state
    seen = {}

    def solve(system, pseudos, density, *, nbnd, calculation, **_):
        # ``nbnd`` here is one past the sum, which measures the gap at the cut
        nk = int(np.asarray(system.kpoints.weights).shape[0])
        eigenvalues = np.tile(np.linspace(-0.5, 1.0, nbnd), (nk, 1))
        return calculation, system, eigenvalues, np.zeros((nk, nbnd, 1))

    def assemble(*args, ddd_paw=None, **_):
        seen["ddd_paw"] = ddd_paw
        return "reached"

    monkeypatch.setattr(geometry, "fixed_density_states", solve)
    monkeypatch.setattr(geometry, "quantum_metric", assemble)
    assert geometry.run_quantum_metric(
        system, pseudos, density, method="kubo", nbnd=NBND, becsum=becsum,
        tau=tau, calculation=calculation) == "reached"
    return np.asarray(seen["ddd_paw"])


def _realtime_ddd(monkeypatch, state):
    """``_solved_states``, the frozen real-time routes' setup, with the solve replaced."""
    from defuqumat.workflows import nscf

    system, pseudos, calculation, density, tau, becsum = state

    def solve(system, pseudos, density, *, nbnd, calculation, **_):
        nk = int(np.asarray(system.kpoints.weights).shape[0])
        eigenvalues = np.tile(np.linspace(-0.5, 1.0, nbnd), (nk, 1))
        return calculation, system, eigenvalues, np.zeros((nk, nbnd, 1))

    # imported inside ``_solved_states``, so it is replaced where it lives
    monkeypatch.setattr(nscf, "fixed_density_states", solve)
    *_, ddd_paw = realtime._solved_states(
        system, pseudos, density, system.kpoints, nbnd=NBND, conv_thr=1.0e-8,
        k_batch="default", calculation=calculation, becsum=becsum, tau=tau)
    return np.asarray(ddd_paw)


@pytest.mark.parametrize("route", [_quantum_metric_ddd, _realtime_ddd],
                         ids=["quantum-metric", "realtime"])
def test_a_paw_tb09_sibling_builds_d_with_its_c(monkeypatch, tb09, route):
    _, _, calculation, *_ = tb09
    assert calculation.is_paw and calculation.functional.is_meta
    assert np.array_equal(route(monkeypatch, tb09), _expected(tb09))


@pytest.mark.parametrize("route", [_quantum_metric_ddd, _realtime_ddd],
                         ids=["quantum-metric", "realtime"])
def test_a_paw_pbe_sibling_builds_the_same_d_as_before(monkeypatch, pbe, route):
    """Without ``tau`` nothing changes: ``onecenter(becsum)`` exactly."""
    _, _, calculation, _, tau, becsum = pbe
    assert calculation.is_paw and not calculation.functional.is_meta and tau is None
    assert np.array_equal(route(monkeypatch, pbe),
                          np.asarray(calculation.onecenter(becsum)[1]))
