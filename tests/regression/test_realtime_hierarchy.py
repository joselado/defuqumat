"""The frequency-domain hierarchy against the dense one and against the propagation.

Two comparisons, each an identity rather than an agreement:

* the iterative hierarchy of :mod:`defuqumat.realtime.hierarchy` against
  :func:`~defuqumat.realtime.dense.dense_orders` with every band of the sphere, on
  zincblende AlAs at 4 Ry at Gamma and a general point, all five components of
  orders one to three at three frequencies, and the result unchanged when the
  projector on the computed bands holds half as many;
* the spectrum's second and third orders against the real-time route's at one
  frequency, which shares the table of the projectors and nothing of the solve;
* the first order with the induced potential, self-consistent, against the dense
  self-consistent solve of :func:`~defuqumat.realtime.dense.dense_first_order`, and
  its static shift against the Sternheimer stack's;
* orders one to three with the induced potential against the propagation with the
  potential updated (I102), at two time steps;
* the same first order on two FFT grids (``ecutrho = 8 ecutwfc``) against the
  dense solve, and with the Hartree kernel against the one-grid run of the same
  states.

``PLAN.md`` P137 has the measurements.
"""

import math
import re
from functools import lru_cache
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pytest

from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.realtime.dense import (
    dense_first_order, dense_ground_states, dense_hamiltonians, dense_orders)
from defuqumat.realtime.hierarchy import (
    HierarchyError, hierarchy_linear_self_consistent, hierarchy_orders)
from defuqumat.scf import Calculation
from defuqumat.system import build_system
from defuqumat.system.kpoints import KPoints

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
#: Gamma and one point of no symmetry, weights summing to one, as in ``test_realtime.py``.
KSET = (np.array([[0.0, 0.0, 0.0], [0.25, 0.1, -0.05]]), np.array([0.5, 0.5]))
COMPONENTS = ((1, 1), (2, 2), (2, 0), (3, 3), (3, 1))


@pytest.fixture(autouse=True)
def _bounded_compilation():
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _dense_cell(pseudo_dir, case, ecut, computed, ecutrho=None):
    """``(calculation, v_scf, hamiltonians, basis, energies)`` on :data:`KSET` at ``ecut``.

    At the superposition of atomic charges, where the identity holds as at any
    frozen potential; ``basis`` is the lowest ``computed`` dense eigenstates of
    each sphere. ``ecutrho`` above ``4 ecut`` puts the density on a second,
    denser grid.
    """
    replacement = f"ecutwfc = {ecut}" + ("" if ecutrho is None else f", ecutrho = {ecutrho}")
    text = re.sub(r"ecutwfc\s*=\s*[0-9.dD+-]+", replacement,
                  (CASES / f"{case}.in").read_text())
    path = Path("/tmp") / f"defuqumat-hierarchy-{case}-{ecut}-{ecutrho}.in"
    path.write_text(text)
    system = build_system(read_pw_input(path))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    system = eqx.tree_at(lambda s: s.kpoints, system, KPoints(coords=KSET[0], weights=KSET[1]))
    calculation = Calculation(system, pseudos)
    v_scf = calculation.potential(calculation.starting_density()).v_scf
    terms = calculation.local_terms(v_scf)
    mask = np.asarray(calculation.basis.planewaves.mask)
    nk, npwx = mask.shape
    hamiltonians, basis = [], np.zeros((nk, computed, npwx), dtype=complex)
    energies = np.zeros((nk, computed))
    for ik in range(nk):
        h = dense_hamiltonians(calculation, terms, ik, np.array([1.0, 0.0, 0.0]), 4)
        values, vectors = np.linalg.eigh(h[0])
        basis[ik][:, np.flatnonzero(mask[ik])] = vectors[:, :computed].T
        energies[ik] = values[:computed]
        hamiltonians.append(h)
    return calculation, v_scf, hamiltonians, basis, energies


def test_the_iterative_hierarchy_is_the_dense_hierarchy(pseudo_dir):
    """All five components at 0.02, 0.05 and 0.11 Ha, ``eta = 0.01`` Ha, on AlAs at 4 Ry.

    Measured with twelve computed bands in the projector: **1e-13 to 8e-12**
    relative at a BiCGStab tolerance of 1e-10, in 18 to 19 iterations; 2e-11 to
    1.6e-10 with six bands (18 to 39 iterations), 1e-11 to 1.3e-9 at a tolerance
    of 1e-8 and 5e-13 to 3.5e-12 at 1e-12. The frequencies put ``3w`` below,
    across and above the gap of this cell, and the even orders are nonzero
    here (no inversion), so every component is compared and none is a residue.
    """
    calculation, v_scf, hamiltonians, basis, energies = _dense_cell(
        pseudo_dir, "alas-shg", 4.0, 12)
    nocc = int(round(calculation.nelec / 2))
    weights = np.asarray(calculation.system.kpoints.weights)
    volume = float(calculation.system.cell.volume)
    omegas, eta = (0.02, 0.05, 0.11), 0.01
    w = np.repeat(weights[:, None], nocc, axis=1)
    results = {}
    for computed in (12, 6):
        results[computed] = hierarchy_orders(
            calculation, basis[:, :computed], energies[:, :computed], w, v_scf,
            omegas=omegas, eta=eta, direction=(1.0, 0.0, 0.0), order=3, tolerance=1e-10)
    for iw, omega in enumerate(omegas):
        dense = {}
        for ik, h in enumerate(hamiltonians):
            e, u = dense_ground_states(h[0], nocc)
            for key, value in dense_orders(h, e, u, np.full(nocc, weights[ik]), 2 * omega,
                                           2 * eta, nmax=3).items():
                dense[key] = dense.get(key, 0.0) + value
        for key in COMPONENTS:
            reference = -dense[key] / (2.0 * volume)
            for computed, tolerance in ((12, 1e-10), (6, 1e-9)):
                value = results[computed]["components"][key][iw, 0]
                assert abs(value - reference) < tolerance * abs(reference), (
                    computed, omega, key, value, reference)
    assert results[12]["iterations"].max() < 40


def test_a_component_that_does_not_converge_is_refused(pseudo_dir):
    calculation, v_scf, _, basis, energies = _dense_cell(pseudo_dir, "alas-shg", 4.0, 12)
    nocc = int(round(calculation.nelec / 2))
    weights = np.repeat(np.asarray(calculation.system.kpoints.weights)[:, None], nocc, axis=1)
    with pytest.raises(HierarchyError, match="did not converge"):
        hierarchy_orders(calculation, basis, energies, weights, v_scf, omegas=(0.05,), eta=0.01,
                         direction=(1.0, 0.0, 0.0), order=1, max_iterations=2)


def test_the_spectrum_is_the_propagation_at_one_frequency(pseudo_dir, tmp_path):
    """The hierarchy's ``J_(n,m)`` against ``get_harmonic_orders`` on AlAs at 6 Ry, 2x2x2, [111].

    1.5 eV and ``eta = 0.3`` eV, ``eta_t = 12``, the field along [111], whose
    little group keeps six operations; the propagation's error is its ``dt^2``
    and its start transient, ``exp(-eta_t)`` times a resonance factor. Measured,
    relative to each component, at 400 and 800 steps a period:

    ======  ==========  ==========
    (n,m)   400         800
    ======  ==========  ==========
    (1,1)   3.5e-5      4.2e-6
    (2,2)   1.1e-4      1.6e-5
    (2,0)   9.6e-4      1.7e-5
    (3,3)   5.6e-5      4.1e-5
    (3,1)   3.5e-5      1.2e-5
    ======  ==========  ==========

    falling with the step except ``(3,3)``, which sits on the start transient;
    ``(2,0)`` is a fifth of ``(2,2)`` here, so each component is asserted
    against the largest of its order. The hierarchy took 13.6 s for this
    frequency, the fixed-density solve included, and the propagation 302 s at
    400 steps a period (four of this workstation's efficiency cores).
    """
    from defuqumat import Calculator

    text = (CASES / "alas-shg.in").read_text()
    text = re.sub(r"ecutwfc\s*=\s*[0-9.dD+-]+", "ecutwfc = 6.0", text)
    text = re.sub(r"^\s*no(sym|inv)\s*=.*\n", "", text, flags=re.M)
    path = tmp_path / "alas-shg-6.in"
    path.write_text(text)
    calculator = Calculator.from_file(path, pseudo_dir=pseudo_dir, announce=False)
    calculator.get_scf(conv_thr=1e-12)
    direction = tuple(np.ones(3) / math.sqrt(3.0))
    options = dict(broadening=0.3, direction=direction, grid=(2, 2, 2), conv_thr=1e-12)
    spectrum = calculator.get_nonlinear_spectrum([1.5], order=3, **options)
    orders = calculator.get_harmonic_orders(1.5, order=3, eta_t=12.0, steps_per_period=400,
                                            **options)
    references = {key: complex(orders.component(*key, axis=0)) for key in COMPONENTS}
    for key in COMPONENTS:
        value = spectrum.component(*key, axis=0)[0]
        scale = max(abs(v) for k, v in references.items() if k[0] == key[0])
        assert abs(value - references[key]) < 3e-4 * scale, (key, value, references[key])


@pytest.mark.parametrize("potential", ["hartree", "hxc"])
def test_the_self_consistent_first_order_is_the_dense_one(pseudo_dir, potential):
    """``J_(1,1)`` with ``dv_+ = K drho_+`` against ``dense_first_order``, AlAs at 4 Ry.

    Gamma and a general point, ``eta = 0.01`` Ha, at 0.02, 0.05 and 0.11 Ha.
    Measured: **9e-12 to 5e-11** against the dense self-consistent solve, in 9
    to 12 outer GMRES products, where the induced potential moves the current
    by 2.1 to 11.5 per cent with the Hartree kernel and 0.08 to 2.1 per cent
    with the exchange-correlation one; the same run's frozen current against
    the dense frozen one 2e-13 to 5e-13.
    """
    calculation, v_scf, _, basis, energies = _dense_cell(pseudo_dir, "alas-shg", 4.0, 12)
    terms = calculation.local_terms(v_scf)
    nocc = int(round(calculation.nelec / 2))
    weights = np.repeat(np.asarray(calculation.system.kpoints.weights)[:, None], nocc, axis=1)
    volume = float(calculation.system.cell.volume)
    direction = np.array([1.0, 0.0, 0.0])
    omegas, eta = (0.02, 0.05, 0.11), 0.01
    out = hierarchy_linear_self_consistent(calculation, basis, energies, weights, v_scf,
                                           omegas=omegas, eta=eta, direction=direction,
                                           potential=potential)
    for iw, omega in enumerate(omegas):
        reference, _, residual = dense_first_order(calculation, terms, direction, 2 * omega,
                                                   2 * eta, nocc, potential=potential)
        frozen, _, _ = dense_first_order(calculation, terms, direction, 2 * omega, 2 * eta, nocc)
        reference, frozen = -reference / (2.0 * volume), -frozen / (2.0 * volume)
        value = out["components"][(1, 1)][iw, 0]
        assert abs(value - reference) < 1e-9 * abs(reference), (omega, value, reference)
        assert abs(out["frozen"][(1, 1)][iw, 0] - frozen) < 1e-11 * abs(frozen)
        assert abs(reference - frozen) > 5e-4 * abs(frozen), "the update is load-bearing"


@pytest.mark.parametrize("potential", ["hartree", "hxc"])
def test_the_self_consistent_first_order_on_two_grids_is_the_dense_one(pseudo_dir, potential):
    """The same comparison on two FFT grids, AlAs at 4 Ry with ``ecutrho = 32``.

    The hierarchy carries ``dv_+`` and the cross density on the smooth grid and
    goes through the dense one for the kernel; ``dense_first_order`` builds the
    density in the dense box by Miller index and reads ``v(G - G')`` off the
    dense transform, so the two share the kernel and nothing of the
    interpolation. Measured (D22, one sample): **1.9e-11 to 5.1e-11** with the
    Hartree kernel and 9.5e-12 to 2.5e-11 with ``'hxc'``, in 9 to 12 outer
    products, where the update moves the current by 2.1 to 11.5 and 0.08 to
    2.1 per cent; the frozen current 2.6e-13 to 9.2e-13.
    """
    calculation, v_scf, _, basis, energies = _dense_cell(pseudo_dir, "alas-shg", 4.0, 12,
                                                         ecutrho=32.0)
    assert calculation.basis.smooth.grid != calculation.basis.dense.grid
    terms = calculation.local_terms(v_scf)
    nocc = int(round(calculation.nelec / 2))
    weights = np.repeat(np.asarray(calculation.system.kpoints.weights)[:, None], nocc, axis=1)
    volume = float(calculation.system.cell.volume)
    direction = np.array([1.0, 0.0, 0.0])
    omegas, eta = (0.02, 0.05, 0.11), 0.01
    out = hierarchy_linear_self_consistent(calculation, basis, energies, weights, v_scf,
                                           omegas=omegas, eta=eta, direction=direction,
                                           potential=potential)
    for iw, omega in enumerate(omegas):
        reference, _, _ = dense_first_order(calculation, terms, direction, 2 * omega,
                                            2 * eta, nocc, potential=potential)
        frozen, _, _ = dense_first_order(calculation, terms, direction, 2 * omega, 2 * eta, nocc)
        reference, frozen = -reference / (2.0 * volume), -frozen / (2.0 * volume)
        value = out["components"][(1, 1)][iw, 0]
        assert abs(value - reference) < 1e-9 * abs(reference), (omega, value, reference)
        assert abs(out["frozen"][(1, 1)][iw, 0] - frozen) < 1e-11 * abs(frozen)
        assert abs(reference - frozen) > 5e-4 * abs(frozen), "the update is load-bearing"


def test_the_hartree_first_order_on_two_grids_is_the_one_grid_one(pseudo_dir):
    """A norm-conserving density is band-limited at ``4 ecutwfc``, so with the Hartree
    kernel, diagonal in ``G``, the two-grid run is the one-grid run of the same states.

    The states, energies and potential are the two-grid cell's, the potential
    handed to the one-grid cell resampled onto its grid; the wavefunction
    spheres are checked to hold the same Miller indices in the same order
    before the states are shared. The exchange-correlation kernel has no such
    identity: ``f_xc drho`` is a pointwise product and is formed on a finer
    grid in the two-grid run, and the two differ there by 2.0e-6 to 1.5e-5.
    Measured (D22, one sample): **1.8e-14 to 3.5e-14**.
    """
    from defuqumat.basis.interpolate import to_smooth

    two, v_two, _, basis, energies = _dense_cell(pseudo_dir, "alas-shg", 4.0, 12, ecutrho=32.0)
    one, _, _, _, _ = _dense_cell(pseudo_dir, "alas-shg", 4.0, 12)
    assert tuple(one.basis.dense.grid) == tuple(two.basis.smooth.grid)
    assert tuple(one.basis.dense.grid) != tuple(two.basis.dense.grid)
    mask = np.asarray(one.basis.planewaves.mask)
    assert np.array_equal(mask, np.asarray(two.basis.planewaves.mask))
    miller = [np.asarray(c.basis.smooth.miller)[np.asarray(c.basis.planewaves.indices)][mask]
              for c in (one, two)]
    assert np.array_equal(*miller)
    v_one = to_smooth(v_two, two.basis.dense, two.basis.smooth)
    nocc = int(round(two.nelec / 2))
    weights = np.repeat(np.asarray(two.system.kpoints.weights)[:, None], nocc, axis=1)
    options = dict(omegas=(0.02, 0.05, 0.11), eta=0.01, direction=(1.0, 0.0, 0.0),
                   potential="hartree")
    a = hierarchy_linear_self_consistent(two, basis, energies, weights, v_two, **options)
    b = hierarchy_linear_self_consistent(one, basis, energies, weights, v_one, **options)
    value, reference = a["components"][(1, 1)][:, 0], b["components"][(1, 1)][:, 0]
    frozen = a["frozen"][(1, 1)][:, 0]
    assert np.all(np.abs(value - reference) < 1e-9 * np.abs(reference)), (value, reference)
    assert np.all(np.abs(reference - frozen) > 5e-4 * np.abs(frozen)), "the update is load-bearing"


def test_the_self_consistent_static_shift_is_the_sternheimer_shift(pseudo_dir):
    """The local fields' shift of ``eps`` at 0.1 eV against the static Sternheimer one.

    Two-atom silicon at 12 Ry on its 4x4x4 mesh's [100] wedge (eight operations,
    so the induced density is completed by the field's little group), 0.1 eV
    and ``eta = 0.1`` eV. The shift ``eps - eps_frozen`` of one run carries
    none of the velocity gauge's band-curvature term, which is the size of
    ``eps`` itself on this mesh (``-3760i`` at 0.1 eV), and the static
    response with every band is ``get_dielectric_tensor`` with
    ``screening = 'none'``, ``'hartree'`` and ``'full'``: 23.7876, 21.6676 and
    22.9042. Measured: the hierarchy's shifts -2.119995 - 0.0054i and
    -0.883446 - 0.0018i against -2.120012 and -0.883444, **8e-6 and 2e-6**,
    what ``|z|^2`` over the gap squared leaves at that frequency and less.
    """
    from defuqumat import Calculator

    calculator = Calculator.from_file(CASES / "si2-symmetric.in", pseudo_dir=pseudo_dir,
                                      announce=False)
    calculator.get_scf(conv_thr=1e-12)
    static = {screening: float(calculator.get_dielectric_tensor(
        screening=screening, born_charges=False, threshold=1e-14).epsilon[0, 0])
        for screening in ("none", "hartree", "full")}
    for potential, screening in (("hartree", "hartree"), ("hxc", "full")):
        spectrum = calculator.get_nonlinear_spectrum([0.1], broadening=0.1, order=1,
                                                     potential=potential, conv_thr=1e-12)
        shift = complex(spectrum.epsilon(0)[0] - spectrum.epsilon(0, frozen=True)[0])
        reference = static[screening] - static["none"]
        assert abs(shift.real - reference) < 5e-5 * abs(reference), (potential, shift, reference)
        assert abs(shift.imag) < 1e-2 * abs(reference)


@pytest.mark.parametrize("potential", ["hartree", "hxc"])
def test_the_self_consistent_orders_are_the_self_consistent_propagation(pseudo_dir, potential):
    """Orders one to three with the potential updated against the propagation's, AlAs at 4 Ry.

    Gamma and a general point at the superposition of atomic charges, ``w = 0.05``
    and ``eta = 0.01`` Hartree, the propagation over ``eta T = 20`` with the
    potential updated (``propagate_orders_self_consistent``, whose nested
    ``jvp`` carries ``K``, ``K2`` and ``K3`` of the potential by itself) against
    :func:`~defuqumat.realtime.hierarchy.hierarchy_self_consistent` on the same
    states. The two share the table of the projectors and the potential's
    functional and nothing of the solve. Measured (D22, one sample each), relative
    to the largest component of each order, at 400 and 800 steps a period:

    ======  =================  =================  ========  ===========
    (n,m)   hartree 400 / 800  hxc 400 / 800      frozen    linear K
    ======  =================  =================  ========  ===========
    (1,1)   1.4e-5 / 8.3e-7    1.6e-5 / 3.7e-6    3.5e-3    3.7e-6
    (2,2)   4.0e-6 / 1.3e-6    2.0e-5 / 3.5e-6    1.7e-2    5.2e-3
    (2,0)   1.7e-5 / 3.3e-6    3.0e-5 / 5.3e-6    1.9e-2    5.6e-3
    (3,3)   3.0e-5 / 9.8e-6    3.3e-5 / 5.2e-6    5.0e-2    1.3e-2
    (3,1)   2.6e-5 / 3.7e-6    4.7e-5 / 7.6e-6    4.2e-2    2.3e-2
    ======  =================  =================  ========  ===========

    falling with the step by 3 to 16, the propagation's ``dt^2`` over its start
    transient; "frozen" is the same run at ``dv = 0`` against the ``'hxc'``
    propagation at 800 steps (9.98e-2 to 1.8e-1 for ``'hartree'``) and "linear
    K" the ``'hxc'`` hierarchy without ``K2`` and ``K3``, so the nonlinear
    kernel is seen by a factor of 700 to 3000 and the update by 3000 and more.
    The hierarchy took 48 s for the three runs, the propagation 151 to 235 s at
    800 steps.
    """
    import jax.numpy as jnp

    from defuqumat.realtime.hierarchy import hierarchy_self_consistent
    from defuqumat.realtime.orders import propagate_orders
    from defuqumat.realtime.pulse import Adiabatic

    calculation, v_scf, _, basis, energies = _dense_cell(pseudo_dir, "alas-shg", 4.0, 12)
    nocc = int(round(calculation.nelec / 2))
    weights = np.repeat(np.asarray(calculation.system.kpoints.weights)[:, None], nocc, axis=1)
    omega, eta = 0.05, 0.01
    out = hierarchy_self_consistent(calculation, basis, energies, weights, v_scf,
                                    omegas=(omega,), eta=eta, direction=(1.0, 0.0, 0.0),
                                    order=3, potential=potential)
    period = 2 * math.pi / omega
    length = math.ceil(20.0 / eta / period) * period
    shape = Adiabatic(amplitude=1.0, omega=omega, eta=eta, eta_t=length * eta)
    result = propagate_orders(calculation, jnp.asarray(basis[:, :nocc]), weights, v_scf, shape,
                              dt=period / 800, order=3, start=-length, duration=length,
                              k_batch=None, potential=potential)
    references = {key: complex(result.component(*key, axis=0)) for key in COMPONENTS}
    for key in COMPONENTS:
        value = out["components"][key][0, 0]
        frozen = out["frozen"][key][0, 0]
        scale = max(abs(v) for k, v in references.items() if k[0] == key[0])
        assert abs(value - references[key]) < 2e-5 * scale, (key, value, references[key])
        if key[0] > 1:
            assert abs(frozen - references[key]) > 1e-3 * scale, "the update is load-bearing"
    if potential == "hxc":
        linear = hierarchy_self_consistent(calculation, basis, energies, weights, v_scf,
                                           omegas=(omega,), eta=eta, direction=(1.0, 0.0, 0.0),
                                           order=3, potential=potential, nonlinear_kernel=False)
        for key in COMPONENTS[1:]:
            scale = max(abs(v) for k, v in references.items() if k[0] == key[0])
            value = linear["components"][key][0, 0]
            assert abs(value - references[key]) > 1e-3 * scale, "the nonlinear kernel is seen"
