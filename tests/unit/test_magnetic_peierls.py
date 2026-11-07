"""Peierls substitution on the nearest-neighbour honeycomb model: the guards and the exact facts.

The model is the review's (``plans/orbital-field-review/peierls_torus.py``): ``t = -2.7``
eV, ``a = 2.46`` A, B at ``(a1 + a2)/3``. What is checked, at ``q <= 401`` so that every
test is a dense diagonalisation of at most 802 rows:

- the magnetic Bloch Hamiltonian is Hermitian to round-off at a ``k`` with both wraps
  active, and the guard raises once the Bloch-sum factor is dropped (4.7 eV, the review);
- the Landau levels are flat in ``k`` to round-off, ``2 p`` states per level per ``k``;
- the hexagon's plaquette product is ``exp(-i 2 pi p/q)``, and a flipped sign raises;
- a wide ribbon from the same bonds has the torus levels in its bulk;
- the Hofstadter spectrum is symmetric under ``E -> -E`` (chiral) and under
  ``p/q -> (q - p)/q`` over a mesh closed under inversion;
- the banded :class:`PeierlsOperator` is the CSR matrix;
- P162's bond expansion (each Wigner-Seitz image its own bond) is the zero-field
  interpolation, and the workflow's Dirac comparison and zero-mode traces.
"""

import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.magnetic.conventions import MagneticCell
from defuqumat.magnetic.landau import dirac_levels
from defuqumat.magnetic.peierls import (
    Bonds, PeierlsTorus, bonds_from_wannier, cluster_levels, hermiticity, honeycomb_nn_bonds,
    levels_near, plaquette_check,
)
from defuqumat.units import RY_TO_EV, TPI
from defuqumat.wannier.interpolation import interpolate
from defuqumat.workflows.landau import run_landau_levels

pytestmark = pytest.mark.unit

KAPPAS = ((0.0, 0.0), (0.37, 0.0), (0.0, 0.29), (0.21, 0.63))
BONDS = honeycomb_nn_bonds()


def _torus(p, q, **kw):
    return PeierlsTorus(MagneticCell.build(BONDS.lattice, p, q), BONDS, **kw)


def _hexagon():
    """The six sites around the hexagon centred at 2 (a1 + a2)/3, counterclockwise."""
    lattice = BONDS.lattice
    sites = [(0, (1, 0, 0)), (0, (0, 1, 0)), (0, (1, 1, 0)), (1, (0, 0, 0)), (1, (1, 0, 0)),
             (1, (0, 1, 0))]
    centre = 2 * (lattice[0] + lattice[1]) / 3
    pos = [BONDS.centres[o] + np.array(R) @ lattice - centre for o, R in sites]
    order = np.argsort([math.atan2(x[1], x[0]) for x in pos])
    return [sites[i] for i in order]


def test_torus_is_hermitian_and_flat_and_the_guard_fires_without_the_bloch_factor():
    torus = _torus(1, 400)
    spectra = []
    for kappa in KAPPAS:
        h = torus.matrix(kappa)
        assert hermiticity(h) * RY_TO_EV < 1e-13
        spectra.append(levels_near(h, 0.0, 22, vectors=False))
    spread = max(np.max(np.abs(s - spectra[0])) for s in spectra[1:]) * RY_TO_EV
    assert spread < 1e-12, spread
    broken = _torus(1, 400, bloch_factor=False)
    assert hermiticity(broken.matrix((0.37, 0.29), check=False)) * RY_TO_EV > 1.0
    with pytest.raises(ValueError, match="not Hermitian"):
        broken.matrix((0.37, 0.29))


def test_hexagon_plaquette_and_the_flipped_sign():
    cell = MagneticCell.build(BONDS.lattice, 1, 400)
    loop = _hexagon()
    product = plaquette_check(cell, BONDS, loop)
    assert abs(product - np.exp(-1j * TPI / 400)) < 1e-12
    with pytest.raises(ValueError, match="plaquette"):
        plaquette_check(cell, BONDS, loop, hop_phase=lambda a, b: -cell.peierls_phase(a, b))
    with pytest.raises(ValueError, match="not a bond"):
        plaquette_check(cell, BONDS, [(0, (0, 0, 0)), (0, (1, 0, 0)), (1, (0, 0, 0))])


def test_two_valleys_times_p_states_per_level():
    """p = 2, q = 401 (2/400 is not reduced): the zero level and the positive ones are 4-fold."""
    torus = _torus(2, 401)
    for kappa in KAPPAS[:2]:
        w = levels_near(torus.matrix(kappa), 0.0, 44, vectors=False)
        levels, counts = cluster_levels(w, 1e-6 / RY_TO_EV)
        zero = int(np.argmin(np.abs(levels)))
        assert counts[zero] == 4
        assert np.all(counts[zero + 1:-1] == 4), counts


def test_ribbon_bulk_levels_are_the_torus_levels():
    q, kappa2 = 400, 0.29
    torus = _torus(1, q)
    wt = levels_near(torus.matrix((0.0, kappa2)), 0.0, 40, vectors=False)
    first = np.unique(np.round(wt[wt > 1e-8] * RY_TO_EV, 9))[:6]
    ribbon = torus.ribbon(q)
    wr, vr = np.linalg.eigh(ribbon.matrix((0.0, kappa2)).toarray())
    nw = BONDS.nw
    edge = np.zeros(ribbon.dim, bool)
    edge[: 20 * nw] = True
    edge[-20 * nw:] = True
    bulk = wr[np.sum(np.abs(vr[edge]) ** 2, axis=0) < 1e-6] * RY_TO_EV
    assert len(first) == 6
    for e in first:
        assert np.min(np.abs(bulk - e)) < 1e-6, e


def _dense(torus, kappa):
    return np.linalg.eigvalsh(torus.matrix(kappa).toarray())


def test_hofstadter_identities():
    """E -> -E at every k (sublattice symmetry) and p/q against (q - p)/q over a 6 x 6 mesh.

    ``(q - p)/q`` is ``-p/q`` up to one flux quantum per hexagon, a gauge transformation
    that is not periodic over the magnetic cell, composed with time reversal: it maps
    ``kappa`` to another point, so the two spectra agree as sets over a mesh closed under
    that map, not at one ``kappa``. Measured (``plans/orbital-field/foundation.md``): 1.4 eV
    apart at one ``kappa``, 0.63 and 0.78 eV over the 2 x 2 and 3 x 3 meshes, 1.5e-14 eV
    over 6 x 6 and 12 x 12.
    """
    mesh = [(i / 6, j / 6) for i in range(6) for j in range(6)]
    for q in (5, 7, 12):
        for p in range(1, q):
            if math.gcd(p, q) != 1:
                continue
            a = np.sort(np.concatenate([_dense(_torus(p, q), k) for k in mesh]))
            b = np.sort(np.concatenate([_dense(_torus(q - p, q), k) for k in mesh]))
            np.testing.assert_allclose(a, -a[::-1], atol=1e-12)
            np.testing.assert_allclose(a, b, atol=1e-12)


def test_banded_operator_is_the_matrix():
    for torus in (_torus(3, 10), _torus(1, 10).ribbon(7)):
        kappa = (0.37, 0.29)
        h = torus.matrix(kappa).toarray()
        op = torus.operator(kappa)
        rng = np.random.default_rng(0)
        v = rng.normal(size=(3, torus.dim)) + 1j * rng.normal(size=(3, torus.dim))
        np.testing.assert_allclose(np.asarray(op.apply(jnp.asarray(v))), v @ h.T, atol=1e-12)


def test_wannier_bonds_reproduce_the_zero_field_interpolation():
    """Each Wigner-Seitz image a bond of weight hr/(ndegen count): H(k) is the interpolation's."""
    rng = np.random.default_rng(1)
    lattice = BONDS.lattice
    rvec = np.array([(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)])
    nw = 2
    hr = rng.normal(size=(5, nw, nw)) + 1j * rng.normal(size=(5, nw, nw))
    for r, rm in ((1, 2), (3, 4)):
        hr[rm] = hr[r].conj().T
    hr[0] = hr[0] + hr[0].conj().T
    ndegen = np.array([1, 2, 2, 1, 1])
    shifted = np.zeros((5, nw, nw, 8, 3), np.int64)
    count = np.ones((5, nw, nw), np.int64)
    for ir in range(5):
        shifted[ir, :, :, 0] = rvec[ir]
    shifted[1, 0, 1, 1] = rvec[1] + (0, 1, 0)          # a tied image for one pair
    shifted[2, 1, 0, 1] = rvec[2] - (0, 1, 0)          # and its Hermitian partner
    count[1, 0, 1] = count[2, 1, 0] = 2
    w = SimpleNamespace(rvectors=rvec, degeneracy=ndegen, hoppings=hr, centres=BONDS.centres,
                        lattice=lattice, ws_shifts=(shifted, count), spin=None)
    bonds = bonds_from_wannier(w)
    assert bonds.nb == 5 * nw * nw + 2
    for kappa in ((0.1, 0.3, 0.0), (0.37, -0.2, 0.0)):
        ref, _ = interpolate(np.array([kappa]), rvec, ndegen, hr, shifted, count)
        np.testing.assert_allclose(np.linalg.eigvalsh(bonds.zero_field(kappa)), ref[0], atol=1e-13)
    torus = PeierlsTorus(MagneticCell.build(lattice, 1, 9), bonds)
    assert hermiticity(torus.matrix((0.37, 0.29))) < 1e-13


def _model():
    """The NN model as a stand-in Wannier set: bands from the bonds, edge at zero."""
    def bands(kappa):
        return np.array([np.linalg.eigvalsh(BONDS.zero_field(k)) for k in np.atleast_2d(kappa)])

    return SimpleNamespace(bands=bands, edge=0.0, spin=None)


def test_workflow_dirac_comparison_and_zero_modes():
    """q = 400 (197 T): E_1 within 1 per cent of Dirac at the NN velocity, two zero modes
    with tr P_A = 1 over the pair, and a Zeeman term refused."""
    out = run_landau_levels(_model(), bonds=BONDS, flux=(1, 400), kappa=((0.11, 0.23),),
                            nlevels=30, dirac=True)
    d = out.dirac
    v_nn = 3 * 2.7 * 1.602176634e-19 * (2.46e-10 / np.sqrt(3)) / (2 * 6.62607015e-34 / TPI)
    assert d["v_F"] == pytest.approx(v_nn, rel=1e-3)
    assert d["E_D"] == pytest.approx(0.0, abs=1e-12)
    entry = d["per_k"][0]
    assert entry["zero_multiplicity"] == 2
    assert entry["zero_site_traces"][0] == pytest.approx(1.0, abs=1e-6)
    assert entry["ratios"][1] == pytest.approx(1.0, abs=1e-2)
    assert entry["ratios"][-1] == pytest.approx(1.0, abs=1e-2)
    assert dirac_levels(out.field_tesla, d["v_F"], 1)[2] == pytest.approx(d["E1_dirac_ev"])
    assert out.relative[0] < 1e-13
    # the shift-invert path (dim 3950): ARPACK's zero-mode pair is not orthonormal by itself
    # (tr P_A read 0.65 there before the Rayleigh-Ritz step of levels_near)
    big = run_landau_levels(_model(), bonds=BONDS, flux=(1, 1975), kappa=((0.11, 0.23),),
                            nlevels=24, dirac=True)
    entry = big.dirac["per_k"][0]
    assert big.dim > 1600
    assert entry["zero_multiplicity"] == 2
    assert entry["zero_site_traces"][0] == pytest.approx(1.0, abs=1e-6)
    assert entry["ratios"][1] == pytest.approx(1.0 - 4.6e-4, abs=1e-4)
    with pytest.raises(NotImplementedError, match="Zeeman"):
        run_landau_levels(_model(), bonds=BONDS, flux=(1, 400), zeeman=True)
