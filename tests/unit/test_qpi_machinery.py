"""The T-matrix QPI machinery, on synthetic bands rather than converged cells.

``PLAN.md`` P157. Every identity here holds for *any* set of orthonormal
Bloch states and *any* eigenvalues, because the model is defined inside the
band window: the impurity is ``V |phi><phi|`` projected on the window and the
Green's function is the window's. So random bands on a small sphere test the
whole algebra with no SCF:

* the closed-form T-matrix with the Bloch factorisation over the supercell is
  the **dense inverse** of the supercell Hamiltonian, sampled at every tip point
  directly, for both readers and for either exit leg of ``landauer`` (the
  literal resolvent and the on-shell amplitude);
* the ``landauer`` reader is the **real-space integral** of ``|G(r, r')|^2``
  over the supercell's exit plane, evaluated on a grid with no Gram matrix,
  which is the only check on the index order in ``A S A^dagger``;
* the Born approximation is the ``V -> 0`` limit, linearly, and is measurably
  wrong at a strong ``V``.
"""

import numpy as np
import pytest

from defuqumat.basis.sample import sample_wavefunctions
from defuqumat.stm.qpi import (
    QPIStates,
    dense_reference,
    fourier_map,
    gaussian_transform,
    impurity_maps,
    impurity_overlaps,
    lattice_translations,
)
from defuqumat.transport.green import amplitude_weights
from defuqumat.transport.substrate import exit_overlap


class _Cell:
    def __init__(self, at):
        self.at = np.asarray(at, dtype=float)
        self.volume = float(abs(np.linalg.det(self.at)))
        self.bg = 2.0 * np.pi * np.linalg.inv(self.at).T


#: An oblique surface cell, so that a crystal/cartesian mix-up in the lattice
#: phases or the Gaussian's transform cannot cancel.
CELL = _Cell([[4.0, 0.0, 0.0], [-1.3, 3.7, 0.0], [0.0, 0.0, 9.0]])
DIVISIONS = (3, 2)
SHAPE = (3, 2)
TIP_HEIGHT = 0.8
EXIT_HEIGHT = 0.2
IMPURITY = np.array([0.31, 0.12, 0.5])
SIGMA = 0.9
BROADENING = 0.08


def _sphere(n=2):
    grid = np.arange(-n, n + 1)
    h = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1)
    return h.reshape((-1, 3))


def _bands(nbnd=4, seed=3):
    """Orthonormal random bands at every k of a shifted 3x2x1 grid."""
    rng = np.random.default_rng(seed)
    miller = _sphere()
    npw = miller.shape[0]
    kcrystal = np.array([[(i + 0.5) / DIVISIONS[0], j / DIVISIONS[1], 0.0]
                         for i in range(DIVISIONS[0])
                         for j in range(DIVISIONS[1])])
    nk = kcrystal.shape[0]
    coefficients = []
    for _ in range(nk):
        raw = rng.normal(size=(npw, nbnd)) + 1.0j * rng.normal(size=(npw, nbnd))
        q, _ = np.linalg.qr(raw)
        coefficients.append(q.T)
    eigenvalues = np.sort(rng.uniform(-0.6, 0.6, size=(nk, nbnd)), axis=1)
    return (np.array(coefficients), np.broadcast_to(miller, (nk, npw, 3)),
            kcrystal, np.ones((nk, npw), dtype=bool), eigenvalues)


def _tip_points():
    m1, m2 = SHAPE
    return np.array([[j / m1, l / m2, TIP_HEIGHT]
                     for j in range(m1) for l in range(m2)])


def _states(coefficients, miller, kcrystal, mask, eigenvalues):
    tip = _tip_points()
    nk = kcrystal.shape[0]
    sampled = np.stack([sample_wavefunctions(coefficients[ik], miller[ik],
                                             kcrystal[ik], tip, CELL.volume,
                                             mask=mask[ik]) for ik in range(nk)])
    overlaps = np.stack([impurity_overlaps(coefficients[ik], miller[ik],
                                           kcrystal[ik], IMPURITY, CELL.bg,
                                           CELL.volume, SIGMA, mask=mask[ik])
                         for ik in range(nk)])
    gram = np.stack([exit_overlap(coefficients[ik], miller[ik], EXIT_HEIGHT, 2,
                                  CELL, mask=mask[ik]) for ik in range(nk)])
    return QPIStates(eigenvalues=eigenvalues, tip=sampled, overlaps=overlaps,
                     kcrystal=kcrystal, exit=gram)


def _supercell_points(translations):
    tip = _tip_points()
    shifts = np.zeros((translations.shape[0], 3))
    shifts[:, :2] = translations
    return (shifts[:, None, :] + tip[None, :, :]).reshape(-1, 3)


@pytest.fixture(scope="module")
def case():
    coefficients, miller, kcrystal, mask, eigenvalues = _bands()
    return (coefficients, miller, kcrystal, mask, eigenvalues,
            _states(coefficients, miller, kcrystal, mask, eigenvalues))


@pytest.mark.parametrize("on_shell", [False, True])
@pytest.mark.parametrize("potential", [0.4, -2.5])
def test_the_t_matrix_is_the_dense_inverse_of_the_supercell(case, potential,
                                                             on_shell):
    coefficients, miller, kcrystal, mask, eigenvalues, states = case
    translations = lattice_translations(DIVISIONS)
    energy = 0.13
    weights = (amplitude_weights(eigenvalues, energy, BROADENING)
               if on_shell else None)
    maps, _, _ = impurity_maps(states, energy, BROADENING, potential,
                               translations, exit_weights=weights)
    reference = dense_reference(
        eigenvalues, coefficients, miller, kcrystal, mask, CELL, IMPURITY,
        SIGMA, _supercell_points(translations), energy, BROADENING, potential,
        exit=states.exit, exit_weights=weights)
    for reader in ("ldos", "landauer"):
        got = maps[reader].reshape(-1)
        want = reference[reader]
        scale = np.abs(want).max()
        assert scale > 0.0
        assert np.abs(got - want).max() < 1e-10 * scale, reader


def test_the_landauer_reader_is_the_plane_integral_of_the_green_function(case):
    """``int_exit |G(r, r')|^2`` on a grid, with no Gram matrix anywhere."""
    coefficients, miller, kcrystal, mask, eigenvalues, states = case
    translations = lattice_translations(DIVISIONS)
    energy, potential = -0.21, 0.9
    maps, _, _ = impurity_maps(states, energy, BROADENING, potential,
                               translations, readers=("landauer",))
    nk, nbnd = eigenvalues.shape
    # The dense dressed and clean resolvents, as in ``dense_reference``.
    p = np.stack([impurity_overlaps(coefficients[ik], miller[ik], kcrystal[ik],
                                    IMPURITY, CELL.bg, CELL.volume, SIGMA)
                  for ik in range(nk)]).reshape(-1) / np.sqrt(nk)
    z = energy + 1.0j * BROADENING
    h = np.diag(eigenvalues.reshape(-1)) + potential * np.outer(p, p.conj())
    dressed = np.linalg.inv(z * np.eye(nk * nbnd) - h)
    clean = np.diag(1.0 / (z - eigenvalues.reshape(-1)))

    def sample(points):
        return np.stack([sample_wavefunctions(coefficients[ik], miller[ik],
                                              kcrystal[ik], points, CELL.volume)
                         for ik in range(nk)]).reshape(nk * nbnd, -1)

    tips = sample(_supercell_points(translations))
    # A band-limited integrand on a periodic grid is integrated exactly once
    # the grid resolves twice the largest Miller index plus the k spread.
    m = 12
    n1, n2 = DIVISIONS
    grid = np.array([[a / m, b / m, EXIT_HEIGHT]
                     for a in range(n1 * m) for b in range(n2 * m)])
    exits = sample(grid)
    area = abs(np.linalg.det(CELL.at[:2, :2])) / m ** 2
    values = []
    for green in (dressed, clean):
        g = tips.T @ green @ exits.conj() / nk            # G(r, r')
        values.append(np.sum(np.abs(g) ** 2, axis=1) * area)
    want = values[0] - values[1]
    got = maps["landauer"].reshape(-1)
    assert np.abs(got - want).max() < 1e-9 * np.abs(want).max()


def test_the_born_approximation_is_the_weak_scatterer_limit(case):
    *_, states = case
    translations = lattice_translations(DIVISIONS)

    def ratio(potential, born):
        maps, _, _ = impurity_maps(states, 0.05, BROADENING, potential,
                                   translations, born=born)
        return np.concatenate([maps[r].ravel() for r in ("ldos",)]) / potential

    born = ratio(1.0, True)
    errors = [np.abs(ratio(v, False) - born).max() for v in (1e-2, 1e-3)]
    # first order in V: the error falls tenfold with V
    assert 8.0 < errors[0] / errors[1] < 12.0
    assert np.abs(ratio(3.0, False) - born).max() > 0.1 * np.abs(born).max()


def test_the_lattice_sum_is_blind_to_where_the_tip_grid_starts(case):
    """A translation by a lattice vector moves the map, not its values."""
    *_, states = case
    maps, _, _ = impurity_maps(states, 0.1, BROADENING, 1.0,
                               np.array([[1, 1], [-2, 1]]))
    again, _, _ = impurity_maps(states, 0.1, BROADENING, 1.0,
                                np.array([[1 - 3, 1], [-2 + 3, 1 - 2]]))
    assert np.allclose(maps["ldos"], again["ldos"], atol=1e-14)


def test_the_gaussian_is_normalised_in_reciprocal_space():
    sigma = 0.7
    k = np.linspace(-12.0, 12.0, 121)
    grid = np.stack(np.meshgrid(k, k, k, indexing="ij"), axis=-1)
    dk = (k[1] - k[0]) ** 3
    norm = np.sum(gaussian_transform(grid, sigma) ** 2) * dk / (2 * np.pi) ** 3
    assert abs(norm - 1.0) < 1e-10


def test_a_zero_width_is_refused():
    with pytest.raises(ValueError, match="width"):
        gaussian_transform(np.zeros((1, 3)), 0.0)


def test_the_fourier_map_tiles_the_supercell_and_finds_a_wave():
    n, m = (4, 3), (5, 2)
    translations = lattice_translations(n)
    u = np.array([[j / m[0], l / m[1]] for j in range(m[0]) for l in range(m[1])])
    x = translations[:, None, :] + u[None, :, :]            # (nR, np, 2)
    # a standing wave on a harmonic of the supercell, q = (3/4, -1/3) in
    # crystal units, so q N = (3, -1)
    q = np.array([3 / 4, -1 / 3])
    values = np.cos(2 * np.pi * (x @ q)) + 0.25
    grid, transform = fourier_map(values, n, m)
    assert grid.shape == (n[0] * m[0], n[1] * m[1])
    centre = (grid.shape[0] // 2, grid.shape[1] // 2)
    assert abs(transform[centre] - 0.25) < 1e-12
    p = (centre[0] + 3, centre[1] - 1)
    assert abs(transform[p] - 0.5) < 1e-12
    assert abs(np.abs(transform).sum() - 1.25) < 1e-10


def test_an_unknown_reader_is_refused(case):
    *_, states = case
    with pytest.raises(ValueError, match="reader"):
        impurity_maps(states, 0.0, BROADENING, 1.0,
                      lattice_translations(DIVISIONS), readers=("didv",))
