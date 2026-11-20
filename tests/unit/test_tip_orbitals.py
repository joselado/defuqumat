"""The orbital of the tip in the vertical-transport map, on synthetic states.

``PLAN.md`` P179, ``plans/tip-orbitals.md``. What needs an SCF (the sign of a
hybrid on a real vacuum tail, kappa against a fitted decay, the wedge, the
graphene contrast) is in ``tests/regression/test_tip_orbitals_slow.py``. What is
here is everything that holds for any set of states, or for the exact states of
a small model sheet:

* the multiplier ``i (k+G)`` is the derivative, through the plain and the folded
  sampler, on a hexagonal and a sheared cell (V2);
* the s tip through the orbital path is the point tip (V1);
* the map of a channel integrated over the tip plane is the momentum trace with
  the tip Gram matrix of the multiplied coefficients (V3);
* in the volume limit the three p channels are the kinetic energy density of
  :func:`~defuqumat.scf.density.band_kinetic_density` over ``kappa^2`` (V4);
* the frame turns with the states, and ``pxy`` has the symmetry of the lattice
  (V5);
* a chiral tip couples to one winding of the Dirac pair, which pins the
  conjugate in ``C^*`` (V7);
* the refusals.
"""

import types

import numpy as np
import pytest

from defuqumat.basis.gvectors import _fft_index
from defuqumat.basis.sample import sample_wavefunctions
from defuqumat.scf.density import band_kinetic_density
from defuqumat.transport.green import TransportGeometry, amplitude_weights
from defuqumat.transport.momentum import momentum_weights
from defuqumat.transport.substrate import (
    exit_overlap,
    fold_onto_plane,
    folded_gram,
    surface_area,
)
from defuqumat.transport.tip import (
    ORBITALS,
    TipAssembly,
    TipOrbital,
    decay_constants_at,
    multiplier,
    rotation_representation,
    tip_frame,
    tip_side,
    tip_vectors,
    vacuum_level,
)
from defuqumat.transport.tip import LAPLACIAN, continued_components, continued_wavevectors
from defuqumat.workflows.transport import (
    _assemble,
    _continued_groups,
    _feet,
    _sample_sphere,
    _tip_assembly,
    _tip_sphere,
    _turned_tips,
)


class _Cell:
    """The three things the transport asks a cell for."""

    def __init__(self, at):
        self.at = np.asarray(at, dtype=float)
        self.volume = float(abs(np.linalg.det(self.at)))
        self.bg = 2.0 * np.pi * np.linalg.inv(self.at).T


HEXAGONAL = _Cell([[4.6, 0.0, 0.0], [-2.3, 3.983716857408418, 0.0],
                   [0.0, 0.0, 14.0]])
SHEARED = _Cell([[4.0, 0.3, 0.0], [-1.2, 3.7, 0.0], [0.9, -0.6, 13.0]])
SQUARE = _Cell(np.diag([4.0, 4.0, 13.0]))


def _ball(cell, gmax, kcrystal=(0.0, 0.0, 0.0), reach=4):
    """Miller indices with ``|k + G| < gmax``: closed under the point group."""
    grid = np.arange(-reach, reach + 1)
    h = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
    q = (h + np.asarray(kcrystal)) @ cell.bg
    return h[np.linalg.norm(q, axis=1) < gmax]


def _orthonormal(npw, nbnd, seed):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(npw, nbnd)) + 1.0j * rng.normal(size=(npw, nbnd))
    q, _ = np.linalg.qr(raw)
    return q.T


def _geometry(cell, kpoints, nbnd=5, npol=1, gmax=3.2, seed=0, states=None):
    """A synthetic transport geometry: every k-point has the same ball."""
    kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
    miller = _ball(cell, gmax)
    npw = miller.shape[0]
    nk = kpoints.shape[0]
    if states is None:
        states = np.stack([_orthonormal(npol * npw, nbnd, seed + ik)
                           for ik in range(nk)])
    geometry = TransportGeometry(
        miller=np.repeat(miller[None], nk, axis=0),
        mask=np.ones((nk, npw), dtype=bool),
        kcrystal=kpoints, kweights=np.full(nk, 2.0 / nk), cell=cell, npol=npol)
    return geometry, states[None]


def _map(geometry, wavefunctions, eigenvalues, points, tip=None, **options):
    settings = dict(exit_height=0.2, exit_axis=2, energies=np.array([0.0]),
                    broadening=0.05, spin=None, polarization=1.0,
                    tip_spin=None, tip_polarization=1.0, incoherent=True,
                    exit_region="plane", method="spectral",
                    smearing="gaussian", tip=tip)
    settings.update(options)
    values, extras = _assemble(geometry, wavefunctions, eigenvalues, points,
                               **settings)
    return values, extras


def _tip(spec, cell, kappa=(0.7,), frame=None, **extra):
    orbital = TipOrbital.resolve(spec)
    basis = orbital.basis()
    kappa = np.asarray(kappa, dtype=float)
    return TipAssembly(orbital=orbital, basis=basis,
                       coefficients=orbital.coefficients(kappa, basis),
                       frame=tip_frame(cell, 2, 1) if frame is None else frame,
                       kappa=kappa, **extra)


def _plane_points(n, height):
    return np.array([[i / n, j / n, height] for i in range(n)
                     for j in range(n)], dtype=float)


# --------------------------------------------------------------------------
# the orbitals
# --------------------------------------------------------------------------


def test_the_aliases_and_the_channels_are_what_the_user_decided():
    """``"pxy"`` is two incoherent channels (the user, 2026-11-15)."""
    assert TipOrbital.resolve("pxy").channels == ((("px", 1.0),), (("py", 1.0),))
    assert TipOrbital.resolve(None).is_point
    assert TipOrbital.resolve("s").is_point
    assert not TipOrbital.resolve({"s": 2.0}).is_point
    hybrid = TipOrbital.resolve([{"s": 1, "pz": -0.5}, "pxy"])
    assert len(hybrid.channels) == 3
    assert hybrid.names == ("s", "py", "pz", "px")
    with pytest.raises(ValueError, match="unknown tip orbital"):
        TipOrbital.resolve("p_z")
    with pytest.raises(ValueError, match="every weight"):
        TipOrbital.resolve({"s": 0.0})


def test_the_coefficients_carry_the_conjugate_and_kappa_to_the_minus_l():
    orbital = TipOrbital.resolve({"s": 2.0, "px": 1.0j, "dz2": 3.0})
    basis = orbital.basis()
    values = orbital.coefficients([0.5, 2.0], basis)
    row = dict(zip(basis, values[0, 0]))
    assert row["s"] == pytest.approx(2.0)
    assert row["px"] == pytest.approx(-1.0j / 0.5)
    assert row["dz2"] == pytest.approx(3.0 / 0.25)
    assert dict(zip(basis, values[1, 0]))["px"] == pytest.approx(-0.5j)


@pytest.mark.parametrize("cell", [HEXAGONAL, SHEARED], ids=["hexagonal", "sheared"])
def test_the_multiplier_is_the_derivative_through_both_samplers(cell):
    """V2: ``i (k+G)`` against a central difference, ``h = 1e-3`` bohr.

    In the tip frame, on a cell whose ``a3`` is not normal to the plane, and
    through the folded sampler a flat tip plane uses: a crystal-for-cartesian
    mix-up or a lost ``2 pi`` reads as a factor here and nowhere else.
    """
    kcrystal = np.array([0.13, -0.21, 0.25])
    miller = _ball(cell, 2.6, kcrystal)
    coefficients = _orthonormal(miller.shape[0], 3, seed=4)
    frame = tip_frame(cell, 2, 1)
    vectors = tip_vectors(miller, kcrystal, cell, frame)
    centre = np.array([[0.31, 0.47, 0.62]]) @ cell.at
    inverse = np.linalg.inv(cell.at)
    h = 1.0e-3

    def psi(offsets):
        return sample_wavefunctions(coefficients, miller, kcrystal,
                                    (centre + offsets) @ inverse, cell.volume)

    def derivative(name):
        return sample_wavefunctions(coefficients * multiplier(name, vectors)[None],
                                    miller, kcrystal, centre @ inverse,
                                    cell.volume)[:, 0]

    x, y, z = frame
    first = {"px": x, "py": y, "pz": z}
    for name, axis in first.items():
        difference = (psi(h * axis[None]) - psi(-h * axis[None]))[:, 0] / (2 * h)
        assert np.allclose(derivative(name), difference, rtol=2e-5, atol=0), name

    def second(u, v):
        return ((psi(h * (u + v)[None]) - psi(h * (u - v)[None])
                 - psi(h * (v - u)[None]) + psi(-h * (u + v)[None]))[:, 0]
                / (4 * h * h))

    checks = {
        "dxy": second(x, y),
        "dyz": second(y, z), "dxz": second(x, z),
        "dx2-y2": second(x, x) - second(y, y),
        "dz2": 3 * second(z, z) - (second(x, x) + second(y, y) + second(z, z)),
    }
    for name, expected in checks.items():
        assert np.allclose(derivative(name), expected, rtol=2e-4,
                           atol=2e-4 * np.abs(expected).max()), name

    # the folded sampler on a flat plane: the same amplitudes to round-off
    points = _plane_points(4, 0.62)
    window = coefficients * multiplier("pz", vectors)[None]
    folded = _sample_sphere(
        _tip_sphere(window, miller, np.ones(miller.shape[0], bool), kcrystal, 1,
                    2, points, 2, None), points, cell.volume)[0]
    plain = sample_wavefunctions(window, miller, kcrystal, points, cell.volume)
    assert np.allclose(folded, plain, rtol=1e-12, atol=1e-13 * np.abs(plain).max())


# --------------------------------------------------------------------------
# V1: the s tip is the point tip
# --------------------------------------------------------------------------


@pytest.mark.parametrize("npol, tip_spin", [(1, None), (2, None),
                                            (2, (0.3, -0.4, 0.86))])
def test_the_s_tip_through_the_orbital_path_is_the_point_tip(npol, tip_spin):
    """V1, the half that can fail: the ``F = 1`` path against the old branch."""
    cell = HEXAGONAL
    geometry, wavefunctions = _geometry(
        cell, [[0.0, 0.0, 0.0], [0.25, -0.1, 0.0]], npol=npol, seed=11)
    eigenvalues = np.linspace(-0.06, 0.06, 5)[None, None].repeat(2, axis=1)
    points = _plane_points(5, 0.7)
    old, _ = _map(geometry, wavefunctions, eigenvalues, points,
                  tip_spin=tip_spin)
    new, extras = _map(geometry, wavefunctions, eigenvalues, points,
                       tip_spin=tip_spin, tip=_tip("s", cell))
    for key in ("coherent", "incoherent"):
        assert np.allclose(new[key], old[key], rtol=1e-13, atol=0), key
    assert "helmholtz_residue" in extras["notes"]["tip"]


def test_the_channels_add_up_to_the_map():
    cell = HEXAGONAL
    geometry, wavefunctions = _geometry(cell, [[0.1, 0.2, 0.0]], seed=5)
    eigenvalues = np.linspace(-0.05, 0.05, 5)[None, None]
    points = _plane_points(4, 0.7)
    values, _ = _map(geometry, wavefunctions, eigenvalues, points,
                     tip=_tip([{"s": 1, "pz": -0.6}, "pxy"], cell,
                              channel_maps=True))
    assert values["channels"].shape == (1, 3, points.shape[0])
    assert np.allclose(values["channels"].sum(axis=1), values["coherent"],
                       rtol=1e-13)
    assert np.all(values["coherent"] >= 0.0)


# --------------------------------------------------------------------------
# V3: the map over the tip plane is the momentum trace
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ["px", {"s": 1.0, "pz": -0.5}, "dz2"])
def test_the_map_over_the_tip_plane_is_the_momentum_trace(spec):
    """V3: ``int_tip T_c d^2 r = sum_k w_k Tr[D S^exit D S^tip_c]``.

    The tip Gram matrix is :func:`exit_overlap`'s integral of the coefficients
    multiplied by the channel's ``F_c``. It shares the multiplier with the map,
    so this tests the fold, the pixel blocks and the per-channel combination,
    not the factor (V2 does that). A 12 x 12 grid integrates the map exactly:
    the in-plane frequencies of ``|a|^2`` are below 12 on this ball.
    """
    cell = HEXAGONAL
    kpoints = [[0.0, 0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0, 0.0]]
    geometry, wavefunctions = _geometry(cell, kpoints, gmax=2.4, seed=21)
    eigenvalues = np.linspace(-0.05, 0.05, 5)[None, None].repeat(2, axis=1)
    height, exit_height = 0.71, 0.2
    n = 12
    points = _plane_points(n, height)
    tip = _tip(spec, cell, kappa=(0.8,))
    values, _ = _map(geometry, wavefunctions, eigenvalues, points, tip=tip,
                     exit_height=exit_height, incoherent=False)
    integral = values["coherent"][0].sum() * surface_area(cell, 2) / n**2

    trace = 0.0
    weights = np.real(amplitude_weights(eigenvalues[0], 0.0, 0.05))
    scale = surface_area(cell, 2) / cell.volume
    for ik in range(len(kpoints)):
        miller = geometry.miller[ik]
        block = wavefunctions[0, ik]
        vectors = tip_vectors(miller, geometry.kcrystal[ik], cell, tip.frame)
        factor = sum(tip.coefficients[0, 0, ib] * multiplier(name, vectors)
                     for ib, name in enumerate(tip.basis))
        folded, _ = fold_onto_plane(block * factor[None], miller, height, 2)
        tip_gram = folded_gram(folded, scale)[None]
        exit_gram = exit_overlap(block, miller, exit_height, 2, cell)[None]
        trace += momentum_weights(exit_gram, tip_gram, geometry.kweights[ik:ik + 1],
                                  weights[ik:ik + 1])["weight"][0]
    assert integral == pytest.approx(trace, rel=1e-11)


@pytest.mark.parametrize("npol, spin", [(1, None), (2, "z")], ids=["scalar", "spinor"])
def test_the_momentum_route_builds_basis_grams_once_for_every_energy(monkeypatch, npol, spin):
    """Audit 2026-11-20 #7: the tip plane's Gram matrix of a channel at an
    energy is ``sum_bb' C_b^* C_b' G_bb'``, with the basis Grams built once.

    :func:`_assemble_momentum` on synthetic states against the direct form,
    one :func:`folded_gram` of the combined coefficients per energy and per
    channel and :func:`momentum_weights` on it, for a ``p`` tip and a hybrid
    channel and a complex one, five energies, a spinor with a spin-selective
    substrate too; and
    the number of :func:`folded_gram` calls does not grow with the energies.
    Before, it was one per energy and channel and k-point, and the Grams of
    every energy were held at once.
    """
    import defuqumat.workflows.transport as transport
    from defuqumat.workflows.transport import _assemble_momentum, _substrate_acceptance

    cell = HEXAGONAL
    kpoints = np.array([[0.0, 0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0, 0.0], [0.2, -0.1, 0.0]])
    geometry, wavefunctions = _geometry(cell, kpoints, gmax=2.4, seed=31, npol=npol)
    nk, nbnd = wavefunctions.shape[1:3]
    eigenvalues = np.stack([np.linspace(-0.05, 0.05, nbnd) + 0.01 * ik
                            for ik in range(nk)])[None]
    planewaves = types.SimpleNamespace(miller=lambda _smooth: geometry.miller,
                                       mask=geometry.mask)
    monkeypatch.setattr(transport, "build_basis", lambda _system: types.SimpleNamespace(
        planewaves=planewaves, smooth=None))
    kset = types.SimpleNamespace(weights=geometry.kweights,
                                 crystal=lambda _cell: kpoints)
    calculation = types.SimpleNamespace(
        system=types.SimpleNamespace(cell=cell, kpoints=kset), noncolin=npol == 2)
    calls = []
    original = transport.folded_gram
    monkeypatch.setattr(transport, "folded_gram",
                        lambda *a, **k: calls.append(1) or original(*a, **k))
    exit_height, height = 0.2, 0.71
    scale = surface_area(cell, 2) / cell.volume
    projector, _ = _substrate_acceptance(spin, 0.7, npol, 1)

    for spec in ("p", [{"s": 1.0, "pz": -0.6}, "px"], {"px": 1.0, "py": 0.5j}):
        counts = []
        for energies in (np.array([0.0]), np.linspace(-0.04, 0.03, 5)):
            tip = _tip(spec, cell, kappa=0.8 + 0.1 * energies)
            calls.clear()
            columns, _ = _assemble_momentum(
                calculation, wavefunctions, eigenvalues, exit_height=exit_height,
                heights=[height], exit_axis=2, energies=energies, broadening=0.05,
                spin=spin, polarization=0.7, method="spectral", smearing="gaussian",
                degeneracy_tol=1e-6, tip=tip)
            counts.append(len(calls))
        assert counts[0] == counts[1], counts

        exit_grams = np.stack([
            exit_overlap(wavefunctions[0, ik], geometry.miller[ik], exit_height, 2,
                         cell, npol=npol, projector=projector) for ik in range(nk)])
        reference = {name: np.zeros((len(energies), nk))
                     for name in ("weight", "tersoff_hamann", "incoherent")}
        for ie, energy in enumerate(energies):
            weights = np.real(amplitude_weights(eigenvalues[0], float(energy), 0.05))
            for ic in range(tip.nchannels):
                grams = []
                for ik in range(nk):
                    vectors = tip_vectors(geometry.miller[ik], kpoints[ik], cell, tip.frame)
                    factor = sum(tip.coefficients[ie, ic, ib] * multiplier(name, vectors)
                                 for ib, name in enumerate(tip.basis))
                    block = (wavefunctions[0, ik].reshape(nbnd, npol, -1)
                             * factor[None, None]).reshape(nbnd, -1)
                    folded, _ = fold_onto_plane(block, geometry.miller[ik], height, 2,
                                                npol=npol)
                    grams.append(original(folded, scale, npol=npol, projector=projector))
                here = momentum_weights(exit_grams, np.stack(grams), geometry.kweights,
                                        weights, eigenvalues=eigenvalues[0], tol=1e-6)
                for name in reference:
                    reference[name][ie] += here[name]
        for name, value in reference.items():
            peak = np.abs(value).max()
            assert peak > 0.0
            assert np.abs(columns[name][0] - value).max() < 1e-12 * peak, name


# --------------------------------------------------------------------------
# V4: the volume limit is the kinetic energy density
# --------------------------------------------------------------------------


def test_the_p_channels_in_the_volume_limit_are_the_kinetic_energy_density():
    """V4: ``sum_a T_pa = kappa^-2 sum_kn w_k delta/eta |grad psi|^2``.

    The right side is :func:`band_kinetic_density` on the FFT grid -- the
    meta-GGA ``tau``, ``i(k+G) c_G`` transformed one direction at a time, in Ry
    with no factor of one half -- sampled at grid points, so nothing is
    interpolated. With ``exit_region="volume"`` the exit Gram matrix is the
    identity, which needs orthonormal rows.
    """
    cell = SHEARED
    kcrystal = np.array([0.11, 0.27, 0.0])
    geometry, wavefunctions = _geometry(cell, [kcrystal], nbnd=4, gmax=2.4,
                                        seed=8)
    miller = geometry.miller[0]
    eigenvalues = np.array([[[-0.03, 0.0, 0.01, 0.04]]])
    grid = (10, 10, 24)
    points = np.array([[i / grid[0], j / grid[1], l / grid[2]]
                       for i in range(0, 10, 3) for j in range(0, 10, 2)
                       for l in (5, 17)], dtype=float)
    kappa = 0.6
    values, _ = _map(geometry, wavefunctions, eigenvalues, points,
                     tip=_tip("p", cell, kappa=(kappa,)), exit_region="volume",
                     incoherent=False)

    weights = geometry.kweights[0] * np.real(
        amplitude_weights(eigenvalues[0, 0], 0.0, 0.05)) ** 2
    kplusg = (miller + kcrystal) @ cell.bg
    tau = np.asarray(band_kinetic_density(
        wavefunctions[0, 0], _fft_index(miller, grid), grid, weights, cell,
        kplusg))
    index = np.rint(points * np.asarray(grid)).astype(int)
    expected = tau[index[:, 0], index[:, 1], index[:, 2]] / kappa**2
    assert np.allclose(values["coherent"][0], expected, rtol=1e-11, atol=0)


# --------------------------------------------------------------------------
# V5: the frame
# --------------------------------------------------------------------------


def _symmetric_states(cell, miller, rotation, nbnd, seed):
    """States at Gamma invariant under ``rotation`` (cartesian, columns)."""
    q = miller @ cell.bg
    lookup = {tuple(h): i for i, h in enumerate(miller)}
    raw = _orthonormal(miller.shape[0], nbnd, seed).T          # (npw, nbnd)
    images = [np.arange(miller.shape[0])]
    for power in (1, 2):
        turned = q @ np.linalg.matrix_power(rotation, power).T
        back = np.rint(turned @ np.linalg.inv(cell.bg)).astype(int)
        images.append(np.array([lookup[tuple(h)] for h in back]))
    symmetric = sum(raw[i] for i in images) / 3.0
    out, _ = np.linalg.qr(symmetric)
    return out.T


def test_the_frame_turns_with_the_states_and_pxy_keeps_the_lattice_symmetry():
    """V5: ``T_px[F](R r) = T_px[F R](r)`` for states invariant under ``R``.

    ``R`` is the threefold rotation of the hexagonal cell. If ``psi(R r) =
    psi(r)`` then ``d_u psi(R r) = d_(R^T u) psi(r)``, so turning the points and
    turning the frame the other way give one map; and the incoherent ``pxy``
    pair, whose ``sum_c c c^dagger`` is the identity on the plane, gives the
    same map at ``r`` and ``R r``. A rotation of the frame alone would not test
    the cartesian components, since it passes with crystal ones too.
    """
    cell = HEXAGONAL
    angle = 2.0 * np.pi / 3.0
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                         [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    miller = _ball(cell, 3.0)
    states = _symmetric_states(cell, miller, rotation, 4, seed=13)
    geometry, wavefunctions = _geometry(cell, [[0.0, 0.0, 0.0]], gmax=3.0,
                                        states=states[None])
    eigenvalues = np.array([[[-0.04, -0.01, 0.02, 0.05]]])
    rng = np.random.default_rng(2)
    cartesian = np.c_[rng.uniform(0, 4, size=(6, 2)), np.full(6, 0.7 * 14.0)]
    inverse = np.linalg.inv(cell.at)
    points = cartesian @ inverse
    turned = (cartesian @ rotation.T) @ inverse

    frame = tip_frame(cell, 2, 1)
    at_turned, _ = _map(geometry, wavefunctions, eigenvalues, turned,
                        tip=_tip("px", cell, frame=frame))
    with_frame, _ = _map(geometry, wavefunctions, eigenvalues, points,
                         tip=_tip("px", cell, frame=frame @ rotation))
    plain, _ = _map(geometry, wavefunctions, eigenvalues, points,
                    tip=_tip("px", cell, frame=frame))
    assert np.allclose(at_turned["coherent"], with_frame["coherent"], rtol=1e-11)
    # and the test can fail: the frame matters for a single p orbital here
    assert not np.allclose(at_turned["coherent"], plain["coherent"], rtol=1e-3)

    pxy_turned, _ = _map(geometry, wavefunctions, eigenvalues, turned,
                         tip=_tip("pxy", cell))
    pxy, _ = _map(geometry, wavefunctions, eigenvalues, points,
                  tip=_tip("pxy", cell))
    assert np.allclose(pxy_turned["coherent"], pxy["coherent"], rtol=1e-11)


# --------------------------------------------------------------------------
# V7: the conjugate, pinned by a winding measured without a derivative
# --------------------------------------------------------------------------


def _honeycomb_at_k():
    """The two lowest states at K of a honeycomb of Gaussian s wells.

    Exact eigenvectors of a plane-wave Hamiltonian small enough to diagonalise
    densely; the vacuum level is 0. The pair at K is degenerate by symmetry,
    the same E irrep as graphene's pi pair, which is all this test uses.
    """
    a, c, depth, width, ecut = 4.65, 16.0, 3.0, 1.0, 12.0
    cell = _Cell([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    sites = np.array([[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.5]])
    k = np.array([1 / 3, 1 / 3, 0.0])
    reach = [int(np.ceil(np.sqrt(ecut) * np.linalg.norm(cell.at[i]) / (2 * np.pi))) + 1
             for i in range(3)]
    h = np.stack(np.meshgrid(*[np.arange(-m, m + 1) for m in reach], indexing="ij"),
                 -1).reshape(-1, 3)
    q = (h + k) @ cell.bg
    h = h[(q**2).sum(1) <= ecut]
    q = (h + k) @ cell.bg
    dh = h[:, None, :] - h[None, :, :]
    structure = np.exp(-2j * np.pi * (dh @ sites.T)).sum(-1)
    hamiltonian = (-depth * (2 * np.pi) ** 1.5 * width**3 / cell.volume
                   * np.exp(-0.5 * ((dh @ cell.bg) ** 2).sum(-1) * width**2)
                   * structure)
    hamiltonian = hamiltonian + np.diag((q**2).sum(1))
    energies, vectors = np.linalg.eigh(hamiltonian)
    return cell, h, k, energies[:2], vectors[:, :2].T.copy(), sites


def test_a_chiral_tip_couples_to_one_winding_of_the_dirac_pair():
    """V7: on atom A, ``{"px": 1, "py": 1j}`` against its conjugate.

    The hollow site cannot pin the conjugate: there the pair carries both
    windings and both chiral tips couple equally (the review, 9.968e-08 each).
    On an atom one winding is missing. It is measured here **without a
    derivative**, from ``psi`` on a circle of 0.2 bohr around the site, and the
    tip whose angular momentum matches it must couple while the other must not.
    Both are invariant under a rotation inside the pair (rule D4), which the
    test also applies.
    """
    cell, miller, k, energies, pair, sites = _honeycomb_at_k()
    assert abs(energies[1] - energies[0]) < 1e-10
    kappa = float(np.sqrt(-energies[0]))
    geometry = TransportGeometry(miller=miller[None], mask=np.ones((1, len(miller)), bool),
                                 kcrystal=k[None], kweights=np.ones(1), cell=cell)
    rng = np.random.default_rng(3)
    unitary, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    eigenvalues = np.full((1, 1, 2), float(energies[0]))
    height = 0.5 + 3.0 / 16.0
    inverse = np.linalg.inv(cell.at)

    def coupling(spec, site, states):
        point = np.array([[site[0], site[1], height]])
        values, _ = _map(geometry, states[None, None], eigenvalues, point,
                         tip=_tip(spec, cell, kappa=(kappa,)),
                         energies=np.array([float(energies[0])]),
                         exit_height=0.5 - 4.0 / 16.0, incoherent=False)
        return float(values["coherent"][0, 0])

    def winding(site):
        centre = np.array([site[0], site[1], height]) @ cell.at
        phi = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
        circle = centre + 0.2 * np.c_[np.cos(phi), np.sin(phi), 0.0 * phi]
        psi = sample_wavefunctions(pair, miller, k, circle @ inverse, cell.volume)
        return {m: float((np.abs((psi * np.exp(-1j * m * phi)).mean(1)) ** 2).sum())
                for m in (1, -1)}

    site = sites[0][:2]
    turns = winding(site)
    present = 1 if turns[1] > turns[-1] else -1
    assert min(turns.values()) < 1e-20 * max(turns.values())
    chiral = {1: {"px": 1.0, "py": 1.0j}, -1: {"px": 1.0, "py": -1.0j}}
    for states in (pair, unitary @ pair):
        matching = coupling(chiral[present], site, states)
        other = coupling(chiral[-present], site, states)
        assert matching > 0.0 and other < 1e-20 * matching
    # the second atom carries the other winding
    second = winding(sites[1][:2])
    assert (1 if second[1] > second[-1] else -1) == -present
    # at the hollow s and pz see nothing, on an atom they do
    hollow = (0.0, 0.0)
    for spec in ("s", "pz"):
        assert coupling(spec, hollow, pair) < 1e-20 * coupling(spec, site, pair)
    assert coupling("pxy", hollow, pair) > 1e-6 * coupling("s", site, pair)


# --------------------------------------------------------------------------
# rotations, the wedge, kappa and the refusals
# --------------------------------------------------------------------------


def test_the_rotation_representation_is_the_rotation_on_whole_shells():
    rng = np.random.default_rng(7)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    other, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(other) < 0:
        other[:, 0] *= -1
    p = ("py", "pz", "px")
    rep = rotation_representation(p, rotation)
    # a p channel is a polar vector: (px, py, pz) -> R (px, py, pz)
    order = [p.index(n) for n in ("px", "py", "pz")]
    assert np.allclose(rep[np.ix_(order, order)], rotation, atol=1e-12)
    d = tuple(n for n in ORBITALS if ORBITALS[n] == 2)
    both = rotation_representation(d, rotation @ other)
    assert np.allclose(both, rotation_representation(d, rotation)
                       @ rotation_representation(d, other), atol=1e-11)
    # Chen's d operators are not orthonormal, so ``D`` is not unitary: it keeps
    # the metric of their angular norms, ``D^T W D = W`` (``tip.py``)
    metric = np.diag([4 * np.pi / 15, 4 * np.pi / 15, 16 * np.pi / 5,
                      4 * np.pi / 15, 16 * np.pi / 15])
    assert np.allclose(both.T @ metric @ both, metric, atol=1e-10)
    assert not np.allclose(both.T @ both, np.eye(5), atol=1e-3)
    with pytest.raises(ValueError, match="whole shells"):
        rotation_representation(("px",), rotation)


def test_an_invariant_tip_is_not_turned_and_a_hybrid_is():
    cell = HEXAGONAL
    tip = _tip("pxy", cell)
    turn = np.array([[0, 1, 0], [-1, -1, 0], [0, 0, 1]], dtype=float)   # C3, rows
    mirror = np.diag([1.0, 1.0, -1.0])
    assert _turned_tips(tip, [np.eye(3), turn, mirror], cell, 4) is None
    assert _turned_tips(_tip("pz", cell), [np.eye(3), mirror], cell, 4) is None
    hybrid = _turned_tips(_tip({"s": 1.0, "pz": -0.5}, cell),
                          [np.eye(3), mirror], cell, 4)
    assert hybrid is not None and hybrid.per_point
    # one set per operation, not per point (audit 2026-11-20 #34): the points
    # are operation-major, four to an operation, so point 4 is the mirror's
    assert hybrid.coefficients.shape[:2] == (1, 2) and hybrid.points_per_image == 4
    flipped = dict(zip(hybrid.basis, hybrid.at_points(0, 0, [4])[0]))
    kept = dict(zip(hybrid.basis, hybrid.at_points(0, 0, [3])[0]))
    assert flipped["pz"] == pytest.approx(-kept["pz"])
    assert flipped["s"] == pytest.approx(kept["s"])


def _star(cell, kcrystal, rotations, nbnd, seed, gmax=3.0):
    """A wedge of one k-point and the whole star it stands for.

    ``rotations`` are cartesian (columns) and keep the stacking axis. The
    states of the image ``R k`` are those of ``k`` turned, ``psi_Rk(r) =
    psi_k(R^-1 r)``, which puts the coefficient of ``G`` at ``R G`` in the same
    Gamma-centred ball. Returns ``(wedge, whole, eigenvalues, group)``: two
    ``(geometry, wavefunctions)`` pairs, the wedge carrying the weight of the
    whole star, and the group as :func:`_completed_on_a_wedge` reads it, the
    crystal rows ``M = A R^T A^-1``.
    """
    miller = _ball(cell, gmax)
    lookup = {tuple(h): i for i, h in enumerate(miller)}
    bg, at = np.asarray(cell.bg), np.asarray(cell.at)
    states = _orthonormal(miller.shape[0], nbnd, seed)            # (nbnd, npw)
    kpoints, whole = [], []
    for rotation in rotations:
        turn = bg @ rotation.T @ np.linalg.inv(bg)                # on crystal rows
        moved = np.rint(miller @ turn).astype(int)
        order = np.array([lookup[tuple(h)] for h in moved])
        image = np.zeros_like(states)
        image[:, order] = states
        kpoints.append(np.asarray(kcrystal) @ turn)
        whole.append(image)
    nk = len(rotations)
    star = _geometry(cell, kpoints, nbnd=nbnd, gmax=gmax, states=np.stack(whole))
    geometry, wavefunctions = _geometry(cell, [kcrystal], nbnd=nbnd, gmax=gmax,
                                        states=states[None])
    geometry = TransportGeometry(
        miller=geometry.miller, mask=geometry.mask, kcrystal=geometry.kcrystal,
        kweights=np.array([2.0]), cell=cell, npol=1)
    assert np.allclose(star[0].kweights.sum(), 2.0) and nk == star[0].kweights.size
    group = types.SimpleNamespace(
        crystal=np.stack([at @ r.T @ np.linalg.inv(at) for r in rotations]),
        translations=np.zeros((nk, 3)), antiunitary=np.zeros(nk, dtype=bool))
    return (geometry, wavefunctions), star, group


@pytest.mark.parametrize("spec", ["pxy", "p", [{"s": 1.0, "px": 0.7}, "py"]], ids=str)
def test_the_channel_maps_on_a_wedge_are_those_of_the_whole_star(spec):
    """Audit 2026-11-20 #4 and #5: each channel's map, not only their sum.

    A general k-point of the hexagonal cell and its six images under C6, with
    the states of each image the turned states of the first. The map completed
    from the one point is the map of the six, channel by channel. Before the
    fix the ``pxy`` and ``p`` tips, whose sum over channels is invariant, were
    not turned, and each channel map came back averaged over the group; the
    total, which is checked first, was right then too.
    """
    from defuqumat.workflows.transport import _completed_on_a_wedge

    cell = HEXAGONAL
    rotations = []
    for n in range(6):
        angle = n * np.pi / 3.0
        rotations.append(np.array([[np.cos(angle), -np.sin(angle), 0.0],
                                   [np.sin(angle), np.cos(angle), 0.0],
                                   [0.0, 0.0, 1.0]]))
    wedge, star, group = _star(cell, np.array([0.17, 0.06, 0.0]), rotations,
                               nbnd=5, seed=21)
    eigenvalues = np.linspace(-0.05, 0.05, 5)
    points = np.r_[_plane_points(4, 0.7), [[0.31, 0.12, 0.7], [0.55, 0.83, 0.7]]]
    tip = _tip(spec, cell, kappa=(0.7, 0.6), channel_maps=True)
    settings = dict(exit_axis=2, energies=np.array([0.0, 0.02]), broadening=0.05,
                    spin=None, polarization=1.0, tip_spin=None,
                    tip_polarization=1.0, incoherent=False, exit_region="plane",
                    method="spectral", smearing="gaussian", tip=tip)
    right, _ = _assemble(star[0], star[1], np.broadcast_to(eigenvalues, (1, 6, 5)),
                         points, exit_height=0.2, **settings)
    got, _ = _completed_on_a_wedge(
        group, wedge[0], wedge[1], eigenvalues[None, None], points, 0.2, 2,
        _fake_system([[0, 0, 0.5]], cell), (), settings)
    scale = np.abs(right["coherent"]).max()
    assert scale > 0.0
    assert np.abs(got["coherent"] - right["coherent"]).max() < 1e-12 * scale
    assert got["channels"].shape == right["channels"].shape
    assert np.abs(got["channels"] - right["channels"]).max() < 1e-12 * scale
    # and the test can fail: the channels of the star are not the group average
    averaged = np.broadcast_to(right["channels"].mean(axis=1, keepdims=True),
                               right["channels"].shape)
    assert np.abs(averaged - right["channels"]).max() > 1e-2 * scale


def _fake_system(positions, cell, nspin=1, **extra):
    structure = types.SimpleNamespace(
        positions_crystal=lambda _cell: np.asarray(positions, dtype=float))
    return types.SimpleNamespace(structure=structure, cell=cell, nspin=nspin,
                                 spiral=False, efield=None, berry_field=None,
                                 **extra)


def test_the_tip_side_is_fixed_by_where_the_material_is():
    cell = SQUARE
    system = _fake_system([[0, 0, 0.5]], cell)
    assert tip_side(system, 2, 0.3, [[0, 0, 0.7]]) == 1
    assert tip_side(system, 2, 0.7, [[0, 0, 0.3]]) == -1
    with pytest.raises(NotImplementedError, match="both sides"):
        tip_side(system, 2, 0.1, [[0, 0, 0.7], [0, 0, 0.3]])
    # z points from the material to the tip
    assert tip_frame(cell, 2, -1)[2] @ np.array([0, 0, 1.0]) < 0


def test_the_vacuum_level_is_the_plateau_and_averages_the_two_spins():
    cell = SQUARE
    system = _fake_system([[0, 0, 0.5]], cell, nspin=2)
    z = np.arange(26) / 26
    profile = np.where(np.abs(z - 0.5) < 0.2, -1.0, 0.3)
    potential = np.stack([np.broadcast_to(profile + shift, (4, 4, 26))
                          for shift in (0.1, -0.1)])
    level = vacuum_level(system, potential, 2)
    assert level["plateau"] == pytest.approx(0.3)
    assert level["spread"] == pytest.approx(0.0)
    assert decay_constants_at(0.3, [0.1]) == pytest.approx([np.sqrt(0.2)])
    with pytest.raises(ValueError, match="no barrier"):
        decay_constants_at(0.3, [0.35])


def test_the_refusals_are_by_name():
    cell = SQUARE
    result = types.SimpleNamespace(potential=np.zeros((1, 4, 4, 26)))
    points = np.array([[0.0, 0.0, 0.7]])
    plain = _fake_system([[0, 0, 0.5]], cell)
    args = (result, 2, 0.3, points, np.array([-0.5]), None, None, None, False)
    assert _tip_assembly("s", plain, *args)[0] is None
    spiral = _fake_system([[0, 0, 0.5]], cell)
    spiral.spiral = True
    with pytest.raises(NotImplementedError, match="spin spiral"):
        _tip_assembly("pz", spiral, *args)
    with pytest.raises(NotImplementedError, match="symmetry-reduced"):
        _tip_assembly({"px": 1, "py": 1j}, plain, *args, wedge=True)
    _tip_assembly({"px": 1, "py": 1}, plain, *args, wedge=True)   # real: fine
    field = _fake_system([[0, 0, 0.5]], cell, )
    field.efield = (2, 0.001, 0.5, 0.1, False)
    with pytest.raises(NotImplementedError, match="tip_kappa"):
        _tip_assembly("pz", field, *args)
    tip, notes = _tip_assembly("pz", field, result, 2, 0.3, points,
                               np.array([-0.5]), 0.9, None, None, False)
    assert notes["kappa_source"] == "given" and tip.kappa[0] == 0.9
    with pytest.raises(ValueError, match="orthonormal"):
        tip_frame(cell, 2, 1, np.eye(3) * 2)
    with pytest.raises(ValueError, match="right-handed"):
        tip_frame(cell, 2, 1, np.diag([1.0, 1.0, -1.0]))


# --------------------------------------------------------------------------
# the vacuum continuation
# --------------------------------------------------------------------------


def _match(cell, height, plateau):
    bg = np.asarray(cell.bg)
    return {"height": height, "axis": 2, "normal": bg[2] / np.linalg.norm(bg[2]),
            "side": 1, "spacing": cell.volume / surface_area(cell, 2),
            "plateau": plateau, "distances": []}


@pytest.mark.parametrize("cell", [HEXAGONAL, SHEARED], ids=["hexagonal", "sheared"])
def test_the_continued_state_obeys_the_vacuum_equation_and_meets_the_raw_one(cell):
    """At the matching plane the continued state is the raw one, and so are its
    in-plane derivatives; above it ``nabla^2 = kappa_n^2`` exactly.

    On the sheared cell the foot of a tip point is not at its in-plane crystal
    coordinates, which is what :func:`_feet` is for.
    """
    kcrystal = np.array([0.13, -0.21, 0.0])
    miller = _ball(cell, 2.6, kcrystal)
    states = _orthonormal(miller.shape[0], 3, seed=9)
    energies = np.array([-0.31, -0.27, -0.22])
    plateau, height = 0.05, 0.6
    frame = tip_frame(cell, 2, 1)
    tip = TipAssembly(orbital=TipOrbital.resolve("p"), basis=("py", "pz", "px"),
                      coefficients=np.zeros((1, 3, 3)), frame=frame,
                      kappa=np.array([0.6]), match=_match(cell, height, plateau))
    mask = np.ones(miller.shape[0], bool)
    on_plane = _plane_points(3, height)
    (index, feet, spheres), = _continued_groups(
        tip, states, 1, miller, mask, kcrystal, energies, on_plane, cell, None)
    assert np.allclose(feet, on_plane)
    vectors = tip_vectors(miller, kcrystal, cell, frame)
    for name, sphere in zip(tip.basis, spheres):
        continued = _sample_sphere(sphere, feet, cell.volume)[0]
        raw = sample_wavefunctions(states * multiplier(name, vectors)[None],
                                   miller, kcrystal, on_plane, cell.volume)
        if name == "pz":
            assert not np.allclose(continued, raw, rtol=1e-3)
        else:
            assert np.allclose(continued, raw, rtol=1e-11,
                               atol=1e-12 * np.abs(raw).max()), name
    # Helmholtz by construction: -Q.Q = plateau - e_n
    folded, inplane = fold_onto_plane(states, miller, height, 2,
                                      k_along=float(kcrystal[2]))
    lateral = kcrystal.copy()
    lateral[2] = 0.0
    wavevectors, kappa = continued_wavevectors(inplane, lateral, cell,
                                               tip.match["normal"], plateau, energies)
    laplacian = multiplier(LAPLACIAN, wavevectors @ frame.T)
    assert np.allclose(laplacian, (plateau - energies)[:, None], rtol=1e-12)
    # and a tip 2 bohr above has its foot straight below along the normal
    above = on_plane.copy()
    above[:, 2] += 2.0 / tip.match["spacing"]
    feet, distance = _feet(tip.match, above, cell)
    assert np.allclose(distance, 2.0)
    shift = (above - feet) @ cell.at
    assert np.allclose(shift, 2.0 * tip.match["normal"][None], atol=1e-12)


def test_a_matching_plane_outside_the_gap_is_refused():
    cell = SQUARE
    result = types.SimpleNamespace(potential=np.zeros((1, 4, 4, 26)))
    plain = _fake_system([[0, 0, 0.5]], cell)
    args = (result, 2, 0.3, np.array([[0.0, 0.0, 0.7]]), np.array([-0.5]), None,
            None, None, False)
    tip, notes = _tip_assembly("pz", plain, *args, match_height=0.6)
    assert tip.match is not None and notes["match_distance_bohr"] == [pytest.approx(1.3)]
    for wrong in (0.45, 0.8, 0.2):
        with pytest.raises(ValueError, match="between the material and the tip"):
            _tip_assembly("pz", plain, *args, match_height=wrong)
    with pytest.raises(NotImplementedError, match="symmetry-reduced"):
        _tip_assembly("pz", plain, *args, match_height=0.6, wedge=True)
    # the s tip takes the orbital path once it is continued
    tip, _ = _tip_assembly("s", plain, *args, match_height=0.6)
    assert tip is not None and tip.orbital.is_point


def test_a_matching_plane_inside_an_augmentation_sphere_is_refused():
    """The rule of the exit and tip planes, now on the matching plane (audit
    2026-11-20 #12): folded there, a state is the pseudo-wavefunction.

    An ultrasoft-like dataset reaching 2.4 bohr, the atom at 0.5 of a 13 bohr
    cell and the tip at 0.7: a plane 1.3 bohr above the atom is refused, one
    2.47 bohr above it is not, and without an augmented dataset (the
    norm-conserving case of every recorded run) 1.3 bohr is kept.
    """
    cell = SQUARE
    result = types.SimpleNamespace(potential=np.zeros((1, 4, 4, 26)))
    plain = _fake_system([[0, 0, 0.5]], cell)
    args = (result, 2, 0.3, np.array([[0.0, 0.0, 0.7]]), np.array([-0.5]), None,
            None, None, False)
    r = np.linspace(0.0, 5.0, 501)
    ultrasoft = types.SimpleNamespace(
        augmentation=object(), paw=None, r=r,
        projectors=[types.SimpleNamespace(cutoff_index=240)])     # r = 2.4
    with pytest.raises(NotImplementedError, match="the matching plane"):
        _tip_assembly("pz", plain, *args, match_height=0.6, pseudos=(ultrasoft,))
    tip, _ = _tip_assembly("pz", plain, *args, match_height=0.69,
                           pseudos=(ultrasoft,))
    assert tip.match["height"] == 0.69
    tip, _ = _tip_assembly("pz", plain, *args, match_height=0.6, pseudos=())
    assert tip.match is not None


# --------------------------------------------------------------------------
# the presets
# --------------------------------------------------------------------------


def _square_well_orbital(l, kappa, radius, tmp_path):
    """A bound state of a spherical well at ``-kappa^2``, as a UPF file tabulates it.

    Inside ``j_l(k r)``, outside ``A k_l(kappa r)``, matched in value and slope,
    so the tail amplitude ``A`` is known exactly. Written as a minimal UPF with
    ``pseudo_energy``, so :func:`bardeen_weight` reads it the way it reads a
    real dataset.
    """
    from scipy.optimize import brentq
    from scipy.special import spherical_jn, spherical_kn

    def mismatch(k):
        inside = k * spherical_jn(l, k * radius, True) / spherical_jn(l, k * radius)
        outside = (kappa * spherical_kn(l, kappa * radius, True)
                   / spherical_kn(l, kappa * radius))
        return inside - outside

    grid = np.linspace(0.3, 3.0 + 2.0 * l, 4000) / radius
    values = [mismatch(k) for k in grid]
    k = next(brentq(mismatch, grid[i], grid[i + 1]) for i in range(len(grid) - 1)
             if np.sign(values[i]) != np.sign(values[i + 1])
             and abs(values[i]) < 50 and abs(values[i + 1]) < 50)
    r = np.exp(np.linspace(np.log(1e-5), np.log(60.0), 6000))
    scale = spherical_jn(l, k * radius) / spherical_kn(l, kappa * radius)
    radial = np.where(r < radius, spherical_jn(l, k * r),
                      scale * spherical_kn(l, kappa * r))
    chi = r * radial
    norm = np.sqrt(np.sum(0.5 * (chi[1:] ** 2 + chi[:-1] ** 2) * np.diff(r)))
    chi, tail = chi / norm, scale / norm
    label = "1S" if l == 0 else "2P"
    path = tmp_path / f"well-{l}.UPF"
    numbers = lambda a: " ".join(f"{x:.16e}" for x in a)
    path.write_text(
        '<UPF version="2.0.1">\n<PP_HEADER element="X" pseudo_type="NC" '
        'number_of_wfc="1" number_of_proj="0" mesh_size="%d" z_valence="1.0" '
        'l_max="0" core_correction="F" is_ultrasoft="F" is_paw="F" '
        'functional="LDA" relativistic="scalar"/>\n'
        '<PP_MESH><PP_R>%s</PP_R><PP_RAB>%s</PP_RAB></PP_MESH>\n'
        '<PP_LOCAL>%s</PP_LOCAL>\n<PP_NONLOCAL></PP_NONLOCAL>\n'
        '<PP_PSWFC><PP_CHI.1 label="%s" l="%d" occupation="1.0" '
        'pseudo_energy="%.16e">%s</PP_CHI.1></PP_PSWFC>\n'
        '<PP_RHOATOM>%s</PP_RHOATOM>\n</UPF>\n'
        % (r.size, numbers(r), numbers(r * np.gradient(np.log(r))), numbers(0 * r),
           label, l, -kappa**2, numbers(chi), numbers(chi**2)))
    return path, label, tail


def test_the_bardeen_weight_is_the_tail_with_one_constant_for_every_l(tmp_path):
    """``K_l = (pi / 2 kappa) A_l`` for both ``l``, ``A_l`` the exact tail.

    The constant is the addition theorem's, and what matters is that it does
    not depend on ``l``: a weight computed numerically from a tabulated orbital
    (a second derivative on a log mesh, as for a UPF file) then gives the right
    s against p weight. A well of radius 2.5 bohr binding its state at the
    tunnelling energy, so its tail is exactly ``k_l(kappa r)``.
    """
    from defuqumat.transport.tip_presets import bardeen_weight

    kappa = 0.6
    ratios = []
    for l in (0, 1):
        path, label, tail = _square_well_orbital(l, kappa, 2.5, tmp_path)
        weight = bardeen_weight(path, label, kappa, cutoff=20.0)
        ratios.append(abs(weight) * 2.0 * kappa / np.pi / abs(tail))
    # the same constant for both l, which is what a relative weight needs:
    # 1.006983 and 1.006990 measured; the 0.7 per cent from the exact tail is
    # the well's kink under a numerical second derivative, and it falls
    # linearly with the mesh (0.19 per cent at four times the points)
    assert ratios[0] == pytest.approx(ratios[1], rel=1e-4)
    assert ratios[0] == pytest.approx(1.0, rel=1e-2)


def test_the_angular_norms_turn_a_normalised_orbital_into_chens_weight():
    """``(N_b / N_s)^2 int P_b^2 dOmega = 4 pi`` for every orbital of Table I."""
    from defuqumat.transport.tip_presets import ANGULAR_NORM

    theta = np.linspace(0.0, np.pi, 400)
    phi = np.linspace(0.0, 2.0 * np.pi, 800, endpoint=False)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    n = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], -1)
    area = np.sin(t) * (theta[1] - theta[0]) * (phi[1] - phi[0])
    for name, ratio in ANGULAR_NORM.items():
        polynomial = np.real(multiplier(name, -1.0j * n))
        if ORBITALS[name] == 2:
            polynomial = -np.real(multiplier(name, n))
        norm = np.sum(polynomial**2 * area) * ratio**2
        assert norm == pytest.approx(4 * np.pi, rel=2e-3), name


def test_the_presets_are_single_families_with_the_weight_of_their_orbital():
    from defuqumat.transport.tip_presets import PRESETS, bardeen_weight, preset, PSEUDO_DIR

    assert set(PRESETS) >= {"CO", "W", "Pb"}
    co = TipOrbital.resolve("CO")
    assert co.names == ("py", "px") and len(co.channels) == 2
    weights = co.coefficients([0.57], co.basis())
    expected = np.sqrt(3.0) * abs(bardeen_weight(PSEUDO_DIR / "O.pbe-n-rrkjus_psl.1.0.0.UPF",
                                                 "2P", 0.57)) / 0.57
    assert np.abs(weights[0]).max() == pytest.approx(expected, rel=1e-12)
    assert preset("Pb").names == ("py", "pz", "px")
    assert preset("W").names == ("dz2",)
    # converged in the radial cutoff
    o2p = [bardeen_weight(PSEUDO_DIR / "O.pbe-n-rrkjus_psl.1.0.0.UPF", "2P", 0.57, c)
           for c in (12.0, 16.0)]
    assert o2p[0] == pytest.approx(o2p[1], rel=5e-3)


def test_a_label_two_orbitals_share_is_refused_and_a_single_one_is_kept():
    """A fully relativistic file has one orbital per ``j`` under one label.

    ``Bi.rel-pbe-dn-rrkjus_psl.1.0.0.UPF`` has two ``6P`` (-0.4085 and -0.2726
    Ry) and two ``5D``: a tip built from the first would carry the weight of
    one ``j`` alone (audit 2026-11-20 #43). Its ``6S`` is one orbital and is
    kept, and so are the presets, which are scalar-relativistic.
    """
    from defuqumat.transport.tip_presets import (
        PRESETS, PSEUDO_DIR, bardeen_weight, from_upf, orbital_eigenvalue, preset)

    bismuth = PSEUDO_DIR / "Bi.rel-pbe-dn-rrkjus_psl.1.0.0.UPF"
    with pytest.raises(NotImplementedError, match="spinor tip channel"):
        from_upf(bismuth, {"pz": ("6P", 1.0)})
    with pytest.raises(NotImplementedError, match="2 orbitals labelled '5D'"):
        bardeen_weight(bismuth, "5D", 0.6)
    with pytest.raises(NotImplementedError, match="fully relativistic"):
        orbital_eigenvalue(bismuth, "6P")
    assert from_upf(bismuth, {"s": ("6S", 1.0)}).names == ("s",)
    for name in PRESETS:
        assert preset(name).channels


def test_the_wkb_continuation_is_the_flat_one_in_a_flat_vacuum_and_integrates_a_slope():
    """``exp(-int kappa dt)`` with ``kappa(t)^2 = V(t) - e + |q|^2``.

    A flat profile gives :func:`continued_wavevectors` to round-off; a profile
    linear in the height gives the closed form
    ``(2 / 3b) [(a + b d)^(3/2) - a^(3/2)]`` of the integral, to the 0.05 bohr
    steps' trapezoid error, and the derivative at the tip takes the local
    ``kappa``.
    """
    cell = HEXAGONAL
    inplane = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [-1, 2, 0]])
    lateral = np.array([0.1, 0.2, 0.0])
    energies = np.array([-0.4, -0.3])
    match = _match(cell, 0.6, 0.05)
    n_points = 120
    flat = dict(match, profile=np.full(n_points, 0.05))
    distance = 3.7
    q_flat, decay_flat = continued_components(inplane, lateral, cell, flat, energies, distance)
    q_old, kappa = continued_wavevectors(inplane, lateral, cell, match["normal"], 0.05, energies)
    assert np.allclose(decay_flat, np.exp(-kappa * distance), rtol=1e-12)
    assert np.allclose(q_flat, q_old, rtol=1e-12)
    # a potential rising linearly from the plane, V = V0 + slope t
    v0, slope = -0.1, 0.03
    heights = np.arange(n_points) / n_points
    above = np.mod(heights - 0.6, 1.0) * match["spacing"]
    profile = np.where(above < 6.0, v0 + slope * above, v0 + slope * 6.0)
    sloped = dict(match, profile=profile)
    q_sloped, decay = continued_components(inplane, lateral, cell, sloped, energies, distance)
    q = (inplane + lateral) @ cell.bg
    q_par2 = np.sum((q - np.outer(q @ match["normal"], match["normal"])) ** 2, axis=1)
    a = v0 - energies[:, None] + q_par2[None, :]
    exact = 2.0 / (3.0 * slope) * ((a + slope * distance) ** 1.5 - a ** 1.5)
    assert np.allclose(-np.log(decay), exact, rtol=2e-3)
    local = np.sqrt(a + slope * distance)
    assert np.allclose(np.imag(q_sloped @ match["normal"]), local, rtol=5e-3)
