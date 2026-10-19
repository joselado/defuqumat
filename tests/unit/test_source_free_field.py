"""``projsbf``: the longitudinal part of a vector field, removed.

Elk's ``nosource`` makes the exchange-correlation magnetic field divergence-free
by projecting out its longitudinal component, which on a plane-wave grid is

    B(G) -> B(G) - G (G . B(G)) / |G|^2,

with ``G = 0`` left alone. Everything here is about that one operation, on
fields built by hand where the answer is known in advance, because the run-level
consequences (what it does to a moment, and to the torque that is identically
zero without it) belong to a cell rather than to an array.

**The two cases that have to be told apart.** A gradient is *all* longitudinal
and must be removed entirely; a curl is *all* transverse and must come back
untouched. A projector that did nothing at all would pass the second on its own,
which is why they are tested as a pair and not one at a time.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.basis.gradients import divergence, gradient
from defuqumat.basis.gvectors import generate_gvectors
from defuqumat.scf.sourcefree import longitudinal_field, project_source_free
from defuqumat.system.cell import Cell

pytestmark = [pytest.mark.unit]


@pytest.fixture(scope="module")
def grid():
    """A cubic cell and its dense G set: nothing here depends on the crystal."""
    cell = Cell.from_vectors(np.eye(3) * 10.0, alat=10.0)
    return cell, generate_gvectors(cell, 40.0)


def _phase(gvectors, cell, index):
    """``exp(i G . r)`` as a real field: one plane wave, on the grid."""
    n = gvectors.grid
    coords = np.stack(np.meshgrid(
        np.arange(n[0]) / n[0], np.arange(n[1]) / n[1], np.arange(n[2]) / n[2],
        indexing="ij",
    ))
    return np.cos(2.0 * np.pi * sum(index[d] * coords[d] for d in range(3)))


def test_a_gradient_is_removed_entirely(grid):
    """``B = grad f`` has no transverse part, so the projection must return zero."""
    cell, gvectors = grid
    scalar = jnp.asarray(_phase(gvectors, cell, (1, 2, 0))
                         + 0.5 * _phase(gvectors, cell, (2, -1, 1)))
    from defuqumat.basis.fft import r_to_g
    field = gradient(r_to_g(scalar, gvectors.fft_index), gvectors, cell)

    transverse = field - longitudinal_field(field, gvectors, cell)
    assert np.abs(np.asarray(transverse)).max() < 1.0e-10 * np.abs(
        np.asarray(field)).max()


def test_a_divergence_free_field_is_untouched(grid):
    """The other half of the pair, and the one a do-nothing projector passes.

    ``B = G x e`` for a single ``G`` is transverse by construction, so nothing
    may be subtracted from it -- and *nothing* means to round-off, not to the
    accuracy of a round trip through the sphere, which is why the projection is
    written as a subtraction of the longitudinal part rather than as a rebuild
    of the whole field.
    """
    cell, gvectors = grid
    index = (1, 0, 2)
    g = np.asarray(gvectors.cartesian(cell))
    # The cartesian G of that Miller index, from the reciprocal cell directly.
    gvec = 2.0 * np.pi * np.asarray(index) @ np.linalg.inv(np.asarray(cell.at))
    direction = np.cross(gvec, [0.0, 0.0, 1.0])
    if np.linalg.norm(direction) < 1.0e-12:
        direction = np.cross(gvec, [1.0, 0.0, 0.0])
    profile = _phase(gvectors, cell, index)
    field = jnp.asarray(direction[:, None, None, None] * profile[None])

    removed = longitudinal_field(field, gvectors, cell)
    assert np.abs(np.asarray(removed)).max() < 1.0e-10 * np.abs(
        np.asarray(field)).max()


def test_the_projection_kills_the_divergence(grid):
    """A general field, and the quantity the whole thing is named after."""
    cell, gvectors = grid
    rng = np.random.default_rng(0)
    scalar = _phase(gvectors, cell, (1, 1, 0)) + 0.3 * _phase(gvectors, cell, (0, 2, 1))
    field = np.stack([scalar, 0.7 * np.roll(scalar, 3, axis=0),
                      -0.4 * np.roll(scalar, 5, axis=1)])
    v_xc = jnp.asarray(np.concatenate([np.zeros((1,) + field.shape[1:]), field]))

    before = divergence(jnp.asarray(field), gvectors, cell)
    projected = project_source_free(v_xc, gvectors, cell)
    after = divergence(jnp.real(projected[1:]), gvectors, cell)

    assert float(jnp.sum(after ** 2)) < 1.0e-16 * float(jnp.sum(before ** 2))
    # The scalar component is not a field and must not move.
    assert np.abs(np.asarray(projected[0] - v_xc[0])).max() == 0.0


def test_the_uniform_component_survives(grid):
    """``G = 0`` is excluded, and it has to be: a uniform field is already
    divergence-free, and a Poisson equation cannot move a constant."""
    cell, gvectors = grid
    uniform = jnp.asarray(np.broadcast_to(
        np.array([0.3, -0.2, 0.5])[:, None, None, None],
        (3,) + tuple(gvectors.grid),
    ).copy())
    removed = longitudinal_field(uniform, gvectors, cell)
    assert np.abs(np.asarray(removed)).max() < 1.0e-12


def test_the_tangent_is_finite_at_g_zero(grid):
    """The ``0/0`` at ``G = 0`` is masked rather than divided and repaired, so
    a derivative taken through the potential has to come back finite.

    This is the trap ``CLAUDE.md`` lists first, and the failure it produces is
    not an error: ``jnp.where`` after a division leaves the *primal* right and
    the *tangent* a nan, which then reaches every response built on the
    potential and nothing before it complains.
    """
    import jax

    cell, gvectors = grid
    profile = _phase(gvectors, cell, (1, 0, 1))
    field = jnp.asarray(np.stack([profile, 0.5 * profile, -profile]))

    def total(scale):
        removed = longitudinal_field(scale * field, gvectors, cell)
        return jnp.sum(removed ** 2)

    tangent = jax.grad(total)(1.0)
    assert np.isfinite(float(tangent))


def test_a_spin_spiral_is_refused():
    """``nosource`` on a spiral: the guard fires before anything is built."""
    from types import SimpleNamespace

    from defuqumat.scf.sourcefree import refuse_source_free

    system = SimpleNamespace(noncolin=True, spiral_q=(0.0, 0.0, 0.25), nosource=True)
    with pytest.raises(NotImplementedError, match="spin spiral"):
        refuse_source_free(system)


def test_the_source_free_field_of_a_spiral_is_not_a_spiral(grid):
    """Why the spiral refusal is permanent: the projection leaves the ansatz.

    The box is read as four primitive cells along ``z`` and the field as a
    spiral at ``q = b3/4`` of the primitive cell, which is one step of the box
    along ``z``: ``B_+ = B_x + i B_y = exp(i (G - q).r)`` with ``G`` along
    ``x``, ``B_z = 0``. A spiral's ``B_z`` is lattice periodic in the primitive
    cell. The projected field's is not: its longitudinal part has a ``z``
    component at ``G - q``, which changes sign under a translation by half
    the box, so no spiral carries it.
    """
    cell, gvectors = grid
    n = gvectors.grid
    assert n[2] % 4 == 0
    coords = np.stack(np.meshgrid(
        np.arange(n[0]) / n[0], np.arange(n[1]) / n[1], np.arange(n[2]) / n[2],
        indexing="ij",
    ))
    phase = 2.0 * np.pi * (1 * coords[0] - 1 * coords[2])   # (G - q) . r
    field = np.stack([np.cos(phase), np.sin(phase), np.zeros_like(phase)])
    v_xc = jnp.asarray(np.concatenate([np.zeros((1,) + phase.shape), field]))

    projected = np.asarray(jnp.real(project_source_free(v_xc, gvectors, cell)))
    b_z = projected[3]
    assert np.abs(b_z).max() > 0.1  # the longitudinal part has a z component
    # A primitive cell is a quarter of the box along z; the spiral's B_z would
    # repeat under that translation, and this one does not.
    shifted = np.roll(b_z, n[2] // 4, axis=2)
    assert np.abs(shifted - b_z).max() > 0.1



# --- I108 and I109: a field that is not parallel to m, and a meta-GGA ---------
#
# The projection is linear and ``v_of_rho`` applies it to whatever field the
# functional built, so ``'grad_m'`` (whose field has a transverse part before any
# projection) and Tran-Blaha (whose potential is added before it) reach it the
# way the LDA does. What these tests measure is what each regime leaves behind.
#
# **Two places a divergence is read, and only one of them is exact.** On the
# dense G sphere, which is every component a Hamiltonian matrix element between
# two states inside the wavefunction cutoff reads, ``G . B(G) = 0`` to
# round-off. On the whole FFT box the projection leaves the corners alone (the
# module says why), so a residue stays there and falls with ``ecutrho``. Measured
# on the texture below (2026-10-19), the box residue over its value without the
# projection:
#
#     ecutrho (Ry)        40        80        160
#     PBE  'grad_m'    6.03e-3   1.70e-3   1.82e-4
#     PBE  'kubler'    4.92e-2   2.56e-2   1.02e-2
#     TB09 'grad_m'    2.98e-3   1.12e-3   5.29e-4
#     TB09 'kubler'    7.02e-2   4.93e-2   2.51e-2
#     PZ               3.58e-7   2.07e-8   3.42e-8
#
# ``'grad_m'`` falls faster than ``'kubler'`` because its field has no kink
# where ``|m|`` is small, which is the opposite of what the old refusal of
# ``'grad_m'`` under ``nosource`` implied.


def _texture(gvectors):
    """Two Gaussian moments, along z and along x, on a floor of charge.

    ``|m| < n`` everywhere and ``m`` turns between the two sites, so the field
    of every functional here has a longitudinal part to remove. ``tau`` is a
    Thomas-Fermi-like profile with ``|tau_m| < tau_0``, only so that Tran-Blaha
    has a kinetic energy density to read.
    """
    n = gvectors.grid
    x = np.stack(np.meshgrid(*[np.arange(k) / k for k in n], indexing="ij"))

    def gauss(centre, width=1.6):
        d = x - np.asarray(centre)[:, None, None, None]
        d -= np.round(d)
        return np.exp(-np.sum((10.0 * d) ** 2, axis=0) / width**2)

    g1, g2 = gauss((0.3, 0.3, 0.3)), gauss((0.7, 0.6, 0.55))
    rho = 0.003 + 0.05 * (g1 + g2)
    m = 0.03 * (np.array([0.0, 0.0, 1.0])[:, None, None, None] * g1
                + np.array([1.0, 0.0, 0.0])[:, None, None, None] * g2)
    tau0 = 0.3 * rho ** (5.0 / 3.0)
    return (np.concatenate([rho[None], m]),
            np.concatenate([tau0[None], 0.5 * tau0 * m / rho]))


def _sphere_divergence(field, gvectors, cell):
    d = divergence(jnp.asarray(field), gvectors, cell)
    return float(cell.volume / d.size * jnp.sum(d ** 2))


def _box_divergence(field, gvectors, cell):
    """``sum |G . B(G)|^2`` over every frequency of the box, not the sphere."""
    n = gvectors.grid
    f = np.fft.fftn(np.asarray(field), axes=(1, 2, 3))
    miller = np.stack(np.meshgrid(*[np.fft.fftfreq(k) * k for k in n],
                                  indexing="ij"), -1)
    g = miller @ np.asarray(cell.bg)
    return float(np.sum(np.abs(np.einsum("...i,i...->...", g, f)) ** 2))


def _pair(name, choice, ecut, cell):
    from defuqumat.scf.potential import v_of_rho
    from defuqumat.xc.functional import get_functional

    gvectors = generate_gvectors(cell, ecut)
    rho, tau = _texture(gvectors)
    functional = get_functional(name)
    options = dict(functional=functional, noncollinear_gga=choice)
    if functional.is_meta:
        options["tau"] = jnp.asarray(tau)
    off = v_of_rho(jnp.asarray(rho), gvectors, cell, **options)
    on = v_of_rho(jnp.asarray(rho), gvectors, cell, source_free=True, **options)
    return gvectors, off, on


@pytest.mark.parametrize("name, choice", [
    ("PBE", "grad_m"), ("TB09", "kubler"), ("TB09", "grad_m"),
])
def test_the_sphere_is_divergence_free_for_a_field_not_parallel_to_m(grid, name, choice):
    """I108 and I109: ``int |div B|^2`` on the sphere, with the projection off
    (measured 0.405, 0.209 and 0.141 at 40 Ry) and on (1e-31, round-off)."""
    cell, _ = grid
    gvectors, off, on = _pair(name, choice, 40.0, cell)
    before = _sphere_divergence(jnp.real(off.v_scf[1:]), gvectors, cell)
    after = _sphere_divergence(jnp.real(on.v_scf[1:]), gvectors, cell)
    assert before > 0.1
    assert after < 1.0e-25 * before
    # The scalar part, the energy and Tran-Blaha's c are not the field.
    assert np.abs(np.asarray(on.v_scf[0] - off.v_scf[0])).max() == 0.0
    assert float(on.etxc) == float(off.etxc)
    assert float(on.meta_c) == float(off.meta_c)


@pytest.mark.parametrize("name, choice, at40, at80", [
    ("PBE", "grad_m", 6.03e-3, 1.70e-3),
    ("TB09", "grad_m", 2.98e-3, 1.12e-3),
    ("PBE", "kubler", 4.92e-2, 2.56e-2),
])
def test_the_box_residue_falls_with_the_cutoff(grid, name, choice, at40, at80):
    """The convergence half: what is left on the box corners, over its value
    without the projection, at 40 and 80 Ry, to 2 per cent of the measured
    numbers (the table above). It has to fall, and ``'grad_m'`` has to sit
    below ``'kubler'`` at the same cutoff."""
    cell, _ = grid
    fractions = []
    for ecut in (40.0, 80.0):
        gvectors, off, on = _pair(name, choice, ecut, cell)
        fractions.append(_box_divergence(jnp.real(on.v_scf[1:]), gvectors, cell)
                         / _box_divergence(jnp.real(off.v_scf[1:]), gvectors, cell))
    assert fractions[0] == pytest.approx(at40, rel=0.02)
    assert fractions[1] == pytest.approx(at80, rel=0.02)
    assert fractions[1] < 0.6 * fractions[0]


def test_grad_m_and_kubler_project_to_the_same_field_when_m_is_collinear(grid):
    """I108's second check: ``m`` along one tilted axis with no node, where the
    two noncollinear gradient corrections are the same functional, so their
    projected fields must agree; and with a node, where ``'kubler'`` has a kink
    ``'grad_m'`` does not, they must not (the comparison can fail)."""
    from defuqumat.scf.potential import v_of_rho
    from defuqumat.xc.functional import get_functional

    cell, _ = grid
    gvectors = generate_gvectors(cell, 40.0)
    rho, _ = _texture(gvectors)
    axis = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    functional = get_functional("PBE")

    def projected(mz, choice):
        rho4 = np.concatenate([rho[:1], axis[:, None, None, None] * mz[None]])
        return np.asarray(jnp.real(v_of_rho(
            jnp.asarray(rho4), gvectors, cell, functional=functional,
            source_free=True, noncollinear_gga=choice).v_scf))

    nodeless = 0.4 * (rho[0] - 0.002)
    assert (nodeless > 0).all()
    one, two = projected(nodeless, "grad_m"), projected(nodeless, "kubler")
    assert np.abs(two - one).max() < 1.0e-10
    # The projection did something: the field is no longer along the axis.
    along = np.einsum("i,i...->...", axis, one[1:])
    transverse = one[1:] - axis[:, None, None, None] * along[None]
    assert np.abs(transverse).max() > 1.0e-4

    noded = nodeless - 0.4 * 0.003 * 2.0
    assert (noded > 0).any() and (noded < 0).any()
    one, two = projected(noded, "grad_m"), projected(noded, "kubler")
    assert np.abs(two - one).max() > 1.0e-6


def test_the_doors_now_open_and_the_ones_that_stay_shut():
    """I108 and I109 lifted; PAW, a collinear run and a spiral still refused,
    each by its own message, at the guard itself."""
    from types import SimpleNamespace

    from defuqumat.scf.potential import refuse_noncollinear_gga, resolve_noncollinear_gga
    from defuqumat.scf.sourcefree import refuse_source_free
    from defuqumat.xc.functional import get_functional

    pbe, tb09 = get_functional("PBE"), get_functional("TB09")
    noncollinear = SimpleNamespace(noncolin=True, spiral_q=None, nosource=True)
    refuse_source_free(noncollinear, tb09)
    refuse_source_free(noncollinear, get_functional("BJ06"))
    assert refuse_noncollinear_gga("grad_m", 4, pbe, False, True) == "grad_m"
    assert refuse_noncollinear_gga("grad_m", 4, tb09, False, True) == "grad_m"
    # Nothing asked for under nosource stays 'kubler', as every nosource run
    # with a gradient functional ran before I108 (the user's question, recorded
    # in notes/incompat/I108.md); an explicit choice is kept.
    assert resolve_noncollinear_gga(None, 4, pbe, False, True) == "kubler"
    assert resolve_noncollinear_gga(None, 4, tb09, False, True) == "kubler"
    assert resolve_noncollinear_gga("grad_m", 4, pbe, False, True) == "grad_m"

    with pytest.raises(NotImplementedError, match="PAW"):
        refuse_source_free(noncollinear, tb09, (SimpleNamespace(is_paw=True),))
    with pytest.raises(NotImplementedError, match="noncollinear run"):
        refuse_source_free(SimpleNamespace(noncolin=False), tb09)
    with pytest.raises(NotImplementedError, match="spin spiral"):
        refuse_source_free(SimpleNamespace(noncolin=True, spiral_q=(0, 0, 0.25)), tb09)
