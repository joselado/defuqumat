"""``mixing_space = 'g'`` on a magnetic spin spiral (I74).

A spiral's transverse pair is the rotating-frame ``m'_+``, whose coefficients
lie on the sphere displaced by ``q`` (``OPEN.md`` Part XXI item 1), so the
resident :class:`~defuqumat.scf.mixing.SphereLayout` cuts it at every step;
:class:`~defuqumat.scf.mixing.SpiralSphereLayout` keeps it there, or on the
whole box where the run builds no displaced set. What is pinned here: a field
with the spiral's band limits goes through the layout unchanged, smooth part and
shell, while the resident layout loses the pair's shell; the stored flat dot is
the real-space dot; at dual 4 with the whole-box pair the layout is the
real-space mixer to round-off, with ``beta_mag``, Kerker and local-TF; and a
converged run lands on the real-space fixed point (slow, two cells).
"""

from functools import lru_cache
import warnings

import numpy as np
import pytest

from defuqumat.basis.builder import build_basis
from defuqumat.basis.gvectors import displaced_gvectors
from defuqumat.io.pwin import parse_pw_input
from defuqumat.scf.mixing import (
    SphereLayout,
    SpiralSphereLayout,
    get_mixer,
    kerker_preconditioner,
    kerker_preconditioner_g,
    local_tf_preconditioner,
    local_tf_preconditioner_g,
)
from defuqumat.system import build_system
from defuqumat.system.spiral import spiral_cartesian
from tests.conftest import GENERATED


@lru_cache(maxsize=2)
def _pieces(name):
    """The system, basis and cartesian ``q`` (1/bohr) of an input, built without an SCF."""
    system = build_system(parse_pw_input((GENERATED / name).read_text()))
    basis = build_basis(system)
    q = np.asarray(spiral_cartesian(system.spiral_q, system.cell)) * float(system.cell.tpiba)
    return system, basis, q


def _displaced_layout(name="co-helix4-spiral.in"):
    """The layout on the displaced sets, on a cell with a separate smooth grid."""
    system, basis, q = _pieces(name)
    dense, smooth = displaced_gvectors(system.cell, basis.dense, -q, smooth=basis.smooth)
    layout = SpiralSphereLayout(basis.dense, basis.ngms, system.cell,
                                (4,) + basis.dense.grid, np.asarray(dense.fft_index),
                                smooth.ngm)
    return system, basis, layout


def _whole_box_layout(name="h-chain-spiral.in"):
    """The layout a norm-conserving spiral at dual 4 gets: ``m'_+`` on the whole box."""
    system, basis, _ = _pieces(name)
    points = int(np.prod(basis.dense.grid))
    layout = SpiralSphereLayout(basis.dense, basis.ngms, system.cell,
                                (4,) + basis.dense.grid, np.arange(points), points)
    return system, basis, layout


def _stored(layout, rng):
    """A random stored block, its padding zero as a step keeps it."""
    stored = rng.normal(size=layout.stored_shape)
    stored[::3, layout.ngms:] = 0.0
    stored[1:3, layout.npair:] = 0.0
    return stored


def _shell(layout, rng):
    return 1e-2 * (rng.normal(size=layout.nshell) + 1j * rng.normal(size=layout.nshell))


def _field(layout, rng, shell=True):
    return np.asarray(layout.field(_stored(layout, rng),
                                   _shell(layout, rng) if shell else None))


def test_a_spiral_field_goes_through_the_layout_unchanged():
    """``m'_+`` on the displaced spheres, smooth part and shell, comes back to round-off.

    The control: the resident layout's own round trip of the same field keeps
    the charge and ``m_z`` and loses a part of ``m_x`` and ``m_y`` of the
    order of the field, which is the cut the refusal was about.
    """
    system, basis, layout = _displaced_layout()
    assert layout.npair != layout.ngms and layout.npair_shell > 0
    rng = np.random.default_rng(1)
    stored, shell = _stored(layout, rng), _shell(layout, rng)
    field = np.asarray(layout.field(stored, shell))
    again, again_shell = layout.pack(layout.forward(field))
    np.testing.assert_allclose(again, stored, rtol=0, atol=1e-12 * np.max(np.abs(stored)))
    np.testing.assert_allclose(again_shell, shell, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(layout.field(again, again_shell)), field,
                               rtol=0, atol=1e-12 * np.max(np.abs(field)))

    resident = SphereLayout(basis.dense, basis.ngms, system.cell, layout.shape)
    coefficients = resident.forward(field)
    head, tail = resident.pack(np.asarray(coefficients))
    cut = np.asarray(resident.field(head, tail)) - field
    scale = np.max(np.abs(field[1:3]))
    assert np.max(np.abs(cut[::3])) < 1e-12 * np.max(np.abs(field))
    assert np.max(np.abs(cut[1:3])) > 1e-3 * scale


def test_the_stored_dot_is_the_real_space_dot():
    """Parseval on both packings, the pair's with no conjugate partner: ``N Re sum conj(c) c``."""
    _, _, layout = _displaced_layout()
    rng = np.random.default_rng(2)
    f, g = _field(layout, rng, shell=False), _field(layout, rng, shell=False)
    stored = float(layout.stored_of(f).ravel() @ layout.stored_of(g).ravel())
    assert stored == pytest.approx(float(np.sum(f * g)), rel=1e-12)


def test_the_whole_box_pair_keeps_any_transverse_field():
    """On the hydrogen chain the displaced sphere does not fit the box, and nothing is cut."""
    system, basis, _ = _pieces("h-chain-spiral.in")
    with pytest.raises(NotImplementedError, match="share a box index"):
        displaced_gvectors(system.cell, basis.dense, -_pieces("h-chain-spiral.in")[2],
                           smooth=basis.smooth)
    _, _, layout = _whole_box_layout()
    rng = np.random.default_rng(3)
    field = _field(layout, rng, shell=False)
    field[1:3] = rng.normal(size=field[1:3].shape)
    np.testing.assert_allclose(np.asarray(layout.field(layout.stored_of(field)))[1:3],
                               field[1:3], rtol=0, atol=1e-12 * np.max(np.abs(field)))


@pytest.mark.parametrize("beta_mag", [None, 0.2])
def test_at_dual_four_the_whole_box_layout_is_the_real_space_mixer(beta_mag):
    """No shell and nothing cut, so four Anderson steps match real space to round-off.

    With ``beta_mag`` the magnetic step scales rows 1 to 3 of the padded block,
    which is ``m_z`` and ``m'_+``, and the result is still the real-space one.
    """
    from defuqumat.scf.driver import _mix

    _, _, layout = _whole_box_layout()
    rng = np.random.default_rng(4)
    pairs = [(_field(layout, rng, shell=False), _field(layout, rng, shell=False))
             for _ in range(4)]
    real = get_mixer("anderson", beta=0.5, beta_mag=beta_mag)
    sphere = get_mixer("anderson", beta=0.5, beta_mag=beta_mag)
    real.shape = layout.shape
    sphere.layout, sphere.shape = layout, layout.stored_shape
    for rho_in, rho_out in pairs:
        a = np.asarray(_mix(real, rho_in, rho_out, (), ())[0])
        b = np.asarray(_mix(sphere, rho_in, rho_out, (), ())[0])
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-12 * np.max(np.abs(a)))


def test_kerker_and_local_tf_on_the_spiral_layout_are_the_real_space_operators():
    """The charge row screened, the padding left at zero, the same operator as on the grid."""
    system, basis, layout = _whole_box_layout()
    cell = system.cell
    rng = np.random.default_rng(5)
    residual = 1e-3 * (_field(layout, rng, shell=False) - _field(layout, rng, shell=False))
    stored = layout.stored_of(residual).ravel()

    real = kerker_preconditioner(basis.dense, cell, layout.shape, beta=0.6, nelec=1.0)
    sphere = kerker_preconditioner_g(layout, cell, beta=0.6, nelec=1.0)
    expected = np.asarray(real(residual.ravel())).reshape(layout.shape)
    stepped = sphere(stored).reshape(layout.stored_shape)
    assert not np.any(stepped[::3, layout.ngms:])
    np.testing.assert_allclose(np.asarray(layout.field(stepped)), expected, rtol=0,
                               atol=1e-12 * np.max(np.abs(expected)))

    ripples = 1e-3 * _stored(layout, rng)
    ripples[0, 0] = 0.05 * np.sqrt(layout.points)
    density = np.asarray(layout.field(ripples))
    assert np.all(density[0] > 0.0)
    real = local_tf_preconditioner(basis.dense, cell, layout.shape, beta=0.6)
    sphere = local_tf_preconditioner_g(layout, basis.dense, cell, beta=0.6)
    expected = np.asarray(real(residual.ravel(), density.ravel())).reshape(layout.shape)
    stepped = sphere(stored, layout.stored_of(density).ravel())
    got = np.asarray(layout.field(stepped))
    np.testing.assert_allclose(got, expected, rtol=0,
                               atol=1e-9 * np.max(np.abs(expected)))


def test_a_history_written_at_another_q_is_tagged_apart():
    """The tag names the displaced sphere, so a resume at another ``q`` drops the history."""
    system, basis, layout = _displaced_layout()
    _, _, box = _whole_box_layout()
    assert layout.tag.startswith("g-spiral-") and layout.tag != box.tag
    q = _pieces("co-helix4-spiral.in")[2]
    dense, smooth = displaced_gvectors(system.cell, basis.dense, -0.5 * q,
                                       smooth=basis.smooth)
    other = SpiralSphereLayout(basis.dense, basis.ngms, system.cell, layout.shape,
                               np.asarray(dense.fft_index), smooth.ngm)
    assert other.tag != layout.tag


# --- whole runs ----------------------------------------------------------------


def _run(name, pseudo_dir, space, **options):
    from defuqumat.pseudo import read_upf
    from defuqumat.scf import run_scf

    system = build_system(parse_pw_input((GENERATED / name).read_text()))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_scf(system, pseudos, mixing_space=space, verbose=False, **options)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["h-chain-spiral.in", "o-chain-spiral-us.in"])
def test_a_spiral_mixed_in_g_reaches_the_real_space_fixed_point(name, pseudo_dir):
    """The same fixed point as ``mixing_space = 'r'``: energy and density.

    The hydrogen chain is norm-conserving at dual 4, so ``m'_+`` is kept on the
    whole box; the oxygen chain is ultrasoft at dual 8, so it is kept on the
    displaced smooth sphere and its shell is the displaced augmentation charge
    rebuilt from the mixed ``becsum``. Numbers in ``notes/incompat/I74.md``.
    """
    results = {space: _run(name, pseudo_dir, space) for space in ("r", "g")}
    assert results["r"].converged and results["g"].converged
    assert results["g"].total_energy == pytest.approx(results["r"].total_energy,
                                                      abs=1e-9)
    rho_r, rho_g = (np.asarray(results[s].density) for s in ("r", "g"))
    assert np.max(np.abs(rho_g - rho_r)) < 1e-5 * np.max(np.abs(rho_r))
