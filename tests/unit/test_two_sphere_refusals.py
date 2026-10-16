"""I82: what the phonon at ``q != 0`` refuses, checked without a ground state.

Every refusal of :func:`~defuqumat.response.phononq.require_a_two_sphere_regime`
is a function of the calculation's flags alone, so it is driven here with a
stand-in object at gate speed. ``nspin = 2`` collinear runs since I82 (its
number is ``tests/regression/test_phonons_at_q.py::
test_a_magnetic_insulator_at_half_b3_is_the_doubled_cell_at_gamma``); each
sibling refusal must still fire, and so must the ones that stay for a spin
axis: a spinor, a metal, the k-chunked route, and the electron-phonon coupling.
A gradient-corrected functional is refused at every ``nspin``: the ``q``
route's XC kernel has no ``iq`` in its gradients (``dgradcorr.f90``'s ``xq``).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from defuqumat.response.phononq import require_a_two_sphere_regime


def _calculation(**changes):
    """A stand-in carrying exactly the flags the guard reads, every one off."""
    system = dict(nspin=1, noncolin=False, spiral_q=None, occupations="fixed")
    flags = dict(is_ultrasoft=False, rho_core_g=None, use_symmetry=False,
                 functional=SimpleNamespace(is_meta=False, is_gradient=False),
                 source_free=False,
                 is_hubbard=False)
    for key, value in changes.items():
        (system if key in system else flags)[key] = value
    return SimpleNamespace(system=SimpleNamespace(**system), **flags)


def test_the_stand_in_passes():
    require_a_two_sphere_regime(_calculation(), None)


def test_a_collinear_spin_polarized_insulator_passes():
    """The lift itself: two channels, no spinor, fixed occupations."""
    require_a_two_sphere_regime(_calculation(nspin=2), None)


def test_an_unpolarized_metal_still_passes():
    require_a_two_sphere_regime(_calculation(occupations="smearing"), None)


def test_a_core_correction_at_one_channel_passes():
    """I80's lift; at ``nspin = 2`` the core is the sibling refusal below."""
    require_a_two_sphere_regime(_calculation(rho_core_g=np.zeros(1)), None)


@pytest.mark.parametrize("changes, match", [
    (dict(nspin=4, noncolin=True), "spinor"),
    (dict(nspin=2, occupations="smearing"), "metal"),
    (dict(is_ultrasoft=True), "ultrasoft"),
    (dict(nspin=2, rho_core_g=np.zeros(1)), "nonlinear core"),
    (dict(use_symmetry=True), "small group"),
    (dict(spiral_q=(0.0, 0.0, 0.1)), "spin spiral"),
    (dict(functional=SimpleNamespace(is_meta=True, is_gradient=True)),
     "meta-GGA"),
    (dict(functional=SimpleNamespace(is_meta=False, is_gradient=True)),
     "gradient-corrected"),
    (dict(nspin=2, functional=SimpleNamespace(is_meta=False, is_gradient=True)),
     "gradient-corrected"),
    (dict(source_free=True), "nosource"),
    (dict(is_hubbard=True), "Hubbard"),
], ids=["noncolin", "lsda-metal", "ultrasoft", "lsda-nlcc", "symmetry", "spiral",
        "meta", "gga", "lsda-gga", "nosource", "hubbard"])
def test_each_sibling_refusal_still_fires(changes, match):
    with pytest.raises(NotImplementedError, match=match):
        require_a_two_sphere_regime(_calculation(**changes), None)


def test_the_k_chunked_route_refuses_two_channels():
    from defuqumat.response.phononq import _require_one_channel_when_walked

    _require_one_channel_when_walked(_calculation(), streamed=True)
    _require_one_channel_when_walked(_calculation(nspin=2), streamed=False)
    with pytest.raises(NotImplementedError, match="k-chunked"):
        _require_one_channel_when_walked(_calculation(nspin=2), streamed=True)


def test_a_spin_polarized_coupling_is_refused():
    """``elphsum_simple`` counts one channel; the guard fires before anything
    reads the ground state."""
    from defuqumat.response.elph import electron_phonon_at_q

    calculation = SimpleNamespace(system=SimpleNamespace(
        occupations="smearing", nspin=2, noncolin=False))
    with pytest.raises(NotImplementedError, match="nspin = 2"):
        electron_phonon_at_q(calculation, None, None, None, q=(0.0, 0.0, 0.5))


@pytest.mark.parametrize("name", ["PBE", "PBESOL", "REVPBE"])
@pytest.mark.parametrize("nspin", [1, 2])
def test_a_real_gradient_corrected_functional_is_refused(name, nspin):
    """The stand-in's flag is the real one: ``Functional.is_gradient``.

    Measured before the refusal (the I82 verifier, D22): the supercell identity
    holding to 1.52e-5 Ry/bohr^2 under the LDA read 7.97e-3 and 60.55 cm^-1
    under PBE on the magnetic H3 chain at b3/2, and 1.33e-2 and 77 cm^-1 on a
    dimerized H2 chain at nspin = 1, with no error raised.
    """
    from defuqumat.xc.functional import get_functional

    functional = get_functional(name)
    assert functional.is_gradient and not functional.is_meta
    with pytest.raises(NotImplementedError, match="gradient-corrected"):
        require_a_two_sphere_regime(
            _calculation(nspin=nspin, functional=functional), None)


@pytest.mark.parametrize("nspin", [1, 2])
def test_a_real_lda_functional_passes(nspin):
    from defuqumat.xc.functional import get_functional

    functional = get_functional("PZ")
    assert not functional.is_gradient
    require_a_two_sphere_regime(_calculation(nspin=nspin, functional=functional),
                                None)
