"""The ``Gamma`` phonon of an ultrasoft or PAW metal: its pair weight, and what it refuses.

The dynamical matrix of such a metal is measured against ``ph.x`` in
``tests/regression/test_phonons.py`` (``notes/incompat/I86.md``). What is here
needs no ground state:

* :func:`~defuqumat.response.phonon.pair_weights` is ``compute_weight``'s
  ``wgg``, symmetric in its two bands, the smeared projector's own weight
  without its ``alpha_pv`` piece, and the occupied-block mask in the
  insulating limit;
* the regimes beside the measured one each still refuse, by name, before any
  state is read: symmetry on, ``nspin = 2``, a subset of the atoms, and the
  route that walks the k axis a chunk at a time.
"""

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.response import phonon
from defuqumat.response.phonon import dynamical_matrix, pair_weights
from defuqumat.response.sternheimer import Smearing
from defuqumat.scf import Calculation
from defuqumat.scf.occupations import wgauss
from defuqumat.system import build_system

pytestmark = pytest.mark.unit

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


def _calculation(case, edit=None, tmp_path=None):
    text = (CASES / f"{case}.in").read_text()
    if edit is not None:
        text = edit(text)
    deck = tmp_path / f"{case}-edited.in"
    deck.write_text(text)
    system = build_system(read_pw_input(deck))
    pseudos = tuple(read_upf(PSEUDO / sp.pseudo_file)
                    for sp in system.structure.species)
    return Calculation(system, pseudos)


def _solver(eps, ef=0.3, degauss=0.05, ngauss=-1):
    return SimpleNamespace(smearing=Smearing(ef, degauss, ngauss),
                           eigenvalues=jnp.asarray(eps)[None, None])


EPS = np.array([-0.40, -0.10, 0.27, 0.30, 0.30, 0.34, 0.70])


def test_the_pair_weight_is_symmetric_and_is_the_projectors_own():
    """``W_mn = f_n (1 - theta_mn) + f_m theta_mn``, ``theta_mn + theta_nm = 1``.

    Symmetric, which is what lets the occupied block be the hermitian
    ``-1/2 S' W`` and the multipliers' tangent hermitian; and equal to the
    smeared projector's ``wwg`` with the ``alpha_pv`` term taken out, which is
    the weight the Sternheimer solve the block completes was built with.
    """
    solver = _solver(EPS)
    weights = np.asarray(pair_weights(solver))[0, 0]
    assert np.abs(weights - weights.T).max() < 1e-15
    f = np.asarray(wgauss((0.3 - EPS) / 0.05, -1))
    assert np.allclose(np.diag(weights), f, atol=1e-15)
    step = np.asarray(wgauss((EPS[:, None] - EPS[None, :]) / 0.05, 0))
    expected = f[None, :] * (1.0 - step) + f[:, None] * step
    assert np.abs(weights - expected).max() < 1e-15


def test_the_pair_weight_is_the_occupied_mask_when_the_smearing_is_narrow():
    """The insulating limit: 1 on the occupied block and 0 elsewhere, which is
    :func:`~defuqumat.response.phonon.occupied_pairs`' mask and the ``-1/2 S'``
    the insulator's block is validated with."""
    eps = np.array([-0.4, -0.2, -0.1, 0.5, 0.9])
    weights = np.asarray(pair_weights(_solver(eps, ef=0.2, degauss=1e-3)))[0, 0]
    occupied = eps < 0.2
    mask = occupied[:, None] & occupied[None, :]
    assert np.abs(weights - mask).max() < 1e-12


def test_an_insulator_has_no_pair_weight():
    assert pair_weights(SimpleNamespace(smearing=None)) is None


def test_a_degenerate_pair_takes_half_of_each_occupation():
    """``theta = 1/2`` at ``eps_m = eps_n``: the pair carries the mean of the
    two occupations, which at an exact degeneracy is the shared one."""
    solver = _solver(EPS)
    weights = np.asarray(pair_weights(solver))[0, 0]
    assert weights[3, 4] == pytest.approx(
        float(wgauss((0.3 - 0.30) / 0.05, -1)), abs=1e-15)


def _metal(text):
    return text.replace(
        "    ecutwfc = 20.0",
        "    occupations = 'smearing'\n    degauss = 0.02\n    ecutwfc = 20.0")


def test_an_ultrasoft_metal_with_symmetry_is_refused(tmp_path):
    """The measured cells are ``nosym``; a wedge's pair-weighted block is not."""
    calculation = _calculation("si-epsilon-us", _metal, tmp_path)
    assert calculation.use_symmetry
    with pytest.raises(NotImplementedError, match="nosym run only"):
        dynamical_matrix(calculation, None, jnp.zeros((1, 1, 1)), None)


def test_a_spin_polarized_ultrasoft_metal_is_refused(tmp_path):
    def edit(text):
        return text.replace(
            "nosym=.true.", "nosym=.true., nspin=2, starting_magnetization(1)=0.2")
    calculation = _calculation("al2-metal-us", edit, tmp_path)
    assert calculation.nspin == 2
    with pytest.raises(NotImplementedError, match="spin-polarized metal"):
        dynamical_matrix(calculation, None, jnp.zeros((2, 1, 1)), None)


def test_a_subset_of_an_ultrasoft_metal_is_refused(tmp_path):
    calculation = _calculation("al2-metal-us", None, tmp_path)
    with pytest.raises(NotImplementedError, match="partial dynamical matrix"):
        dynamical_matrix(calculation, None, jnp.zeros((1, 1, 1)), None,
                         atoms=(0,))


def test_the_walked_route_refuses_an_ultrasoft_metal(tmp_path):
    """``chunked_phonon`` has its own copies of the tangents, without ``W``."""
    calculation = _calculation("al2-metal-paw", None, tmp_path)
    with pytest.raises(NotImplementedError, match="k-chunk at a time"):
        phonon._require_no_walked_overlap_metal(calculation)


def test_the_measured_regime_passes_the_guard(tmp_path):
    """The guard that fired before I86 is silent on the cell it is measured on,
    and a norm-conserving metal and an ultrasoft insulator never reached it."""
    for case in ("al2-metal-us", "al2-metal-paw", "al2-metal"):
        phonon._require_a_measured_overlap_metal(
            _calculation(case, None, tmp_path))
    # the walked route stays open to a norm-conserving metal
    phonon._require_no_walked_overlap_metal(
        _calculation("al2-metal", None, tmp_path))


def test_the_third_derivative_still_refuses_an_ultrasoft_metal(tmp_path):
    """The Raman tensor keeps the old guard behind its own metal refusal."""
    from defuqumat.response.nonlinear import raman_tensors

    calculation = _calculation("al2-metal-us", None, tmp_path)
    with pytest.raises(NotImplementedError, match="metal"):
        raman_tensors(calculation, None)
    with pytest.raises(NotImplementedError, match="third derivative"):
        phonon._require_a_moving_overlap_regime(calculation)
