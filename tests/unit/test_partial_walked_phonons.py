"""A partial dynamical matrix from a displacement response solved for every atom.

``notes/INCOMPATIBILITIES.md`` I54. The Raman tensors keep the displacement
response (``keep_internals``), solved for every atom, and
:func:`~defuqumat.response.phonon.dynamical_matrix` assembles from it. With
``atoms=`` the assembly reads row ``r`` of every per-mode store as the response
to ``atoms[r]``, so a whole response has to be cut to those atoms' rows first.
The walked route (a k-chunk at a time) refused the subset; the whole-k route
took it and read atom 0's response for ``atoms[0]``, with no error.

These tests need no SCF: the row map, the refusals that stay and the call into
the walked assembly. The numbers, the subset against the same response's whole
matrix bit for bit and against a fresh subset solve, are
``tests/regression/test_streamed_phonons.py::
test_a_partial_matrix_from_a_kept_response_is_its_rows`` (slow).
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat import Calculator
from defuqumat.response.phonon import (
    DisplacementResponse,
    _response_rows,
    dynamical_matrix,
)


def _calculation(case):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Calculator.from_file(
            f"tests/data/qe/{case}.in", pseudo_dir="tests/data/pseudo",
            announce=False).calculation


def _response(nat, walked=None):
    """Rows tagged by atom: every entry of row ``a`` is ``a``."""
    dpsi = np.empty((nat, 3), dtype=object)
    for a in range(nat):
        for c in range(3):
            dpsi[a, c] = np.full((1, 2, 3, 4), float(a))
    tag = np.arange(nat, dtype=float)[:, None, None, None]
    drho = np.broadcast_to(tag, (nat, 3, 1, 2)).copy()
    return DisplacementResponse(
        dpsi=dpsi, drho=drho, history=[], converged=True,
        bare=np.broadcast_to(tag, (nat, 3, 1, 5)).copy(),
        extras={"dvscf": drho.copy(), "onecentre": None, "dbecsum": None},
        walked=walked)


def test_a_whole_response_is_cut_to_the_subsets_rows():
    response = _response(3)
    dpsi, drho, bare, extras = _response_rows(response, (2, 0), 3)
    assert [dpsi[r, 0][0, 0, 0, 0] for r in range(2)] == [2.0, 0.0]
    assert drho[:, :, 0, 0].tolist() == [[2.0] * 3, [0.0] * 3]
    assert bare[:, 0, 0, 0].tolist() == [2.0, 0.0]
    assert extras["dvscf"][:, 0, 0, 0].tolist() == [2.0, 0.0]
    assert extras["onecentre"] is None


def test_a_response_on_the_device_is_cut_too():
    """The whole-k loop hands back jax arrays, which refuse a list as an index."""
    response = _response(3)
    response = DisplacementResponse(
        dpsi=response.dpsi, drho=jnp.asarray(response.drho), history=[],
        bare=jnp.asarray(response.bare),
        extras={"dvscf": jnp.asarray(response.drho), "onecentre": None,
                "dbecsum": None})
    _, drho, bare, extras = _response_rows(response, (2,), 3)
    assert float(drho[0, 0, 0, 0]) == 2.0
    assert float(bare[0, 0, 0, 0]) == 2.0
    assert float(extras["dvscf"][0, 0, 0, 0]) == 2.0


def test_a_whole_cell_or_a_subsets_own_response_is_taken_as_it_is():
    response = _response(3)
    assert _response_rows(response, None, 3)[1] is response.drho
    own = _response(1)
    assert _response_rows(own, (2,), 3)[1] is own.drho


def test_a_response_whose_rows_match_no_atoms_is_refused():
    with pytest.raises(ValueError, match="cannot be matched"):
        _response_rows(_response(2), (0,), 3)
    with pytest.raises(ValueError, match="needs 3"):
        _response_rows(_response(2), None, 3)


class _Walked:
    """The walked assembly's interface, recording what it was asked for."""

    def __init__(self, nat):
        self.atoms = tuple(range(nat))
        self.asked = None

    def force_constants(self, drho, extras, on_row=None, atoms=None):
        self.asked = atoms
        nat = len(self.atoms)
        rows = nat if atoms is None else len(atoms)
        return np.eye(rows * 3, nat * 3).reshape(rows, 3, nat, 3)


def test_a_walked_response_assembles_the_subsets_rows():
    """The refusal this replaces raised here: the subset goes to the walk."""
    calculation = _calculation("alas-epsilon-us-unshifted-nosym")
    assert not calculation.use_symmetry
    walked = _Walked(2)
    response = _response(2, walked=walked)
    phonons = dynamical_matrix(calculation, None, jnp.zeros((8, 4)), None,
                               response=response, atoms=(1,))
    assert walked.asked == (1,)
    assert phonons.matrix.shape == (1, 3, 1, 3)
    assert phonons.atoms == (1,)
    assert np.all(phonons.induced_density == 1.0)


def test_a_walked_response_still_refuses_a_symmetrised_subset():
    calculation = _calculation("si-epsilon")
    assert calculation.use_symmetry
    with pytest.raises(NotImplementedError, match="nosym"):
        dynamical_matrix(calculation, None, jnp.zeros((10, 4)), None,
                         response=_response(2, walked=_Walked(2)), atoms=(1,))


@pytest.mark.parametrize("case, atoms", [("alas-raman-wedge", (0,)),
                                         ("h-chain-afm", None)])
def test_a_regime_the_walk_lacks_takes_the_whole_k_route(case, atoms):
    """A closed subset with symmetry (I51) and ``nspin = 2`` were measured on
    the whole-k route only. They used to be refused where the response walks;
    since memory mode became the CPU default (2026-10-21) they take the whole-k
    route instead, on a device store without a word, and a walked response
    handed in is solved again there rather than assembled."""
    from defuqumat.response.efield import _streams
    from defuqumat.response.phonon import _walk_lacks

    calculation = _calculation(case)
    assert calculation.memory_mode == "memory"
    assert calculation.system.kpoints.nk > 1
    lacks = _walk_lacks(calculation, atoms)
    assert lacks is not None
    states = jnp.zeros((calculation.nspin, calculation.system.kpoints.nk, 2, 4))
    assert _streams(calculation, states, False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not _streams(calculation, states, False, whole_k=lacks)
    with pytest.warns(RuntimeWarning, match="whole-k route"):
        assert not _streams(calculation, np.asarray(states), False,
                            whole_k=lacks)
