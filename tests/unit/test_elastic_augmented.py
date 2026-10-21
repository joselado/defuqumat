"""I87: the elastic constants of an ultrasoft or PAW dataset, and what still refuses.

``require_a_measured_elastic_regime`` reads the calculation, the route and the
strain response, so a real :class:`~defuqumat.scf.Calculation` on
``si-us-nosym.in`` (a basis and a projector table, no SCF) is enough to see each
of its refusals fire. The number that lifted the whole-k route is a slow test,
``tests/regression/test_electrostriction.py::
test_the_elastic_constants_match_a_second_difference_of_the_energy``.
"""

import types
from pathlib import Path

import numpy as np
import pytest

from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf import Calculation
from defuqumat.system import build_system

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


def _calculation(case):
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return Calculation(system, pseudos)


@pytest.mark.parametrize("case", ["si-us-nosym", "si-paw-nosym"])
def test_the_whole_k_route_is_not_refused(case):
    from defuqumat.response.elastic import require_a_measured_elastic_regime

    calculation = _calculation(case)
    require_a_measured_elastic_regime(calculation)
    complete = types.SimpleNamespace(ort=np.empty((3, 3), object),
                                     multipliers=np.empty((3, 3), object))
    require_a_measured_elastic_regime(calculation, response=complete)


@pytest.mark.parametrize("case", ["si-us-nosym", "si-paw-nosym"])
def test_the_walked_route_still_refuses_an_augmented_dataset(case):
    from defuqumat.response.elastic import require_a_measured_elastic_regime

    with pytest.raises(NotImplementedError, match="walked route"):
        require_a_measured_elastic_regime(_calculation(case), streamed=True)


def test_a_host_store_takes_the_whole_k_route_with_a_warning():
    """A host store is what makes :func:`elastic_constants` walk. An augmented
    dataset's ``C_ijkl`` is not written on the walk, so the route is the
    whole-k one, and the warning says the states go on the device whole."""
    from defuqumat.response.efield import _streams
    from defuqumat.response.elastic import _walk_lacks

    calculation = _calculation("si-us-nosym")
    nk = calculation.system.kpoints.nk
    store = np.zeros((1, nk, 4, 8), complex)
    assert _streams(calculation, store, False)
    with pytest.warns(RuntimeWarning, match="multipliers"):
        assert not _streams(calculation, store, False,
                            whole_k=_walk_lacks(calculation))


@pytest.mark.parametrize("missing", ["ort", "multipliers", "both"])
def test_a_response_without_the_two_tangents_is_refused(missing):
    from defuqumat.response.elastic import require_a_measured_elastic_regime

    fields = {"ort": np.empty((3, 3), object), "multipliers": np.empty((3, 3), object)}
    for name in (("ort", "multipliers") if missing == "both" else (missing,)):
        fields[name] = None
    with pytest.raises(NotImplementedError, match="StrainResponse.multipliers"):
        require_a_measured_elastic_regime(
            _calculation("si-us-nosym"), response=types.SimpleNamespace(**fields))


def test_a_norm_conserving_dataset_is_never_refused_by_it():
    from defuqumat.response.elastic import require_a_measured_elastic_regime

    calculation = _calculation("si-electrostriction")
    assert not calculation.is_ultrasoft
    require_a_measured_elastic_regime(
        calculation, streamed=True,
        response=types.SimpleNamespace(ort=None, multipliers=None))


def test_electrostriction_checks_the_walked_route_before_the_responses():
    """The refusal moved below the route decision and has to stay above the
    two self-consistent responses, which is what makes it cost nothing."""
    import importlib
    import inspect

    module = importlib.import_module("defuqumat.response.electrostriction")
    body = inspect.getsource(module.electrostriction)
    guard = body.index("require_a_measured_elastic_regime(calculation, streamed=walked)")
    assert body.index("walked = _streams(") < guard < body.index("dielectric_tensor(")
