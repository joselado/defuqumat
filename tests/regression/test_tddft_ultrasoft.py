"""The sum-over-states ``chi_0`` of an ultrasoft dataset, against the Sternheimer solve.

``notes/incompat/I94.md``. P40 and P94 attempted it and reverted: with ``Q_ij(G)``
in the body and the plain ``dH/dk`` head, ultrasoft silicon's RPA ``eps_M(0)``
sat **+0.2497, +0.5403** from the Sternheimer solve at 30 and 60 bands, rising
where a truncation falls. The head is the generalised velocity of P99,
``<n|dH - e_m dS|m> + (e_m - e_n) K_nm``, and with it the residual is
negative and falls with ``nbnd`` as the norm-conserving control on the same
cell does. The control that would read differently, the plain head, is run
beside it.

Slow: six fixed-density runs on a two-k-point cell, about two minutes.
"""

from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.response.velocity import VelocityOperator
from defuqumat.scf import run_scf
from defuqumat.system import build_system
from defuqumat.workflows.tddft import run_absorption

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"
US = (CASES / "si-us-nosym.in").read_text()
NC = US.replace("Si.pz-n-rrkjus_psl.0.1.UPF", "Si.pz-vbc.UPF").replace(", ecutrho = 96.0", "")


@pytest.fixture(autouse=True)
def _bounded_compilation():
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _converged(text):
    system = build_system(parse_pw_input(text))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    result = run_scf(system, pseudos, conv_thr=1e-12, max_iterations=200)
    return system, pseudos, result


def residual(text, nbnd, ecut_response=8.0):
    """``(eps_M(0) in RPA from chi_0, minus the Sternheimer solve's)``."""
    system, pseudos, result = _converged(text)
    spectrum = run_absorption(system, pseudos, result.density, np.array([0.1]),
                              kernel="rpa", nbnd=nbnd, broadening=0.01,
                              ecut_response=ecut_response, static_residual=True,
                              conv_thr=1e-11)
    return spectrum.static_rpa, spectrum.static_residual


def test_the_ultrasoft_residual_falls_with_nbnd_as_the_norm_conserving_one_does():
    """Measured on D22: US -0.3369, -0.0522; NC -0.2885, -0.0601 (30, 60 bands)."""
    us = {n: residual(US, n)[1] for n in (30, 60)}
    nc = {n: residual(NC, n)[1] for n in (30, 60)}
    for column in (us, nc):
        assert column[30] < column[60] < 0.0
        assert abs(column[60]) < abs(column[30]) / 4.0
    assert abs(us[60]) < 0.08


def test_the_plain_velocity_head_is_what_p94_measured():
    """The control: ``dH/dk`` in the head gives P94's rising, positive residual."""
    tangent, connections = VelocityOperator.tangent_elements, VelocityOperator.connections
    VelocityOperator.tangent_elements = (
        lambda self, psi, energies, sequential=False:
        self.matrix_elements(psi, sequential=sequential))
    VelocityOperator.connections = lambda self, psi, sequential=False: None
    try:
        plain = {n: residual(US, n)[1] for n in (30, 60)}
    finally:
        VelocityOperator.tangent_elements = tangent
        VelocityOperator.connections = connections
    assert plain[30] == pytest.approx(0.2497, abs=2e-3)
    assert plain[60] == pytest.approx(0.5403, abs=2e-3)
