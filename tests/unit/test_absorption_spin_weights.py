"""A k-set handed to ``run_absorption`` gets the weights of the spin regime (audit 2026-11-20c #19).

``KPoints.automatic`` applies the unpolarized factor of two. Without
``for_spin`` at the boundary, a direct call at ``nspin = 2`` or as a spinor,
with ``kpoints=`` and no ``calculation=``, diagonalised on weights summing to 2
where the run needs 1, so the fixed occupations held twice the electrons and
``chi_0``'s pair weights ``(wg_i - wg_j)/Omega`` doubled. The Sternheimer
referee reads the same weights, so ``static_residual`` could not see it.
Setup only: the fixed-density solve is replaced by a sentinel that hands back
the calculation it would have diagonalised in.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.system import build_system
from defuqumat.system.kpoints import KPoints

pytestmark = pytest.mark.unit

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"


class _Solved(Exception):
    """Raised where the fixed-density solve would start."""

    def __init__(self, calculation):
        super().__init__("solve")
        self.calculation = calculation


def _silicon(nspin: int):
    """``si-epsilon-unshifted-nosym``: unpolarized, ``nspin = 2`` at zero moment, or a spinor."""
    parsed = read_pw_input(CASES / "si-epsilon-unshifted-nosym.in")
    if nspin == 4:
        parsed.namelists["system"]["noncolin"] = True
    system = build_system(parsed)
    if nspin == 2:
        system = system.with_spin(2, tot_magnetization=0)
    return system


@pytest.mark.parametrize(("nspin", "total"), [(1, 2.0), (2, 1.0), (4, 1.0)])
def test_a_caller_built_k_set_gets_the_weights_of_the_spin_regime(
        nspin, total, monkeypatch, pseudo_dir):
    from defuqumat.workflows import tddft

    system = _silicon(nspin)
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file)
                    for s in system.structure.species)
    raw = KPoints.automatic((2, 2, 2), (0, 0, 0), system.cell,
                            precision=system.kpoints.precision)
    assert float(np.sum(np.asarray(raw.weights))) == pytest.approx(2.0)

    def sentinel(system, pseudos, density, **options):
        raise _Solved(options["calculation"])

    monkeypatch.setattr(tddft, "fixed_density_states", sentinel)
    with pytest.raises(_Solved) as solved:
        tddft.run_absorption(system, pseudos, None, np.array([0.1]), kernel="rpa",
                             kpoints=raw, nbnd=8, static_residual=False)
    calculation = solved.value.calculation
    weights = np.asarray(calculation.system.kpoints.weights)
    assert weights.sum() == pytest.approx(total, abs=1e-12)

    # The fixed occupations on these weights are what chi_0's pair weights
    # are made of, and they must hold the electrons once.
    channels = 2 if nspin == 2 else 1
    eigenvalues = jnp.broadcast_to(jnp.arange(8.0), (channels, raw.nk, 8))
    wg, _ = calculation.occupations(eigenvalues)
    assert float(np.sum(np.asarray(wg))) == pytest.approx(calculation.nelec, abs=1e-10)
