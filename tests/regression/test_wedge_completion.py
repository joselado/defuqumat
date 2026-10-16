"""A response summed over a symmetry-reduced k-set, completed by its group.

Each test is one identity with no external reference: the same converged
density, the response evaluated on the run's own wedge (completed by
:class:`~defuqumat.system.wedge.WedgeGroup`) and on the whole unshifted grid
passed as ``kpoints=``, which is closed under the point group. Same
Hamiltonian, same FFT grid, only the k-sum differs, so what is left is the
fixed-density solve's threshold.

Two traps decide the setup, and both were measured before the bounds were set:

* **the band count is cut in a real gap.** A cut inside a degenerate multiplet
  keeps a subspace that is not covariant under the group, so the wedge then
  differs from the whole grid by the cut and not by the sum: on silicon at
  ``nbnd = 16`` (cut gap 1.4e-14 Ry) the body of ``chi_0`` was 3.2e-4 apart,
  at ``nbnd = 20`` (5.6e-2 Ry) 3.6e-9;
* **the SCF keeps its symmetry.** A ``nosym`` reference would move the FFT grid
  off the factors the fractional translations need, which the shift-current
  refusal measured at three to four orders on silicon's inversion-forbidden
  tensor.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo.upf import read_upf
from defuqumat.scf.driver import run_scf
from defuqumat.system.builder import build_system
from defuqumat.system.kpoints import KPoints

pytestmark = pytest.mark.slow

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """``CLAUDE.md``'s bound on a file that runs several cells."""
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def converged(name: str):
    """The SCF of a committed input, symmetry on. ``maxsize=2``."""
    system = build_system(read_pw_input(CASES / name))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return system, pseudos, run_scf(system, pseudos, conv_thr=1.0e-11)


def full_mesh(system, n: int) -> KPoints:
    """The whole unshifted ``n x n x n`` grid -- closed under the point group."""
    axis = np.arange(n) / n
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    return KPoints.from_crystal(
        points, np.full(len(points), 1.0 / len(points)), system.cell,
        precision=system.kpoints.precision,
    )


def relative(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _chi0(system, pseudos, density, kpoints, nbnd):
    from defuqumat.tddft.chi0 import independent_response
    from defuqumat.workflows.nscf import fixed_density_states

    calculation, _, eigenvalues, wavefunctions = fixed_density_states(
        system, pseudos, density, kpoints, nbnd=nbnd, conv_thr=1.0e-10)
    potential = calculation.potential(jnp.asarray(density))
    return np.asarray(independent_response(
        calculation, wavefunctions, eigenvalues, potential.v_scf,
        np.linspace(0.0, 1.0, 21), ecut_response=4.0, broadening=0.02).x)


def test_chi0_on_a_wedge_is_the_whole_grid_s():
    """``chi_0(G, G')`` with both labels rotated at once (``PLAN`` I42).

    Silicon, ``si-epsilon-unshifted.in``: 8 of 64 points, ``Oh`` with its
    fractional translations and time reversal. Measured on D22: 2.1e-10 of
    ``max|x|`` over the whole matrix (head 2.1e-10, wings 2.4e-9, body
    3.6e-9), against 0.73 for the uncompleted wedge sum.
    """
    system, pseudos, scf = converged("si-epsilon-unshifted.in")
    assert system.kpoints.nk == 8
    wedge = _chi0(system, pseudos, scf.density, None, 20)
    whole = _chi0(system, pseudos, scf.density, full_mesh(system, 4), 20)
    assert relative(wedge, whole) < 1.0e-8


def test_optics_on_a_wedge_is_the_whole_grid_s():
    """``sigma^ab``, the shift current and SHG on a wedge (``PLAN`` I40, I41).

    AlAs, ``alas-raman-wedge.in``: 8 of 64 points, ``Td`` with time reversal,
    48 operations. ``nbnd = 14`` cuts a 6.7e-2 Ry gap at every point of the
    whole grid. Measured on D22, relative to the largest component of the
    whole grid's tensor: ``sigma`` 2.8e-10, the shift current 2.1e-9, SHG
    1.4e-8; the uncompleted wedge sums were 1.1, 0.96 and 2.5 off, and the
    same cell with ``noinv`` (``Td`` alone, 10 points) read 1.8e-10, 2.8e-9
    and 1.4e-8.
    """
    from defuqumat.workflows.conductivity import run_conductivity
    from defuqumat.workflows.photocurrent import run_shift_current
    from defuqumat.workflows.shg import run_shg

    system, pseudos, scf = converged("alas-raman-wedge.in")
    assert system.kpoints.nk == 8
    whole_k = full_mesh(system, 4)
    runs = {
        "sigma": lambda **k: run_conductivity(
            system, pseudos, scf.density, nbnd=14, nw=60, window=1.2,
            broadening=0.02, **k).sigma,
        "shift": lambda **k: run_shift_current(
            system, pseudos, scf.density, nbnd=14, window=0.9, nw=90,
            broadening=0.02, **k).sigma,
        "shg": lambda **k: run_shg(
            system, pseudos, scf.density, nbnd=14, window=0.6, nw=120, **k).chi,
    }
    for name, run in runs.items():
        error = relative(run(), run(kpoints=whole_k))
        print(f"{name}: wedge against whole grid {error:.3e}")
        assert error < 1.0e-6, name
