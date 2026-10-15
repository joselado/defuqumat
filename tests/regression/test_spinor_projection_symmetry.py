"""``sym_proj_nc``: the group average of a projection when the columns carry a
spin as well as an ``m``.

A noncollinear column without spin-orbit coupling is ``|l m> x |sigma>``, so a
point-group operation turns two things at once -- the real harmonics among
themselves and the spin frame the spinor is written in -- and the operator the
group average runs over is the tensor product ``D^l x conj(U)``. Averaging the
``m`` indices alone leaves the spin frame where it was, which is a smooth,
normalised, plausible projection of the wrong thing.

**What the test is.** The projection is a sum over the Brillouin zone, so a
reduced wedge and the closed grid have to give the same Loewdin charge in every
column, and the wedge only does if the average is over the right operator. The
closed grid is the reference and it shares nothing with the wedge but the
ground state: different k-points, different weights, no symmetrisation at all.

**Two cells, because neither can do the job alone, and that is the point.**

* ``h4-cycloid-90.in`` is four hydrogens with their moments turned 90 degrees
  from one another. Hydrogen carries an ``s`` shell only, so ``D^l`` is the
  number one and the *whole* symmetrisation is the spin factor. Without it the
  wedge misses the closed grid by 3.8e-4 electrons, which is exactly what no
  symmetrisation at all gives; with it, 4.1e-6.
* ``ni-noncol-111.in`` is nickel with its moment along a three-fold axis, and
  it exists because ``ni-ldau-noncol.in`` -- the same crystal with the moment
  along ``z`` -- **cannot tell any spin convention from any other**. Every one
  of that cell's operations turns the spin about ``z``, so its 2x2 matrix is
  diagonal, the phase factors out of a modulus, and the right matrix, its
  transpose, its conjugate and the identity all agree to the last digit. On the
  three-fold cell they do not: ``U``, ``U^T`` and ``U^dagger`` are wrong by
  3.9e-2, 2.1e-2 and 2.1e-2 electrons where ``conj(U)`` is 1.1e-5.

Only ``conj(U)`` passes both, which is what pins it, and it is what the
harmonic factor already does one index over: the contraction is on the *first*
index, so what multiplies the projection is ``D^T = D^{-1}``, the inverse
operation, and the spin factor has to be the inverse of the same operation.

**The spin-orbit case is the same operator in another basis.** Its columns are
``|j m_j>`` and ``sym_proj_so`` contracts ``d_matrix_so``'s ``D^j``, which is
``conj(M_j) (D^l x conj(U)) M_j^T`` with ``M_j`` the spin-angle map the orbitals
are built with; a nonmagnetic run averages each column with its Kramers partner
as well. It is checked on fcc platinum, ultrasoft and PAW, by the same wedge
against the closed grid, and the closed grid's columns against ``projwfc.x``'s
own (``lsym = .false.``). A magnetic spin-orbit run stays refused, and the
refusal is tested by name in ``tests/unit/test_projwfc.py``.
"""

from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

from defuqumat import Calculator
from defuqumat.projwfc.projections import atomic_projections, calculation_channels

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """``CLAUDE.md``'s rule for a file that runs several cells: keep the
    results, drop the executables."""
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _converged(stem):
    calc = Calculator.from_file(CASES / stem, pseudo_dir=PSEUDO)
    return calc, calc.get_scf()


def _column_charges(calc, scf, symmetrize):
    """``sum_k sum_n f_nk |<phi_c|S|psi_nk>|^2``, one number per column.

    ``occupations`` already carries the k-point weight -- summing it over ``k``
    and ``n`` gives the electron count -- so the weights are not applied again
    here, which would square them.
    """
    projections = np.asarray(
        atomic_projections(calc.calculation, scf.wavefunctions,
                           symmetrize=symmetrize)
    )
    occupations = np.asarray(scf.occupations)
    if occupations.ndim == 2:
        occupations = occupations[None]
    return np.einsum("skcn,skn->c", projections, occupations)


#: ``(wedge, closed, what the symmetrisation has to reach, what dropping it
#: costs)``. The last two are measured, not chosen: see the module docstring.
PAIRS = (
    ("ni-noncol-111.in", "ni-noncol-111-nosym.in", 1.0e-4, 1.0e-2),
    ("h4-cycloid-90.in", "h4-cycloid-90-nosym.in", 1.0e-5, 1.0e-4),
)


@pytest.mark.parametrize("wedge, closed, tolerance, floor", PAIRS)
def test_the_wedge_reproduces_the_closed_grid(wedge, closed, tolerance, floor):
    calc_w, scf_w = _converged(wedge)
    calc_c, scf_c = _converged(closed)
    assert scf_w.converged and scf_c.converged
    # The two runs have to be the same ground state before their projections can
    # be compared at all.
    assert abs(scf_w.total_energy - scf_c.total_energy) < 1.0e-7

    symmetrised = _column_charges(calc_w, scf_w, symmetrize=True)
    reference = _column_charges(calc_c, scf_c, symmetrize=False)
    assert np.abs(symmetrised - reference).max() < tolerance

    # **And the guard has to fire.** A test that only checks the symmetrised
    # number passes on a cell where the symmetrisation does nothing, which is
    # how a group average over the wrong operator survives: the unsymmetrised
    # wedge must be measurably wrong on the same cell.
    raw = _column_charges(calc_w, scf_w, symmetrize=False)
    assert np.abs(raw - reference).max() > floor


def test_the_operator_is_unitary_per_operation():
    """``d_matrix_nc``'s own check, and it is cheap enough to keep.

    Each operation's ``D^l x conj(U)`` is a product of two unitary matrices, so
    the assembled coefficient block has to be unitary shell by shell. It is the
    one property that catches a mis-shaped gather without running anything.
    """
    from defuqumat.projwfc.projections import build_projection_symmetry

    calc, _ = _converged("ni-noncol-111.in")
    channels = calculation_channels(calc.calculation)
    symmetry = build_projection_symmetry(
        channels, calc.system.cell, calc.system.structure,
        calc.calculation.symmetries,
    )
    coefficients = np.asarray(symmetry.coefficients)  # (nsym, nwfc, mmax)
    assert np.iscomplexobj(coefficients)

    # One shell at a time -- ``(atom, wfc, l)``, not ``l`` alone, since two
    # shells of the same ``l`` are two gather blocks and stacking them is not a
    # square matrix.
    shells = {(c.atom, c.wfc, c.l) for c in channels}
    for atom, wfc, l in shells:
        columns = [c.index for c in channels
                   if (c.atom, c.wfc, c.l) == (atom, wfc, l)]
        block = coefficients[:, columns, : 2 * (2 * l + 1)]
        identity = np.eye(len(columns))
        for s in range(symmetry.nsym):
            product = block[s] @ block[s].conj().T
            assert np.abs(product - identity).max() < 1.0e-10


def _read_filproj(path):
    """``projwfc.x``'s ``filproj``: ``proj[nwfc, nbnd, nk]``, ten decimals."""
    lines = path.read_text().splitlines()
    index = 0
    while True:
        parts = lines[index].split()
        if len(parts) == 3 and all(p.lstrip("-").isdigit() for p in parts):
            natomwfc, nk, nbnd = map(int, parts)
            break
        index += 1
    index += 2
    proj = np.zeros((natomwfc, nbnd, nk))
    for column in range(natomwfc):
        index += 1
        for _ in range(nk * nbnd):
            ik, ibnd, value = lines[index].split()
            proj[column, int(ibnd) - 1, int(ik) - 1] = float(value)
            index += 1
    return proj


#: ``(wedge, closed, the stem of projwfc.x's files for the closed grid)``.
SPIN_ORBIT = (
    ("pt-soc.in", "pt-soc-nosym.in", "pt-soc"),
    pytest.param("pt-soc-paw.in", "pt-soc-paw-nosym.in", "pt-soc-paw",
                 id="pt-soc-paw"),
)


@pytest.mark.parametrize("wedge, closed, stem", SPIN_ORBIT)
def test_the_spin_orbit_wedge_reproduces_the_closed_grid(wedge, closed, stem):
    """``sym_proj_so``: the wedge's ``|j m_j>`` charges, column by column.

    Three numbers, each on the occupation-weighted column charge, which is blind
    to the rotation a degenerate eigensolver is free in (rule D4) where a single
    band's projection is not:

    * the symmetrised wedge against the closed grid of this code;
    * the symmetrised wedge against ``projwfc.x``'s columns on the closed grid,
      weighted with the closed run's occupations, which shares nothing with the
      wedge but the input;
    * the unsymmetrised wedge, which has to miss, or the first two prove nothing.

    The Kramers partner average is invisible here: platinum has inversion, so a
    Kramers pair at one ``k`` already carries equal ``m_j`` and ``-m_j`` weight
    once summed, and only a per-band projection could see it.
    """
    filproj = CASES / f"reference.{stem}.projwfc_up"
    calc_w, scf_w = _converged(wedge)
    assert calc_w.calculation.use_symmetry
    assert calc_w.calculation.symmetries.nsym == 48
    assert not calc_w.system.domag
    calc_c, scf_c = _converged(closed)
    assert scf_w.converged and scf_c.converged
    energy_gap = abs(scf_w.total_energy - scf_c.total_energy)
    assert energy_gap < 1.0e-6

    symmetrised = _column_charges(calc_w, scf_w, symmetrize=True)
    closed_grid = _column_charges(calc_c, scf_c, symmetrize=False)
    raw = _column_charges(calc_w, scf_w, symmetrize=False)

    occupations = np.asarray(scf_c.occupations).reshape(
        np.asarray(scf_c.eigenvalues).shape[-2:])  # (nk, nbnd)
    reference = np.einsum("cnk,kn->c", _read_filproj(filproj), occupations)

    against_closed = np.abs(symmetrised - closed_grid).max()
    against_qe = np.abs(symmetrised - reference).max()
    unsymmetrised = np.abs(raw - closed_grid).max()
    print(f"\n{stem}: |E_wedge - E_closed| = {energy_gap:.3e} Ry; columns: "
          f"wedge-closed {against_closed:.3e}, wedge-projwfc {against_qe:.3e}, "
          f"closed-projwfc {np.abs(closed_grid - reference).max():.3e}, "
          f"unsymmetrised {unsymmetrised:.3e}")
    assert against_closed < 1.0e-4
    assert against_qe < 2.0e-4
    assert unsymmetrised > 1.0e-3
