"""The Hubbard term's share of ``dH/dk`` and ``d^2H/dk^2``, against a difference of ``H(k)``.

``notes/incompat/I92.md``. The shift current and the second-harmonic tensor
refused a DFT+U run on the ground that the Hubbard term, a separable operator
built from ``wfcU(k+G)``, carries a velocity and a second velocity nothing had
threaded through. Reading the operator showed both were already there:
:class:`~defuqumat.response.velocity.VelocityOperator` differentiates
``calculation.hamiltonian`` with the converged ``ns`` through
:meth:`~defuqumat.scf.driver.Calculation.at_kcart`, which rebuilds ``wfcU``
from the traced ``k``. What had never been done is to measure them, and this
file does, on ``alas-ldau.in`` (AlAs with ``U = 3`` eV on the As 4p shell).

The identity is exact for *any* states, potential and occupation matrix, so
there is no SCF here: the starting potential, an ``ns`` that is the starting one
plus an arbitrary Hermitian part, and states that overlap the Hubbard orbitals.
At a generic k-point (``l = 1`` orbitals have an origin trap at ``k + G = 0``,
``tests/unit/test_velocity_locality.py``) each derivative is compared with a
central difference on the frozen sphere at two steps, and the discrepancy must
fall as ``h^2``. The same comparison with the Hubbard term left out of the
``jvp`` (the operator of the same cell without its ``U``) is the size of the
term, and it is what says the agreement is not one a missing term would also
have given.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.response import VelocityOperator
from defuqumat.scf import Calculation
from defuqumat.system import build_system
from defuqumat.system.kpoints import KPoints

pytestmark = pytest.mark.unit

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"

#: The generic point of ``tests/regression/test_photocurrent.py``.
GENERIC_K = np.array([0.11, 0.23, 0.31])

#: The two central-difference steps, 1/bohr. Their ratio is 4, so a truncation
#: error ``O(h^2)`` falls by 16 between them.
STEPS = (4.0e-3, 1.0e-3)


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """Three projector kinds and a second derivative each: clear the executables."""
    yield
    jax.clear_caches()


def _input(tmp_path: Path, kind: str, hubbard: bool) -> Path:
    text = (CASES / "alas-ldau.in").read_text()
    if not hubbard:
        text = text[: text.index("HUBBARD")]
    else:
        text = text.replace("HUBBARD {atomic}", f"HUBBARD {{{kind}}}")
    path = tmp_path / f"alas-{kind}-{int(hubbard)}.in"
    path.write_text(text)
    return path


@lru_cache(maxsize=2)
def _pieces(path: str, pseudo_dir: str):
    """``(calculation at GENERIC_K, v_scf, ns, psi)`` with no SCF."""
    system = build_system(read_pw_input(Path(path)))
    pseudos = tuple(
        read_upf(Path(pseudo_dir) / s.pseudo_file) for s in system.structure.species
    )
    whole = Calculation(system, pseudos)
    v_scf = whole.potential(whole.starting_density()).v_scf
    kp = KPoints.from_crystal(
        GENERIC_K.reshape(1, 3), np.ones(1), system.cell,
        precision=system.kpoints.precision,
    )
    calculation = whole.at_kpoints(kp)
    rng = np.random.default_rng(3)
    ns = None
    width = calculation.basis.planewaves.npwx * calculation.npol
    noise = rng.normal(size=(1, 1, 6, width)) + 1j * rng.normal(size=(1, 1, 6, width))
    mask = np.asarray(calculation.basis.planewaves.mask, dtype=float)[0]
    noise = noise * mask
    if calculation.is_hubbard:
        start = np.asarray(calculation.starting_ns())
        extra = rng.normal(size=start.shape) * 0.1
        ns = jnp.asarray(start + extra + np.swapaxes(extra, -1, -2))
        # States with weight on the Hubbard orbitals: the term is invisible to
        # states orthogonal to them, and a check that cannot see it is a null.
        orbitals = np.asarray(calculation.wfcU)[0].T  # (nwfcU, npwx)
        noise[0, 0, : orbitals.shape[0]] += 30.0 * orbitals
    psi = noise / np.linalg.norm(noise, axis=-1, keepdims=True)
    return calculation, v_scf, ns, jnp.asarray(psi)


def _first(calculation, v_scf, ns, psi, kcart=None):
    return np.asarray(VelocityOperator(calculation, v_scf, ns=ns, kcart=kcart)
                      .matrix_elements(psi))


def _difference_of_h(calculation, v_scf, ns, psi, h):
    """``<m|[H(k + h e_a) - H(k - h e_a)]/2h|n>`` on the frozen sphere."""
    operator = VelocityOperator(calculation, v_scf, ns=ns)
    k0 = jnp.asarray(operator.kcart)
    rows = []
    for axis in range(3):
        step = jnp.zeros_like(k0).at[:, axis].set(h)
        moved = (operator._operator(psi, k0 + step, overlap=False)
                 - operator._operator(psi, k0 - step, overlap=False)) / (2.0 * h)
        rows.append(np.asarray(jnp.einsum("skmg,skng->skmn", psi.conj(), moved)))
    return np.stack(rows)


def _difference_of_v(calculation, v_scf, ns, psi, h):
    """``[v_a(k + h e_b) - v_a(k - h e_b)]/2h``, ``(3, 3, ...)`` indexed ``[a, b]``."""
    k0 = np.asarray(VelocityOperator(calculation, v_scf, ns=ns).kcart)
    columns = []
    for axis in range(3):
        step = np.zeros_like(k0)
        step[:, axis] = h
        columns.append((_first(calculation, v_scf, ns, psi, k0 + step)
                        - _first(calculation, v_scf, ns, psi, k0 - step)) / (2.0 * h))
    return np.stack(columns, axis=1)


def hubbard_velocity_errors(tmp_path, pseudo_dir, kind):
    """Every number the check reads, for one projector kind.

    ``first``/``second`` are ``max|jvp - difference|`` at each step,
    ``first_off``/``second_off`` the same with the Hubbard term left out of the
    ``jvp`` (at the smaller step), and ``hubbard_first``/``hubbard_second`` the
    size of the Hubbard term itself (the ``jvp`` with ``U`` minus without).
    """
    calculation, v_scf, ns, psi = _pieces(
        str(_input(tmp_path, kind, True)), str(pseudo_dir))
    bare, bare_v, _, _ = _pieces(str(_input(tmp_path, kind, False)), str(pseudo_dir))
    v = _first(calculation, v_scf, ns, psi)
    w = np.asarray(VelocityOperator(calculation, v_scf, ns=ns)
                   .second_matrix_elements(psi))
    v_off = _first(bare, bare_v, None, psi)
    w_off = np.asarray(VelocityOperator(bare, bare_v).second_matrix_elements(psi))
    first, second = [], []
    for h in STEPS:
        dv = _difference_of_h(calculation, v_scf, ns, psi, h)
        dw = _difference_of_v(calculation, v_scf, ns, psi, h)
        first.append(float(np.abs(v - dv).max()))
        second.append(float(np.abs(w - dw).max()))
    return dict(
        first=first, second=second,
        first_off=float(np.abs(v_off - dv).max()),
        second_off=float(np.abs(w_off - dw).max()),
        hubbard_first=float(np.abs(v - v_off).max()),
        hubbard_second=float(np.abs(w - w_off).max()),
        scale_first=float(np.abs(v).max()), scale_second=float(np.abs(w).max()),
    )


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["atomic", "ortho-atomic", "norm-atomic"])
def test_the_hubbard_velocity_matches_a_difference_of_h(tmp_path, pseudo_dir, kind):
    """Both derivatives, every projector kind, falling as ``h^2``.

    The tolerances are set from the measured numbers in ``notes/incompat/I92.md``,
    not from a single passing point: the smaller step's discrepancy is at least
    ten times below the larger one's (``h^2`` gives sixteen), and at least a
    thousand times below the Hubbard term it is checking.
    """
    errors = hubbard_velocity_errors(tmp_path, pseudo_dir, kind)
    for name in ("first", "second"):
        coarse, fine = errors[name]
        term = errors[f"hubbard_{name}"]
        # The term is there, at a size the difference resolves.
        assert term > 1.0e-3 * errors[f"scale_{name}"]
        # Left out of the jvp, the comparison reads the whole term.
        assert errors[f"{name}_off"] > 0.5 * term
        # With it, the comparison is truncation and nothing else.
        assert fine < coarse / 10.0
        assert fine < 1.0e-3 * term
