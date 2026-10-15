"""A spiral on the wedge of its spin space group, against the same spiral with ``nosym``.

The check for the spin space group (``notes/INCOMPATIBILITIES.md`` I50) is the
``nosym`` run: the same input on the full grid, with nothing symmetrised. Two
cells, because each discriminates something the other cannot:

* fcc hydrogen with ``q = b3/4`` (``h-fcc-spiral-scan.in`` at a larger
  smearing), a three-dimensional metal on a non-orthogonal cell whose wedge is
  20 of 64 points under ``C3v``, all with ``f = 0``;
* a zigzag hydrogen chain whose screw ``{C2z|c/2}`` and glide ``{m_x|c/2}`` carry
  ``q . f = 45`` degrees, which is the only arm where permuting the rotated-frame
  components as four scalars and leaving a ``Rot_z(q . f)`` behind read
  differently; 16 of 36 points.

The seed residuals that make the four-scalar rule a discriminator rather than
a null result, and the group counts, are ``tests/unit/test_spiral_symmetry.py``.
"""

from functools import lru_cache
from pathlib import Path

import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf import run_scf
from defuqumat.system import build_system
from tests.unit.test_spiral_symmetry import FCC, ZIGZAG, _nosym

pytestmark = [pytest.mark.regression, pytest.mark.slow]


@pytest.fixture(autouse=True)
def _bound_compilation_cache():
    """Four cells of different shapes in one process; see ``CLAUDE.md`` on memory."""
    yield
    import jax

    jax.clear_caches()


@lru_cache(maxsize=2)
def _run(text: str, pseudo_dir: Path):
    pwin = parse_pw_input(text)
    system = build_system(pwin)
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    return system, run_scf(
        system, pseudos,
        conv_thr=float(pwin.get("electrons", "conv_thr")),
        mixing_beta=float(pwin.get("electrons", "mixing_beta")),
        max_iterations=200,
    )


@pytest.mark.parametrize("text, nk_sym, nk_full, tolerance", [
    (FCC, 20, 64, 1e-8),
    (ZIGZAG, 16, 36, 1e-8),
], ids=["fcc", "zigzag"])
def test_the_spiral_on_its_wedge_is_the_nosym_spiral(text, nk_sym, nk_full,
                                                     tolerance, pseudo_dir):
    """The total energy and the moment on the wedge, against the full grid."""
    full_system, full = _run(_nosym(text), pseudo_dir)
    system, reduced = _run(text, pseudo_dir)

    assert system.symmetry_group().nsym > 1 and not system.nosym
    assert system.kpoints.nk == nk_sym and full_system.kpoints.nk == nk_full
    assert full.converged and reduced.converged
    # A magnetic state, or the comparison would not see the magnetization's rule.
    assert full.absolute_magnetization > 0.1

    assert reduced.total_energy == pytest.approx(full.total_energy, abs=tolerance)
    assert reduced.absolute_magnetization == pytest.approx(
        full.absolute_magnetization, abs=1e-5
    )
