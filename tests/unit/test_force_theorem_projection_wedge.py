"""The projected force theorem at the system's own axis on a wedge (audit 2026-11-20c #26).

The decomposition by site, orbital and spin is summed with the k-set's own
weights and nothing symmetrises it. On the wedge of the moment's magnetic group
that completes the band energy and not the labels the group permutes: for a
moment along ``z`` in tetragonal Co, ``C4z`` maps ``d_xz`` onto ``d_yz``. The
refusal fired only off the system's own axis. Setup only: the fixed-density
solve is replaced by a sentinel, so a run the guard lets through stops there.
"""

from pathlib import Path

import numpy as np
import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.system.builder import build_system

pytestmark = pytest.mark.unit

CASE = Path(__file__).resolve().parents[1] / "data" / "qe" / "co-tetragonal-anisotropy-soc.in"


class _Solved(Exception):
    """Raised where the fixed-density solve would start."""


def _system(nosym: bool):
    text = CASE.read_text()
    if not nosym:
        text = text.replace("nosym = .true.,", "").replace("nosym = .true.", "")
    return build_system(parse_pw_input(text))


@pytest.fixture
def solve_stops(monkeypatch):
    from defuqumat.workflows import anisotropy

    def sentinel(*args, **options):
        raise _Solved()

    monkeypatch.setattr(anisotropy, "fixed_density_states", sentinel)


def _pseudos(system, pseudo_dir):
    return tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)


def test_a_projection_at_the_own_axis_is_refused_on_the_wedge(solve_stops, pseudo_dir):
    from defuqumat.workflows.anisotropy import _reference_axis, run_force_theorem

    reduced = _system(nosym=False)
    assert reduced.kpoints.reduced and reduced.kpoints.nk == 6
    pseudos = _pseudos(reduced, pseudo_dir)
    density = np.zeros((2, 4, 4, 4))
    own = _reference_axis(reduced)
    assert np.allclose(own, (0.0, 0.0, 1.0))
    for direction in (None, own):
        with pytest.raises(ValueError, match="projected force-theorem decomposition"):
            run_force_theorem(reduced, pseudos, density, direction=direction,
                              projected=True)


def test_the_regimes_next_to_it_reach_the_solve(solve_stops, pseudo_dir):
    """The band energy alone on the wedge, and the projection on the whole grid."""
    from defuqumat.workflows.anisotropy import run_force_theorem

    reduced = _system(nosym=False)
    whole = _system(nosym=True)
    assert not whole.kpoints.reduced and whole.kpoints.nk == 18
    density = np.zeros((2, 4, 4, 4))
    for system, projected in ((reduced, False), (whole, True)):
        with pytest.raises(_Solved):
            run_force_theorem(system, _pseudos(system, pseudo_dir), density,
                              direction=(0.0, 0.0, 1.0), projected=projected)
