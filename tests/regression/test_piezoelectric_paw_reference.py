"""The PAW piezoelectric tensor against ``pw.x``'s own Berry-phase finite difference.

Until 2026-10-19 the tensor refused every PAW dataset, because nothing about it
had been compared with a reference outside this package: the PAW wedge
completion (``test_piezoelectric_paw.py``) is an identity between two k-sets of
one calculation. This file is the comparison that lifted the refusal
(``notes/incompat/I89.md``).

**The reference shares no code with the quantity.** ``e_14`` is ``dP_x/d eps_4``
at clamped ions, and ``pw.x`` gives ``P`` on each of two sheared cells
(``E[1,2] = E[2,1] = +-0.005``, so ``eps_4 = +-0.01``): an scf on ``6 6 6`` and
three ``lberry`` runs per cell, 15 k-points a string over 8x8 strings. The
outputs are committed as ``reference.out.alas-piezo-paw-{plus,minus}-g{1,2,3}``
and the difference is taken here from them, so the number is ``pw.x``'s.

**The convergence parameter is the response's k-mesh**, and what is asserted is
the way the deficit falls with it, not one point. The taped route on the wedges
of ``4 4 4``, ``6 6 6`` and ``8 8 8`` reads 0.817718, 0.720006 and 0.699317
C/m^2 against ``pw.x``'s 0.693273, deficits of 15.22, 3.71 and 0.86 per cent,
beside 15.30, 3.73, 0.86 for the ultrasoft cell and 14.54, 3.65, 1.01 for the
norm-conserving calibration. Two more rungs were run and are not in the test
for their cost: 0.694466 at ``10 10 10`` and 0.693264 at ``12 12 12``, deficits
of 0.17 and -0.001 per cent. The tolerance on the converged comparison is set
from what the reference itself is worth, not from that lucky last point:
``pw.x`` at 21 strings over 12x12 reads 0.693927 at 30 Ry against 0.693240 at
15 over 8x8, so the string mesh alone is 0.10 per cent, and the ``12 12 12``
rung scatters by 0.05 per cent between 25, 30 and 35 Ry (0.693264, 0.693625,
0.693404). The extrapolated PAW value of the three rungs here is held to 0.2
per cent, where it reads 0.07.

**What this comparison cannot see** is the one-centre cross term that
``method='zstar_eu'`` lacks on PAW: it moves ``e_14`` by 5.3e-04 C/m^2 at
every mesh from ``8 8 8`` to ``12 12 12``, 0.08 per cent, below that 0.10.
The tensor is checked as a whole here, and the term is carried because the
taped route differentiates the energy.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.response.piezo import PAW_KMESH_DEFICIT
from defuqumat.system import build_system
from defuqumat.units import BOHR_RADIUS_SI, ELECTRON_SI

pytestmark = pytest.mark.regression

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: The shear of the two committed cells, ``E[1,2] = E[2,1]``.
SHEAR = 0.005

#: ``e_14`` in C/m^2 of the taped route on the wedges of 4 4 4, 6 6 6, 8 8 8,
#: measured on D22 (CPU, ``memory_mode='memory'``), 2026-10-19.
RECORDED_E14 = (0.817717551, 0.720005531, 0.699317427)


def _pw_x_phase(name: str) -> tuple[float, int]:
    """The total Berry phase off a committed ``lberry`` output, and its quantum.

    Read as ``P / mod * MOD_TOT`` from the ``(e/Omega).bohr`` line, which has
    seven decimals where the ``TOTAL PHASE`` line has five.
    """
    text = (CASES / f"reference.out.{name}").read_text()
    p, mod = map(float, re.search(
        r"P =\s+(\S+)\s+\(mod\s+(\S+)\)\s+\(e/Omega\)\.bohr", text).groups())
    mod_tot = int(re.search(r"MOD_TOT:\s+(\d+)", text).group(1))
    return p / mod * mod_tot, mod_tot


def _cell(name: str) -> np.ndarray:
    """The ``CELL_PARAMETERS bohr`` rows of a committed input."""
    lines = (CASES / f"{name}.in").read_text().splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.strip().startswith("CELL_PARAMETERS"))
    return np.array([[float(x) for x in lines[start + 1 + i].split()]
                     for i in range(3)])


def pw_x_e14() -> float:
    """``e_14`` in C/m^2 from ``pw.x``'s polarization on the two sheared cells.

    ``piezo_measure.finite_difference``'s contraction: the wrapped phase
    difference along each ``gdir`` times the ``+s`` cell's lattice vector, over
    its volume. For a pure ``y``-``z`` shear the ``x`` components of the two
    cells' vectors are equal and so are the volumes, so this is exact for
    ``e_14``.
    """
    vectors = _cell("alas-piezo-paw-plus-g1")
    volume = abs(np.linalg.det(vectors))
    derivative = np.zeros(3)
    for gdir in (1, 2, 3):
        plus, quantum = _pw_x_phase(f"alas-piezo-paw-plus-g{gdir}")
        minus, _ = _pw_x_phase(f"alas-piezo-paw-minus-g{gdir}")
        delta = plus - minus
        delta = quantum * np.angle(np.exp(2j * np.pi * delta / quantum)) / (2 * np.pi)
        derivative += delta * vectors[gdir - 1]
    factor = ELECTRON_SI / BOHR_RADIUS_SI**2
    return float(derivative[0] / volume / (4.0 * SHEAR) * factor)


def test_the_pw_x_reference_reads_what_the_record_says():
    """The committed outputs give 0.693273 C/m^2.

    Cheap (no calculation, so it is in the fast gate), and it pins the parser: the record's number, ``pw_x_e14`` and the
    constants in ``response/piezo.py`` are one number.
    """
    assert pw_x_e14() == pytest.approx(0.693273, abs=2e-6)


@pytest.mark.slow
def test_the_paw_tensor_falls_onto_pw_x_berry_value_as_the_mesh_grows(monkeypatch):
    """Three rungs, the deficit at each, and its limit against ``pw.x``.

    ``memory_mode='memory'`` so the taped route walks the k axis a point at a
    time, about 3 GB at most and about seven minutes on one core for the three
    rungs. On the code before 2026-10-19 the ladder raised "not implemented
    for a PAW dataset" before its first ground state.
    """
    from defuqumat.workflows.piezo_ladder import piezoelectric_kmesh_ladder

    monkeypatch.setenv("DEFUQUMAT_MEMORY_MODE", "memory")
    system = build_system(read_pw_input(CASES / "alas-piezo-paw.in"))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    assert all(p.is_paw for p in pseudos)

    ladder = piezoelectric_kmesh_ladder(system, pseudos, meshes=(4, 6, 8))
    e14 = np.array(ladder.e14)
    assert e14 == pytest.approx(RECORDED_E14, abs=2e-5)

    reference = pw_x_e14()
    deficit = (e14 - reference) / e14 * 100.0
    assert deficit == pytest.approx(PAW_KMESH_DEFICIT, abs=0.02)
    # Falls the way the calibration's does, about a quarter left per step ...
    assert deficit[1] / deficit[0] < 0.3
    assert deficit[2] / deficit[1] < 0.3
    # ... and its limit is pw.x's value within the calibration's own floor.
    steps = np.diff(e14)
    limit = e14[2] - steps[1] ** 2 / (steps[1] - steps[0])
    assert abs(limit - reference) / reference < 2.0e-3, (
        f"the extrapolated tensor {limit:.6f} is "
        f"{abs(limit - reference) / reference * 100:.2f} per cent from pw.x's "
        f"{reference:.6f}, where 0.07 was measured"
    )
