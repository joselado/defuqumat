"""A ``c=`` projection site is in angstrom, as in a ``.win`` file.

Wannier90's ``param_get_projections`` reads ``c=x,y,z`` in angstrom unless the
block opens with a ``bohr`` line (``parameters.F90:5219-5223``), and the
``.win`` that ``WannierFunctions.write_w90`` writes has no such line. The
projection lines given to ``run_wannier`` and to the Landau ultracell's
``trials=`` were parsed with the reciprocal lattice in 1/bohr, so ``c=`` was
read in bohr, and the same line meant a site 1.89 times farther out to
``wannier90.x`` than to this code (audit 2026-11-20c #5).

No band is solved: ``run_wannier`` is stopped right after the parse, and the
ultracell's trial resolver is called directly.
"""

import numpy as np
import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo.upf import read_upf
from defuqumat.system.builder import build_system
from defuqumat.units import BOHR_TO_ANGSTROM

pytestmark = pytest.mark.unit

PSEUDO = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "pseudo"

_DIAMOND = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 6.74, nat = 2, ntyp = 1, ecutwfc = 30.0
/
&electrons
/
ATOMIC_SPECIES
 C 12.011 C.pbe-hgh.UPF
ATOMIC_POSITIONS crystal
 C 0.00 0.00 0.00
 C 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""

#: A site off every symmetry element, so that a factor in the unit cannot hide.
SITE = np.array([0.31, -0.12, 0.07])


@pytest.fixture(scope="module")
def diamond():
    system = build_system(parse_pw_input(_DIAMOND))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return system, pseudos


def _lines(system):
    """The same site twice: ``c=`` in angstrom, then ``f=``."""
    at = np.asarray(system.cell.at, dtype=float)
    angstrom = SITE @ at * BOHR_TO_ANGSTROM
    cart = ",".join(repr(float(x)) for x in angstrom)
    frac = ",".join(repr(float(x)) for x in SITE)
    return [f"c={cart}:s", f"f={frac}:s"]


class _Parsed(Exception):
    def __init__(self, parsed):
        self.parsed = parsed


def test_run_wannier_reads_a_cartesian_site_in_angstrom(monkeypatch, diamond):
    """The ``c=`` line and its ``f=`` twin are one site."""
    import defuqumat.wannier.projections as projections
    from defuqumat.workflows.wannier import run_wannier

    def stop(parsed, num_wann, select):
        raise _Parsed(parsed)

    monkeypatch.setattr(projections, "apply_projection_selection", stop)
    system, pseudos = diamond
    with pytest.raises(_Parsed) as caught:
        run_wannier(system, pseudos, None, num_wann=2, projections=_lines(system),
                    symmetry=False)
    cartesian, fractional = caught.value.parsed
    assert np.allclose(fractional.site_frac, SITE, rtol=0.0, atol=1e-14)
    assert np.allclose(cartesian.site_frac, fractional.site_frac, rtol=0.0, atol=1e-12)


def test_the_landau_trials_read_a_cartesian_site_in_angstrom(diamond):
    from defuqumat.workflows.landau_ultracell import _resolve_trials

    system, pseudos = diamond
    cartesian, fractional = _resolve_trials(_lines(system), system, pseudos)
    assert np.allclose(cartesian.site_frac, fractional.site_frac, rtol=0.0, atol=1e-12)


def test_a_unit_line_is_still_refused(diamond):
    """``bohr`` would make the next lines bohr in Wannier90; it is not ported."""
    from defuqumat.wannier.projections import parse_projections

    system, _ = diamond
    bg = np.asarray(system.cell.bg, dtype=float) / BOHR_TO_ANGSTROM
    for unit in ("bohr", "Ang"):
        with pytest.raises(NotImplementedError, match="unit line"):
            parse_projections([unit, "c=0,0,0:s"], ["C"], np.zeros((3, 1)), bg)
