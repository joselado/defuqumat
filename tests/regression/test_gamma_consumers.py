"""The post-SCF consumers under gamma-only storage, against the whole sphere.

``K_POINTS gamma`` stores one plane wave of each ``(G, -G)`` pair. The consumers
that read the states after the SCF -- the projected density of states, the
site angular momenta, the sum-over-states ``chi_0``, the transverse ``chi_0``,
the STM image, the tunnelling spectrum and both tunnelling transmissions --
used to refuse it, and now put the states on the whole sphere first
(``scf/whole_sphere.py``), except the STM image, whose density routine already
carries ``g_to_r_gamma``.

**The reference is the same cell at ``K_POINTS automatic 1 1 1 0 0 0``**, which
is the same physics on the whole sphere and needs no other code. The two SCFs
are independent runs in two storages, so what is compared is what does not
depend on the gauge of the states: Si at Gamma has threefold shells (rule D4),
and every quantity below is a sum over whole shells (the band counts are chosen
so that none is cut) or weighted by a smooth function of the band energy.
Each test prints the relative difference it measured.

Three cells: the undisplaced Si pair of ``test_gamma_only.py`` for the
projections and the optical spectrum, ``h-sheet.in`` for everything that needs
vacuum, and ``h-fcc-magnon.in`` for the two that need a moment.
"""

import re
import warnings
from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

from defuqumat.calculator import Calculator

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

GAMMA = "K_POINTS gamma\n"
WHOLE = "K_POINTS automatic\n 1 1 1 0 0 0\n"

_SILICON = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1, ecutwfc = 12.0,
  nosym = .true., nbnd = 8
/
&electrons
  conv_thr = 1.0d-12, diago_full_acc = .true.
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
"""

#: Where the two planes go on ``h-sheet.in``, whose atom sits at 0.5
#: (``test_transport.py``'s).
SHEET = dict(exit_height=0.15, height=0.85, broadening=0.05)


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """Three cells in two storages each: ``CLAUDE.md``'s rule for a file that
    sweeps cells. The converged pairs stay cached; the executables go."""
    yield
    jax.clear_caches()


def _text(case: str, kpoints: str) -> str:
    if case == "silicon":
        return _SILICON + kpoints
    text = (CASES / f"{case}.in").read_text()
    head = text[: text.index("K_POINTS")]
    if case == "h-fcc-magnon":
        # one shell of eight above the lowest state, so nothing is cut at Gamma
        assert "nbnd = 8" in head
        head = head.replace("nbnd = 8", "nbnd = 9")
    # ``diago_full_acc``: the empty bands are converged as tightly as the
    # occupied ones. Without it the sheet's image and transmission, which weight
    # the partly filled bands at the Fermi level, differed by 3.4e-6 and 4.2e-6
    # between the two storages -- the empty-band floor of two independent SCFs,
    # since the image does not go through the unfold at all.
    return re.sub(r"conv_thr = [0-9.d-]+",
                  "conv_thr = 1.0d-12, diago_full_acc = .true.", head) + kpoints


@lru_cache(maxsize=2)
def _pair(case: str):
    """``(gamma, whole)`` calculators, each converged to 1e-12."""
    out = []
    for kpoints in (GAMMA, WHOLE):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calculator = Calculator.from_text(_text(case, kpoints), PSEUDO,
                                              announce=False)
            calculator.get_scf()
        out.append(calculator)
    assert out[0].calculation.gamma_only
    assert not out[1].calculation.gamma_only
    return tuple(out)


def _relative(name, gamma, whole):
    gamma, whole = np.asarray(gamma), np.asarray(whole)
    assert gamma.shape == whole.shape, (name, gamma.shape, whole.shape)
    scale = float(np.abs(whole).max())
    assert scale > 0.0, f"{name}: the reference is zero, which cannot fail"
    difference = float(np.abs(gamma - whole).max()) / scale
    print(f"{name}: max |gamma - whole| / max |whole| = {difference:.3e}")
    return difference


def _quietly(method, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return method(*args, **kwargs)


def test_the_projected_density_of_states():
    gamma, whole = _pair("silicon")
    a, b = _quietly(gamma.get_pdos), _quietly(whole.get_pdos)
    assert _relative("pdos", a.pdos, b.pdos) < 1e-10  # 1.7e-12 measured
    assert _relative("pdos integrated", a.integrated, b.integrated) < 1e-10  # 6.4e-13


def test_the_optical_spectrum():
    gamma, whole = _pair("silicon")
    options = dict(kernel="rpa", nbnd=8, ecut_response=2.0,
                   static_residual=False)
    frequencies = np.linspace(0.05, 0.6, 6)
    a = _quietly(gamma.get_absorption, frequencies, **options)
    b = _quietly(whole.get_absorption, frequencies, **options)
    assert _relative("eps_M", a.epsilon, b.epsilon) < 1e-8  # 9.9e-10 measured
    assert _relative("eps_M (no local fields)", a.epsilon_no_local_fields,
                     b.epsilon_no_local_fields) < 1e-8  # 5.3e-10


def test_the_stm_image_and_the_tunnelling_spectrum():
    gamma, whole = _pair("h-sheet")
    a = _quietly(gamma.get_stm, height=0.85, shape=(6, 6))
    b = _quietly(whole.get_stm, height=0.85, shape=(6, 6))
    assert _relative("stm image", a.values, b.values) < 1e-8

    fermi = float(whole.get_scf().fermi_energy)
    energies = fermi + np.linspace(-0.1, 0.1, 5)
    a = _quietly(gamma.get_sts, energies, height=0.85, shape=(4, 4))
    b = _quietly(whole.get_sts, energies, height=0.85, shape=(4, 4))
    assert _relative("sts", a.values, b.values) < 1e-8


def test_the_vertical_and_the_momentum_transmission():
    gamma, whole = _pair("h-sheet")
    a = _quietly(gamma.get_vertical_transport, shape=(4, 4), **SHEET)
    b = _quietly(whole.get_vertical_transport, shape=(4, 4), **SHEET)
    assert _relative("vertical transmission", a.image, b.image) < 1e-8

    a = _quietly(gamma.get_momentum_transport, **SHEET)
    b = _quietly(whole.get_momentum_transport, **SHEET)
    assert _relative("momentum weight", a.weight, b.weight) < 1e-8
    assert _relative("momentum tersoff-hamann", a.tersoff_hamann,
                     b.tersoff_hamann) < 1e-8


def test_the_transverse_susceptibility_and_the_site_moments():
    gamma, whole = _pair("h-fcc-magnon")
    density = np.asarray(whole.get_scf().density)
    m = float(np.abs(density[0] - density[1]).max())
    print(f"h-fcc at Gamma: max |rho_up - rho_down| = {m:.3e}")
    assert m > 1e-3  # a moment, or the transverse response tests nothing
    options = dict(nbnd=9, interacting=False, goldstone=False,
                   ecut_response=4.0)
    frequencies = np.array([0.01, 0.05])
    a = _quietly(gamma.get_spin_susceptibility, (0.0, 0.0, 0.0), frequencies,
                 **options)
    b = _quietly(whole.get_spin_susceptibility, (0.0, 0.0, 0.0), frequencies,
                 **options)
    assert _relative("X_0^{+-}", a.chi, b.chi) < 1e-9  # 1.0e-11 measured

    a, b = _quietly(gamma.get_angular_momenta), _quietly(whole.get_angular_momenta)
    assert _relative("site <S>", a.spin, b.spin) < 1e-9  # 3.8e-11 measured
