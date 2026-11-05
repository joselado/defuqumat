"""Wannier functions from the selected columns of the density matrix (SCDM).

The reference is QE 7.5 ``pw2wannier90.x`` with ``scdm_proj = .true.``, then
Wannier90 3.1.0 ``wannier90.x`` (``dis_num_iter = 0``, ``num_iter = 3000``,
``conv_window = 3``, ``conv_tol = 1e-10``), on the inputs in
``tests/data/wannier/scdm/``. The silicon ``nscf`` there pins the FFT grid to
16^3 (``nr1 = nr2 = nr3 = 16``), the grid our SCF uses: QE's ``nosym`` nscf would
drop the factor 4 of the fractional translations and run on 15^3, which moves the
selected points. A tie between columns of equal norm, which the symmetry makes
common at Gamma, can be broken differently by the two LAPACKs, so the checks are
on the spreads; the selected points are compared only where they are measured to
be the same set.
"""

from pathlib import Path

import numpy as np
import pytest

import defuqumat  # noqa: F401
from defuqumat.wannier import scdm as scdmmod

pytestmark = pytest.mark.unit

DATA = Path(__file__).resolve().parents[1] / "data" / "wannier"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: ``wannier90.x`` on ``pw2wannier90.x``'s SCDM files, from the ``.wout`` in
#: ``tests/data/wannier/scdm/`` (A^2).
QE = {"diamond-iso": 2.349146274, "si-iso": 6.385637632, "si-erfc16": 20.007424685}


def test_the_weights_are_pw2wannier90s():
    e = np.array([-2.0, 0.0, 1.0, 3.5])
    assert np.all(scdmmod.occupation(e, "isolated", None, None) == 1.0)
    from scipy.special import erfc
    assert np.allclose(scdmmod.occupation(e, "erfc", 1.0, 2.0), 0.5 * erfc((e - 1.0) / 2.0),
                       rtol=0, atol=1e-15)
    assert np.allclose(scdmmod.occupation(e, "gaussian", 1.0, 2.0),
                       np.exp(-((e - 1.0) ** 2) / 4.0), rtol=0, atol=1e-15)


def test_the_erfc_fit_recovers_its_parameters():
    from scipy.special import erfc
    e = np.linspace(-6.0, 30.0, 400)
    p = 0.5 * erfc((e - 12.5) / 3.25)
    mu, sigma = scdmmod.fit_erfc(e, p)
    assert abs(mu - 12.5) < 1e-6 and abs(sigma - 3.25) < 1e-6


def test_a_fit_without_an_edge_is_refused():
    with pytest.raises(ValueError, match="never drops below 1/2"):
        scdmmod.fit_erfc(np.linspace(0, 10, 20), np.ones(20))


def test_anint_is_fortrans():
    """The image of a pivot: ``ANINT`` rounds a half away from zero, numpy to even."""
    x = np.array([0.5, -0.5, 0.4, 1.5, 2.5, -0.6])
    assert np.all(scdmmod._anint(x) == np.array([1.0, -1.0, 0.0, 2.0, 3.0, -1.0]))


def _text(kpoints="K_POINTS {automatic}\n 2 2 2 0 0 0"):
    return f"""&control
 calculation='scf'
/
&system
 ibrav=2, celldm(1)=10.2, nat=2, ntyp=1, ecutwfc=12.0
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS {{crystal}}
Si 0.00 0.00 0.00
Si 0.25 0.25 0.25
{kpoints}
"""


def test_the_refusals_fire_before_any_state():
    from defuqumat import Calculator
    from defuqumat.workflows.wannier import run_wannier

    calc = Calculator.from_text(_text(), pseudo_dir=PSEUDO)
    run = lambda **kw: run_wannier(calc.system, calc.pseudos, None,  # noqa: E731
                                   projections="scdm", **{"symmetry": False, **kw})
    with pytest.raises(NotImplementedError, match="'scdm' together with symmetry"):
        run(symmetry=True)
    with pytest.raises(ValueError, match="scdm_sigma = 0.0 must be positive"):
        run(scdm_mu=5.0, scdm_sigma=0.0)
    with pytest.raises(ValueError, match="scdm_sigma = -1.0 must be positive"):
        run(scdm_mu=5.0, scdm_sigma=-1.0)
    with pytest.raises(ValueError, match="is not one of isolated, erfc, gaussian"):
        run(scdm_entanglement="fermi")
    with pytest.raises(ValueError, match="give scdm_mu and scdm_sigma together"):
        run(scdm_mu=5.0)
    with pytest.raises(ValueError, match="'isolated' weighs every band by 1"):
        run(num_wann=4, scdm_entanglement="isolated", scdm_mu=5.0, scdm_sigma=1.0)
    gamma = Calculator.from_text(_text("K_POINTS gamma"), pseudo_dir=PSEUDO)
    with pytest.raises(NotImplementedError, match="K_POINTS gamma"):
        run_wannier(gamma.system, gamma.pseudos, None, projections="scdm", symmetry=False)


@pytest.fixture(scope="module")
def silicon():
    from defuqumat import Calculator

    calc = Calculator.from_file(DATA / "si-pdwf" / "scf.in", pseudo_dir=PSEUDO)
    calc.get_scf()
    return calc


ENGINE = dict(num_iter=3000, conv_window=3, conv_tol=1e-10)


def test_isolated_silicon_is_the_qe_chain(silicon):
    """The four valence bands: the same four bond-centred points as
    ``pw2wannier90.x`` (in another order, a tie), and its spread to 1e-6 A^2
    (measured 3.7e-7, two SCFs and Wannier90's bohr)."""
    w = silicon.get_wannier(4, "scdm", scdm_entanglement="isolated", divisions=(4, 4, 4),
                            symmetry=False, engine_options=ENGINE)
    assert w.num_bands == 4 and w.scdm.grid == (16, 16, 16)
    assert abs(w.spread_terms_angstrom2()["omega_total"] - QE["si-iso"]) < 1e-6
    assert np.max(np.abs(w.spreads_angstrom2 - 1.59640941)) < 1e-6
    theirs = np.array([[0.125, 0.125, -0.375], [0.125, -0.375, 0.125], [-0.375, 0.125, 0.125],
                       [0.125, 0.125, 0.125]])
    key = lambda pts: sorted(map(tuple, np.round(np.mod(pts, 1.0), 6)))  # noqa: E731
    assert key(w.scdm.points) == key(theirs)


def test_the_automatic_parameters_and_the_entangled_spread(silicon):
    """Vitale's protocol on 16 bands: the fit, ``mu = mu_fit - 3 sigma_fit``,
    eight functions, and the spread of the QE chain run with these ``mu`` and
    ``sigma`` (measured 5.1e-7 apart, with different tied points)."""
    w = silicon.get_wannier(projections="scdm", divisions=(4, 4, 4), symmetry=False,
                            engine_options=ENGINE)
    s = w.scdm
    assert w.num_wann == 8 and w.num_bands == 16 and s.entanglement == "erfc"
    assert abs(s.mu_fit - 13.224590) < 1e-4 and abs(s.sigma_fit - 6.509102) < 1e-4
    assert abs(s.mu - (s.mu_fit - 3.0 * s.sigma_fit)) < 1e-12 and s.sigma == s.sigma_fit
    assert w.converged
    assert abs(w.spread_terms_angstrom2()["omega_total"] - QE["si-erfc16"]) < 2e-6
    assert w.projectability is not None and w.projectability.shape == (64, 16)


def test_isolated_diamond_is_the_qe_chain():
    from defuqumat import Calculator

    calc = Calculator.from_file(DATA / "diamond" / "diamond.scf", pseudo_dir=PSEUDO)
    calc.get_scf()
    w = calc.get_wannier(4, "scdm", scdm_entanglement="isolated", divisions=(4, 4, 4),
                         symmetry=False, engine_options=ENGINE)
    assert abs(w.spread_terms_angstrom2()["omega_total"] - QE["diamond-iso"]) < 1e-6
    assert np.max(np.abs(w.bands(w.kpoints) - w.eigenvalues)) < 1e-10
