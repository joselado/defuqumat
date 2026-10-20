"""The meta-GGA exchange torque on a converged noncollinear texture (I110).

``h-chain-90deg.in`` under TB09 with the default ``noncollinear_gga =
'grad_m'``, in a box 5 bohr across the chain rather than 12 (in the 12 bohr box
TB09's potential in the vacuum kept the SCF from converging in 300 iterations):
four hydrogen moments turned 90 degrees from one another, no field and no
constraint.

Two comparisons, with numbers from ``notes/incompat/I110.md`` (D22, run 3):

* ``SCFResult.potential`` is rebuilt at convergence on the output ``(rho,
  tau)``, so ``m x B_xc`` read off it is the field ``torque_of_result``
  rebuilds; the site gap is round-off, 1.4e-18 to 5.2e-18 Ry at ``conv_thr``
  1e-6 to 1e-12, while dropping ``tau``'s vector part moves the site torques by
  1.3e-5 Ry at 1e-10.
* ``potential - potential_change`` is the field the last iteration applied; the
  gap to it is the self-consistency residual, 2.0e-5, 3.3e-5, 7.7e-6 and 7.1e-7
  Ry at ``conv_thr`` 1e-6, 1e-8, 1e-10 and 1e-12.

**The site torques of this cell are a residue**: the 90-degree spiral's spin
space group (inversion through a site with a two-fold spin rotation about that
site's moment) forces the component normal to the spin plane to zero on every
site, which is the component that carries the reading, and the largest falls
with ``conv_thr`` from 2.9e-5 to 9.6e-7 Ry as the texture settles. The test checks the plumbing, not a
nonzero physical torque.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat import Calculator
from defuqumat.scf.spin_torque import exchange_torque, torque_of_result

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASE = Path(__file__).resolve().parents[1] / "data" / "qe" / "h-chain-90deg.in"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: The site gap to the stored (output) potential, in Ry: measured <= 5.2e-18.
SITE_GAP = 1.0e-15
#: The site gap to the field the last iteration applied at ``conv_thr = 1e-10``,
#: in Ry: measured 7.7e-6, against 2.0e-5 at 1e-6 and 7.1e-7 at 1e-12.
SITE_GAP_APPLIED = 2.0e-5
#: How far above the identity gap the gap with ``tau``'s vector part dropped
#: must sit. Relative rather than a fixed floor: the dropped-``tau`` gap tracks
#: the site torque, which is a symmetry residue on this cell and falls with
#: where the loop stopped (1.3e-5 Ry at 1e-10, 3.7e-6 at 1e-12), while the
#: identity gap is round-off (<= 5.2e-18), so the ratio holds at every
#: measured ``conv_thr`` by about nine orders beyond this factor.
WITHOUT_TAU_FACTOR = 1.0e3


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    yield
    jax.clear_caches()


def test_the_meta_torque_is_the_scf_field_torque():
    text = CASE.read_text().replace(
        "    nosym = .true.", "    nosym = .true.\n    input_dft = 'tb09'", 1
    ).replace("conv_thr = 1.0d-11", "conv_thr = 1.0d-10", 1).replace(
        "celldm(1) = 12.0, celldm(3) = 1.66666666666667",
        "celldm(1) = 5.0, celldm(3) = 4.00000000000000", 1)
    calc = Calculator.from_text(text, pseudo_dir=PSEUDO, max_iterations=300)
    calculation = calc.calculation
    scf = calc.get_scf()
    assert scf.converged
    assert calculation.noncollinear_gga == "grad_m"

    rebuilt = torque_of_result(calculation, scf)
    own = exchange_torque(scf.density, scf.potential, calculation.system.cell,
                          calculation.local_regions())
    sites, reference = np.asarray(rebuilt.sites), np.asarray(own.sites)
    # The field has a part off m (measured 0.98647), so the torque is not the
    # kubler null.
    assert float(rebuilt.parallel_fraction) < 0.99
    assert np.abs(sites - reference).max() < SITE_GAP

    applied = exchange_torque(
        scf.density, jnp.asarray(scf.potential) - jnp.asarray(scf.potential_change),
        calculation.system.cell, calculation.local_regions())
    assert np.abs(sites - np.asarray(applied.sites)).max() < SITE_GAP_APPLIED

    tau = jnp.asarray(scf.tau)
    dropped = torque_of_result(calculation, scf, density=scf.density,
                               tau=tau.at[1:].set(0.0))
    identity = max(float(np.abs(sites - reference).max()), 1.0e-17)
    assert (np.abs(np.asarray(dropped.sites) - reference).max()
            > WITHOUT_TAU_FACTOR * identity)
