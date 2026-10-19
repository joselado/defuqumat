"""The meta-GGA exchange torque on a converged noncollinear texture (I110).

``h-chain-90deg.in`` under TB09 with the default ``noncollinear_gga =
'grad_m'``: four hydrogen moments turned 90 degrees from one another, no field
and no constraint. The torque ``torque_of_result`` rebuilds from the result's
``(density, tau)`` is compared with ``m x B_xc`` read straight off the SCF's own
last potential. The two differ by the self-consistency residual, so the
tolerance is set from the gap measured at ``conv_thr = 1e-10``
(``notes/incompat/I110.md``), and the gap with ``tau``'s vector part dropped is
asserted to be far larger, so that the agreement is ``tau``'s.
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

#: The largest site-torque gap allowed at ``conv_thr = 1e-10``, in Ry; set
#: from the harness's measurement in ``notes/incompat/I110.md``.
SITE_GAP = 1.0e-6
#: The smallest site-torque gap with ``tau``'s vector part dropped, in Ry.
SITE_GAP_WITHOUT_TAU = 1.0e-4


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    yield
    jax.clear_caches()


def test_the_meta_torque_is_the_scf_field_torque():
    text = CASE.read_text().replace(
        "    nosym = .true.", "    nosym = .true.\n    input_dft = 'tb09'", 1
    ).replace("conv_thr = 1.0d-11", "conv_thr = 1.0d-10", 1)
    calc = Calculator.from_text(text, pseudo_dir=PSEUDO, max_iterations=300)
    calculation = calc.calculation
    scf = calc.get_scf()
    assert scf.converged
    assert calculation.noncollinear_gga == "grad_m"

    rebuilt = torque_of_result(calculation, scf)
    own = exchange_torque(scf.density, scf.potential, calculation.system.cell,
                          calculation.local_regions())
    sites, reference = np.asarray(rebuilt.sites), np.asarray(own.sites)
    # The guard that the comparison is not a null: the field has a part off m.
    assert float(rebuilt.parallel_fraction) < 0.999
    assert np.abs(sites).max() > 100.0 * SITE_GAP
    assert np.abs(sites - reference).max() < SITE_GAP

    tau = jnp.asarray(scf.tau)
    dropped = torque_of_result(calculation, scf, density=scf.density,
                               tau=tau.at[1:].set(0.0))
    assert np.abs(np.asarray(dropped.sites) - reference).max() > SITE_GAP_WITHOUT_TAU
