"""I80: the phonon at ``q != 0`` with a nonlinear core correction, at gate speed.

The guard of :func:`~defuqumat.response.phononq.require_a_two_sphere_regime` is
a function of a handful of flags, so it is checked here on a stand-in
calculation rather than on a ground state.
"""

from types import SimpleNamespace

import pytest

from defuqumat.response.phononq import require_a_two_sphere_regime


def _calculation(**changes):
    """A stand-in carrying exactly the fields the guard reads, every one in the
    regime the route carries; ``changes`` moves one of them out of it."""
    system = SimpleNamespace(nspin=1, noncolin=False, spiral_q=None)
    fields = dict(
        is_ultrasoft=False, rho_core_g=None, use_symmetry=False,
        functional=SimpleNamespace(is_meta=False), source_free=False,
        is_hubbard=False, system=system,
    )
    for name, value in changes.items():
        if name in ("nspin", "noncolin", "spiral_q"):
            setattr(system, name, value)
        else:
            fields[name] = value
    return SimpleNamespace(**fields)


def test_a_core_correction_is_refused():
    with pytest.raises(NotImplementedError, match="nonlinear core correction"):
        require_a_two_sphere_regime(_calculation(rho_core_g=object()), None)
