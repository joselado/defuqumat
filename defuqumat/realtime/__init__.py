"""Real-time propagation of the Kohn-Sham states under a uniform field.

The field is carried as a wavevector, ``kappa(t) = A(t)/c``, and the
Hamiltonian at time ``t`` is ``H(k + kappa(t))`` on the plane-wave sphere built
for ``k`` (``notes/HARMONICS-NEXT.md``). :mod:`~defuqumat.realtime.pulse` holds the
fields, :mod:`~defuqumat.realtime.propagators` the time steps,
:mod:`~defuqumat.realtime.radial` the table the projectors at ``k + kappa`` are
built from, :mod:`~defuqumat.realtime.propagate` the driver and the current, and
:mod:`~defuqumat.realtime.dense` the dense frequency-domain reference the
perturbative orders are checked against.
"""

from defuqumat.realtime.pulse import (  # noqa: E402, F401  (the one import a script needs)
    Adiabatic, Gaussian, Kick, Ramp, Sin2, Sum, field_amplitude, get_pulse)

__all__ = ["Adiabatic", "Gaussian", "Kick", "Ramp", "Sin2", "Sum", "field_amplitude",
           "get_pulse"]
