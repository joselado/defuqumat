"""The calls of the gauge sweep (``tools/gauge_sweep.py``): one cheap call per ``get_*``.

A call the cell does not admit raises, and the sweep records the refusal with its message. The
arguments are the cheapest that exercise each assembly. Left out, by fable's review of
``plans/gauge-sweep-p151.md``: the calls that run SCFs of their own (``get_relax``, the spiral
and ``get_relaxed_*`` family, ``get_elk_seed``, ``get_magnetoelectric_tensor``,
``get_piezoelectric_kmesh_ladder``), whose move is the SCF's noise under either gauge, and the
expensive members of families a cheaper member already exercises (the real-time and harmonic
calls, ``get_raman_tensors``, ``get_vibrational_spectrum``, ``get_electrostriction``, the
ultracell, the full ``get_phonons``). The force-theorem family takes a second, spinor
calculator and is left to a cell built for it.
"""

from __future__ import annotations

import numpy as np

FREQUENCIES = np.array([0.05, 0.1, 0.2])


#: The discriminating reruns of the first pass, each the A/B that tells a gauge defect from the
#: thing it could also be: the responses at a tighter self-consistency (a move that shrinks with
#: ``tr2`` is the response's own noise), the conductivity with ``nbnd`` off a multiplet (a move
#: that goes with the cut is the truncation ``band_cut_gap`` reports), and the calls the first
#: pass did not reach.
FOLLOWUPS = {
    "tight:get_born_charges": lambda c: c.get_born_charges(tr2=1e-20, threshold=1e-14),
    "tight:get_phonons_at_q": lambda c: c.get_phonons_at_q((0.0, 0.0, 0.5), tr2=1e-20,
                                                           threshold=1e-14),
    "tight:get_strain_response": lambda c: c.get_strain_response(tr2=1e-20, threshold=1e-14),
    "get_electrostriction": lambda c: c.get_electrostriction(),
    "get_shg": lambda c: c.get_shg(nbnd=8),
    "get_shift_current": lambda c: c.get_shift_current(nbnd=8),
    "get_sts": lambda c: c.get_sts(energies=np.array([0.3, 0.4, 0.5])),
    "nbnd8:get_optical_conductivity": lambda c: c.get_optical_conductivity(nbnd=8),
    "nbnd10:get_optical_conductivity": lambda c: c.get_optical_conductivity(nbnd=10),
}


def CALLS(cell: str, followups: bool = False) -> dict:
    grid = (2, 2, 2)
    if followups:
        return dict(FOLLOWUPS)
    return {
        "get_bands": lambda c: c.get_bands(),
        "get_nscf": lambda c: c.get_nscf(),
        "get_dos": lambda c: c.get_dos(),
        "get_pdos": lambda c: c.get_pdos(),
        "get_forces": lambda c: c.get_forces(),
        "get_stress": lambda c: c.get_stress(),
        "get_band_velocities": lambda c: c.get_band_velocities(),
        "get_effective_mass": lambda c: c.get_effective_mass((0.0, 0.0, 0.0)),
        "get_angular_momenta": lambda c: c.get_angular_momenta(),
        "get_dielectric_tensor": lambda c: c.get_dielectric_tensor(),
        "get_born_charges": lambda c: c.get_born_charges(),
        "get_phonons_at_q": lambda c: c.get_phonons_at_q((0.0, 0.0, 0.5)),
        "get_electron_phonon": lambda c: c.get_electron_phonon((0.0, 0.0, 0.5)),
        "get_strain_response": lambda c: c.get_strain_response(),
        "get_piezoelectric_tensor": lambda c: c.get_piezoelectric_tensor(),
        "get_absorption": lambda c: c.get_absorption(FREQUENCIES),
        "get_optical_conductivity": lambda c: c.get_optical_conductivity(),
        "get_structure_factors": lambda c: c.get_structure_factors(),
        "get_stm": lambda c: c.get_stm(height=0.5),
        "get_sts": lambda c: c.get_sts(),
        "get_vertical_transport": lambda c: c.get_vertical_transport(exit_height=0.1, height=0.6),
        "get_momentum_transport": lambda c: c.get_momentum_transport(exit_height=0.1, height=0.6),
        "get_nesting": lambda c: c.get_nesting(grid=grid),
        "get_spin_susceptibility": lambda c: c.get_spin_susceptibility(np.zeros(3), FREQUENCIES),
        "get_magnon_dispersion": lambda c: c.get_magnon_dispersion(np.zeros((1, 3)), FREQUENCIES),
        "get_shift_current": lambda c: c.get_shift_current(),
        "get_shg": lambda c: c.get_shg(),
        "get_berry_curvature": lambda c: c.get_berry_curvature(),
        "get_orbital_magnetization": lambda c: c.get_orbital_magnetization(),
        "get_polarization": lambda c: c.get_polarization(),
        "get_exchange_torque": lambda c: c.get_exchange_torque(),
        "get_z2": lambda c: c.get_z2(),
        "get_z2_3d": lambda c: c.get_z2_3d(),
    }
