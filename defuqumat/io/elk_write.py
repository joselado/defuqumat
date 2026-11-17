"""Writing an Elk ``STATE.OUT``: the inverse of :func:`defuqumat.io.elk.read_elk_state`.

The records are ``writestate.f90``'s (Elk 11.0.2), in its order: the version,
``spinpol``, the species and their radial meshes, the grid sizes and flags,
``efermi`` and ``dlefe``, then one record per ``(muffin-tin, interstitial)``
pair -- the density, the Coulomb, exchange-correlation and Kohn-Sham
potentials -- and for a spin-polarized state the magnetization, ``B_xc`` and
``B_s`` (the last on the coarse radial mesh ``nrcmtmax``). Potentials and
energies are in Hartree on file and in Rydberg on :class:`ElkState`, so they
are halved here; the factor is a power of two, and a state read and written
back is Elk's file byte for byte, the uninitialised padding rows past
``nrmt(is)`` included (``tests/unit/test_elk_write.py``).

**What Elk does with the file is the point to understand before writing one.**
``gndstate.f90`` with ``task = 1`` reads the state and goes straight to
``gencore`` and the first diagonalisation with ``vsmt``/``vsir`` (and
``bsmt``/``bsir``): it starts from the **potential**, and the density in the
file enters only the mixer. With the default potential mixing it does not
enter at all. So a state written for Elk carries Elk's potential of its
density: :func:`defuqumat.io.elk_allelectron.write_elk_state_from` computes it
with the transcribed ``potks`` (:mod:`defuqumat.io.elk_potential`). The other
route, kept for a density written with any potential, is one hand-off loop
that keeps the density and builds Elk's own potential from it: ``task = 1``, ``mixrho = .true.``, ``mixtype = 0``, ``beta0 = 0``,
``maxscl = 1``, ``nwrite = 1``. ``mixlinear`` then sets the mixed density to
the old one exactly, ``potks`` runs on it, and the in-loop ``writestate``
leaves the pair on file (:data:`HANDOFF_INPUT`, ``PLAN.md`` P181). ``nwrite``
is not optional: the write after the loop happens only when ``maxscl > 1``
(``gndstate.f90:335``), and without it the hand-off ran, kept the density and
wrote nothing, so the next run started from the old potential.

Refused, because the records are not carried by :class:`ElkState`: a fixed
spin moment state (``bfsmc``, ``bfsmcmt``), a meta-GGA state (``wxcmt``,
``wxcir``) and a fixed tensor moment (``vmftm``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["HANDOFF_INPUT", "write_elk_state"]

HARTREE_TO_RY = 2.0

#: The ``elk.in`` lines of the hand-off loop that turns a written density into
#: a state Elk can continue from: one loop, the density mixed with weight zero
#: on the new one, and Elk's own ``potks`` on the result.
HANDOFF_INPUT = """tasks
  1

mixrho
  .true.

mixtype
  0

beta0
  0.0

maxscl
  1

nwrite
  1
"""


def write_elk_state(path, state) -> Path:
    """Write ``state`` (an :class:`~defuqumat.io.elk.ElkState`) as ``STATE.OUT``.

    ``path`` is the file, or a run directory to write ``STATE.OUT`` into.
    """
    from scipy.io import FortranFile

    path = Path(path)
    if path.is_dir():
        path = path / "STATE.OUT"
    if state.fsmtype != 0:
        raise NotImplementedError(
            "a fixed spin moment state carries bfsmc and bfsmcmt after the "
            "magnetic records, which ElkState does not hold"
        )
    if state.xcgrad in (3, 4, 5, 6):
        raise NotImplementedError(
            "a meta-GGA state carries wxcmt and wxcir, which ElkState does not hold"
        )
    if state.ftmtype != 0:
        raise NotImplementedError(
            "a fixed tensor moment state carries vmftm, which ElkState does not hold"
        )
    spinpol = state.magmt is not None
    geometry = state.geometry
    natmtot = int(sum(geometry.natoms))
    ngtot = int(np.prod(state.ngridg))
    i32 = np.int32

    def ints(*values):
        return np.array(values, dtype=i32)

    def pair(mt, ir, scale=1.0):
        mt = np.asarray(mt, dtype=np.float64) / scale
        ir = np.asarray(ir, dtype=np.float64) / scale
        return np.concatenate([mt.reshape(-1, order="F"), ir.reshape(-1, order="F")])

    with FortranFile(path, "w") as f:
        f.write_record(ints(*state.version))
        f.write_record(ints(1 if spinpol else 0))
        f.write_record(ints(len(geometry.natoms)))
        f.write_record(ints(state.lmmaxo))
        f.write_record(ints(state.nrmtmax))
        f.write_record(ints(state.nrcmtmax))
        for natoms, nrmt, rsp, nrcmt, rcmt in zip(
                geometry.natoms, state.nrmt, state.rsp, state.nrcmt, state.rcmt):
            f.write_record(ints(natoms))
            f.write_record(ints(nrmt))
            f.write_record(np.asarray(rsp, dtype=np.float64)[:nrmt])
            f.write_record(ints(nrcmt))
            f.write_record(np.asarray(rcmt, dtype=np.float64)[:nrcmt])
        f.write_record(ints(*state.ngridg))
        f.write_record(ints(state.ngvec))
        f.write_record(ints(state.ndmag))
        f.write_record(ints(state.nspinor))
        f.write_record(ints(state.fsmtype))
        if state.version[0] > 2:
            f.write_record(ints(state.ftmtype))
        f.write_record(ints(state.dftu))
        f.write_record(ints(state.lmmaxdm))
        f.write_record(ints(state.xcgrad))
        f.write_record(np.array([state.efermi / HARTREE_TO_RY]))
        f.write_record(np.array([state.dlefe / HARTREE_TO_RY]))
        for mt, ir, scale in (
                (state.rhomt, state.rhoir, 1.0),
                (state.vclmt, state.vclir, HARTREE_TO_RY),
                (state.vxcmt, state.vxcir, HARTREE_TO_RY),
                (state.vsmt, state.vsir, HARTREE_TO_RY)):
            if np.shape(mt) != (state.lmmaxo, state.nrmtmax, natmtot) or np.size(ir) != ngtot:
                raise ValueError(f"a muffin-tin block {np.shape(mt)} or interstitial "
                                 f"{np.size(ir)} that is not this state's mesh")
            f.write_record(pair(mt, ir, scale))
        if spinpol:
            for mt, ir, scale in (
                    (state.magmt, state.magir, 1.0),
                    (state.bxcmt, state.bxcir, HARTREE_TO_RY),
                    (state.bsmt, state.bsir, HARTREE_TO_RY)):
                f.write_record(pair(mt, ir, scale))
        if state.dftu != 0:
            vmat = np.asarray(state.vmatmt, dtype=np.complex128) / HARTREE_TO_RY
            f.write_record(vmat.reshape(-1, order="F"))
    return path
