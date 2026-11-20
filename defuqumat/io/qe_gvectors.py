"""``pw.x``'s global order of the G-vectors, for a file ``pw.x`` will read.

``read_rhog`` (``Modules/io_base.f90``) takes the coefficients of a
``charge-density.dat`` in the order they are stored and scatters them with
``ig_l2g``, ``pw.x``'s own global G index. It reads the Miller indices only
when the file's ``gamma_only`` flag differs from the run's (``charge_k_to_g``),
and it zero-pads a short file with an ``infomsg`` and nothing else. So a file
written for ``pw.x`` must list its G-vectors in exactly ``pw.x``'s order, or
``pw.x`` reads a scrambled density without saying so.

That order is ``ggen``'s (``Modules/recvec_subs.f90``): the Miller indices are
enumerated with ``i`` outermost and ``k`` innermost, each from ``-n`` to ``n``
(from 0 for the first non-zero index of a ``gamma_only`` run, the half
sphere), the ones with ``|G|^2 <= gcutm`` kept in that order, and then sorted by
``hpsort_eps`` (``FFTXlib/tests/sort.f90``, the same routine as
``upflib/sort.f90``) on ``|G|^2`` with ``eps8``. Both are transcribed
literally rather than replaced by a stable sort: ``hpsort_eps`` treats two
values within ``1e-8`` as equal and breaks the tie by the enumeration index, a
comparison that is not transitive, so what it returns for a near tie is a
property of the heap and not of any ordering a stable sort could be given.

``|G|^2`` is accumulated as ``ggen`` does, ``tx = i b1``, ``ty = tx + j b2``,
``t = ty + k b3``, in units of ``(2 pi / alat)^2``, and a value below ``eps8``
is set to zero. The loop bounds do not enter the result: the enumeration index
only orders the triples lexicographically, and every triple inside the sphere
is enumerated whatever the bounds, as long as they hold the sphere.

Measured against the Miller record of ``pw.x`` 7.5's own files: identical on
the cubic, the molecule-in-a-box and the ``K_POINTS gamma`` fixtures under
``tests/data/qe/save/`` and on a triclinic cell (``PLAN.md`` P180).

**The order is the global one, and ``memory = 'small'`` does not use it on
more than one FFT rank** (audit 2026-11-20 #3, read in the source and not run).
``init_run.f90:89-90`` calls ``ggen`` with ``no_global_sort = smallmem``, and
``smallmem`` is ``memory == 'small'`` (``input.f90:902``). Then each rank
enumerates only its own sticks (``recvec_subs.f90:140-142``), sorts them
locally, and numbers them ``ig_l2g = ng + ngm_offset`` (``:238-239``), so the
global index is blocked by rank. ``read_rhog`` scatters with that ``ig_l2g``
and ``ngm_g`` is unchanged, so a file in the global order is placed on the
wrong G-vectors with no message, while ``G = 0`` and so the charge usually
stay right. On one FFT rank the local sort is the global one. Writing the
rank-blocked order would need ``pw.x``'s stick distribution, and is not done.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

__all__ = ["qe_gvector_order"]

#: ``constants.f90``'s ``eps8``, the tie tolerance of ``hpsort_eps`` and the
#: threshold below which ``ggen`` stores ``|G|^2`` as zero.
EPS8 = 1.0e-8


@lru_cache(maxsize=1)
def _kernels():
    """The two loops, compiled on first use (numba, host side only)."""
    import numba

    @numba.njit(cache=True)
    def enumerate_sphere(bg, gcutm, ni, nj, nk, gamma_only):
        # ``bg[:, a]`` is ``b_a``, Fortran's ``bg(1:3, a)``, in 2 pi / alat.
        capacity = (2 * ni + 1) * (2 * nj + 1) * (2 * nk + 1)
        mill = np.zeros((capacity, 3), np.int64)
        g2 = np.zeros(capacity + 1)  # 1-indexed for hpsort_eps
        tt = np.zeros(2 * nk + 1)
        ngm = 0
        istart = 0 if gamma_only else -ni
        for i in range(istart, ni + 1):
            jstart = 0 if (gamma_only and i == 0) else -nj
            tx0 = i * bg[0, 0]
            tx1 = i * bg[1, 0]
            tx2 = i * bg[2, 0]
            for j in range(jstart, nj + 1):
                kstart = 0 if (gamma_only and i == 0 and j == 0) else -nk
                ty0 = tx0 + j * bg[0, 1]
                ty1 = tx1 + j * bg[1, 1]
                ty2 = tx2 + j * bg[2, 1]
                for k in range(kstart, nk + 1):
                    t0 = ty0 + k * bg[0, 2]
                    t1 = ty1 + k * bg[1, 2]
                    t2 = ty2 + k * bg[2, 2]
                    tt[k - kstart] = t0 * t0 + t1 * t1 + t2 * t2
                for k in range(kstart, nk + 1):
                    value = tt[k - kstart]
                    if value <= gcutm:
                        ngm += 1
                        g2[ngm] = value if value > 1.0e-8 else 0.0
                        mill[ngm - 1, 0] = i
                        mill[ngm - 1, 1] = j
                        mill[ngm - 1, 2] = k
        return mill[:ngm].copy(), g2[: ngm + 1].copy()

    @numba.njit(cache=True)
    def hpsort_eps(n, ra, ind, eps):
        # Literal, 1-indexed: ra[1..n], ind[1..n]; index 0 is unused.
        if ind[1] == 0:
            for i in range(1, n + 1):
                ind[i] = i
        if n < 2:
            return
        l = n // 2 + 1
        ir = n
        while True:
            if l > 1:
                l = l - 1
                rra = ra[l]
                iind = ind[l]
            else:
                rra = ra[ir]
                iind = ind[ir]
                ra[ir] = ra[1]
                ind[ir] = ind[1]
                ir = ir - 1
                if ir == 1:
                    ra[1] = rra
                    ind[1] = iind
                    break
            i = l
            j = l + l
            while j <= ir:
                if j < ir:
                    if abs(ra[j] - ra[j + 1]) >= eps:
                        if ra[j] < ra[j + 1]:
                            j = j + 1
                    else:
                        if ind[j] < ind[j + 1]:
                            j = j + 1
                if abs(rra - ra[j]) >= eps:
                    if rra < ra[j]:
                        ra[i] = ra[j]
                        ind[i] = ind[j]
                        i = j
                        j = j + j
                    else:
                        j = ir + 1
                else:
                    if iind < ind[j]:
                        ra[i] = ra[j]
                        ind[i] = ind[j]
                        i = j
                        j = j + j
                    else:
                        j = ir + 1
            ra[i] = rra
            ind[i] = iind

    return enumerate_sphere, hpsort_eps


def qe_gvector_order(at, alat: float, ecutrho: float,
                     gamma_only: bool = False) -> np.ndarray:
    """The Miller indices of ``pw.x``'s dense G set, in its global order.

    Args:
        at: ``(3, 3)`` lattice vectors as rows, bohr.
        alat: the lattice parameter ``pw.x`` ran with, bohr (``celldm(1)``, or
            ``A`` in bohr); ``|G|^2`` and the tie tolerance are in its units.
        ecutrho: the density cutoff, Ry.
        gamma_only: the half sphere of a ``K_POINTS gamma`` run.

    Returns ``(ngm_g, 3)`` integers, ``G = 0`` first.
    """
    at_alat = np.asarray(at, dtype=float) / float(alat)
    # recips: b_a . a_b = delta_ab, in units of 2 pi / alat.
    bg = np.ascontiguousarray(np.linalg.inv(at_alat))  # columns are b_a
    tpiba = 2.0 * np.pi / float(alat)
    gcutm = float(ecutrho) / tpiba**2
    # |m_a| <= |G| |a_a| / (2 pi) = sqrt(gcutm) |a_a| in these units; one more
    # for safety, since the bounds do not change the order.
    bounds = [int(np.sqrt(gcutm) * np.linalg.norm(a)) + 1 for a in at_alat]
    enumerate_sphere, hpsort_eps = _kernels()
    mill, g2 = enumerate_sphere(bg, gcutm, *bounds, bool(gamma_only))
    ngm = len(mill)
    ind = np.zeros(ngm + 1, dtype=np.int64)
    hpsort_eps(ngm, g2, ind, EPS8)
    order = mill[ind[1:] - 1]
    if ngm and np.any(order[0] != 0):
        raise AssertionError("ggen's order puts G = 0 first")
    return order
