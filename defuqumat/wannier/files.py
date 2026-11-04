"""The ``.mmn``, ``.amn`` and ``.eig`` files of Wannier90, read and written.

These are the three files ``pw2wannier90.x`` writes and ``wannier90.x`` reads
(Wannier90 3.1.0 user guide, chapter 8; ``PP/src/pw2wannier90.f90``
``compute_mmn``/``compute_amn``/``write_band``). Writing them lets the matrices
of this code be handed to ``wannier90.x``; reading them lets this code's engine
run on ``pw2wannier90.x``'s matrices. Both directions are how the two halves
are validated separately.

Conventions, all 1-indexed in the files:

- ``.mmn``: a comment, ``num_bands num_kpts nntot``, then per ``(k, b)`` a line
  ``k k2 G1 G2 G3`` and ``num_bands^2`` complex numbers with the *first* band
  index running fastest, ``M_mn(k, b) = <u_mk|u_n,k+b>`` with ``k + b = k2 + G``.
- ``.amn``: a comment, ``num_bands num_kpts num_proj``, then ``m n k Re Im``.
- ``.eig``: ``n k e_nk`` in eV.

Arrays here are in Wannier90's memory layout, ``M[m, n, nn, k]`` and
``A[m, n, k]``, because the engine (a port of the Fortran) indexes them so.
"""

from __future__ import annotations

import gzip

import numpy as np

__all__ = ["read_eig", "read_amn", "read_mmn", "write_eig", "write_amn", "write_mmn"]


def _open(path):
    """A text file, or its ``.gz``."""
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def read_eig(path, num_bands: int, num_kpts: int) -> np.ndarray:
    """``(num_bands, num_kpts)`` eigenvalues, in the file's unit (eV)."""
    with _open(path) as f:
        data = np.loadtxt(f).reshape(-1, 3)
    eig = np.zeros((num_bands, num_kpts))
    eig[data[:, 0].astype(int) - 1, data[:, 1].astype(int) - 1] = data[:, 2]
    return eig


def read_amn(path, num_bands: int, num_kpts: int, num_wann: int) -> np.ndarray:
    """``(num_bands, num_wann, num_kpts)`` projections."""
    with _open(path) as f:
        f.readline()
        header = tuple(int(x) for x in f.readline().split())
        data = np.loadtxt(f).reshape(-1, 5)
    if header != (num_bands, num_kpts, num_wann):
        raise ValueError(f"{path}: header {header} is not {(num_bands, num_kpts, num_wann)}")
    amn = np.zeros((num_bands, num_wann, num_kpts), dtype=complex)
    m, n, k = (data[:, i].astype(int) - 1 for i in range(3))
    amn[m, n, k] = data[:, 3] + 1j * data[:, 4]
    return amn


def read_mmn(path, num_bands: int, num_kpts: int, nnlist: np.ndarray,
             nncell: np.ndarray) -> np.ndarray:
    """``(num_bands, num_bands, nntot, num_kpts)`` overlaps, matched to the
    neighbour tables of :func:`~defuqumat.wannier.kmesh.kmesh_get` (1-indexed
    ``nnlist``, ``nncell`` shaped ``(3, num_kpts, nntot)``)."""
    nntot = nnlist.shape[1]
    with _open(path) as f:
        f.readline()
        header = tuple(int(x) for x in f.readline().split())
        if header != (num_bands, num_kpts, nntot):
            raise ValueError(f"{path}: header {header} is not {(num_bands, num_kpts, nntot)}")
        lines = f.read().split("\n")
    mmn = np.zeros((num_bands, num_bands, nntot, num_kpts), dtype=complex)
    block = num_bands * num_bands
    row = 0
    for _ in range(num_kpts * nntot):
        k1, k2, g1, g2, g3 = (int(x) for x in lines[row].split())
        values = np.array(" ".join(lines[row + 1: row + 1 + block]).split(), dtype=float)
        row += 1 + block
        matrix = (values[0::2] + 1j * values[1::2]).reshape(num_bands, num_bands).T
        hit = [nn for nn in range(nntot) if nnlist[k1 - 1, nn] == k2
               and tuple(nncell[:, k1 - 1, nn]) == (g1, g2, g3)]
        if len(hit) != 1:
            raise ValueError(f"{path}: neighbour {k1} -> {k2} ({g1},{g2},{g3}) is not in the table")
        mmn[:, :, hit[0], k1 - 1] = matrix
    return mmn


def write_eig(path, eig_ev: np.ndarray) -> None:
    """Write ``(num_bands, num_kpts)`` eigenvalues in eV."""
    nb, nk = eig_ev.shape
    with open(path, "w") as f:
        for k in range(nk):
            for n in range(nb):
                f.write(f"{n + 1:5d}{k + 1:5d}{eig_ev[n, k]:18.12f}\n")


def write_amn(path, amn: np.ndarray, comment: str = "written by defuqumat") -> None:
    nb, nw, nk = amn.shape
    with open(path, "w") as f:
        f.write(comment + "\n")
        f.write(f"{nb:12d}{nk:12d}{nw:12d}\n")
        for k in range(nk):
            for n in range(nw):
                for m in range(nb):
                    a = amn[m, n, k]
                    f.write(f"{m + 1:5d}{n + 1:5d}{k + 1:5d}{a.real:18.12f}{a.imag:18.12f}\n")


def write_mmn(path, mmn: np.ndarray, nnlist: np.ndarray, nncell: np.ndarray,
              comment: str = "written by defuqumat") -> None:
    nb, _, nntot, nk = mmn.shape
    with open(path, "w") as f:
        f.write(comment + "\n")
        f.write(f"{nb:12d}{nk:12d}{nntot:12d}\n")
        for k in range(nk):
            for nn in range(nntot):
                g = nncell[:, k, nn]
                f.write(f"{k + 1:5d}{int(nnlist[k, nn]):5d}{g[0]:5d}{g[1]:5d}{g[2]:5d}\n")
                for n in range(nb):
                    for m in range(nb):
                        z = mmn[m, n, nn, k]
                        f.write(f"{z.real:18.12f}{z.imag:18.12f}\n")
