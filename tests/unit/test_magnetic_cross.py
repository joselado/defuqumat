"""The crosscheck of the orbital-field routes: the exact facts that hold on the committed level sets.

The crosscheck stage (``plans/orbital-field/cross.md``, probe ``plans/orbital-field/mag_cross.py``)
compares route P (Lu et al., PRL 135, 196401 (2025)), route U (the magnetic ultracell of Wang
and Vafek, PRB 108, 245109 (2023)) and the Peierls model on one zero-field graphene state, always
as sorted level sets pooled over the magnetic zone, never as ``eps(k)`` at a named ``k``. That rule
rests on the strong translational symmetry: ``T(a1)`` sends ``kappa2 -> kappa2 - p/q``, so on the
closed ``(2, 2q)`` mesh the full set is ``q`` copies of the set at the representatives
``kappa2 in {0, 1/(2q)}``. Checked here, from the committed files and the Peierls model only:

- route P's full mesh is ``q`` copies of its representatives, to 1e-9 Ry, in every saved set;
- the Peierls model on the ``p_z`` Wannier set obeys the same theorem at ``q = 6`` to round-off,
  and is Hermitian there;
- the guard fires: an open set of "representatives" (``kappa2 in {0, 2/(2q)}``) misses by far
  more than the tolerance.

Numpy and the Peierls model only (dimension 12 at ``q = 6``); about a second.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from defuqumat.magnetic.conventions import MagneticCell
from defuqumat.magnetic.peierls import PeierlsTorus, bonds_from_wannier

pytestmark = pytest.mark.unit

DATA = Path(__file__).resolve().parents[2] / "plans" / "orbital-field"


def _copies(eigenvalues, kpoints, q, members=(0, 1)):
    rep = np.isin(np.round(np.asarray(kpoints)[:, 1] * 2 * q).astype(int), list(members))
    return np.sort(np.tile(np.asarray(eigenvalues)[rep].ravel(), q))


def _route_p_files():
    files = sorted(DATA.glob("route-p-q*.npz"))
    if not files:
        pytest.skip("the route P level sets are not in plans/orbital-field")
    return files


def test_route_p_full_mesh_is_q_copies_of_the_representatives():
    for path in _route_p_files():
        z = np.load(path)
        q = int(z["flux"][1])
        assert len(z["kpoints"]) == 4 * q, path.name
        d = np.max(np.abs(z["sorted_levels"] - _copies(z["eigenvalues"], z["kpoints"], q)))
        assert d < 1e-9, (path.name, d)


def test_an_open_set_of_representatives_is_refused_by_the_comparison():
    z = np.load(DATA / "route-p-q3-full.npz") if (DATA / "route-p-q3-full.npz").exists() else None
    if z is None:
        pytest.skip("route-p-q3-full.npz is not in plans/orbital-field")
    wrong = _copies(z["eigenvalues"], z["kpoints"], 3, members=(0, 2))
    assert np.max(np.abs(z["sorted_levels"] - wrong)) > 1e-4


def _pz_set():
    path = DATA / "graphene-wannier-pz.npz"
    if not path.exists():
        pytest.skip("graphene-wannier-pz.npz is not in plans/orbital-field")
    z = np.load(path)
    shifted = z["ws_shifted"]
    ws = None if shifted.size == 0 else (shifted, z["ws_count"])
    return SimpleNamespace(centres=z["centres"], rvectors=z["rvectors"], degeneracy=z["degeneracy"],
                           hoppings=z["hoppings"], lattice=z["lattice"], ws_shifts=ws, spin=None)


def test_peierls_on_the_pz_set_obeys_the_strong_translational_symmetry_at_q_6():
    w = _pz_set()
    q = 6
    torus = PeierlsTorus(MagneticCell.build(w.lattice, 1, q), bonds_from_wannier(w))
    kpoints = np.array([(i / 2, j / (2 * q)) for i in range(2) for j in range(2 * q)])
    levels = []
    for kap in kpoints:
        h = torus.matrix(kap).toarray()
        assert np.max(np.abs(h - h.conj().T)) < 1e-12 * np.max(np.abs(h))
        levels.append(np.linalg.eigvalsh(h))
    levels = np.array(levels)
    assert levels.shape == (4 * q, 2 * q)
    full = np.sort(levels.ravel())
    assert np.max(np.abs(full - _copies(levels, kpoints, q))) < 1e-9
    assert np.max(np.abs(full - _copies(levels, kpoints, q, members=(0, 2)))) > 1e-4
