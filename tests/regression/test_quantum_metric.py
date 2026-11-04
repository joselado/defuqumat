"""The quantum metric of a plane-wave crystal, two routes that share no machinery.

The Kubo route is a sum over empty states of the generalised velocity, one
``jvp`` of ``H(k)`` per direction (:mod:`defuqumat.response.geometry`). The
overlap route is ``N - ||<u(k - d/2)|S|u(k + d/2)>||^2 / d^2``: no ``jvp``, no
empty states, the Miller-index alignment and the ultrasoft ``q_ij(b)`` that
every invariant here is built from. They share the converged density and the
fixed-density states and nothing after them, so a factor, a transpose or a
missing augmentation term in either shows as a difference that does not fall
with the band count.

Measured at the generic k ``(0.13, 0.21, 0.07)`` (crystal), not on a mirror
plane, so the off-diagonal elements are nonzero, at 40 bands against
``d = 2e-3`` bohr^-1 with the three spheres equal::

    alas-raman (NC)  g_xx 15.4335 / 15.4575   g_yz -6.0983 / -6.1006
    si2-us           g_xx 22.5406 / 22.5482   g_yz -8.3386 / -8.3395
    si2-paw          g_xx 22.5503 / 22.5579   g_yz -8.3369 / -8.3377

and 16 bands sit 0.10 further away on AlAs, so the residual is the truncation.

The memory rule: four cells, each its own compile, so the compiled code is
dropped between tests and the converged states are bounded at two.
"""

from functools import lru_cache
from pathlib import Path
import re
import warnings

import jax
import numpy as np
import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo.upf import read_upf
from defuqumat.scf.driver import run_scf
from defuqumat.system.builder import build_system
from defuqumat.system.kpoints import KPoints
from defuqumat.workflows.geometry import run_quantum_metric
from defuqumat.workflows.topology import DFTSource

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"
GENERIC_K = np.array([0.13, 0.21, 0.07])


@pytest.fixture(autouse=True)
def _bounded_compilation():
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def converged(name: str, extra: str = "", mesh: str = "3 3 3"):
    text = (CASES / name).read_text()
    text = re.sub(r"K_POINTS.*\n.*\n", f"K_POINTS automatic\n {mesh} 0 0 0\n", text)
    if extra:
        text = re.sub(r"(&system\s*\n)", r"\1    " + extra + "\n", text, count=1, flags=re.I)
    system = build_system(parse_pw_input(text))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    result = run_scf(system, pseudos, conv_thr=1e-11, max_iterations=200)
    return system, pseudos, result


def _by_overlaps(system, pseudos, result, nocc, delta):
    """``g_ab`` at :data:`GENERIC_K` from central-difference overlaps, and the sphere sizes."""
    source = DFTSource(system=system, pseudos=pseudos, density=result.density,
                       nocc=nocc, nbnd=nocc + 4, conv_thr=1e-13, becsum=result.becsum)
    bg = np.asarray(system.cell.bg)
    directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                           [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float)
    values, spheres = [], set()
    centre = source.states(GENERIC_K.reshape(1, 3))
    spheres.add(int(np.sum(np.asarray(centre.valid[0]))))
    for direction in directions:
        step = delta * direction @ np.linalg.inv(bg)
        two = source.states(np.stack([GENERIC_K - step / 2, GENERIC_K + step / 2]))
        spheres.update(int(np.sum(np.asarray(two.valid[i]))) for i in (0, 1))
        matrix = np.asarray(two.overlap(0, 1))
        values.append((nocc - np.sum(np.abs(matrix) ** 2)) / delta ** 2)
    g = np.diag(values[:3])
    for (a, b), value in zip(((0, 1), (0, 2), (1, 2)), values[3:]):
        g[a, b] = g[b, a] = 0.5 * (value - values[a] - values[b])
    return g, spheres


def _by_kubo(system, pseudos, result, nbnd):
    point = KPoints.from_crystal(GENERIC_K.reshape(1, 3), np.array([2.0]), system.cell,
                                 precision=system.kpoints.precision)
    metric = run_quantum_metric(system, pseudos, result.density, nbnd=nbnd, kpoints=point,
                                becsum=result.becsum, conv_thr=1e-12)
    return metric.metric[0]


@pytest.mark.parametrize("name", ["alas-raman.in", "si2-us.in", "si2-paw.in"])
def test_the_kubo_metric_is_the_overlap_metric_at_a_generic_k(name):
    """Two routes, one number, and the gap closes with the band count.

    The 16-band Kubo metric is farther from the overlap one than the 40-band
    metric is, by the truncation; at 40 bands what is left is below 0.2 per
    cent of the largest element (measured 0.16 per cent on AlAs, 0.04 per cent
    on the two augmented cells). A missing augmentation connection would be a
    difference that does not move with the band count.
    """
    system, pseudos, result = converged(name)
    overlap, spheres = _by_overlaps(system, pseudos, result, 4, 2.0e-3)
    assert len(spheres) == 1                      # one sphere, so no jump in the difference
    few = _by_kubo(system, pseudos, result, 16)
    many = _by_kubo(system, pseudos, result, 40)
    scale = float(np.max(np.abs(overlap)))
    assert abs(overlap[1, 2]) > 0.2 * scale        # a generic k: off-diagonals are there
    assert np.max(np.abs(many - overlap)) < 2.0e-3 * scale
    assert np.max(np.abs(many - overlap)) < np.max(np.abs(few - overlap))
    assert np.allclose(many, many.T, atol=1e-12)


def test_the_wedge_the_channels_and_the_spinor_agree_with_the_whole_grid():
    """Four regimes, one localization tensor, to the truncation they carry.

    Silicon at 4x4x4 and 28 bands (56 for the spinor): the wedge against the
    whole grid 2.4e-5, ``nspin = 2`` 1.2e-5, the spinor 1.9e-5, on a tensor of
    3.955 bohr^2 whose truncation is 9e-5; at 16 bands the same three were
    1.2e-4, 5.6e-5 and 9.4e-5 against a truncation of 3.3e-4, so they fall with
    it. ``Omega_I`` doubles at ``nspin = 2`` and for the spinor, which count
    spin-orbitals, and the tensor per electron does not.
    """
    warnings.simplefilter("ignore")
    whole_args = "nosym = .true., noinv = .true."
    reference = None
    for extra, nbnd in (("", 28), (whole_args, 28),
                        (whole_args + ", nspin = 2, starting_magnetization(1) = 0.0, "
                         "tot_magnetization = 0", 28),
                        (whole_args + ", noncolin = .true.", 56)):
        system, pseudos, result = converged("si2-symmetric.in", extra, "4 4 4")
        metric = run_quantum_metric(system, pseudos, result.density, nbnd=nbnd)
        tensor = metric.localization_tensor
        if reference is None:
            reference = metric
            assert np.max(np.abs(tensor - np.diag(np.diag(tensor)))) < 1e-14
            continue
        assert np.max(np.abs(tensor - reference.localization_tensor)) < 5.0e-5
        doubled = system.nspin != 1
        assert metric.spread_invariant == pytest.approx(
            reference.spread_invariant * (2.0 if doubled else 1.0), rel=1e-5)


def test_a_metal_is_refused_by_both_routes():
    """The guard fires on aluminium, which has a Fermi surface on any mesh."""
    system, pseudos, result = converged("al-conductivity.in", mesh="2 2 2")
    with pytest.raises(ValueError, match="refused on a metal"):
        run_quantum_metric(system, pseudos, result.density, nbnd=8)
    with pytest.raises(ValueError, match="refused on a metal"):
        run_quantum_metric(system, pseudos, result.density, method="overlap",
                           divisions=(2, 2, 2))
