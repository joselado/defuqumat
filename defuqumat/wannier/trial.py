"""Trial orbitals ``g_n``, in the plane-wave basis of each k-point.

These are the functions the projections ``A_mn(k) = <psi_mk|S|g_n>`` are taken
against: a hydrogenic radial part times one of Wannier90's real angular
functions, centred on a site and turned to a chosen ``z`` and ``x`` axis. The
code is a transcription of ``pw2wannier90.f90`` (QE 7.5, ``PP/src``):

- ``generate_guiding_functions``: ``g(k+G) = sum_l f_l(k+G) R_l(|k+G|) (-i)^l
  e^{-i(k+G).tau}``, with ``f_l`` the part of the angular function with angular
  momentum ``l``. A hybrid such as ``sp3`` mixes ``l = 0`` and ``l = 1``, and
  each part takes its own radial transform and its own phase.
- ``radialpart``: ``R_l(q) = (4 pi/sqrt(Omega)) int r^2 j_l(q r) R(r) dr`` on
  the log mesh ``r = e^x/alpha``, ``x`` from -6 in steps of 0.025 to
  ``ln 10``, by QE's Simpson rule. ``R(r)`` is the hydrogenic function with
  ``r = 1, 2, 3`` nodes-plus-one and ``alpha`` the ``zona``.
- ``ylm_wannier``: the real angular functions of Wannier90's user guide,
  Tables 3.1 and 3.2, evaluated at the direction in the orbital's local frame
  ``(x . d, y . d, z . d)``, ``y = z cross x`` (``set_u_matrix``).
  ``pw2wannier90`` evaluates them through a numerical expansion in QE's real
  spherical harmonics (``ylm_expansion``), which is exact to ``l = 3``; here
  they are evaluated directly, which is the same function.

**Units.** ``zona`` is used in 1/bohr, as ``pw2wannier90`` uses it
(``radialpart`` takes ``alpha`` with ``r`` in bohr), although Wannier90's user
guide writes it in 1/angstrom. The default, 1, is the same number in both
codes, so a calculation run through both reads the same ``A``.

``g`` is not normalised at each k, since an atomic orbital's Bloch sum is not
(the comment at the head of ``generate_guiding_functions``); the Lowdin step of
the engine removes the norm.

Host-side NumPy: the table is built once per k-point, outside any compiled
path, like the projectors' radial tables.
"""

from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn

__all__ = ["angular_parts", "radial_transform", "guiding_functions"]

_XMIN, _DX, _RMAX = -6.0, 0.025, 10.0
_LMAX = 3


def _real_harmonics(cost, phi):
    """The real functions of ``ylm_wannier``'s helpers, by name."""
    fpi = 4.0 * np.pi
    sint = np.sqrt(np.abs(1.0 - cost * cost))
    return {
        "s": np.full_like(cost, 1.0 / np.sqrt(fpi)),
        "pz": np.sqrt(3.0 / fpi) * cost,
        "px": np.sqrt(3.0 / fpi) * sint * np.cos(phi),
        "py": np.sqrt(3.0 / fpi) * sint * np.sin(phi),
        "dz2": np.sqrt(1.25 / fpi) * (3.0 * cost * cost - 1.0),
        "dxz": np.sqrt(15.0 / fpi) * sint * cost * np.cos(phi),
        "dyz": np.sqrt(15.0 / fpi) * sint * cost * np.sin(phi),
        "dx2-y2": np.sqrt(3.75 / fpi) * sint * sint * np.cos(2.0 * phi),
        "dxy": np.sqrt(3.75 / fpi) * sint * sint * np.sin(2.0 * phi),
        "fz3": 0.25 * np.sqrt(7.0 / np.pi) * (5.0 * cost * cost - 3.0) * cost,
        "fxz2": 0.25 * np.sqrt(10.5 / np.pi) * (5.0 * cost * cost - 1.0) * sint * np.cos(phi),
        "fyz2": 0.25 * np.sqrt(10.5 / np.pi) * (5.0 * cost * cost - 1.0) * sint * np.sin(phi),
        "fz(x2-y2)": 0.25 * np.sqrt(105.0 / np.pi) * sint * sint * cost * np.cos(2.0 * phi),
        "fxyz": 0.25 * np.sqrt(105.0 / np.pi) * sint * sint * cost * np.sin(2.0 * phi),
        "fx(x2-3y2)": 0.25 * np.sqrt(17.5 / np.pi) * sint ** 3 * np.cos(3.0 * phi),
        "fy(3x2-y2)": 0.25 * np.sqrt(17.5 / np.pi) * sint ** 3 * np.sin(3.0 * phi),
    }


_L_OF = {"s": 0, "pz": 1, "px": 1, "py": 1, "dz2": 2, "dxz": 2, "dyz": 2, "dx2-y2": 2,
         "dxy": 2, "fz3": 3, "fxz2": 3, "fyz2": 3, "fz(x2-y2)": 3, "fxyz": 3,
         "fx(x2-3y2)": 3, "fy(3x2-y2)": 3}

_B2, _B3, _B6, _B12 = 1 / np.sqrt(2.0), 1 / np.sqrt(3.0), 1 / np.sqrt(6.0), 1 / np.sqrt(12.0)

#: ``(l, mr)`` -> the linear combination ``ylm_wannier`` builds, as
#: ``{name: coefficient}``.
_COMBINATIONS = {
    (0, 1): {"s": 1.0},
    (1, 1): {"pz": 1.0}, (1, 2): {"px": 1.0}, (1, 3): {"py": 1.0},
    (2, 1): {"dz2": 1.0}, (2, 2): {"dxz": 1.0}, (2, 3): {"dyz": 1.0},
    (2, 4): {"dx2-y2": 1.0}, (2, 5): {"dxy": 1.0},
    (3, 1): {"fz3": 1.0}, (3, 2): {"fxz2": 1.0}, (3, 3): {"fyz2": 1.0},
    (3, 4): {"fz(x2-y2)": 1.0}, (3, 5): {"fxyz": 1.0},
    (3, 6): {"fx(x2-3y2)": 1.0}, (3, 7): {"fy(3x2-y2)": 1.0},
    (-1, 1): {"s": _B2, "px": _B2}, (-1, 2): {"s": _B2, "px": -_B2},
    (-2, 1): {"s": _B3, "px": -_B6, "py": _B2},
    (-2, 2): {"s": _B3, "px": -_B6, "py": -_B2},
    (-2, 3): {"s": _B3, "px": 2.0 * _B6},
    (-3, 1): {"s": 0.5, "px": 0.5, "py": 0.5, "pz": 0.5},
    (-3, 2): {"s": 0.5, "px": 0.5, "py": -0.5, "pz": -0.5},
    (-3, 3): {"s": 0.5, "px": -0.5, "py": 0.5, "pz": -0.5},
    (-3, 4): {"s": 0.5, "px": -0.5, "py": -0.5, "pz": 0.5},
    (-4, 1): {"s": _B3, "px": -_B6, "py": _B2},
    (-4, 2): {"s": _B3, "px": -_B6, "py": -_B2},
    (-4, 3): {"s": _B3, "px": 2.0 * _B6},
    (-4, 4): {"pz": _B2, "dz2": _B2}, (-4, 5): {"pz": -_B2, "dz2": _B2},
    (-5, 1): {"s": _B6, "px": -_B2, "dz2": -_B12, "dx2-y2": 0.5},
    (-5, 2): {"s": _B6, "px": _B2, "dz2": -_B12, "dx2-y2": 0.5},
    (-5, 3): {"s": _B6, "py": -_B2, "dz2": -_B12, "dx2-y2": -0.5},
    (-5, 4): {"s": _B6, "py": _B2, "dz2": -_B12, "dx2-y2": -0.5},
    (-5, 5): {"s": _B6, "pz": -_B2, "dz2": _B3},
    (-5, 6): {"s": _B6, "pz": _B2, "dz2": _B3},
}


def _frame(z_axis, x_axis) -> np.ndarray:
    """``set_u_matrix``: rows ``x``, ``y = z cross x``, ``z``, all unit."""
    x = np.asarray(x_axis, dtype=float)
    z = np.asarray(z_axis, dtype=float)
    x = x / np.linalg.norm(x)
    z = z / np.linalg.norm(z)
    if abs(x @ z) > 1.0e-6:
        raise ValueError("the x and z axes of a projection are not orthogonal")
    return np.stack([x, np.cross(z, x), z])


def angular_parts(l: int, mr: int, directions: np.ndarray, z_axis=(0, 0, 1),
                  x_axis=(1, 0, 0)) -> np.ndarray:
    """``(4, n)``: the ``l = 0..3`` parts of Wannier90's ``(l, mr)`` function.

    ``directions`` is ``(n, 3)`` cartesian and need not be normalised; a zero
    vector gets the ``l = 0`` value and zero for every other ``l``, which is
    what survives the radial transform there anyway (``j_l(0) = 0`` for
    ``l > 0``).
    """
    if (l, mr) not in _COMBINATIONS:
        raise ValueError(f"no Wannier90 angular function with l = {l}, mr = {mr}")
    local = np.asarray(directions, dtype=float) @ _frame(z_axis, x_axis).T
    norm = np.linalg.norm(local, axis=1)
    zero = norm < 1.0e-8
    safe = np.where(zero[:, None], np.array([0.0, 0.0, 1.0]), local)
    rr = np.linalg.norm(safe, axis=1)
    cost = safe[:, 2] / rr
    # ``ylm_wannier``'s phi: atan(y/x), shifted by pi for x < 0, +-pi/2 at x = 0.
    phi = np.arctan2(safe[:, 1], safe[:, 0])
    table = _real_harmonics(cost, phi)
    parts = np.zeros((_LMAX + 1, len(local)))
    for name, c in _COMBINATIONS[(l, mr)].items():
        parts[_L_OF[name]] += c * table[name]
    parts[1:, zero] = 0.0
    return parts


def _hydrogenic(r, alpha: float, nodes: int) -> np.ndarray:
    if nodes == 1:
        return 2.0 * alpha ** 1.5 * np.exp(-alpha * r)
    if nodes == 2:
        return (1.0 / np.sqrt(8.0)) * alpha ** 1.5 * (2.0 - alpha * r) * np.exp(-alpha * r * 0.5)
    if nodes == 3:
        return (np.sqrt(4.0 / 27.0) * alpha ** 1.5
                * (1.0 - 2.0 / 3.0 * alpha * r + 2.0 * (alpha * r) ** 2 / 27.0)
                * np.exp(-alpha * r / 3.0))
    raise ValueError(f"the radial part r = {nodes} is not one of Wannier90's 1, 2, 3")


def _simpson_weights(rab: np.ndarray) -> np.ndarray:
    """QE's ``simpson`` as weights, even-mesh closure included."""
    mesh = len(rab)
    c = np.empty(mesh)
    c[0] = 1.0 / 3.0
    c[1:mesh - 1] = np.where(np.arange(2, mesh) % 2 == 0, 4.0, 2.0) / 3.0
    if mesh % 2 == 1:
        c[mesh - 1] = 1.0 / 3.0
    else:
        c[mesh - 3] -= 0.25 / 3.0
        c[mesh - 2] += 1.0 / 3.0
        c[mesh - 1] = 1.25 / 3.0
    return c * rab


def radial_transform(q: np.ndarray, alpha: float, nodes: int, volume: float) -> np.ndarray:
    """``(4, n)``: ``radialpart``'s ``R_l(q)`` for ``l = 0..3``, ``q`` in 1/bohr."""
    mesh = int(round((np.log(_RMAX) - _XMIN) / _DX + 1))
    x = _XMIN + _DX * np.arange(mesh)
    r = np.exp(x) / alpha
    weights = _simpson_weights(_DX * r)
    func = _hydrogenic(r, alpha, nodes) * r * r
    qr = np.asarray(q, dtype=float)[:, None] * r[None, :]
    pref = 4.0 * np.pi / np.sqrt(volume)
    return np.stack([pref * (spherical_jn(l, qr) @ (func * weights)) for l in range(_LMAX + 1)])


def guiding_functions(projections, kg: np.ndarray, at: np.ndarray, volume: float,
                      valid: np.ndarray | None = None, radial: dict | None = None) -> np.ndarray:
    """``(nproj, npw)``: the trial orbitals on the plane waves ``k + G``.

    ``kg`` is ``(npw, 3)``, ``k + G`` cartesian in 1/bohr; ``at`` the direct
    lattice vectors as rows, bohr; each projection carries Wannier90's
    ``site_frac``, ``l``, ``m`` (``mr``), ``z_axis``, ``x_axis``, ``radial``
    and ``zona``. Padding entries (``valid`` false) are zeroed. ``radial``, if
    given, maps ``(zona, r)`` to the ``(4, npw)`` table of
    :func:`radial_transform` at these ``|k + G|``, computed once for a whole
    mesh by the caller. A projection with a ``radial_key`` attribute (a
    pseudo-atomic orbital) must find its table there.
    """
    kg = np.asarray(kg, dtype=float)
    q = np.linalg.norm(kg, axis=1)
    out = np.zeros((len(projections), len(kg)), dtype=complex)
    radial_cache: dict = dict(radial or {})
    phase_l = (-1j) ** np.arange(_LMAX + 1)
    for n, p in enumerate(projections):
        # A pseudo-atomic orbital carries its own key (``("pao", species, orbital)``)
        # and its table comes from the caller; a hydrogenic one is built here.
        key = getattr(p, "radial_key", None) or (float(p.zona), int(p.radial))
        if key not in radial_cache:
            if getattr(p, "radial_key", None) is not None:
                raise KeyError(f"no radial table was given for {key}")
            radial_cache[key] = radial_transform(q, key[0], key[1], volume)
        parts = angular_parts(p.l, p.m, kg, p.z_axis, p.x_axis)
        g = np.einsum("l,ln,ln->n", phase_l, parts, radial_cache[key])
        tau = np.asarray(p.site_frac, dtype=float) @ at
        out[n] = g * np.exp(-1j * (kg @ tau))
    if valid is not None:
        out[:, ~np.asarray(valid)] = 0.0
    return out
