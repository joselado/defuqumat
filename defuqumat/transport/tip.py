"""The orbital of the tip: Chen's derivative rule, written as Landauer channels.

``PLAN.md`` P179. The design, and the review it went through before any code,
are ``plans/tip-orbitals.md`` and ``plans/tip-orbitals-review/``.

**The point tip of P66 is the s tip, exactly.** Bardeen's element between a tip
state whose vacuum tail is ``C k_0(kappa |r - r0|)`` and a sample state that
obeys ``(nabla^2 - kappa^2) psi = 0`` around ``r0`` is ``M ~ psi(r0)``, which is
Tersoff and Hamann. Chen (PRB 42, 8841 (1990)) expands a general tip state in
the same vacuum solutions, ``chi = sum_b C_b k_l(kappa rho) Y_b``, and the
element becomes a derivative of the sample state at the apex,

    M_n ~ sum_b C_b^* D_b psi_n(r0),

with the dimensionless operators of Mandi and Palotas (PRB 91, 165406 (2015),
arXiv:1503.09189, Table I): ``D_s = 1``, ``D_pa = (1/kappa) d/dx_a``,
``D_dxy = (1/kappa^2) d^2/dx dy`` and its partners,
``D_dx2-y2 = (1/kappa^2)(d^2/dx^2 - d^2/dy^2)``. For ``dz2`` the table has
``(3/kappa^2) d^2/dz^2 - 1``, and this module uses ``(3 d^2/dz^2 -
nabla^2)/kappa^2``: the two agree exactly where ``nabla^2 psi = kappa^2 psi``,
and the second is a pure ``l = 2`` operator with one power of ``kappa``, so it
turns under a rotation as the other four do. The conjugate ``C_b^*`` is the one
``<chi|psi>`` carries and is what ``a^T S a^*`` takes
(:func:`~defuqumat.transport.green.transmission`); it matters only for a complex
combination such as ``px + i py``.

**The weights are those of unnormalised angular functions.** Table I's
operators are what the tip's tail ``C_b k_l(kappa rho) P_b(n)`` couples
through, with ``P_b`` the plain polynomials ``1``, ``n_a``, ``n_x n_y``,
``n_x^2 - n_y^2``, ``3 n_z^2 - 1`` and ``k_l`` as :func:`scipy.special.spherical_kn`
defines it: ``(1/kappa^l) D_b k_0(kappa |r - r0|) = k_l(kappa rho) P_b(n)``, checked
by finite differences for every orbital (2026-11-15). These ``P_b`` are not
orthonormal on the sphere -- their norms are ``4 pi`` for s, ``4 pi / 3`` for each
p, ``4 pi / 15`` for ``dxy``, ``dyz``, ``dxz``, ``16 pi / 15`` for ``dx2-y2`` and
``16 pi / 5`` for ``dz2`` -- so equal weights on the three p orbitals are
isotropic (the alias ``"p"``) and equal weights on the five d ones are not. A
weight taken from a normalised orbital ``R Y_b`` is its tail on ``k_l Y_b``
times ``N_b`` with ``Y_b = N_b P_b`` (:mod:`defuqumat.transport.tip_presets`).

**A tip is a set of incoherent channels.** ``Gamma_t = sum_c |chi_c><chi_c|``:
one channel is one tip state, a coherent combination of orbitals that interfere
inside it (the s-pz cross term of Mandi and Palotas, eq. 7); two channels are
two independent tip states, which add as ``|M_1|^2 + |M_2|^2`` -- the pi pair of
a CO tip, and how the probe-particle model of Krejci, Hapala, Ondracek and
Jelinek (PRB 95, 045407 (2017), arXiv:1609.09462) treats every tip orbital. Each
channel is the existing Gram form with ``a_kn -> sum_b C_cb^* D_b a_kn``, so the
map stays non-negative channel by channel.

**In plane waves ``D_b`` is diagonal in G**: ``d/dx_a -> i (k+G)_a``, the
cartesian components taken in the tip frame. :func:`multiplier` is that
polynomial, without the ``kappa^-l``, which goes into the channel coefficients
(:meth:`TipOrbital.coefficients`) so that one set of sampled amplitudes serves
every energy. Checked against a central difference of
:func:`~defuqumat.basis.sample.sample_wavefunctions` to 0.9e-6 to 1.2e-6 at
``h = 1e-3`` bohr, slope 2.00 in ``h``, on a hexagonal and a sheared cell,
through the folded and the unfolded sampler.

**The frame** (:func:`tip_frame`): z is the normal of the exit plane, pointing
from the material to the tip, fixed once from the caller's geometry
(:func:`tip_side`); x is the first in-plane lattice vector made orthogonal to
z; y is ``z x x``. With that z, ``(1/kappa) d psi/dz ~ -psi`` on a vacuum tail,
so ``{"s": 1, "pz": -1}`` is the hybrid whose lobe points at the sample.

**Kappa** is the vacuum decay at the tip energy, ``kappa^2 = V_vac - E`` in Ry,
with ``V_vac`` the plateau of the planar average of the local potential in the
vacuum (:func:`vacuum_level`). The rule is exact only where the potential is
that flat, and how flat it is at the tip is a diagnostic of the assembly, not an
assumption of this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ORBITALS", "TipOrbital", "TipAssembly", "multiplier", "tip_frame",
           "tip_side", "tip_vectors", "rotation_representation", "vacuum_level",
           "decay_constants_at", "continued_wavevectors", "continued_components"]

#: The orbitals of Chen's table and their angular momentum, in the order the
#: real spherical harmonics are listed there.
ORBITALS = {"s": 0, "py": 1, "pz": 1, "px": 1, "dxy": 2, "dyz": 2, "dz2": 2,
            "dxz": 2, "dx2-y2": 2}

#: Every orbital of one shell, so that a rotated channel can be written out.
SHELLS = {0: ("s",), 1: ("py", "pz", "px"),
          2: ("dxy", "dyz", "dz2", "dxz", "dx2-y2")}

#: Names that stand for several incoherent channels at once. ``"pxy"`` is the
#: user's decision of 2026-11-15 ("pxy means two incoherent channel, like px
#: and py"): the pi pair of a CO tip, which does not depend on the frame.
ALIASES = {
    "pxy": ({"px": 1.0}, {"py": 1.0}),
    "p": ({"px": 1.0}, {"py": 1.0}, {"pz": 1.0}),
}

#: Not an orbital: ``nabla^2``, sampled for the Helmholtz residue.
LAPLACIAN = "laplacian"


def multiplier(name: str, q) -> np.ndarray:
    """``D_b`` on a plane wave, times ``kappa^l``: a polynomial in ``k + G``.

    ``q`` is ``(..., 3)``, the components of ``k + G`` in 1/bohr along the tip
    frame's x, y and z (:func:`tip_vectors`). ``d/dx_a`` is ``i q_a``, so the p
    multipliers are imaginary and the d ones real. ``q`` may be complex: a state
    continued into the vacuum (:func:`continued_components`) carries
    ``q_par + i kappa_G n``, so that ``d/dn`` is ``-kappa_G``.
    """
    q = np.asarray(q)
    q = q.astype(complex) if np.iscomplexobj(q) else q.astype(float)
    x, y, z = q[..., 0], q[..., 1], q[..., 2]
    if name == "s":
        return np.ones(x.shape, dtype=complex)
    if name == "px":
        return 1.0j * x
    if name == "py":
        return 1.0j * y
    if name == "pz":
        return 1.0j * z
    if name == "dxy":
        return (-x * y).astype(complex)
    if name == "dyz":
        return (-y * z).astype(complex)
    if name == "dxz":
        return (-x * z).astype(complex)
    if name == "dx2-y2":
        return (-(x * x - y * y)).astype(complex)
    if name == "dz2":
        # (3 d^2/dz^2 - nabla^2): -3 qz^2 + |q|^2
        return (-3.0 * z * z + (x * x + y * y + z * z)).astype(complex)
    if name == LAPLACIAN:
        return (-(x * x + y * y + z * z)).astype(complex)
    raise ValueError(_unknown(name))


def _unknown(name) -> str:
    return (f"unknown tip orbital {name!r}: use one of "
            f"{', '.join(repr(n) for n in ORBITALS)}, the aliases "
            f"{', '.join(repr(n) for n in ALIASES)}, a dict of weights such as "
            "{'s': 1, 'pz': -0.5} (one coherent channel), or a list of these "
            "(incoherent channels)")


@dataclass(frozen=True)
class TipOrbital:
    """A tip as a list of incoherent channels, each a combination of orbitals.

    ``channels`` holds, per channel, ``((name, weight), ...)``: the ``C_b`` of
    Chen's expansion in the normalisation of Mandi and Palotas, complex in
    general. ``radial`` holds, per channel, ``((name, callable), ...)``: an
    optional factor of the weight that depends on ``kappa``, which is how a
    preset carries the Bardeen weight of a real orbital
    (:mod:`defuqumat.transport.tip_presets`); empty for a weight given by hand.
    """

    channels: tuple
    label: str = "s"
    radial: tuple = ()

    @classmethod
    def resolve(cls, spec) -> "TipOrbital":
        """``tip_orbital=`` as the caller wrote it, as channels.

        ``None`` or ``"s"`` is the point tip. A name is one orbital, an alias
        (``"pxy"``, ``"p"``) or a preset of
        :mod:`defuqumat.transport.tip_presets`; a dict is one coherent channel;
        a list is several incoherent channels, each a name, an alias or a dict.
        """
        if isinstance(spec, TipOrbital):
            return spec
        if spec is None:
            return cls(channels=((("s", 1.0 + 0.0j),),), label="s")
        if isinstance(spec, str):
            key = spec.strip()
            if key in ORBITALS:
                return cls(channels=(((key, 1.0 + 0.0j),),), label=key)
            if key in ALIASES:
                return cls(channels=tuple(_channel(c) for c in ALIASES[key]),
                           label=key)
            from defuqumat.transport.tip_presets import PRESETS, preset

            if key in PRESETS:
                return preset(key)
            raise ValueError(_unknown(spec))
        if isinstance(spec, dict):
            return cls(channels=(_channel(spec),), label=_describe((spec,)))
        if isinstance(spec, (list, tuple)):
            if len(spec) == 0:
                raise ValueError("tip_orbital is an empty list: give at least "
                                 "one channel")
            channels = []
            for item in spec:
                if isinstance(item, str) and item.strip() in ALIASES:
                    channels.extend(_channel(c) for c in ALIASES[item.strip()])
                elif isinstance(item, str):
                    if item.strip() not in ORBITALS:
                        raise ValueError(_unknown(item))
                    channels.append(_channel({item.strip(): 1.0}))
                elif isinstance(item, dict):
                    channels.append(_channel(item))
                else:
                    raise ValueError(_unknown(item))
            return cls(channels=tuple(channels), label=_describe(spec))
        raise ValueError(_unknown(spec))

    @property
    def is_point(self) -> bool:
        """The P66 tip: one channel, the s orbital alone, weight 1."""
        return (len(self.channels) == 1 and len(self.channels[0]) == 1
                and self.channels[0][0][0] == "s"
                and self.channels[0][0][1] == 1.0 and not self.radial)

    @property
    def names(self) -> tuple:
        """Every orbital with a weight in some channel, in Chen's order."""
        used = {name for channel in self.channels for name, _ in channel}
        return tuple(name for name in ORBITALS if name in used)

    @property
    def shells(self) -> tuple:
        return tuple(sorted({ORBITALS[name] for name in self.names}))

    @property
    def is_real(self) -> bool:
        """No weight with an imaginary part: what a time reversal leaves alone."""
        return all(complex(w).imag == 0.0 for c in self.channels for _, w in c)

    def basis(self, full_shells: bool = False) -> tuple:
        """The orbitals whose amplitudes are sampled; whole shells if asked."""
        if not full_shells:
            return self.names
        return tuple(name for name in ORBITALS if ORBITALS[name] in self.shells)

    def coefficients(self, kappa, basis=None) -> np.ndarray:
        """``(nE, n_channels, n_basis)``: ``C_cb^* kappa^-l`` at each energy.

        What multiplies the sampled amplitude of orbital ``b`` (whose multiplier
        carries no ``kappa``) in channel ``c``; a preset's radial factor is
        evaluated at the same ``kappa``.
        """
        kappa = np.atleast_1d(np.asarray(kappa, dtype=float))
        basis = self.names if basis is None else tuple(basis)
        out = np.zeros((kappa.size, len(self.channels), len(basis)),
                       dtype=complex)
        radial = dict(enumerate(self.radial))
        for ic, channel in enumerate(self.channels):
            factors = dict(radial.get(ic, ()))
            for name, weight in channel:
                ib = basis.index(name)
                l = ORBITALS[name]
                value = np.conj(complex(weight)) * kappa ** (-float(l))
                if name in factors:
                    value = value * np.asarray([factors[name](k) for k in kappa])
                out[:, ic, ib] = value
        return out

    def describe(self) -> list:
        """The channels as plain dicts, for a result's notes."""
        return [{name: complex(w) for name, w in c} for c in self.channels]


def _channel(weights) -> tuple:
    if not isinstance(weights, dict) or not weights:
        raise ValueError(
            f"a tip channel is a non-empty dict of orbital weights, got {weights!r}")
    out = []
    for name, weight in weights.items():
        key = str(name).strip()
        if key not in ORBITALS:
            raise ValueError(_unknown(name))
        value = complex(weight)
        if not np.isfinite(value):
            raise ValueError(f"the weight of {key!r} is not finite: {weight!r}")
        if value != 0.0:
            out.append((key, value))
    if not out:
        raise ValueError(
            f"every weight of the tip channel {weights!r} is zero, so it "
            "carries no current; drop it or give it a weight")
    order = list(ORBITALS)
    return tuple(sorted(out, key=lambda pair: order.index(pair[0])))


def _describe(spec) -> str:
    if isinstance(spec, dict):
        return "+".join(f"{w}*{n}" for n, w in spec.items())
    parts = []
    for item in spec:
        parts.append(item if isinstance(item, str) else _describe(item))
    return " | ".join(parts)


@dataclass(frozen=True)
class TipAssembly:
    """What :func:`~defuqumat.workflows.transport._assemble` needs of a tip.

    ``coefficients`` is ``(nE, n_channels, n_basis)``, shared by every tip
    point, or ``(nE, n_images, n_channels, n_basis)`` with one set per image,
    which is what a wedge's images need (each image carries the orbital turned
    by its operation). The tip points are then laid out image-major,
    ``points_per_image`` to an image, so point ``p`` takes the set of image ``p
    // points_per_image`` (:meth:`at_points`). ``kappa`` is ``(nE,)`` in
    1/bohr. ``ethr`` is the threshold the bands were solved to, for the
    tail-noise estimate, or ``None``.
    """

    orbital: TipOrbital
    basis: tuple
    coefficients: np.ndarray
    frame: np.ndarray
    kappa: np.ndarray
    ethr: float | None = None
    channel_maps: bool = False
    #: tip points the diagnostics are taken on, at most this many
    diagnostic_points: int = 64
    #: the scalar local potential for the diagnostics: its Fourier
    #: coefficients on the FFT grid and the vacuum plateau, or ``None``
    potential: object = None
    plateau: float | None = None
    #: the vacuum continuation, or ``None``: ``{"height": s_m, "axis": a,
    #: "normal": n, "side": +-1, "spacing": bohr}`` with ``s_m`` the matching
    #: plane's crystal coordinate and ``n`` the unit normal towards the tip
    match: object = None
    #: with one set of coefficients per image, how many tip points each image has
    points_per_image: int | None = None

    def potential_at(self, points) -> np.ndarray | None:
        """``V(r)`` in Ry at crystal points, by its Fourier series (exact)."""
        if self.potential is None:
            return None
        coefficients = np.asarray(self.potential)
        shape = coefficients.shape
        frequencies = [np.fft.fftfreq(n, 1.0 / n) for n in shape]
        points = np.atleast_2d(np.asarray(points, dtype=float))
        phase = [np.exp(2j * np.pi * np.outer(points[:, a], frequencies[a]))
                 for a in range(3)]
        return np.real(np.einsum("ijk,pi,pj,pk->p", coefficients, *phase,
                                 optimize=True))

    @property
    def per_point(self) -> bool:
        return np.ndim(self.coefficients) == 4

    def at_points(self, ie: int, ic: int, points) -> np.ndarray:
        """``(len(points), n_basis)``: channel ``ic``'s coefficients at energy
        ``ie`` for the tip points ``points`` (indices), when they vary per
        image. The images are not repeated per point: ``(nE, n_images, nc,
        nb)`` is held, not ``(nE, n_images npts, nc, nb)`` (audit 2026-11-20
        #34)."""
        images = np.asarray(points) // int(self.points_per_image)
        return np.asarray(self.coefficients)[ie, images, ic]

    @property
    def nchannels(self) -> int:
        return int(np.shape(self.coefficients)[-2])

    def with_coefficients(self, coefficients, basis=None,
                          points_per_image=None) -> "TipAssembly":
        return TipAssembly(self.orbital, self.basis if basis is None else tuple(basis),
                           np.asarray(coefficients), self.frame, self.kappa,
                           self.ethr, self.channel_maps, self.diagnostic_points,
                           self.potential, self.plateau, self.match,
                           points_per_image)


# --------------------------------------------------------------------------
# where the tip points
# --------------------------------------------------------------------------


def tip_side(system, exit_axis: int, exit_height: float, points) -> int:
    """``+1`` if the tip is on the side of increasing crystal coordinate.

    The cell is periodic, so "above" means the side the material is between:
    going from the exit plane towards the tip, the atoms have to come first.
    Both directions are tried, and **every** tip point must give the same
    answer -- a list of ``tip=`` points on both sides of the material has no
    single "towards the tip", and a non-s orbital is refused there, since the
    sign of pz would depend on the point.
    """
    positions = np.asarray(system.structure.positions_crystal(system.cell))
    slab = positions[:, exit_axis]
    heights = np.unique(np.round(np.atleast_2d(points)[:, exit_axis], 12))
    sides = set()
    for tip in heights:
        up = np.mod(slab - float(exit_height), 1.0)
        edge_up = np.mod(float(tip) - float(exit_height), 1.0)
        down = np.mod(float(exit_height) - slab, 1.0)
        edge_down = np.mod(float(exit_height) - float(tip), 1.0)
        if edge_up > 0.0 and up.min() > 0.0 and up.max() < edge_up:
            sides.add(1)
        elif edge_down > 0.0 and down.min() > 0.0 and down.max() < edge_down:
            sides.add(-1)
        else:
            sides.add(0)
    if sides == {1}:
        return 1
    if sides == {-1}:
        return -1
    raise NotImplementedError(
        "a tip orbital other than s needs to know which way the sample is, and "
        "here it is not one way: "
        + ("some tip points do not have the material between them and the "
           "exit plane" if 0 in sides else
           "the tip points lie on both sides of the material")
        + ". The sign of a pz lobe would then depend on the point. Put every "
        "tip point on one side, with the material between the tip and the exit "
        "plane, or use the s tip")


def tip_frame(cell, exit_axis: int, side: int, frame=None) -> np.ndarray:
    """``(3, 3)``: the rows are the tip's x, y and z in cartesian components.

    ``frame`` overrides the default, which is a tilted tip; it has to be a
    proper rotation, because a mirror would turn the handedness of a complex
    channel without saying so.
    """
    if frame is not None:
        out = np.asarray(frame, dtype=float)
        if out.shape != (3, 3):
            raise ValueError(f"tip_frame is a 3x3 array of rows x, y, z, got "
                             f"{out.shape}")
        if not np.allclose(out @ out.T, np.eye(3), atol=1e-8):
            raise ValueError("tip_frame must be orthonormal: its rows are the "
                             "unit vectors x, y, z of the tip")
        if np.linalg.det(out) < 0.0:
            raise ValueError("tip_frame must be right-handed (z = x cross y); a "
                             "mirrored frame would swap px + i py for px - i py")
        return out
    at = np.asarray(cell.at, dtype=float)
    bg = np.asarray(cell.bg, dtype=float)
    z = float(side) * bg[exit_axis] / np.linalg.norm(bg[exit_axis])
    first = [i for i in range(3) if i != exit_axis][0]
    x = at[first] - (at[first] @ z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def continued_wavevectors(inplane, lateral_k, cell, normal, plateau, energies):
    """``(Q, kappa)``: the folded components continued into the flat vacuum.

    Above a matching plane where the potential is ``plateau``, the in-plane
    component ``G_par`` of a state at energy ``e_n`` is
    ``b_n(G_par) exp(i q_par . r - kappa_nG d)`` with ``kappa_nG^2 = plateau - e_n
    + |q_par|^2``, ``q_par`` the in-plane part of ``k + G_par`` (the projection
    off ``normal``, which is the in-plane reciprocal vector also when ``a3`` is
    tilted) and ``d`` the height above the plane. Every derivative of it is a
    multiplication by ``i Q`` with the complex ``Q = q_par + i kappa_nG n``, and
    ``-Q.Q = kappa_n^2``, so the continued state obeys ``(nabla^2 - kappa_n^2)
    psi = 0`` exactly: Chen's condition by construction.

    ``inplane`` is ``(ng, 3)`` Miller indices with the normal column zero,
    ``lateral_k`` the k-point with it zeroed, ``energies`` ``(nbnd,)`` in Ry.
    Returns ``Q`` ``(nbnd, ng, 3)`` complex, cartesian, and ``kappa`` ``(nbnd, ng)``.
    """
    normal = np.asarray(normal, dtype=float)
    q = (np.asarray(inplane, dtype=float) + np.asarray(lateral_k, dtype=float)) \
        @ np.asarray(cell.bg, dtype=float)
    q_par = q - np.outer(q @ normal, normal)
    energies = np.asarray(energies, dtype=float).reshape(-1)
    barrier = float(plateau) - energies
    if np.any(barrier <= 0.0):
        raise ValueError(
            f"a state at {float(energies[np.argmin(barrier)]):.6f} Ry is at or "
            f"above the vacuum level {float(plateau):.6f} Ry, so it does not "
            "decay into the vacuum and cannot be continued from a matching "
            "plane; narrow the energy window or drop match_height")
    kappa = np.sqrt(barrier[:, None] + np.sum(q_par * q_par, axis=1)[None, :])
    wavevectors = q_par[None, :, :] + 1.0j * kappa[:, :, None] * normal[None, None, :]
    return wavevectors, kappa


def continued_components(inplane, lateral_k, cell, match, energies, distance):
    """``(Q, decay)`` of the folded components ``distance`` bohr above the plane.

    The WKB exponent of :func:`continued_wavevectors`, **without the WKB
    amplitude** (below): each in-plane component
    decays as ``exp(-int_0^d kappa_nG(t) dt)`` with ``kappa_nG(t)^2 = V(t) -
    e_n + |q_par|^2`` and ``V(t)`` the planar average of the local potential
    between the matching plane and the tip (``match["profile"]``), and its
    derivatives are taken with the local ``kappa_nG(d)``. In a flat vacuum it
    is :func:`continued_wavevectors` exactly. It exists because the potential
    above a matching plane close to the material is still below the plateau:
    on graphene, continued from 3 bohr with the plateau's decay, the s map
    over an atom is 0.885 and 0.778 of the raw plane-wave tail at 3.5 and 4
    bohr, where that tail is still good (``PLAN.md`` P179). A component whose
    ``kappa^2`` is negative somewhere on the path (a classically allowed
    stretch) does not decay there; the lateral corrugation of the potential is
    not in a planar average.

    **The amplitude factor is left out on purpose, and measured.** The WKB
    solution of ``psi'' = kappa(t)^2 psi`` is ``kappa^-1/2 exp(-int kappa)``, so
    a component matched at the plane would carry ``sqrt(kappa_nG(0) /
    kappa_nG(d))`` as well, which is below 1 on a rising potential. With it
    (audit 2026-11-20 #49, the record's probe: graphene at 40 Ry, the 6x6x1
    grid and 20 bands, the plane 3 bohr up), the s map over an atom is 0.959,
    0.914 and 0.903 of the raw tail at 3.5, 4 and 5 bohr, against 1.0105,
    0.9837 and 1.0125 for the exponent alone, and the ``pxy`` hollow over an
    atom at 4 bohr is 1.9504 against 1.9520. On the hydrogen sheet at 15 Ry
    (the momentum route, tip plane 3 bohr up, plane 2 bohr up, the total
    weight against the raw tail) the factor gives 0.951, 0.873 and 0.937 for
    s, pz and ``pxy``, the exponent alone 1.121, 1.022 and 1.087; there the
    raw tail is not good itself (its Helmholtz residue is 1.12 at 3 bohr,
    ``PLAN.md`` P179), so this pair does not decide. The exponent alone is kept
    because it is the closer of the two where the raw tail can be trusted. Why
    the factor makes it worse there is not known: one candidate is the lateral
    change of the potential, which the planar average leaves out.

    Returns ``Q`` ``(nbnd, ng, 3)`` complex, cartesian, ``q_par + i kappa_nG(d)
    n``, and ``decay`` ``(nbnd, ng)``.
    """
    normal = np.asarray(match["normal"], dtype=float)
    q = (np.asarray(inplane, dtype=float) + np.asarray(lateral_k, dtype=float)) \
        @ np.asarray(cell.bg, dtype=float)
    q_par = q - np.outer(q @ normal, normal)
    q2 = np.sum(q_par * q_par, axis=1)[None, :]
    energies = np.asarray(energies, dtype=float).reshape(-1)
    plateau = float(match["plateau"])
    if np.any(plateau - energies <= 0.0):
        raise ValueError(
            f"a state at {float(energies[np.argmax(energies)]):.6f} Ry is at or "
            f"above the vacuum level {plateau:.6f} Ry, so it does not decay into "
            "the vacuum and cannot be continued from a matching plane; narrow the "
            "energy window or drop match_height")
    profile = match.get("profile")
    distance = float(distance)

    def kappa_at(t):
        if profile is None:
            level = plateau
        else:
            level = potential_at(profile, float(match["height"])
                                 + int(match["side"]) * t / float(match["spacing"]))
        return np.sqrt(np.maximum(level - energies[:, None] + q2, 0.0))

    steps = max(8, int(np.ceil(distance / 0.05)))
    t = np.linspace(0.0, distance, steps + 1)
    integral = np.zeros((energies.size, q2.shape[1]))
    previous = kappa_at(0.0)
    for left, right in zip(t[:-1], t[1:]):
        current = kappa_at(right)
        integral += 0.5 * (previous + current) * (right - left)
        previous = current
    wavevectors = q_par[None, :, :] + 1.0j * previous[:, :, None] * normal[None, None, :]
    return wavevectors, np.exp(-integral)


def tip_vectors(miller, kcrystal, cell, frame) -> np.ndarray:
    """``(npw, 3)``: ``k + G`` in 1/bohr, in components along the tip frame."""
    miller = np.asarray(miller, dtype=float)
    kcrystal = np.asarray(kcrystal, dtype=float).reshape(3)
    cartesian = (miller + kcrystal) @ np.asarray(cell.bg, dtype=float)
    return cartesian @ np.asarray(frame, dtype=float).T


def rotation_representation(basis, rotation) -> np.ndarray:
    """``D`` with ``f_b(R^-1 q) = sum_b' D[b', b] f_b'(q)``, on whole shells.

    ``rotation`` is ``(3, 3)``, acting on column vectors in tip-frame
    components. A channel ``F(q) = sum_b c_b f_b(q)`` carried by the rotation is
    ``F(R^-1 q)``, whose coefficients are ``D @ c``. Found by a least-squares fit
    on random wavevectors rather than from a table, since Table I is not equally
    normalised (a 45 degree turn maps ``dxy`` onto half of ``dx2-y2``) and a
    transcribed table is one more place for a factor to go missing; the fit is
    exact to round-off because every multiplier is a polynomial of degree ``l``
    and ``basis`` holds whole shells.
    """
    basis = tuple(basis)
    rotation = np.asarray(rotation, dtype=float)
    rng = np.random.default_rng(12345)
    q = rng.normal(size=(64, 3))
    plain = np.stack([multiplier(name, q) for name in basis], axis=1)
    # (R^-1 q) as a row is q^T R^-T = q^T R for an orthogonal R.
    turned = np.stack([multiplier(name, q @ rotation) for name in basis], axis=1)
    representation, *_ = np.linalg.lstsq(plain, turned, rcond=None)
    if np.abs(plain @ representation - turned).max() > 1e-9 * max(
            1.0, np.abs(turned).max()):
        raise ValueError(
            "the tip orbitals do not span whole shells, so a rotation takes them "
            f"outside {basis}; sample every orbital of each shell")
    return representation


# --------------------------------------------------------------------------
# kappa
# --------------------------------------------------------------------------


def vacuum_level(system, potential, exit_axis: int) -> dict:
    """The plateau of the local potential in the vacuum, and how flat it is.

    ``potential`` is :attr:`SCFResult.potential`, ``(nspin, n1, n2, n3)`` in Ry
    and ``vltot + v_scf`` (``scf/driver.py``). At ``nspin = 2`` it is
    ``(up, down)`` with the local pseudopotential in both, so the two are
    averaged; at 1 and 4 the first component is the scalar potential.

    The vacuum is the largest gap between atoms along ``exit_axis``, and the
    plateau is the planar average at its middle. ``spread`` is how much the
    planar average moves within 1 bohr of that middle, which is the check that
    there is a plateau at all.
    """
    scalar = scalar_potential(system, potential)
    others = tuple(i for i in range(3) if i != exit_axis)
    profile = scalar.mean(axis=others)
    n = profile.size
    coordinates = np.arange(n) / n

    positions = np.asarray(system.structure.positions_crystal(system.cell))
    along = np.sort(np.mod(positions[:, exit_axis], 1.0))
    gaps = np.diff(np.concatenate([along, along[:1] + 1.0]))
    widest = int(np.argmax(gaps))
    middle = float(np.mod(along[widest] + 0.5 * gaps[widest], 1.0))

    from defuqumat.transport.substrate import surface_area

    spacing = float(system.cell.volume) / surface_area(system.cell, exit_axis)
    window = 1.0 / spacing
    distance = np.abs(np.mod(coordinates - middle + 0.5, 1.0) - 0.5)
    near = profile[distance <= window]
    return {
        "plateau": float(_periodic_interp(profile, middle)),
        "middle": middle,
        "spread": float(near.max() - near.min()) if near.size else float("nan"),
        "profile": profile,
        "gap_bohr": float(gaps[widest] * spacing),
    }


def scalar_potential(system, potential) -> np.ndarray:
    """``(n1, n2, n3)``: the spin-independent local potential, Ry."""
    v = np.asarray(potential, dtype=float)
    return v.mean(axis=0) if int(system.nspin) == 2 else v[0]


def potential_at(profile, coordinate: float) -> float:
    """The planar average at a crystal coordinate, interpolated periodically."""
    return float(_periodic_interp(np.asarray(profile, dtype=float), coordinate))


def _periodic_interp(profile, coordinate):
    n = profile.size
    x = np.mod(float(coordinate), 1.0) * n
    i = int(np.floor(x)) % n
    t = x - np.floor(x)
    return (1.0 - t) * profile[i] + t * profile[(i + 1) % n]


def decay_constants_at(plateau: float, energies) -> np.ndarray:
    """``kappa(E) = sqrt(V_vac - E)`` in 1/bohr, Ry units (``hbar^2/2m = 1``)."""
    energies = np.atleast_1d(np.asarray(energies, dtype=float))
    barrier = float(plateau) - energies
    if np.any(barrier <= 0.0):
        worst = float(energies[np.argmin(barrier)])
        raise ValueError(
            f"the tip energy {worst:.6f} Ry is at or above the vacuum level "
            f"{plateau:.6f} Ry, so there is no barrier and no vacuum tail for "
            "Chen's rule to expand in. Pass tip_kappa= if a decay constant is "
            "meant anyway")
    return np.sqrt(barrier)
