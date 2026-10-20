"""Quasiparticle interference from the T-matrix of one impurity.

``PLAN.md`` P157. A single impurity scatters the Bloch states of a surface, the
scattered waves interfere with the incident ones, and the local density of
states acquires standing waves whose Fourier transform is a map of the
scattering vectors ``q = k - k'`` that join two states at the same energy.
That map is what STM quasiparticle-interference (QPI) imaging measures. The
formalism is Rhodes, Kohsaka, Hanaguri, de Almeida Marques and Wahl,
arXiv:2607.22487, Sec. IV (Eqs. 17 to 24, 27 to 33, 36 to 38) for the T-matrix
and Sec. V.3 (Eqs. 87 to 91) for the continuum Green's function. Neither
``pw.x`` nor Elk computes it.

**The scatterer is a separable projector**, ``V_imp = V |phi><phi|``, with
``phi`` a normalised Gaussian of width ``sigma`` centred on ``r_imp``. Then
the Dyson series of Eq. 21 sums in closed form (Eq. 24 for a rank-one
``V_imp``):

    T(w) = V / (1 - V g0(w)),
    g0(w) = <phi| G0(w) |phi> = (1/N) sum_kn |P_kn|^2 / (w - e_kn + i eta),
    P_kn = <psi_kn|phi>,

with ``psi_kn`` normalised in one cell and ``N`` the number of k-points, so
that ``psi_kn / sqrt(N)`` is normalised in the Born-von Karman supercell the
k-grid defines. ``V`` is in Ry: it is the energy shift of a state that lies
entirely inside ``phi``. A true delta is not offered, because ``Re g0`` of a
delta in a plane-wave basis depends on the cutoff; ``sigma`` is the scatterer's
size and is a parameter of the model, not a numerical one.

**The Green's function is the one of the band window**, ``P G0 P`` with ``P``
the projector on the ``nbnd`` bands at every k-point. This is exactly the
object the review's continuum (Wannier) method builds (Eq. 87): a Wannier
basis spans an isolated set of bands, and ``G`` written in it is ``P G0 P``
for that set. It is *not* the complete ``G0``: P66 measured that the off-shell
``G0(r, r')`` between two separated points is built by cancellation between
bands far from ``w`` (a ratio of 349 on a hydrogen sheet), and ``Re G0``
enters here through the Friedel phase. So the window is part of the model and
its dependence is a number recorded in ``PLAN.md``, as for a Wannier model.

**Everything the impurity does goes through one function of the tip point**,
``G0(r, phi)``, which the Bloch theorem makes cheap over a whole supercell:

    G0(R + u, phi; w) = (1/N) sum_k e^{i k.R} F_k(u),
    F_k(u) = sum_n psi_kn(u) P_kn / (w - e_kn + i eta),

so the bands are sampled at the tip points ``u`` of **one** cell and the
``N`` lattice translations ``R`` come from one ``(N, nk)`` phase matrix. The
two readers of the dressed ``G = G0 + G0 T G0`` (Eq. 22) are:

* ``ldos``, the Tersoff-Hamann reader, Eq. 33 at the tip,
  ``delta rho(r) = -(1/pi) Im[ G0(r, phi) T G0(phi, r) ]``;
* ``landauer``, the tunnelling reader of :mod:`defuqumat.transport.green`, an
  electron injected at the tip point ``r`` and collected on an exit plane,
  ``T(r) = int_exit d^2r' |G(r, r')|^2``. Writing
  ``G(r, r') = (1/N) sum_mk A_mk(r) psi*_mk(r')`` and integrating over the
  supercell's exit plane gives ``T(r) = (1/N) sum_k A_k S_k A_k^dagger`` with
  ``S_k[m, m'] = int_plane psi*_km psi_km'``
  (:func:`~defuqumat.transport.substrate.exit_overlap`), and the impurity's part
  of ``A`` factorises, ``A_mk = psi_mk(r)/D_mk + G0(r, phi) c_mk`` with
  ``c_mk = T P*_mk / D_mk``. So

      delta T(r) = 2 Re[ G0*(r, phi) H(r) ] + K |G0(r, phi)|^2,
      H(R + u) = (1/N) sum_k e^{i k.R} sum_m psi_km(u) (S_k c*_k)_m / D_km,
      K = (1/N) sum_k c_k S_k c_k^dagger.

  The index order in ``S_k`` is P66's trap: ``A S A^dagger`` and not
  ``A S^T A^dagger``.

  **The exit leg is on shell, and that is P66's finding a second time.**
  ``A_mk = psi+_mk(r) / D_mk``, where ``psi+_mk = (1 + G0 T)|mk>`` is the
  Lippmann-Schwinger scattering state, so ``1/D_mk`` is the propagator from
  the scattered state to the exit plane through the vacuum. With the literal
  ``1/D_mk`` the reader does not converge with the window -- measured on the
  hydrogen sheet, 6x6 grid, ``eta = 0.05``, against 128 bands, the
  ``q != 0`` transform is off by 0.22, 0.046, 0.075 and 0.094 at 8, 16, 32
  and 64 bands -- while ``ldos``, which has no exit leg, is off by 1.5e-2,
  6.9e-3, 1.2e-3 and 8.1e-4. So the exit leg takes P66's on-shell amplitude
  ``sqrt(delta(w - e_mk))`` (``exit_weights``) and the scattering state keeps
  the literal resolvent, which carries the Friedel phase; at ``V = 0`` this is
  :func:`~defuqumat.workflows.transport.run_vertical_transport`'s map
  exactly.

The supercell map of either reader is Fourier transformed in the tip plane
(Eq. 34) to give ``delta rho(q)``; its first Brillouin zone is the ``N1 x N2``
block around ``q = 0`` and the tip sampling of one cell decides how many zones
are seen. What validates the T-matrix, the Bloch factorisation and both
readers at once is :func:`dense_reference`, which builds the impurity
Hamiltonian in the supercell's band basis and inverts it, with no T-matrix and
no phase matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from defuqumat.basis.sample import sample_wavefunctions

__all__ = ["QPIStates", "QPI", "gaussian_transform", "impurity_overlaps",
           "t_matrix", "lattice_translations", "impurity_maps",
           "fourier_map", "dense_reference", "READERS"]

#: The two ways the dressed Green's function is read at the tip.
READERS = ("ldos", "landauer")


@dataclass(frozen=True)
class QPIStates:
    """One spin channel's bands, read where the T-matrix needs them.

    Attributes:
        eigenvalues: ``(nk, nbnd)`` in Ry.
        tip: ``(nk, nbnd, npoints)`` complex, ``psi_kn(u)`` at the tip points of
            one cell, normalised in the cell (bohr^{-3/2}).
        overlaps: ``(nk, nbnd)`` complex, ``P_kn = <psi_kn|phi>``.
        kcrystal: ``(nk, 3)`` the k-points in crystal coordinates.
        exit: ``(nk, nbnd, nbnd)`` the exit-plane Gram matrices, or ``None``
            when only the ``ldos`` reader is wanted.
    """

    eigenvalues: np.ndarray
    tip: np.ndarray
    overlaps: np.ndarray
    kcrystal: np.ndarray
    exit: np.ndarray | None = None

    @property
    def nk(self) -> int:
        return int(np.asarray(self.eigenvalues).shape[0])


def gaussian_transform(kcartesian, sigma: float):
    """``phi~(K) = int phi(r) e^{-i K.r} d^3r`` of the normalised Gaussian.

    ``phi(r) = (pi sigma^2)^{-3/4} exp(-r^2 / (2 sigma^2))``, so that
    ``int |phi|^2 = 1``, and its transform is
    ``(4 pi sigma^2)^{3/4} exp(-sigma^2 K^2 / 2)``, real and even. ``K`` in
    1/bohr, ``sigma`` in bohr.
    """
    if not sigma > 0.0:
        raise ValueError(
            f"the scatterer's width must be positive, got sigma = {sigma}: a "
            "delta in a plane-wave basis has a cutoff-dependent Re g0")
    kcartesian = np.asarray(kcartesian, dtype=float)
    k2 = np.sum(kcartesian ** 2, axis=-1)
    return (4.0 * np.pi * sigma ** 2) ** 0.75 * np.exp(-0.5 * sigma ** 2 * k2)


def impurity_overlaps(coefficients, miller, kcrystal, position, bg, volume,
                      sigma: float, mask=None):
    """``(nbnd,)`` ``P_n = <psi_kn|phi>`` at one k-point.

    ``<psi|phi> = Omega^{-1/2} sum_G c*(G) e^{-i(k+G).r_imp} phi~(k+G)``, which
    is the complex conjugate of the sampler evaluated at ``r_imp`` on the
    coefficients multiplied by ``phi~``. The integral is over all space, which
    for a ``phi`` much smaller than the cell is the cell it sits in.

    Args:
        coefficients: ``(nbnd, npw)`` this k-point's bands.
        miller: ``(npw, 3)`` the sphere's Miller indices.
        kcrystal: ``(3,)`` the k-point in crystal coordinates.
        position: ``(3,)`` the scatterer's centre in crystal coordinates.
        bg: ``(3, 3)`` reciprocal vectors in 1/bohr, rows.
        volume: the cell volume in bohr^3.
        sigma: the Gaussian's width in bohr.
        mask: ``(npw,)`` the padding mask.
    """
    miller = np.asarray(miller, dtype=float)
    kcart = (np.asarray(kcrystal, dtype=float)[None, :] + miller) @ np.asarray(bg)
    weighted = np.asarray(coefficients) * gaussian_transform(kcart, sigma)[None, :]
    value = sample_wavefunctions(weighted, miller, kcrystal,
                                 np.asarray(position, dtype=float)[None, :],
                                 volume, mask=mask)
    return np.conj(value[:, 0])


def t_matrix(states: QPIStates, energy: float, broadening: float,
             potential: float, born: bool = False):
    """``(T, g0)`` at one energy: Eq. 24 for the rank-one scatterer.

    ``born`` gives ``T = V``, the first Born approximation (Eq. 31), which is
    the limit the full ``T`` must reach linearly as ``V -> 0``.
    """
    denominator = energy + 1.0j * broadening - np.asarray(states.eigenvalues)
    g0 = np.sum(np.abs(states.overlaps) ** 2 / denominator) / states.nk
    if born:
        return complex(potential), complex(g0)
    return complex(potential / (1.0 - potential * g0)), complex(g0)


def lattice_translations(divisions) -> np.ndarray:
    """``(n1 n2, 2)`` the supercell's in-plane translations, row-major.

    ``a`` runs over ``range(n1)`` and ``b`` over ``range(n2)``, so the map tiles
    the supercell starting at the impurity's own cell; :func:`fourier_map`
    relies on that order.
    """
    n1, n2 = (int(n) for n in divisions)
    a, b = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
    return np.stack([a.ravel(), b.ravel()], axis=-1)


def impurity_maps(states: QPIStates, energy: float, broadening: float,
                  potential: float, translations, inplane=(0, 1), *,
                  readers=READERS, born: bool = False, exit_weights=None):
    """The impurity's change of each reader on the supercell, at one energy.

    Args:
        states: one channel's :class:`QPIStates`.
        energy, broadening: ``w`` and ``eta`` in Ry.
        potential: ``V`` in Ry.
        translations: ``(nR, 2)`` integer in-plane translations, in the basis
            of the two lattice vectors ``inplane`` names.
        inplane: which two lattice vectors span the surface.
        readers: a subset of :data:`READERS`.
        born: the first Born approximation instead of the full ``T``.
        exit_weights: ``(nk, nbnd)``, the factor on the exit leg of the
            ``landauer`` reader in place of ``1/D_kn``: the on-shell amplitude
            :func:`~defuqumat.transport.green.amplitude_weights` gives, which is
            what converges with the window. ``None`` is the literal resolvent.

    Returns ``(maps, backgrounds, extras)``. ``maps[reader]`` is ``(nR,
    npoints)`` real, the change at ``R + u``; ``backgrounds[reader]`` is
    ``(npoints,)``, the clean surface's value at ``u`` (periodic, so one
    cell's worth); ``extras`` carries ``T`` and ``g0``.
    """
    readers = tuple(readers)
    for reader in readers:
        if reader not in READERS:
            raise ValueError(f"unknown reader {reader!r}; expected one of {READERS}")
    if "landauer" in readers and states.exit is None:
        raise ValueError("the landauer reader needs the exit-plane overlaps")
    nk = states.nk
    tmat, g0 = t_matrix(states, energy, broadening, potential, born=born)
    denominator = energy + 1.0j * broadening - np.asarray(states.eigenvalues)
    inverse = 1.0 / denominator                           # (nk, nbnd)
    psi = np.asarray(states.tip)                          # (nk, nbnd, np)
    overlaps = np.asarray(states.overlaps)                # (nk, nbnd)

    kpar = np.asarray(states.kcrystal, dtype=float)[:, list(inplane)]
    phases = np.exp(2.0j * np.pi * (np.asarray(translations, dtype=float)
                                    @ kpar.T))            # (nR, nk)

    # G0(R + u, phi) and G0(phi, R + u): the Bloch parts, then the lattice sum.
    to_phi = np.einsum("kn,knu->ku", overlaps * inverse, psi)
    from_phi = np.einsum("kn,knu->ku", np.conj(overlaps) * inverse, np.conj(psi))
    g_to = phases @ to_phi / nk                           # (nR, np)
    g_from = np.conj(phases) @ from_phi / nk

    maps, backgrounds = {}, {}
    if "ldos" in readers:
        maps["ldos"] = -np.imag(g_to * tmat * g_from) / np.pi
        backgrounds["ldos"] = -np.imag(
            np.einsum("kn,knu->u", inverse, np.abs(psi) ** 2)) / (np.pi * nk)
    if "landauer" in readers:
        gram = np.asarray(states.exit)                    # (nk, nbnd, nbnd)
        leg = inverse if exit_weights is None else np.asarray(exit_weights)
        c = tmat * np.conj(overlaps) * leg                # (nk, nbnd)
        s_c = np.einsum("kmn,kn->km", gram, np.conj(c))   # S_k c*_k
        k_term = np.real(np.einsum("km,km->", c, s_c)) / nk
        h = phases @ np.einsum("km,kmu->ku", leg * s_c, psi) / nk
        maps["landauer"] = (2.0 * np.real(np.conj(g_to) * h)
                            + k_term * np.abs(g_to) ** 2)
        amplitude = psi * leg[:, :, None]                 # (nk, nbnd, np)
        backgrounds["landauer"] = np.real(np.einsum(
            "kmu,kmn,knu->u", amplitude, gram, np.conj(amplitude))) / nk
    extras = {"t_matrix": tmat, "g0": g0, "g_to": g_to}
    return maps, backgrounds, extras


def fourier_map(values, divisions, shape):
    """The supercell map on its own grid, and its in-plane Fourier transform.

    ``values`` is ``(..., nR, npoints)`` with ``R`` in
    :func:`lattice_translations`' order and the tip points of one cell on an
    ``(m1, m2)`` grid at ``(j/m1, l/m2)``, row-major. The supercell field is
    ``f[a m1 + j, b m2 + l]``, and its transform is

        f(q) = sum_x f(x) e^{-i q.x} / (N1 m1 N2 m2),

    at ``q = (p1 b1 + p2 b2) / N`` with ``p`` from ``-N m / 2`` up, so the zone
    centre is in the middle of the array (``fftshift``). The normalisation
    makes ``f(q = 0)`` the map's mean.

    Returns ``(field, transform)``, each ``(..., N1 m1, N2 m2)``.
    """
    n1, n2 = (int(n) for n in divisions)
    m1, m2 = (int(m) for m in shape)
    values = np.asarray(values)
    leading = values.shape[:-2]
    grid = values.reshape(leading + (n1, n2, m1, m2))
    grid = np.moveaxis(grid, -3, -2).reshape(leading + (n1 * m1, n2 * m2))
    transform = np.fft.fftshift(np.fft.fft2(grid, axes=(-2, -1)),
                                axes=(-2, -1)) / (n1 * m1 * n2 * m2)
    return grid, transform


def dense_reference(eigenvalues, coefficients, miller, kcrystal, mask, cell,
                    position, sigma, points, energy, broadening, potential,
                    exit=None, exit_weights=None):
    """The same two maps with no T-matrix: invert the supercell Hamiltonian.

    The basis is every ``(k, n)`` of the window, normalised in the supercell;
    the impurity is ``V p p^dagger`` with ``p_kn = P_kn / sqrt(N)``, and
    ``G = (w + i eta - H)^{-1}`` is a dense inverse. The tip points are given in
    crystal coordinates of the *unit* cell anywhere in the supercell and the
    states are sampled there directly, so neither the Bloch factorisation nor
    the phase matrix is used. Cost ``(nk nbnd)^3``: a test, not a route.

    The exit leg of ``landauer`` uses ``G = (1 + G V) G0``: the column of
    ``mk`` is the scattering state ``(1 + G V)|mk>`` times ``1/D_mk``, and
    ``exit_weights`` replaces that last factor exactly as in
    :func:`impurity_maps`.

    Returns ``{"ldos": (np,), "landauer": (np,)}``, the change the impurity
    makes; ``landauer`` only when ``exit`` (``(nk, nbnd, nbnd)``) is given.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    nk, nbnd = eigenvalues.shape
    bg = np.asarray(cell.bg)
    volume = float(cell.volume)
    p = np.stack([impurity_overlaps(coefficients[ik], miller[ik], kcrystal[ik],
                                    position, bg, volume, sigma, mask=mask[ik])
                  for ik in range(nk)]).reshape(-1) / np.sqrt(nk)
    sampled = np.stack([sample_wavefunctions(coefficients[ik], miller[ik],
                                             kcrystal[ik], points, volume,
                                             mask=mask[ik])
                        for ik in range(nk)])          # (nk, nbnd, np), per cell
    psi = sampled.reshape(nk * nbnd, -1)               # psi_kn(r), cell-normalised
    z = energy + 1.0j * broadening
    clean = np.diag(1.0 / (z - eigenvalues.reshape(-1)))
    hamiltonian = np.diag(eigenvalues.reshape(-1)).astype(complex)
    hamiltonian += potential * np.outer(p, np.conj(p))
    dressed = np.linalg.inv(z * np.eye(nk * nbnd) - hamiltonian)
    out = {}
    # G(r, r) = (1/N) sum psi(r) G psi*(r) with cell-normalised psi.
    for name, green in (("dressed", dressed), ("clean", clean)):
        local = np.einsum("au,ab,bu->u", psi, green, np.conj(psi)) / nk
        out.setdefault("ldos", 0.0)
        sign = 1.0 if name == "dressed" else -1.0
        out["ldos"] = out["ldos"] + sign * (-np.imag(local) / np.pi)
        if exit is not None:
            leg = (1.0 / (z - eigenvalues.reshape(-1)) if exit_weights is None
                   else np.asarray(exit_weights).reshape(-1))
            scattering = green * (z - eigenvalues.reshape(-1))[None, :]
            amplitude = (psi.T @ scattering * leg[None, :]).reshape(-1, nk, nbnd)
            value = np.real(np.einsum("ukm,kmn,ukn->u", amplitude,
                                      np.asarray(exit), np.conj(amplitude))) / nk
            out["landauer"] = out.get("landauer", 0.0) + sign * value
    return out


@dataclass
class QPI:
    """Quasiparticle interference around one impurity, over an energy axis.

    The maps are on the tip plane of the whole Born-von Karman supercell,
    ``(N1 m1, N2 m2)`` with the impurity's cell at the origin; the transforms
    are on the same grid in ``q``, zone centre in the middle. Units: the
    ``ldos`` reader is in states/(bohr^3 Ry), summed over spin; the
    ``landauer`` reader is ``int_exit |G|^2`` in 1/(bohr^4 Ry^2), with the
    leads' couplings left out exactly as in
    :class:`~defuqumat.transport.green.VerticalTransport`.
    """

    #: ``(nE,)`` the energies in Ry.
    energies: np.ndarray
    #: ``reader -> (nE, N1 m1, N2 m2)`` real, the impurity's change.
    maps: dict
    #: ``reader -> (nE, N1 m1, N2 m2)`` complex, the in-plane Fourier transform.
    transforms: dict
    #: ``reader -> (nE, m1, m2)``, the clean surface in one cell.
    backgrounds: dict
    #: ``(N1, N2)`` the supercell, which is the in-plane k-grid.
    divisions: tuple
    #: ``(m1, m2)`` the tip points per cell.
    shape: tuple
    #: The scatterer: crystal position, ``V`` and ``sigma``.
    impurity: dict
    #: ``(nE, nchannels)`` complex, ``T(w)`` and ``g0(w)`` per spin channel.
    t_matrix: np.ndarray
    g0: np.ndarray
    height: float
    axis: int
    broadening: float
    #: The fraction of ``|G0(r, phi)|``'s peak left on the supercell's edge; the
    #: map wraps around when it is not small (``eta`` too small for ``N``).
    wraparound: np.ndarray
    exit_height: float | None = None
    fermi_energy: float | None = None
    born: bool = False
    #: ``reader -> (nchannels, nE, N1 m1, N2 m2)`` for a collinear run, else
    #: ``None``.
    maps_by_spin: dict | None = None
    #: ``(nE, nchannels)``: how many k-points the ``landauer`` reader's exit
    #: leg samples the constant-energy contour on, the participation number
    #: ``(sum_k W_k)^2 / sum_k W_k^2`` of ``W_k = sum_n |g_kn|^2``. When it is
    #: a handful the map takes the k-grid's shape rather than the contour's:
    #: on graphene's 18x18 grid the C3 residue of ``|delta T(q)|`` was 0.35 at
    #: ``eta = 0.03`` and 5.3e-3 at ``0.08``. ``nan`` without that reader.
    contour_points: np.ndarray | None = None
    notes: dict = field(default_factory=dict)

    @property
    def qpi(self) -> dict:
        """``reader -> |delta rho(q, w)|``, what a QPI figure shows."""
        return {name: np.abs(value) for name, value in self.transforms.items()}

    def q_axes(self, cell) -> tuple[np.ndarray, np.ndarray]:
        """The two ``q`` axes in units of ``|b_i|``, the crystal coordinates.

        ``q = p / N_i`` reciprocal-lattice units along each in-plane direction;
        the first zone is ``|q_i| <= 1/2``.
        """
        axes = []
        for n, m in zip(self.divisions, self.shape):
            size = int(n) * int(m)
            axes.append(np.fft.fftshift(np.fft.fftfreq(size, d=1.0 / size)) / int(n))
        return axes[0], axes[1]
