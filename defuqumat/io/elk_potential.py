"""Elk's Kohn-Sham potential from a density: ``potks`` transcribed, on the host.

A ``STATE.OUT`` that Elk's ``task = 1`` can continue from carries the potential
as well as the density, because ``gndstate.f90`` diagonalises with the file's
``vsmt``/``vsir`` before it ever builds a potential of its own. This module
builds that potential from a density on Elk's meshes, the way Elk does, so a
density written by :func:`defuqumat.io.elk_write.write_elk_state` is a state
Elk starts from directly. Transcribed from the official Elk 11.0.2 release
(``elk_reference/elk-11.0.2/src``, downloaded from elk.sourceforge.io, gitignored
as ``quantum_espresso/`` is), and every piece checked against the potentials the
official binary, built unpatched from that tree, wrote for its own density
(``PLAN.md`` P181).

**The pieces, in ``potks.f90``'s order.**

* ``potcoul``: the Coulomb potential by Weinert's method (*J. Math. Phys.* 22,
  2433 (1981)). Inside each sphere the radial Poisson equation is solved
  exactly per ``(l, m)`` with ``wsplintp``'s cumulative quadrature
  (``zpotclmt``), the point nucleus is added to ``l = 0``, the multipoles are
  read off the sphere boundary, the charge inside the spheres is replaced by a
  smooth pseudocharge with the same multipoles, the interstitial equation is
  solved in G-space, and the harmonic that matches the boundary value is added
  inside (``zpotcoul``). Two details are Elk's and kept: for ``l > lmaxi`` the
  radial integrals start at the first outer point with the full mesh's weights
  from there, and only the G-vectors inside ``gmaxvr`` are divided by
  ``G^2``, the rest of the FFT box keeping the raw density transform.
* ``potxc``: exchange-correlation pointwise on the spherical cover of
  ``lmmax`` points (``sphcover``, ``genshtmat``: ``rbsht`` the real harmonics
  on the points, ``rfsht`` its inverse), inner region at ``lmaxi``, outer at
  ``lmaxo``; in the interstitial on the grid. Then ``symrfmt`` symmetrises the
  muffin-tin potential over the crystal group, which is not the identity on a
  symmetric density, because ``v_xc[rho]`` is not band-limited when ``rho`` is
  (measured on Si2 against the official Elk: up to 4.4e-3 Ha without it,
  all in channels the site symmetry forbids, and 2e-13 with it).
* ``trimrfg``: ``vxcir`` truncated to ``|G| <= 2 gkmax`` (``ngvc``), not to
  ``gmaxvr``.
* ``vsmt = vclmt + vxcmt``, ``vsir = vclir + vxcir``; for a spin-polarized
  state ``genbs``: ``B_s = B_xc`` on the coarse radial mesh (every ``lradstp``-th
  point), stored **on the spherical-cover points** (``rbshtip``), not in
  harmonics.

**The parameters ``STATE.OUT`` does not carry** -- ``nrmti``, ``lmaxi``,
``gmaxvr``, ``gkmax``, ``lradstp``, the nuclear charges, the point-nucleus flag
and ``xctype`` -- are read from the ``INFO.OUT`` of the template run
(:func:`read_elk_parameters`).

**What is transcribed and what is refused.** The LDA (``xctype`` 2, 3) and
Elk's own PBE family (20, 21, 22, ``xcgrad = 1``) are, unpolarized, collinear
and (LDA) noncollinear. Refused by name: a libxc functional (``xcgrad = 2``, the
``ggamt_2a/2b`` route) or a meta-GGA, a noncollinear state under a GGA
(``dncgga``), a finite nucleus, and a non-spin-polarized functional on a
spin-polarized state (Elk refuses it too). External magnetic fields are not
part of what is returned: ``B_s`` is ``B_xc`` here, and Elk adds ``bfieldc``
and ``bfcmt`` from its own input in ``genbs`` every loop.

**The group** is Elk's (:func:`elk_space_group`, ``findsymcrys``), not QE's,
with the spin rotation of each operation chosen as Elk chooses it
(:func:`crystal_symmetrizer`).

Units: Hartree and bohr inside, as Elk; :func:`kohn_sham_potential` returns
Hartree. :class:`~defuqumat.io.elk.ElkState` holds Rydberg.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["ElkParameters", "read_elk_parameters", "kohn_sham_potential",
           "coulomb_potential", "exchange_correlation_potential", "radial_weights",
           "crystal_symmetrizer", "elk_space_group", "symmetrizer"]

FOURPI = 4.0 * np.pi
Y00 = 0.28209479177387814347
EPSLAT = 1.0e-6


# ---------------------------------------------------------------- parameters

@dataclass(frozen=True)
class ElkParameters:
    """What ``potks`` needs and ``STATE.OUT`` lacks, from ``INFO.OUT``."""

    nrmti: tuple            # per species
    lmaxi: int
    lmaxo: int
    gmaxvr: float
    gkmax: float
    lradstp: int
    npsd: int
    spzn: tuple             # per species, negative, Elk's sign
    point_nucleus: bool
    xctype: tuple
    spinpol: bool

    @property
    def lmmaxi(self) -> int:
        return (self.lmaxi + 1) ** 2

    @property
    def lmmaxo(self) -> int:
        return (self.lmaxo + 1) ** 2


def read_elk_parameters(directory) -> ElkParameters:
    """Read the template run's ``INFO.OUT`` (its first, initialisation block)."""
    text = (Path(directory) / "INFO.OUT").read_text()
    text = text.split("Self-consistent loop started")[0]

    def number(pattern, cast=float):
        match = re.search(pattern + r"\s*:\s*([-+0-9.Ee]+)", text)
        if match is None:
            raise ValueError(f"INFO.OUT has no line '{pattern}'")
        return cast(match.group(1))

    spzn = tuple(float(x) for x in re.findall(r"(?m)^\s+nuclear charge\s*:\s*([-+0-9.Ee]+)", text))
    nrmti = tuple(int(x) for x in re.findall(r"number on inner part of muffin-tin\s*:\s*(\d+)", text))
    xc = re.search(r"Exchange-correlation functional\s*:\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)", text)
    lmax = re.search(r"outer part of muffin-tin\s*:\s*(\d+)\s*\n\s*inner part of muffin-tin\s*:\s*(\d+)", text)
    if xc is None or lmax is None or not spzn or not nrmti:
        raise ValueError(f"{directory}/INFO.OUT is not the INFO.OUT of an Elk ground-state run")
    return ElkParameters(
        nrmti=nrmti,
        lmaxi=int(lmax.group(2)),
        lmaxo=int(lmax.group(1)),
        gmaxvr=number(r"Maximum \|G\| for potential and density"),
        gkmax=number(r"Maximum \|G\+k\| for APW functions"),
        lradstp=number(r"Radial integration step length", int),
        npsd=number(r"Constant for pseudocharge density", int),
        spzn=spzn,
        point_nucleus="Nuclei treated as point charges" in text,
        xctype=tuple(int(xc.group(i)) for i in (1, 2, 3)),
        spinpol="spin-unpolarised" not in text,
    )


# ---------------------------------------------------------------- radial

def _polynm_integral(xa, ya, x):
    """``polynm(-1, 4, xa, ya, x)``: the integral from ``xa[0]`` to ``x`` of the cubic."""
    np_ = 4
    c = np.array(ya, dtype=float)
    for i in range(2, np_ + 1):
        for j in range(np_, i - 1, -1):
            c[j - 1] = (c[j - 1] - c[j - 2]) / (xa[j - 1] - xa[j - i])
    x0 = xa[0]
    for j in range(1, np_):
        for i in range(1, np_ - j + 1):
            k = np_ - i
            c[k - 1] = c[k - 1] + (x0 - xa[k - j]) * c[k]
    value = c[np_ - 1] / np_
    t1 = x - x0
    for i in range(np_ - 1, 0, -1):
        value = value * t1 + c[i - 1] / i
    return value * t1


def radial_weights(r) -> np.ndarray:
    """``wsplintp``: ``(n, 4)`` weights of the cumulative integral on mesh ``r``."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    wp = np.zeros((n, 4))
    unit = np.eye(4)
    for k in range(4):
        wp[1, k] = _polynm_integral(r[0:4], unit[k], r[1])
    for i in range(2, n - 1):  # Fortran i = 3 .. n-1
        xa = r[i - 2:i + 2]
        for k in range(4):
            wp[i, k] = (_polynm_integral(xa, unit[k], r[i])
                        - _polynm_integral(xa, unit[k], r[i - 1]))
    xa = r[n - 4:n]
    for k in range(4):
        wp[n - 1, k] = (_polynm_integral(xa, unit[k], r[n - 1])
                        - _polynm_integral(xa, unit[k], r[n - 2]))
    return wp


def _cumulative(wp, f):
    """``splintwp``: ``g(r_i) = int_{r_1}^{r_i} f``, complex ``f``."""
    n = len(f)
    g = np.zeros(n, dtype=complex)
    total = wp[1] @ f[0:4]
    g[1] = total
    for i in range(2, n - 1):  # Fortran i = 2 .. n-2 fills g(i+1)
        total = total + wp[i] @ f[i - 2:i + 2]
        g[i] = total
    g[n - 1] = total + wp[n - 1] @ f[n - 4:n]
    return g


# ---------------------------------------------------------------- harmonics

def _rtoz_matrix(lmax):
    """``rtozflm`` as a matrix ``(n, n)``: complex coefficients = M @ real ones."""
    c1 = 0.7071067811865475244
    n = (lmax + 1) ** 2
    m_ = np.zeros((n, n), dtype=complex)
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            lm1 = l * l + l + m
            lm2 = l * l + l - m
            if m < 0:
                if m % 2:
                    m_[lm1, lm2] += -c1
                    m_[lm1, lm1] += -1j * c1
                else:
                    m_[lm1, lm2] += c1
                    m_[lm1, lm1] += -1j * c1
            elif m == 0:
                m_[lm1, lm1] = 1.0
            else:
                if m % 2:
                    m_[lm1, lm1] += c1
                    m_[lm1, lm2] += -1j * c1
                else:
                    m_[lm1, lm1] += c1
                    m_[lm1, lm2] += 1j * c1
    return m_


def _ylm_4pil(lmax, vectors):
    """``genylmv(.true., ...)``: ``4 pi (-i)^l Y_lm`` (Condon-Shortley) at ``vectors``."""
    from scipy.special import sph_harm_y

    v = np.asarray(vectors, dtype=float)
    r = np.linalg.norm(v, axis=1)
    safe = np.where(r > 1e-14, r, 1.0)
    ct = np.where(r > 1e-14, np.clip(v[:, 2] / safe, -1.0, 1.0), 1.0)
    planar = np.hypot(v[:, 0], v[:, 1])
    phi = np.where(planar > 1e-14, np.arctan2(v[:, 1], v[:, 0]), 0.0)
    theta = np.arccos(ct)
    out = np.zeros((len(v), (lmax + 1) ** 2), dtype=complex)
    for l in range(lmax + 1):
        factor = FOURPI * (-1j) ** l
        for m in range(-l, l + 1):
            out[:, l * l + l + m] = factor * sph_harm_y(l, m, theta, phi)
    return out


def _sphcover(n):
    """``sphcover``: ``(theta, phi)`` of ``n`` points spread on the sphere."""
    dz = 2.0 / n
    z = 1.0 - dz / 2.0
    tp = np.zeros((n, 2))
    tp[0] = np.arccos(z), 0.0
    dp = np.pi * (1.0 - np.sqrt(5.0))
    p = 0.0
    for k in range(1, n):
        z = z - dz
        p = p + dp
        tp[k] = np.arccos(z), np.mod(p, 2.0 * np.pi)
    return tp


def _sht_matrices(lmax):
    """``genshtmat``: ``(rbsht, rfsht)``, harmonics to cover points and back."""
    from defuqumat.io.elk_density import real_spherical_harmonics

    n = (lmax + 1) ** 2
    tp = _sphcover(n)
    v = np.stack([np.sin(tp[:, 0]) * np.cos(tp[:, 1]),
                  np.sin(tp[:, 0]) * np.sin(tp[:, 1]), np.cos(tp[:, 0])], axis=1)
    rb = real_spherical_harmonics(lmax, v)  # (point, lm)
    return rb, np.linalg.inv(rb)


# ---------------------------------------------------------------- the setup

@dataclass
class _Setup:
    params: ElkParameters
    lmmaxo: int
    natmtot: int
    species_of: np.ndarray        # (natmtot,)
    r: list                       # per species, (nrmt,)
    wp: list                      # per species, (nrmt, 4)
    rmt: np.ndarray               # per species
    positions: np.ndarray         # (natmtot, 3) cartesian, bohr
    omega: float
    ngridg: tuple
    miller: np.ndarray            # (ngvec, 3) in Elk's sorted order
    gc: np.ndarray
    vgc: np.ndarray
    ylmg: np.ndarray              # (ngvec, lmmaxo)
    sfacg: np.ndarray             # (ngvec, natmtot)
    jl: list                      # per species, (ngvec, lnpsd + 1)
    gclg: np.ndarray
    lnpsd: int
    ngvc_mask: np.ndarray         # (n1, n2, n3) bool: |G| within 2 gkmax
    avec: np.ndarray = None


def _setup(state, params: ElkParameters) -> _Setup:
    from scipy.special import spherical_jn

    geometry = state.geometry
    avec = np.asarray(geometry.avec, dtype=float)  # rows a_i
    natoms = geometry.natoms
    species_of = np.concatenate([[i] * n for i, n in enumerate(natoms)]).astype(int)
    r = [np.asarray(state.rsp[i], dtype=float)[: state.nrmt[i]] for i in range(len(natoms))]
    rmt = np.array([x[-1] for x in r])
    bvec = 2.0 * np.pi * np.linalg.inv(avec).T  # rows b_i
    ng = np.array(state.ngridg)
    ranges = [np.arange(n // 2 - n + 1, n // 2 + 1) for n in ng]
    mill = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
    vgc = mill @ bvec
    gc = np.linalg.norm(vgc, axis=1)
    order = np.argsort(gc, kind="stable")
    mill, vgc, gc = mill[order], vgc[order], gc[order]
    ngvec = int(np.count_nonzero(gc <= params.gmaxvr))
    if ngvec < 1 or gc[ngvec] <= params.gmaxvr:
        raise AssertionError("G-vectors are not sorted by length")
    lnpsd = params.lmaxo + params.npsd + 1
    # init0: npsd = max(nint(0.25 gmaxvr maxval(rmt)), 1). Fortran's nint rounds
    # a half away from zero, and rmt here is rsp(nrmt), an ulp from Elk's rmt,
    # with gmaxvr printed to ten digits: so the printed npsd is checked against a
    # window of one half around the product, not recomputed.
    t = 0.25 * params.gmaxvr * rmt.max()
    slack = 1e-6 * max(t, 1.0)
    if not (abs(t - params.npsd) <= 0.5 + slack
            or (params.npsd == 1 and t <= 1.5 + slack)):
        raise ValueError(f"INFO.OUT's pseudocharge constant {params.npsd} is not "
                         f"init0's max(nint({t:.6f}), 1) for this state's muffin tins")
    positions = np.asarray(geometry.positions, dtype=float) @ avec
    gmask = np.zeros(tuple(ng), dtype=bool)
    within = gc <= 2.0 * params.gkmax + EPSLAT
    gmask[tuple((mill[within] % ng).T)] = True
    keep = slice(0, ngvec)
    gclg = np.zeros(ngvec)
    gclg[1:] = FOURPI / gc[1:ngvec] ** 2
    ls = np.arange(lnpsd + 1)
    jl = [spherical_jn(ls[None, :], gc[keep, None] * rmt[s]) for s in range(len(natoms))]
    return _Setup(
        params=params, lmmaxo=params.lmmaxo, natmtot=len(species_of), species_of=species_of,
        r=r, wp=[radial_weights(x) for x in r], rmt=rmt, positions=positions,
        omega=float(abs(np.linalg.det(avec))), ngridg=tuple(int(n) for n in ng),
        miller=mill[keep], gc=gc[keep], vgc=vgc[keep],
        ylmg=_ylm_4pil(params.lmaxo, vgc[keep]),
        sfacg=np.exp(1j * vgc[keep] @ positions.T),
        jl=jl, gclg=gclg, lnpsd=lnpsd, ngvc_mask=gmask, avec=avec,
    )


def _l_of(lmmax):
    lmax = int(round(np.sqrt(lmmax))) - 1
    return np.concatenate([np.full(2 * l + 1, l) for l in range(lmax + 1)])


def _factn2(n):
    out = 1.0
    while n > 1:
        out *= n
        n -= 2
    return out


# ---------------------------------------------------------------- potcoul

def _zpotclmt(setup, s, zrho):
    """One sphere's radial Poisson solution, ``(nr, lmmaxo)`` complex."""
    params = setup.params
    r, wp = setup.r[s], setup.wp[s]
    nr, nri = len(r), params.nrmti[s]
    out = np.zeros_like(zrho)
    for l in range(params.lmaxo + 1):
        t0 = FOURPI / (2 * l + 1)
        for lm in range(l * l, (l + 1) ** 2):
            if l <= params.lmaxi:
                lo = 0
            else:
                lo = nri
            rr, w = r[lo:], wp[lo:]
            f1 = rr ** (l + 2) * zrho[lo:, lm]
            f2 = rr ** (1 - l) * zrho[lo:, lm]
            g3 = _cumulative(w, f1)
            g1 = _cumulative(w, f2)
            out[lo:, lm] = t0 * (rr ** (-l - 1) * g3 + rr ** l * (g1[-1] - g1))
    return out


def coulomb_potential(state, params: ElkParameters, setup: _Setup | None = None):
    """``potcoul``: ``(vclmt, vclir)`` in Hartree from ``state.rhomt``/``rhoir``.

    ``vclmt`` is ``(lmmaxo, nrmtmax, natmtot)`` real, as on :class:`ElkState`;
    ``vclir`` is ``ngridg``.
    """
    setup = _setup(state, params) if setup is None else setup
    if not params.point_nucleus:
        raise NotImplementedError("a finite nucleus (ptnucl = .false.) is not transcribed")
    lmmaxo = setup.lmmaxo
    lmaxi, lmaxo = params.lmaxi, params.lmaxo
    rtoz = _rtoz_matrix(lmaxo)
    ztor = np.linalg.inv(rtoz)
    lslot = _l_of(lmmaxo)
    zv = []
    for ias in range(setup.natmtot):
        s = setup.species_of[ias]
        nr = len(setup.r[s])
        rho = np.asarray(state.rhomt[:, :nr, ias], dtype=float).T  # (nr, lmmaxo)
        z = _zpotclmt(setup, s, rho @ rtoz.T)
        # vcln: spzn / r, as a (0, 0) coefficient.
        z[:, 0] += params.spzn[s] / setup.r[s] / Y00
        zv.append(z)

    # zpotcoul
    ng = setup.ngridg
    rhoir = np.asarray(state.rhoir, dtype=float).reshape(ng)
    zir = np.fft.fftn(rhoir) / np.prod(ng)
    index = tuple((setup.miller % np.array(ng)).T)
    gc, ylmg, sfacg = setup.gc, setup.ylmg, setup.sfacg
    big = gc > EPSLAT
    qlm = np.zeros((setup.natmtot, lmmaxo), dtype=complex)
    for ias in range(setup.natmtot):
        s = setup.species_of[ias]
        qlm[ias] = (2 * lslot + 1) / FOURPI * setup.rmt[s] ** (lslot + 1) * zv[ias][-1]
    zg = zir[index]
    for ias in range(setup.natmtot):
        s = setup.species_of[ias]
        rmt, jl = setup.rmt[s], setup.jl[s]
        z1 = np.where(big, zg * sfacg[:, ias] / np.where(big, gc, 1.0), 0.0)
        zlm = np.zeros(lmmaxo, dtype=complex)
        zlm[0] = np.sum(jl[:, 1] * rmt ** 2 * FOURPI * Y00 * z1)
        for l in range(1, lmaxo + 1):
            sl = slice(l * l, (l + 1) ** 2)
            zlm[sl] = np.sum((jl[:, l + 1] * rmt ** (l + 2) * z1)[:, None]
                             * np.conj(ylmg[:, sl]), axis=0)
        zlm[0] += np.sum(np.where(big, 0.0, (FOURPI / 3.0) * rmt ** 3 * Y00 * zg))
        qlm[ias] -= zlm
    t1 = _factn2(2 * setup.lnpsd + 1) / setup.omega
    add = np.zeros(len(gc), dtype=complex)
    for ias in range(setup.natmtot):
        s = setup.species_of[ias]
        rmt, jl = setup.rmt[s], setup.jl[s]
        zlm = np.array([t1 / (_factn2(2 * l + 1) * rmt ** l) for l in lslot]) * qlm[ias]
        t2 = np.where(big, gc * rmt, 1.0)
        t3 = 1.0 / t2 ** setup.lnpsd
        z1 = t3 * FOURPI * Y00 * zlm[0]
        for l in range(1, lmaxo + 1):
            t3 = t3 * t2
            sl = slice(l * l, (l + 1) ** 2)
            z1 = z1 + t3 * (ylmg[:, sl] @ zlm[sl])
        z2 = jl[:, setup.lnpsd] * np.conj(sfacg[:, ias])
        head = FOURPI * Y00 / _factn2(2 * setup.lnpsd + 1) * zlm[0]
        add += np.where(big, z1 * z2, head)
    zg = zg + add
    zg = zg * setup.gclg
    zir[index] = zg
    for ias in range(setup.natmtot):
        s = setup.species_of[ias]
        r, rmt, jl = setup.r[s], setup.rmt[s], setup.jl[s]
        nri = params.nrmti[s]
        z1 = zg * sfacg[:, ias]
        zlm = np.zeros(lmmaxo, dtype=complex)
        zlm[0] = np.sum(jl[:, 0] * FOURPI * Y00 * z1)
        for l in range(1, lmaxo + 1):
            sl = slice(l * l, (l + 1) ** 2)
            zlm[sl] = np.sum((jl[:, l] * z1)[:, None] * np.conj(ylmg[:, sl]), axis=0)
        amplitude = (zlm - zv[ias][-1]) / rmt ** lslot
        harmonic = amplitude[None, :] * r[:, None] ** lslot[None, :]
        harmonic[:nri, lslot > lmaxi] = 0.0
        zv[ias] = zv[ias] + harmonic
    vclir = np.real(np.fft.ifftn(zir) * np.prod(ng))
    vclmt = np.zeros(np.shape(state.rhomt))
    for ias in range(setup.natmtot):
        s = setup.species_of[ias]
        nr = len(setup.r[s])
        vclmt[:, :nr, ias] = np.real(zv[ias] @ ztor.T).T
    return vclmt, vclir


# ---------------------------------------------------------------- potxc

def _xc_pzca(rho):
    """``xc_pzca``: ``(vx, vc)`` of the unpolarized Perdew-Zunger LDA, Hartree."""
    rho = np.asarray(rho, dtype=float)
    vx = np.zeros_like(rho)
    vc = np.zeros_like(rho)
    ok = rho >= 1.0e-12
    t1 = 3.0 / (4.0 * np.pi)
    p1 = t1 ** (1.0 / 3.0)
    p2 = t1 * (9.0 * np.pi / 4.0) ** (1.0 / 3.0)
    g, b1, b2 = -0.1423, 1.0529, 0.3334
    a, b, c, d = 0.0311, -0.048, 0.0020, -0.0116
    rs = p1 / np.where(ok, rho, 1.0) ** (1.0 / 3.0)
    ex = -p2 / rs
    vx = np.where(ok, (4.0 / 3.0) * ex, 0.0)
    sq = np.sqrt(rs)
    high = g / (1.0 + b1 * sq + b2 * rs)
    vhigh = high * (1.0 + (7.0 / 6.0) * b1 * sq + (4.0 / 3.0) * b2 * rs) / (1.0 + b1 * sq + b2 * rs)
    lg = np.log(rs)
    vlow = a * lg + (b - a / 3.0) + (2.0 / 3.0) * c * rs * lg + (1.0 / 3.0) * (2.0 * d - c) * rs
    vc = np.where(ok, np.where(rs >= 1.0, vhigh, vlow), 0.0)
    return vx, vc


def _xc_pwca(rhoup, rhodn):
    """``xc_pwca``: ``(vxup, vxdn, vcup, vcdn)`` of the Perdew-Wang LSDA, Hartree."""
    rup = np.asarray(rhoup, dtype=float)
    rdn = np.asarray(rhodn, dtype=float)
    r = rup + rdn
    bad = (rup < 0.0) | (rdn < 0.0) | (r < 1.0e-20)
    rup = np.where(bad, 0.5, rup)
    rdn = np.where(bad, 0.5, rdn)
    r = rup + rdn
    thrd, thrd4 = 1.0 / 3.0, 4.0 / 3.0
    d2f0 = 1.709921
    A = (0.0310907, 0.01554535, 0.0168869)
    A1 = (0.21370, 0.20548, 0.11125)
    B1 = (7.5957, 14.1189, 10.357)
    B2 = (3.5876, 6.1977, 3.6231)
    B3 = (1.6382, 3.3662, 0.88026)
    B4 = (0.49294, 0.62517, 0.49671)
    t1 = 3.0 / (4.0 * np.pi)
    p1 = t1 ** thrd
    p2 = t1 * (9.0 * np.pi / 4.0) ** thrd
    p3 = 1.0 / (2.0 ** thrd4 - 2.0)
    ri = 1.0 / r
    ri2 = ri ** 2
    rs = p1 * ri ** thrd
    rs2 = rs ** 2
    rs12 = np.sqrt(rs)
    rs32 = rs12 * rs
    rsi = 1.0 / rs
    rs12i = 1.0 / rs12
    mz = rup - rdn
    z = mz / r
    z3 = z ** 3
    z4 = z3 * z
    drs = -thrd * rs * ri
    t1 = mz * ri2
    dzu = ri - t1
    dzd = -ri - t1
    t1 = -p2 * rsi / 2.0
    t2 = 1.0 + z
    t3 = 1.0 - z
    t4 = t2 ** thrd4
    t5 = t3 ** thrd4
    t6 = t4 + t5
    ex = t1 * t6
    ders = -ex * rsi
    fz = p3 * (t6 - 2.0)
    t4 = t4 / t2
    t5 = np.where(t3 > 0.0, t5 / np.where(t3 > 0.0, t3, 1.0), 0.0)
    t6 = t4 - t5
    t7 = thrd4 * t6
    dez = t1 * t7
    dfz = p3 * t7
    t1 = ders * drs
    vxup = ex + r * (t1 + dez * dzu)
    vxdn = ex + r * (t1 + dez * dzd)

    def g(k):
        a2 = 2.0 * A[k]
        q1 = a2 * (B1[k] * rs12 + B2[k] * rs + B3[k] * rs32 + B4[k] * rs2)
        dq1 = a2 * (0.5 * B1[k] * rs12i + B2[k] + 1.5 * B3[k] * rs12 + 2.0 * B4[k] * rs)
        q3 = 1.0 / q1
        q2 = 1.0 + q3
        dq2 = -dq1 * q3 ** 2
        q3 = 1.0 / q2
        q4 = 1.0 + A1[k] * rs
        q5 = np.log(q2)
        return -a2 * q4 * q5, -a2 * (A1[k] * q5 + q4 * q3 * dq2)

    ec0, dec0 = g(0)
    ec1, dec1 = g(1)
    mac, mdac = g(2)
    ac, dac = -mac, -mdac
    t1 = 1.0 - z4
    t2 = (fz / d2f0) * t1
    t3 = ec1 - ec0
    t4 = fz * z4
    ec = ec0 + ac * t2 + t3 * t4
    t5 = dec1 - dec0
    ders = dec0 + dac * t2 + t5 * t4
    t6 = 4.0 * fz * z3
    dez = (ac / d2f0) * (dfz * t1 - t6) + t3 * (dfz * z4 + t6)
    t1 = ders * drs
    vcup = ec + r * (t1 + dez * dzu)
    vcdn = ec + r * (t1 + dez * dzd)
    zero = np.zeros_like(r)
    return (np.where(bad, zero, vxup), np.where(bad, zero, vxdn),
            np.where(bad, zero, vcup), np.where(bad, zero, vcdn))


def _xc_points(params, rho, mag):
    """Pointwise ``(vxc, bxc)``; ``mag`` is ``None``, ``(n,)`` or ``(n, 3)``."""
    xctype = params.xctype[0]
    if mag is None:
        if xctype == 2:
            vx, vc = _xc_pzca(rho)
        elif xctype == 3:
            half = 0.5 * rho
            vx, _, vc, _ = _xc_pwca(half, half)
        else:
            raise NotImplementedError(f"xctype {params.xctype} is not transcribed")
        return vx + vc, None
    if xctype != 3:
        raise NotImplementedError(
            f"a spin-polarized state with xctype {params.xctype}: only the Perdew-Wang "
            "LSDA (xctype 3) is transcribed among the spin-polarized functionals")
    noncollinear = mag.ndim == 2
    t1 = np.linalg.norm(mag, axis=1) if noncollinear else mag
    rhoup, rhodn = 0.5 * (rho + t1), 0.5 * (rho - t1)
    vxup, vxdn, vcup, vcdn = _xc_pwca(rhoup, rhodn)
    a, b = vxup + vcup, vxdn + vcdn
    vxc = 0.5 * (a + b)
    if not noncollinear:
        return vxc, 0.5 * (a - b)
    t3 = 0.5 * (a - b)
    t4 = rhoup - rhodn
    t4 = np.where(np.abs(t4) > 1.0e-8, t3 / np.where(np.abs(t4) > 1.0e-8, t4, 1.0), t4)
    return vxc, mag * t4[:, None]


def _gga_points(params, rho_points, mag_points, gradients):
    """PBE pointwise from densities and the gradient tuples of ``ggamt_1``/``ggair_1``."""
    kappa, mu, beta = _pbe_parameters(params.xctype[0])
    channels, total = gradients
    if mag_points is None:
        grho, g2rho, g3rho = channels[0]
        half = 0.5 * rho_points
        vx, _, vc, _ = _xc_pbe(kappa, mu, beta, half, half, grho, 0.5 * grho, 0.5 * grho,
                               0.5 * g2rho, 0.5 * g2rho, g3rho, 0.25 * g3rho, 0.25 * g3rho)
        return vx + vc, None
    (gup, g2up, g3up), (gdn, g2dn, g3dn) = channels
    grho, g3rho = total
    rhoup, rhodn = 0.5 * (rho_points + mag_points), 0.5 * (rho_points - mag_points)
    vxup, vxdn, vcup, vcdn = _xc_pbe(kappa, mu, beta, rhoup, rhodn, grho, gup, gdn,
                                     g2up, g2dn, g3rho, g3up, g3dn)
    a, b = vxup + vcup, vxdn + vcdn
    return 0.5 * (a + b), 0.5 * (a - b)


def exchange_correlation_potential(state, params: ElkParameters, symmetry=None,
                                   setup: _Setup | None = None):
    """``potxc`` + ``trimrfg``: ``(vxcmt, vxcir, bxcmt, bxcir)`` in Hartree.

    ``symmetry`` is ``symrfmt``'s group as a callable on the muffin-tin
    functions (:func:`symmetrizer`), or ``None`` for Elk's ``symtype = 0``.
    The LDA (``xctype`` 2, 3) and Elk's own PBE family (20, 21, 22, through
    ``ggamt_1``/``ggair_1``) are transcribed, the latter unpolarized and
    collinear.
    """
    setup = _setup(state, params) if setup is None else setup
    xctype = params.xctype[0]
    gga = xctype in (20, 21, 22)
    if any(x != 0 for x in params.xctype[1:]) or not (gga or xctype in (2, 3)):
        raise NotImplementedError(
            f"xctype {params.xctype}: only the LDA (2, 3) and Elk's own PBE family "
            "(20, 21, 22) are transcribed; a libxc functional (xcgrad = 2, the ggamt_2a/2b "
            "route) or a meta-GGA is not")
    lmmaxi, lmmaxo = params.lmmaxi, params.lmmaxo
    rbi, rfi = _sht_matrices(params.lmaxi)
    rbo, rfo = _sht_matrices(params.lmaxo)
    spinpol = state.magmt is not None
    ndmag = state.magmt.shape[-1] if spinpol else 0
    if gga and ndmag == 3:
        raise NotImplementedError("a noncollinear state under a GGA (dncgga) is not transcribed")
    vxcmt = np.zeros(np.shape(state.rhomt))
    bxcmt = np.zeros(np.shape(state.magmt)) if spinpol else None
    for ias in range(setup.natmtot):
        s = setup.species_of[ias]
        nr, nri = len(setup.r[s]), params.nrmti[s]
        if gga:
            rho_lm = np.asarray(state.rhomt[:, :nr, ias], dtype=float).T
            points = np.zeros_like(rho_lm)
            points[:nri, :lmmaxi] = rho_lm[:nri, :lmmaxi] @ rbi.T
            points[nri:] = rho_lm[nri:] @ rbo.T
            grad = _Gradients(setup.r[s], nri, params.lmaxi, params.lmaxo)
            if spinpol:
                m_lm = np.asarray(state.magmt[:, :nr, ias, 0], dtype=float).T
                mpts = np.zeros_like(m_lm)
                mpts[:nri, :lmmaxi] = m_lm[:nri, :lmmaxi] @ rbi.T
                mpts[nri:] = m_lm[nri:] @ rbo.T
                up, dn = 0.5 * (points + mpts), 0.5 * (points - mpts)
                to_lm = []
                for f in (up, dn):
                    g = np.zeros_like(f)
                    g[:nri, :lmmaxi] = f[:nri, :lmmaxi] @ rfi.T
                    g[nri:] = f[nri:] @ rfo.T
                    to_lm.append(g)
                gradients = _ggamt_1(grad, (rbi, rbo), (rfi, rfo), nri, lmmaxi, to_lm)
                v, b = _gga_points(params, points, mpts, gradients)
            else:
                gradients = _ggamt_1(grad, (rbi, rbo), (rfi, rfo), nri, lmmaxi, [rho_lm])
                v, b = _gga_points(params, points, None, gradients)
            for field, target in ((v, vxcmt), (b, bxcmt)):
                if field is None:
                    continue
                out = np.zeros_like(field)
                out[:nri, :lmmaxi] = field[:nri, :lmmaxi] @ rfi.T
                out[nri:] = field[nri:] @ rfo.T
                if target is vxcmt:
                    vxcmt[:, :nr, ias] = out.T
                else:
                    bxcmt[:, :nr, ias, 0] = out.T
            continue
        for rows, nlm, rb, rf in ((slice(0, nri), lmmaxi, rbi, rfi),
                                  (slice(nri, nr), lmmaxo, rbo, rfo)):
            rho = rb @ state.rhomt[:nlm, rows, ias]           # (points, radii)
            mag = None
            if spinpol:
                mag = np.stack([rb @ state.magmt[:nlm, rows, ias, k] for k in range(ndmag)], -1)
                mag = mag[..., 0] if ndmag == 1 else mag
            shape = rho.shape
            v, b = _xc_points(params, rho.reshape(-1),
                              None if mag is None else mag.reshape((-1,) + mag.shape[2:]))
            vxcmt[:nlm, rows, ias] = rf @ v.reshape(shape)
            if spinpol:
                b = b.reshape(shape + ((ndmag,) if ndmag == 3 else ()))
                for k in range(ndmag):
                    bxcmt[:nlm, rows, ias, k] = rf @ (b[..., k] if ndmag == 3 else b)
    if symmetry is not None:
        vxcmt = symmetry(vxcmt)
        if spinpol:
            bxcmt = symmetry(bxcmt, magnetic=True)
    ng = setup.ngridg
    rhoir = np.asarray(state.rhoir, dtype=float).reshape(ng)
    magir = None
    if spinpol:
        magir = np.asarray(state.magir, dtype=float).reshape(ng + (ndmag,))
        magir = magir[..., 0] if ndmag == 1 else magir.reshape(-1, 3)
    if gga:
        if spinpol:
            gradients = _ggair_1(setup, [0.5 * (rhoir + magir), 0.5 * (rhoir - magir)])
            v, b = _gga_points(params, rhoir, magir, gradients)
        else:
            v, b = _gga_points(params, rhoir, None, _ggair_1(setup, [rhoir]))
    else:
        v, b = _xc_points(params, rhoir.reshape(-1),
                          None if magir is None else (magir.reshape(-1) if ndmag == 1 else magir))
    vxcir = _trim(np.reshape(v, ng), setup)
    bxcir = None
    if spinpol:
        bxcir = np.reshape(b, ng + (ndmag,)) if ndmag == 3 else np.reshape(b, ng + (1,))
    return vxcmt, vxcir, bxcmt, bxcir


def _trim(field, setup):
    """``trimrfg``: keep the Fourier components with ``|G| <= 2 gkmax``."""
    z = np.fft.fftn(field)
    z[~setup.ngvc_mask] = 0.0
    return np.real(np.fft.ifftn(z))



# ---------------------------------------------------------------- gradients (PBE)

def _spline4(x, f):
    """``spline(4, x, f, cf)``: ``(3, 4)``, the first derivative, half the second
    and a sixth of the third at the four points."""
    x0 = x[0]
    x1, x2, x3 = x[1] - x0, x[2] - x0, x[3] - x0
    t4, t5, t6 = x1 - x2, x1 - x3, x2 - x3
    y0 = f[0]
    y1, y2, y3 = f[1] - y0, f[2] - y0, f[3] - y0
    t1 = x1 * x2 * y3
    t2 = x2 * x3 * y1
    t3 = x1 * x3
    t0 = 1.0 / (x2 * t3 * t4 * t5 * t6)
    t3 = t3 * y2
    t7 = t0 * (t1 * t4 + t2 * t6 - t3 * t5)
    t4, t5, t6 = x1 ** 2, x2 ** 2, x3 ** 2
    y1 = t3 * t6 - t1 * t5
    y3 = t2 * t5 - t3 * t4
    y2 = t1 * t4 - t2 * t6
    t1 = t0 * (x1 * y1 + x2 * y2 + x3 * y3)
    t2 = -t0 * (y1 + y2 + y3)
    cf = np.zeros((3, 4))
    cf[:, 0] = t1, t2, t7
    cf[:, 1] = t1 + 2.0 * t2 * x1 + 3.0 * t7 * t4, t2 + 3.0 * t7 * x1, t7
    cf[:, 2] = t1 + 2.0 * t2 * x2 + 3.0 * t7 * t5, t2 + 3.0 * t7 * x2, t7
    cf[:, 3] = t1 + 2.0 * t2 * x3 + 3.0 * t7 * t6, t2 + 3.0 * t7 * x3, t7
    return cf


def _wspline(r):
    """``wspline``: ``(n, 4, 3)``, ``cf[k, i] = sum_j wc[i, j, k] f[window_i, j]``."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    wc = np.zeros((n, 4, 3))
    unit = np.eye(4)
    for j in range(4):
        cf = _spline4(r[0:4], unit[j])
        wc[0, j], wc[1, j] = cf[:, 0], cf[:, 1]
    for i in range(2, n - 3):  # Fortran i = 3 .. n-3, window x(i-1 .. i+2)
        for j in range(4):
            wc[i, j] = _spline4(r[i - 1:i + 3], unit[j])[:, 1]
    for j in range(4):
        cf = _spline4(r[n - 4:n], unit[j])
        wc[n - 3, j], wc[n - 2, j], wc[n - 1, j] = cf[:, 1], cf[:, 2], cf[:, 3]
    return wc


def _windows(f):
    """The four-point window of each point (``splined``/``splinew``'s)."""
    n = len(f)
    idx = np.empty((n, 4), dtype=int)
    idx[0] = idx[1] = np.arange(4)
    for i in range(2, n - 2):
        idx[i] = np.arange(i - 1, i + 3)
    idx[n - 2] = idx[n - 1] = np.arange(n - 4, n)
    return f[idx]


def _splined(wc, f):
    """``splined``: the first derivative, ``(n, ...)``."""
    return np.einsum("ij,ij...->i...", wc[:, :, 0], _windows(f))


def _splinew(wc, f):
    """``splinew``: ``(3, n, ...)`` spline coefficients."""
    return np.einsum("ijk,ij...->ki...", wc, _windows(f))


def _clebgor(j1, j2, j3, m1, m2, m3):
    """``<j1 m1 j2 m2 | j3 m3>`` by Racah's formula (Elk's ``clebgor``)."""
    from math import factorial, sqrt

    if (m1 + m2 != m3 or j3 < abs(j1 - j2) or j3 > j1 + j2 or abs(m1) > j1
            or abs(m2) > j2 or abs(m3) > j3):
        return 0.0
    if j1 == j2 == j3 == 0:
        return 1.0
    pre = sqrt((2 * j3 + 1) * factorial(j3 + j1 - j2) * factorial(j3 - j1 + j2)
               * factorial(j1 + j2 - j3) / factorial(j1 + j2 + j3 + 1))
    pre *= sqrt(factorial(j3 + m3) * factorial(j3 - m3) * factorial(j1 - m1)
                * factorial(j1 + m1) * factorial(j2 - m2) * factorial(j2 + m2))
    total = 0.0
    for k in range(0, j1 + j2 - j3 + 1):
        a = (j1 + j2 - j3 - k, j1 - m1 - k, j2 + m2 - k, j3 - j2 + m1 + k, j3 - j1 - m2 + k)
        if min(a) < 0:
            continue
        denominator = factorial(k)
        for x in a:
            denominator *= factorial(x)
        total += (-1) ** k / denominator
    return pre * total


class _Gradients:
    """``gradrfmt`` and ``grad2rfmt`` on one species' mesh, dense ``(nr, lmmax)``."""

    def __init__(self, r, nri, lmaxi, lmaxo):
        self.r, self.nri = np.asarray(r, dtype=float), nri
        self.ri = 1.0 / self.r
        self.wc = _wspline(self.r)
        self.rtoz = _rtoz_matrix(lmaxo)
        self.ztor = np.linalg.inv(self.rtoz)
        self.lmmaxi, self.lmmaxo = (lmaxi + 1) ** 2, (lmaxo + 1) ** 2
        # gradzfmt's coupling, per region: target (lm1, j) += t3 (d_r f_lm + c f_lm / r)
        self.terms = []
        for lmax in (lmaxi, lmaxo):
            terms = []
            for l in range(lmax + 1):
                t1 = -np.sqrt((l + 1) / (2 * l + 1))
                t2 = np.sqrt(l / (2 * l + 1)) if l > 0 else 0.0
                for m in range(-l, l + 1):
                    lm = l * l + l + m
                    for mu, j in ((-1, 0), (0, 2), (1, 1)):
                        if l + 1 <= lmax:
                            lm1 = (l + 1) * (l + 2) + (m - mu)
                            terms.append((lm, lm1, j, t1 * _clebgor(l + 1, 1, l, m - mu, mu, m),
                                          -float(l)))
                        if abs(m - mu) <= l - 1:
                            lm1 = (l - 1) * l + (m - mu)
                            terms.append((lm, lm1, j, t2 * _clebgor(l - 1, 1, l, m - mu, mu, m),
                                          float(l + 1)))
            self.terms.append(terms)

    def _radial_derivative(self, z):
        nri, out = self.nri, np.zeros_like(z)
        out[:, :self.lmmaxi] = _splined(self.wc, z[:, :self.lmmaxi])
        out[nri:, self.lmmaxi:] = _splined(self.wc[nri:], z[nri:, self.lmmaxi:])
        return out

    def gradient(self, f):
        """``gradrfmt``: ``(3, nr, lmmaxo)``, the cartesian gradient of real ``f``."""
        nri = self.nri
        z = f @ self.rtoz.T
        dz = self._radial_derivative(z)
        g = np.zeros((3,) + z.shape, dtype=complex)
        for terms, rows in ((self.terms[0], slice(0, nri)), (self.terms[1], slice(nri, None))):
            ri = self.ri[rows]
            for lm, lm1, j, t3, c in terms:
                g[j, rows, lm1] += t3 * (dz[rows, lm] + c * ri * z[rows, lm])
        g1, g2 = g[0].copy(), g[1].copy()
        c1 = 0.7071067811865475244
        g[0] = c1 * (g1 - g2)
        z1 = c1 * (g1 + g2)
        g[1] = z1.imag - 1j * z1.real
        return np.real(np.einsum("xrb,ab->xra", g, self.ztor))

    def laplacian(self, f):
        """``grad2rfmt``: ``(nr, lmmaxo)``."""
        nri, out = self.nri, np.zeros_like(f)
        lof = _l_of(self.lmmaxo).astype(float)
        li = lof[None, :self.lmmaxi]
        cf = _splinew(self.wc, f[:, :self.lmmaxi])
        out[:, :self.lmmaxi] = (2.0 * (self.ri[:, None] * cf[0] + cf[1])
                                - li * (li + 1) * self.ri[:, None] ** 2 * f[:, :self.lmmaxi])
        lo = lof[None, self.lmmaxi:]
        cf = _splinew(self.wc[nri:], f[nri:, self.lmmaxi:])
        out[nri:, self.lmmaxi:] = (2.0 * (self.ri[nri:, None] * cf[0] + cf[1])
                                   - lo * (lo + 1) * self.ri[nri:, None] ** 2 * f[nri:, self.lmmaxi:])
        return out


def _x_pbe(kappa, mu, rho, s, u, v):
    """``x_pbe``: ``(ex, vx)``."""
    ax = -0.7385587663820224058
    ul = mu / kappa
    exu = ax * rho ** (1.0 / 3.0)
    s2 = s ** 2
    p0 = 1.0 + ul * s2
    fxpbe = 1.0 + kappa - kappa / p0
    fs = 2.0 * kappa * ul / (p0 * p0)
    fss = -4.0 * ul * s * fs / p0
    return exu * fxpbe, exu * ((4.0 / 3.0) * fxpbe - (u - (4.0 / 3.0) * s2 * s) * fss - v * fs)


def _gcor(a, a1, b1, b2, b3, b4, rtrs):
    """``c_pbe_gcor``."""
    q0 = -2.0 * a * (1.0 + a1 * rtrs * rtrs)
    q1 = 2.0 * a * rtrs * (b1 + rtrs * (b2 + rtrs * (b3 + b4 * rtrs)))
    q2 = np.log(1.0 + 1.0 / q1)
    q3 = a * (b1 / rtrs + 2.0 * b2 + rtrs * (3.0 * b3 + 4.0 * b4 * rtrs))
    return q0 * q2, -2.0 * a * a1 * q2 - q0 * q3 / (q1 * (1.0 + q1))


def _c_pbe(beta, rs, z, t, uu, vv, ww):
    """``c_pbe``: ``(vcup, vcdn)``."""
    thrd = 1.0 / 3.0
    gam = 0.5198420997897463295
    fzz = 8.0 / (9.0 * gam)
    gamma = 0.0310906908696548950
    eta = 1.0e-12
    delt = beta / gamma
    rtrs = np.sqrt(rs)
    eu, eurs = _gcor(0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, rtrs)
    ep, eprs = _gcor(0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517, rtrs)
    alfm, alfrsm = _gcor(0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671, rtrs)
    z4 = z ** 4
    f = ((1.0 + z) ** (4 * thrd) + (1.0 - z) ** (4 * thrd) - 2.0) / gam
    ec = eu * (1.0 - f * z4) + ep * f * z4 - alfm * f * (1.0 - z4) / fzz
    ecrs = eurs * (1.0 - f * z4) + eprs * f * z4 - alfrsm * f * (1.0 - z4) / fzz
    fz = (4 * thrd) * ((1.0 + z) ** thrd - (1.0 - z) ** thrd) / gam
    ecz = (4.0 * z ** 3 * f * (ep - eu + alfm / fzz)
           + fz * (z4 * ep - z4 * eu - (1.0 - z4) * alfm / fzz))
    comm = ec - rs * ecrs / 3.0 - z * ecz
    vcup = comm + ecz
    vcdn = comm - ecz
    g = ((1.0 + z) ** (2 * thrd) + (1.0 - z) ** (2 * thrd)) / 2.0
    g3 = g ** 3
    pon = -ec / (g3 * gamma)
    b = delt / (np.exp(pon) - 1.0)
    b2 = b * b
    t2 = t * t
    t4 = t2 * t2
    q4 = 1.0 + b * t2
    q5 = 1.0 + b * t2 + b2 * t4
    h = g3 * (beta / delt) * np.log(1.0 + delt * q4 * t2 / q5)
    g4 = g3 * g
    t6 = t4 * t2
    rsthrd = rs / 3.0
    gz = (((1.0 + z) ** 2 + eta) ** (-thrd / 2.0) - ((1.0 - z) ** 2 + eta) ** (-thrd / 2.0)) / 3.0
    fac = delt / b + 1.0
    bg = -3.0 * b2 * ec * fac / (beta * g4)
    bec = b2 * fac / (beta * g3)
    q8 = q5 * q5 + delt * q4 * q5 * t2
    q9 = 1.0 + 2.0 * b * t2
    hb = -beta * g3 * b * t6 * (2.0 + b * t2) / q8
    hrs = -rsthrd * hb * bec * ecrs
    fact0 = 2.0 * delt - 6.0 * b
    fact1 = q5 * q9 + q4 * q9 * q9
    hbt = 2.0 * beta * g3 * t4 * ((q4 * q5 * fact0 - delt * fact1) / q8) / q8
    hrst = rsthrd * t2 * hbt * bec * ecrs
    hz = 3.0 * gz * h / g + hb * (bg * gz + bec * ecz)
    ht = 2.0 * beta * g3 * q9 / q8
    hzt = 3.0 * gz * ht / g + hbt * (bg * gz + bec * ecz)
    fact2 = q4 * q5 + b * t2 * (q4 * q9 + q5)
    fact3 = 2.0 * b * q5 * q9 + delt * fact2
    htt = 4.0 * beta * g3 * t * (2.0 * b / q8 - (q9 * fact3 / q8) / q8)
    comm = h + hrs + hrst + t2 * ht / 6.0 + 7.0 * t2 * t * htt / 6.0
    pref = hz - gz * t2 * ht / g
    fact5 = gz * (2.0 * ht + t * htt) / g
    comm = comm - pref * z - uu * htt - vv * ht - ww * (hzt - fact5)
    return vcup + comm + pref, vcdn + comm - pref


def _xc_pbe(kappa, mu, beta, rhoup, rhodn, grho, gup, gdn, g2up, g2dn, g3rho, g3up, g3dn):
    """``xc_pbe``: ``(vxup, vxdn, vcup, vcdn)`` in Hartree, pointwise."""
    rup, rdn = np.asarray(rhoup, dtype=float), np.asarray(rhodn, dtype=float)
    r = rup + rdn
    ok = (rup >= 0.0) & (rdn >= 0.0) & (r > 1.0e-12)
    one = np.ones_like(r)
    rup_s, rdn_s, r_s = (np.where(ok, x, one) for x in (rup, rdn, r))
    vx = []
    for rs_, gs, g2s, g3s in ((rup_s, gup, g2up, g3up), (rdn_s, gdn, g2dn, g3dn)):
        r2 = 2.0 * rs_
        kf = (r2 * 3.0 * np.pi ** 2) ** (1.0 / 3.0)
        s = gs / (2.0 * kf * rs_)
        u = g3s / (rs_ ** 2 * (2.0 * kf) ** 3)
        v = g2s / (rs_ * (2.0 * kf) ** 2)
        vx.append(_x_pbe(kappa, mu, r2, s, u, v)[1])
    rs = (3.0 / (4.0 * np.pi * r_s)) ** (1.0 / 3.0)
    z = (rup_s - rdn_s) / r_s
    g = ((1.0 + z) ** (2.0 / 3.0) + (1.0 - z) ** (2.0 / 3.0)) / 2.0
    kf = (r_s * 3.0 * np.pi ** 2) ** (1.0 / 3.0)
    ks = np.sqrt(4.0 * kf / np.pi)
    ksg = 2.0 * ks * g
    t = grho / (ksg * r_s)
    uu = g3rho / (r_s ** 2 * ksg ** 3)
    vv = (g2up + g2dn) / (r_s * ksg ** 2)
    ww = (gup ** 2 - gdn ** 2 - z * grho ** 2) / (r_s * r_s * ksg ** 2)
    vcup, vcdn = _c_pbe(beta, rs, z, t, uu, vv, ww)
    zero = np.zeros_like(r)
    return tuple(np.where(ok, x, zero) for x in (vx[0], vx[1], vcup, vcdn))


def _pbe_parameters(xctype):
    """``(kappa, mu, beta)`` of PBE (20), revPBE of Zhang and Yang (21), PBEsol (22)."""
    kappa, mu, beta = 0.804, 0.2195149727645171, 0.06672455060314922
    if xctype == 21:
        kappa = 1.245
    if xctype == 22:
        mu, beta = 10.0 / 81.0, 0.046
    return kappa, mu, beta


def _rfft_ops(setup):
    """Elk's ``rzfftifc`` pair (the fastest axis halved), its G and the ``ngvc`` cut."""
    ng = np.array(setup.ngridg)
    bvec = 2.0 * np.pi * np.linalg.inv(np.asarray(setup.avec)).T
    k = [np.arange(ng[0] // 2 + 1), np.arange(ng[1]), np.arange(ng[2])]
    m = [np.where(kk > n // 2, kk - n, kk) for kk, n in zip(k, ng)]
    mill = np.stack(np.meshgrid(*m, indexing="ij"), -1)
    G = mill @ bvec
    cut = np.linalg.norm(G, axis=-1) <= 2.0 * setup.params.gkmax + EPSLAT

    def forward(f):
        return np.fft.rfftn(f, axes=(2, 1, 0)) / np.prod(ng)

    def backward(z):
        return np.fft.irfftn(z, s=(ng[2], ng[1], ng[0]), axes=(2, 1, 0)) * np.prod(ng)

    return forward, backward, G, cut


def _ggair_1(setup, rhos):
    """``ggair_1``/``ggair_sp_1``: per channel ``(|grad|, lap, g3)``, and the total's.

    The gradients of the density are cut at ``|G| <= 2 gkmax`` (``ngvc``); the
    gradient of ``|grad rho|`` in ``g3`` is taken with every G, as Elk does.
    """
    forward, backward, G, cut = _rfft_ops(setup)
    channels, vectors = [], []
    for rho in rhos:
        z = forward(rho)
        gv = np.stack([backward(np.where(cut, 1j * G[..., i] * z, 0.0)) for i in range(3)])
        gn = np.sqrt(np.sum(gv ** 2, axis=0))
        lap = backward(np.where(cut, -np.sum(G ** 2, axis=-1) * z, 0.0))
        zg = forward(gn)
        g3 = sum(gv[i] * backward(1j * G[..., i] * zg) for i in range(3))
        channels.append((gn, lap, g3))
        vectors.append(gv)
    if len(rhos) == 1:
        return channels, None
    gv = vectors[0] + vectors[1]
    gn = np.sqrt(np.sum(gv ** 2, axis=0))
    zg = forward(gn)
    return channels, (gn, sum(gv[i] * backward(1j * G[..., i] * zg) for i in range(3)))


def _ggamt_1(grad, rb, rf, nri, lmmaxi, rhos_lm):
    """``ggamt_1``/``ggamt_sp_1`` on one sphere, on the cover points.

    ``rhos_lm`` are channels in harmonics, dense ``(nr, lmmaxo)``; the results
    are on the cover points, dense, the inner region's past ``lmmaxi`` zero.
    """
    def to_points(f):
        out = np.zeros_like(f)
        out[:nri, :lmmaxi] = f[:nri, :lmmaxi] @ rb[0].T
        out[nri:] = f[nri:] @ rb[1].T
        return out

    def to_lm(f):
        out = np.zeros_like(f)
        out[:nri, :lmmaxi] = f[:nri, :lmmaxi] @ rf[0].T
        out[nri:] = f[nri:] @ rf[1].T
        return out

    channels, vectors = [], []
    for f in rhos_lm:
        gv = np.stack([to_points(x) for x in grad.gradient(f)])
        gn = np.sqrt(np.sum(gv ** 2, axis=0))
        lap = to_points(grad.laplacian(f))
        gg = np.stack([to_points(x) for x in grad.gradient(to_lm(gn))])
        channels.append((gn, lap, np.sum(gv * gg, axis=0)))
        vectors.append(gv)
    if len(rhos_lm) == 1:
        return channels, None
    gv = vectors[0] + vectors[1]
    gn = np.sqrt(np.sum(gv ** 2, axis=0))
    gg = np.stack([to_points(x) for x in grad.gradient(to_lm(gn))])
    return channels, (gn, np.sum(gv * gg, axis=0))

# ---------------------------------------------------------------- symmetry

def symmetrizer(state, rotations_cart, atom_maps, lmax: int, spin_matrices=None):
    """``symrfmt`` as a callable: the group average of muffin-tin functions.

    ``rotations_cart`` are the cartesian rotations ``R`` of the crystal group
    and ``atom_maps[k][a]`` the atom that symmetry ``k`` maps atom ``a`` onto
    (``R tau_a + t``, folded). A symmetric function has ``v_a(x) = v_b(R^-1 x)``
    for ``b`` the atom mapped onto ``a``, so the average is ``(1/N) sum_k
    D(R_k) v_b``, with ``D`` the rotation of Elk's real harmonics, computed by
    quadrature (exact to degree 47, far past ``2 lmax``).
    """
    from defuqumat.io.elk_density import real_spherical_harmonics

    xg, wg = np.polynomial.legendre.leggauss(24)
    ph = np.arange(48) * 2.0 * np.pi / 48
    th, pp = np.meshgrid(np.arccos(xg), ph, indexing="ij")
    dirs = np.stack([np.sin(th) * np.cos(pp), np.sin(th) * np.sin(pp), np.cos(th)], -1).reshape(-1, 3)
    w = (np.repeat(wg[:, None], 48, 1) * 2.0 * np.pi / 48).reshape(-1)
    base = real_spherical_harmonics(lmax, dirs)
    blocks = []
    for rot in rotations_cart:
        rot = np.asarray(rot, dtype=float)
        rotated = real_spherical_harmonics(lmax, dirs @ rot)   # R_lm'(R^-1 x), R orthogonal
        blocks.append(np.einsum("q,qa,qb->ab", w, base, rotated))
    natmtot = len(atom_maps[0])
    sources = [[None] * natmtot for _ in rotations_cart]
    for k, amap in enumerate(atom_maps):
        for b, a in enumerate(amap):
            sources[k][a] = b

    def apply(field, magnetic=False):
        """The group average; a magnetic field's components rotate as ``symrvfmt``'s.

        ``spin_matrices[k]`` is the matrix the components are carried with:
        ``(-1)^t_rev det(R) R`` for an axial vector under defuqumat's magnetic
        group, whose spin rotation is the spatial one (a subgroup of Elk's
        spin-space group without spin-orbit coupling, the same group with it).
        A collinear field (one component) takes the ``zz`` element.
        """
        out = np.zeros_like(field)
        n = len(rotations_cart)
        for k in range(n):
            d = blocks[k]
            for a in range(natmtot):
                b = sources[k][a]
                rotated = np.einsum("ab,b...->a...", d, field[:, :, b]) / n
                if magnetic and spin_matrices is not None:
                    sc = np.asarray(spin_matrices[k], dtype=float)
                    if rotated.shape[-1] == 3:
                        rotated = np.einsum("ij,...j->...i", sc, rotated)
                    else:
                        rotated = rotated * sc[2, 2]
                out[:, :, a] += rotated
        return out

    return apply


def elk_space_group(geometry):
    """``(rotations, atom_maps)``: Elk's crystal group of ``geometry`` (``findsymcrys``).

    Every lattice symmetry (integer ``M`` with entries in ``{-1, 0, 1}`` keeping
    the metric, ``findsymlat``) with **any** fractional translation that maps
    each atom onto an atom of its species, to ``epslat``. This is not QE's
    search, which keeps a translation only when it is commensurate with the FFT
    grid: O2 off the origin in a cubic box has 16 operations in Elk and 8 in
    ``sgam_at``, and symmetrising with the 8 left 1.3e-3 Ha in ``vxcmt`` (P181).
    ``rotations`` are cartesian (``v' = R v``); ``atom_maps[k][a]`` is the atom
    operation ``k`` takes atom ``a`` to.
    """
    import itertools

    avec = np.asarray(geometry.avec, dtype=float)
    metric = avec @ avec.T
    frac = np.asarray(geometry.positions, dtype=float) % 1.0
    species = np.concatenate([[i] * n for i, n in enumerate(geometry.natoms)]).astype(int)
    lattice = []
    for entries in itertools.product((-1, 0, 1), repeat=9):
        m = np.array(entries, dtype=float).reshape(3, 3)
        if abs(abs(np.linalg.det(m)) - 1.0) > 1e-8:
            continue
        if np.allclose(m.T @ metric @ m, metric, atol=1e-6 * np.abs(metric).max()):
            lattice.append(m)
    rotations, maps = [], []
    first = 0
    for m in lattice:
        image = frac @ m.T
        for b in np.nonzero(species == species[first])[0]:
            t = frac[b] - image[first]
            moved = image + t
            amap = []
            for a in range(len(frac)):
                d = frac - moved[a]
                d -= np.round(d)
                hit = np.nonzero((np.abs(d).max(axis=1) < EPSLAT) & (species == species[a]))[0]
                if len(hit) != 1:
                    break
                amap.append(int(hit[0]))
            else:
                rotations.append(avec.T @ m @ np.linalg.inv(avec.T))
                maps.append(amap)
                break
    return rotations, maps


def crystal_symmetrizer(state, lmax: int, moments=None, spinorb: bool = False):
    """``symrfmt``/``symrvfmt`` over Elk's group for ``state``, as a callable.

    ``moments`` are the per-atom moments (``(nat,)`` collinear, ``(nat, 3)``
    noncollinear) the magnetic part of the group is chosen from; for each
    spatial operation the spin rotation is the first that carries every moment
    onto its image's, as Elk's search takes the first: the identity, then a
    reversal (a collinear antiferromagnet), then ``det(R) R`` itself, which is
    the only one Elk allows under spin-orbit coupling (``spinorb``). An
    operation with none is not in the group. Without spin-orbit coupling Elk's
    group is a spin-space group, and for moments all parallel to one axis it
    holds every spatial operation with the identity, so the components of
    ``B_xc`` average as scalars (tilted O2: round-off against Elk, P181).
    Returns ``(apply, nsym)``.
    """
    rotations, maps = elk_space_group(state.geometry)
    if moments is None:
        return symmetrizer(state, rotations, maps, lmax), len(rotations)
    m = np.asarray(moments, dtype=float)
    vectors = np.zeros((len(m), 3))
    if m.ndim == 1:
        vectors[:, 2] = m
    else:
        vectors[:] = m
    scale = max(np.abs(vectors).max(), 1e-12)
    kept_r, kept_m, spins = [], [], []
    for rot, amap in zip(rotations, maps):
        axial = np.linalg.det(rot) * rot
        candidates = [axial] if spinorb else [np.eye(3), -np.eye(3), axial]
        for sc in candidates:
            mapped = vectors @ sc.T
            if np.all(np.abs(mapped - vectors[amap]) < 1e-3 * scale):
                kept_r.append(rot)
                kept_m.append(amap)
                spins.append(sc)
                break
    return symmetrizer(state, kept_r, kept_m, lmax, spin_matrices=spins), len(kept_r)


# ---------------------------------------------------------------- potks

def kohn_sham_potential(state, params: ElkParameters, symmetry=None) -> dict:
    """``potks`` (and ``genbs``): every potential ``STATE.OUT`` carries, Hartree."""
    setup = _setup(state, params)
    vclmt, vclir = coulomb_potential(state, params, setup)
    vxcmt, vxcir, bxcmt, bxcir = exchange_correlation_potential(state, params, symmetry, setup)
    out = dict(vclmt=vclmt, vclir=vclir, vxcmt=vxcmt, vxcir=vxcir,
               vsmt=vclmt + vxcmt, vsir=vclir + vxcir)
    if bxcmt is not None:
        out.update(bxcmt=bxcmt, bxcir=bxcir, bsmt=_coarse_points(state, params, setup, bxcmt),
                   bsir=bxcir.copy())
    return out


def _coarse_points(state, params, setup, field):
    """``rfmtftoc`` then ``rbshtip``: every ``lradstp``-th point, on the cover points."""
    rbi, _ = _sht_matrices(params.lmaxi)
    rbo, _ = _sht_matrices(params.lmaxo)
    out = np.zeros(np.shape(field)[:1] + (state.nrcmtmax,) + np.shape(field)[2:])
    for ias in range(setup.natmtot):
        s = setup.species_of[ias]
        nr, nri = len(setup.r[s]), params.nrmti[s]
        fine = np.arange(0, nr, params.lradstp)
        for irc, ir in enumerate(fine):
            if ir < nri:
                out[:params.lmmaxi, irc, ias] = np.einsum(
                    "pa,a...->p...", rbi, field[:params.lmmaxi, ir, ias])
            else:
                out[:, irc, ias] = np.einsum("pa,a...->p...", rbo, field[:, ir, ias])
    return out
