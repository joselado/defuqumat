"""Elk's Kohn-Sham potential from a density: ``potks`` transcribed, on the host.

A ``STATE.OUT`` that Elk's ``task = 1`` can continue from carries the potential
as well as the density, because ``gndstate.f90`` diagonalises with the file's
``vsmt``/``vsir`` before it ever builds a potential of its own. This module
builds that potential from a density on Elk's meshes, the way Elk does, so a
density written by :func:`defuqumat.io.elk_write.write_elk_state` is a state
Elk starts from directly. Transcribed from Elk 11.0.2 (``vendor/elk/src`` of
the user's elkpy checkout), with ``elkjax`` (``src/elkjax/poisson.py``,
``xc.py``) as the cross-check of the reading, and every piece checked against
the potentials Elk wrote for its own density (``PLAN.md`` P181).

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
  (elkpy ``docs/jax_port_phase2.md`` section 2d measured 5.3e-3 Ha without it,
  all in channels the site symmetry forbids).
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

**What is refused, each by name:** a gradient-corrected or meta-GGA functional
(``xcgrad != 0``: the muffin-tin gradients ``ggamt_*`` are not transcribed), a
finite nucleus, a non-spin-polarized functional on a spin-polarized state
(Elk refuses it too), and external magnetic fields (``bfieldc``, ``bfcmt``).

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
           "crystal_symmetrizer", "symmetrizer"]

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
    expected_npsd = max(int(np.rint(0.25 * params.gmaxvr * rmt.max())), 1)
    if expected_npsd != params.npsd:
        raise ValueError(f"INFO.OUT's pseudocharge constant {params.npsd} is not "
                         f"init0's {expected_npsd} for this state's muffin tins")
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
        jl=jl, gclg=gclg, lnpsd=lnpsd, ngvc_mask=gmask,
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


def exchange_correlation_potential(state, params: ElkParameters, symmetry=None,
                                   setup: _Setup | None = None):
    """``potxc`` + ``trimrfg``: ``(vxcmt, vxcir, bxcmt, bxcir)`` in Hartree.

    ``symmetry`` is ``symrfmt``'s group as a callable on the muffin-tin
    functions (:func:`symmetrizer`), or ``None`` for Elk's ``symtype = 0``.
    """
    setup = _setup(state, params) if setup is None else setup
    if any(x != 0 for x in params.xctype[1:]) or params.xctype[0] >= 20 or params.xctype[0] < 0:
        raise NotImplementedError(
            f"xctype {params.xctype}: a gradient-corrected, meta-GGA or OEP functional "
            "needs the muffin-tin gradients (ggamt_*), which are not transcribed")
    lmmaxi, lmmaxo = params.lmmaxi, params.lmmaxo
    rbi, rfi = _sht_matrices(params.lmaxi)
    rbo, rfo = _sht_matrices(params.lmaxo)
    spinpol = state.magmt is not None
    ndmag = state.magmt.shape[-1] if spinpol else 0
    vxcmt = np.zeros(np.shape(state.rhomt))
    bxcmt = np.zeros(np.shape(state.magmt)) if spinpol else None
    for ias in range(setup.natmtot):
        s = setup.species_of[ias]
        nr, nri = len(setup.r[s]), params.nrmti[s]
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
    rhoir = np.asarray(state.rhoir, dtype=float).reshape(-1)
    magir = None
    if spinpol:
        magir = np.asarray(state.magir, dtype=float).reshape(-1, ndmag)
        magir = magir[:, 0] if ndmag == 1 else magir
    v, b = _xc_points(params, rhoir, magir)
    vxcir = _trim(v.reshape(ng), setup)
    bxcir = None
    if spinpol:
        bxcir = (b.reshape(ng + (ndmag,)) if ndmag == 3 else b.reshape(ng + (1,)))
    return vxcmt, vxcir, bxcmt, bxcir


def _trim(field, setup):
    """``trimrfg``: keep the Fourier components with ``|G| <= 2 gkmax``."""
    z = np.fft.fftn(field)
    z[~setup.ngvc_mask] = 0.0
    return np.real(np.fft.ifftn(z))


# ---------------------------------------------------------------- symmetry

def symmetrizer(state, rotations_cart, atom_maps, lmax: int):
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
        out = np.zeros_like(field)
        n = len(rotations_cart)
        for k in range(n):
            d = blocks[k]
            for a in range(natmtot):
                b = sources[k][a]
                out[:, :, a] += np.einsum("ab,b...->a...", d, field[:, :, b]) / n
        return out

    return apply


def crystal_symmetrizer(state, lmax: int, moments=None):
    """``symrfmt`` for the crystal group of ``state``'s geometry.

    The group is defuqumat's own search (:func:`defuqumat.system.symmetry.find_symmetries`,
    QE's ``sgam_at``) on Elk's cell and atoms, which is the space group Elk finds
    for the same structure; for a spin-polarized state the operations are kept
    only where they carry each atom's moment onto an equal one
    (:func:`defuqumat.system.symmetry.collinear_symmetries` when ``moments`` is
    given, the per-atom ``m_z``).
    """
    from defuqumat.system.cell import Cell
    from defuqumat.system.structure import Species, Structure
    from defuqumat.system.symmetry import atom_mapping, collinear_symmetries, find_symmetries

    geometry = state.geometry
    avec = np.asarray(geometry.avec, dtype=float)
    cell = Cell.from_vectors(avec)
    types = tuple(int(i) for i, n in enumerate(geometry.natoms) for _ in range(n))
    species = tuple(Species(name=str(name), mass=1.0, pseudo_file="")
                    for name in geometry.species)
    structure = Structure(positions=np.asarray(geometry.positions, dtype=float) @ avec,
                          types=types, species=species)
    group = find_symmetries(cell, structure)
    if moments is not None:
        m = np.zeros((len(types), 3))
        m[:, 2] = np.asarray(moments, dtype=float)
        group = collinear_symmetries(cell, structure, group, m)
    irt = atom_mapping(cell, structure, group)
    inverse = np.linalg.inv(avec)
    rotations = [(inverse @ np.asarray(m_, dtype=float) @ avec).T for m_ in group.rotation_array()]
    return symmetrizer(state, rotations, irt, lmax), group.nsym


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
