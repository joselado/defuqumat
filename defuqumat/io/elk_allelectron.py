"""A PAW ground state as an Elk state: the all-electron density on Elk's meshes.

PAW holds the all-electron density without ever storing it: inside each
augmentation sphere

    n_AE = n~_grid + sum_ij rho_ij (phi_i phi_j - phi~_i phi~_j - Q^L_ij) + n_core,

where ``n~_grid`` is the density on the FFT grid (it already carries the
compensation charge ``Q^L_ij``, which the pseudo one-centre term subtracts back),
``phi`` and ``phi~`` are the dataset's all-electron and pseudo partial waves,
``rho_ij`` is ``becsum`` and ``n_core`` the dataset's all-electron frozen core.
Outside the spheres the pseudo density is the all-electron one. This module
evaluates that density in Elk's representation -- a real-harmonic expansion on
each muffin tin's logarithmic mesh and values on the interstitial grid -- and
builds Elk's Kohn-Sham potential from it (:mod:`defuqumat.io.elk_potential`),
so the ``STATE.OUT`` written is one Elk's ``task = 1`` starts from directly.
The user's suggestion (2026-11-17); measured in ``PLAN.md`` P181.

**The template.** ``STATE.OUT`` carries Elk's meshes but not everything about
them, so the unit is an Elk *run directory* of the same crystal: its
``STATE.OUT``, ``GEOMETRY.OUT`` and ``INFO.OUT`` (any state of that run will do,
its own initial one included). Only its meshes, parameters, Fermi level and
the shape of its core density at the nucleus are taken from it.

**Three things are not the PAW dataset's, each measured.**

* The muffin-tin part of the grid density is its exact expansion,
  ``4 pi sum_G c_G e^{iG tau} i^l j_l(G r) R_lm(G)``, truncated at Elk's ``lmaxo``
  as Elk truncates its own.
* **Below** ``r_match`` **the core follows Elk's Dirac shape.** A scalar
  relativistic dataset's core is flat at the nucleus where Elk's Dirac 1s
  density diverges weakly, ``r^(2(gamma - 1))``, 6 to 9 per cent over the
  innermost points for Si; those few points carried Elk's first residual
  (0.066 Ha against 0.0050 with the shape matched). Below ``r_match`` the
  template's ``l = 0`` density is used, scaled to ours at ``r_match``.
* The interstitial density is the grid density with Elk's FFT box, its
  coefficients outside Elk's box dropped.

**Refused, each by name:** a dataset that is not PAW (no all-electron partial
waves to rebuild from), a one-centre correction reaching past Elk's
muffin-tin radius (the all-electron density there would not be a smooth
interstitial function; enlarge ``rmt``), a template of another cell, other
atoms or another spin regime, and a noncollinear state (not yet).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

__all__ = ["all_electron_state", "write_elk_state_from"]

FOURPI = 4.0 * np.pi


def _match_atoms(calculation, geometry):
    """``(order, shift)``: defuqumat atom of each Elk atom, and the frame shift.

    Elk's positions are ``ours + shift`` modulo the lattice for one common
    ``shift`` (``tshift`` moves the origin); found from the first atom and
    checked on every other.
    """
    system = calculation.system
    at = np.asarray(system.cell.at, dtype=float)
    avec = np.asarray(geometry.avec, dtype=float)
    if not np.allclose(at, avec, atol=1e-6):
        raise ValueError(
            "the Elk template and this calculation do not have the same lattice "
            f"vectors (bohr):\n  Elk        {avec.tolist()}\n  defuqumat  {at.tolist()}\n"
            "Elk's real harmonics are tied to the cartesian axes, so the vectors "
            "themselves must agree, not only the lattice they span.")
    ours = np.asarray(system.structure.positions, dtype=float) @ np.linalg.inv(at)
    elk = np.asarray(geometry.positions, dtype=float)
    if len(ours) != len(elk):
        raise ValueError(f"the Elk template has {len(elk)} atoms and this calculation {len(ours)}")
    names = [system.structure.species[t].name for t in system.structure.types]
    elk_names = [geometry.species[i] for i, n in enumerate(geometry.natoms) for _ in range(n)]
    best = None
    for candidate in range(len(ours)):
        if _element(names[candidate]) != _element(elk_names[0]):
            continue
        shift = elk[0] - ours[candidate]
        order = []
        for k, position in enumerate(elk):
            d = ours + shift - position
            d -= np.round(d)
            hits = np.nonzero(np.linalg.norm(d @ at, axis=1) < 1e-5)[0]
            if len(hits) != 1 or _element(names[hits[0]]) != _element(elk_names[k]):
                break
            order.append(int(hits[0]))
        else:
            best = (np.array(order), shift @ at)
            break
    if best is None:
        raise ValueError(
            "the atoms of the Elk template and of this calculation do not coincide, "
            "even up to one common translation of the origin")
    return best


def _element(name: str) -> str:
    letters = "".join(ch for ch in str(name) if ch.isalpha())
    return letters[:2].capitalize() if len(letters) > 1 and letters[1].islower() else letters[:1].upper()


def _qe_to_elk(lmax: int) -> np.ndarray:
    """``(nlm, nlm)``: Elk real-harmonic coefficients from QE (``ylmr2``) ones."""
    from defuqumat.io.elk_density import real_spherical_harmonics
    from defuqumat.pseudo.harmonics import real_spherical_harmonics as ylm_qe

    xg, wg = np.polynomial.legendre.leggauss(24)
    ph = np.arange(48) * 2.0 * np.pi / 48
    th, pp = np.meshgrid(np.arccos(xg), ph, indexing="ij")
    dirs = np.stack([np.sin(th) * np.cos(pp), np.sin(th) * np.sin(pp), np.cos(th)], -1).reshape(-1, 3)
    w = (np.repeat(wg[:, None], 48, 1) * 2.0 * np.pi / 48).reshape(-1)
    return np.einsum("q,qa,qb->ab", w, real_spherical_harmonics(lmax, dirs),
                     np.asarray(ylm_qe(dirs, lmax)))


def _onto(r_from, f, r_to):
    """Cubic spline of ``f(..., r_from)`` onto ``r_to``; zero past its support."""
    from scipy.interpolate import CubicSpline

    f = np.asarray(f, dtype=float)
    out = np.zeros(f.shape[:-1] + (len(r_to),))
    nonzero = np.nonzero(np.any(np.abs(f.reshape(-1, f.shape[-1])) > 0, axis=0))[0]
    if len(nonzero) == 0:
        return out
    last = nonzero[-1] + 1
    rr = r_from[:last]
    inside = r_to <= rr[-1]
    spline = CubicSpline(rr, f[..., :last], axis=-1)
    out[..., inside] = spline(np.clip(r_to[inside], rr[0], None))
    return out


def _grid_coefficients(field, at):
    """``(c, G)``: the nonzero Fourier coefficients of a real field and their G (1/bohr)."""
    field = np.asarray(field, dtype=float)
    grid = np.array(field.shape[-3:])
    coef = np.fft.fftn(field, axes=(-3, -2, -1)) / np.prod(grid)
    idx = np.indices(grid).reshape(3, -1).T
    mill = np.where(idx > grid // 2, idx - grid, idx)
    flat = coef.reshape(coef.shape[:-3] + (-1,))
    keep = np.any(np.abs(flat.reshape(-1, flat.shape[-1])) > 1e-15, axis=0)
    b = 2.0 * np.pi * np.linalg.inv(at).T
    return flat[..., keep], mill[keep], mill[keep] @ b


def _muffin_tin_expansion(c, G, center, lmax, r):
    """``f_lm(r)`` of ``sum_G c_G e^{iGr}`` about ``center``, Elk harmonics, real."""
    from scipy.special import spherical_jn

    from defuqumat.io.elk_density import real_spherical_harmonics

    c = np.atleast_2d(c)
    gn = np.linalg.norm(G, axis=1)
    phase = np.exp(1j * G @ center)[None, :] * c
    rg = real_spherical_harmonics(lmax, np.where(gn[:, None] > 0, G, [0.0, 0.0, 1.0]))
    shells, inverse = np.unique(np.round(gn, 10), return_inverse=True)
    out = np.zeros((c.shape[0], (lmax + 1) ** 2, len(r)))
    for l in range(lmax + 1):
        jl = spherical_jn(l, np.outer(shells, r))
        for m in range(-l, l + 1):
            lm = l * (l + 1) + m
            amplitude = np.zeros((c.shape[0], len(shells)), dtype=complex)
            for k in range(c.shape[0]):
                np.add.at(amplitude[k], inverse, phase[k] * rg[:, lm])
            out[:, lm] = (FOURPI * (1j ** l) * amplitude @ jl).real
    return out


def _onecentre(pseudo, paw_species, becsum):
    """``(nspin, nlm, mesh)`` all-electron minus pseudo one-centre density, QE harmonics.

    Untruncated partial waves: the all-electron density is ``phi phi`` wherever
    the two sets differ, also past QE's ``iraug`` (which the one-centre *energy*
    cuts at); on Si2 the two forms agree to every digit measured.
    """
    paw = pseudo.paw
    r = np.asarray(pseudo.r, dtype=float)
    ae = np.asarray(paw.ae_wfc, dtype=float)
    ps = np.asarray(paw.ps_wfc, dtype=float)
    pf = np.einsum("nr,mr->nmr", ae, ae)
    if paw.ae_wfc_rel is not None:
        small = np.asarray(paw.ae_wfc_rel, dtype=float)
        pf = pf + np.einsum("nr,mr->nmr", small, small)
    pt = np.einsum("nr,mr->nmr", ps, ps)
    coefficients = np.asarray(paw_species.coefficients)
    channel = np.asarray(paw_species.channel_of)
    nlm = coefficients.shape[0]
    qfuncl = np.asarray(pseudo.augmentation.qfuncl, dtype=float)
    pairs = np.einsum("sij,lij,ib,jc->sbcl", becsum, coefficients, channel, channel)
    d = np.einsum("sbcl,bcr->slr", pairs, pf - pt)
    for lm in range(nlm):
        l = int(np.sqrt(lm))
        if l < qfuncl.shape[2]:
            d[:, lm] -= np.einsum("sbc,bcr->sr", pairs[:, :, :, lm], qfuncl[:, :, l, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r > 0.0, d / r ** 2, 0.0)


def all_electron_state(result, calculation, template, r_match: float = 1.0e-4):
    """``(state, params)``: ``template`` with our density and Elk's potential of it.

    ``template`` is an Elk run directory of the same crystal; ``state`` is an
    :class:`~defuqumat.io.elk.ElkState` ready for
    :func:`~defuqumat.io.elk_write.write_elk_state`.
    """
    from defuqumat.io.elk import ElkState
    from defuqumat.io.elk_potential import (crystal_symmetrizer, kohn_sham_potential,
                                            read_elk_parameters)

    template = Path(template)
    base = ElkState.read(template)
    params = read_elk_parameters(template)
    if not calculation.is_paw or any(getattr(p, "paw", None) is None for p in calculation.pseudos):
        raise NotImplementedError(
            "an Elk state is rebuilt from the all-electron partial waves of a PAW "
            "dataset, and this calculation has a species that is not PAW; a "
            "norm-conserving or ultrasoft dataset has no all-electron partial "
            "waves to rebuild the density inside the spheres from")
    system = calculation.system
    nspin_mag = int(calculation.nspin_mag)
    if nspin_mag == 4 or system.noncolin:
        raise NotImplementedError(
            "a noncollinear state is not written yet: its magnetization is a "
            "vector field whose symmetrisation (symrvfmt) rotates it as an axial "
            "vector, which is not transcribed")
    spinpol = nspin_mag == 2
    if spinpol != (base.magmt is not None) or (spinpol and base.ndmag != 1):
        raise ValueError(
            f"this calculation has nspin_mag = {nspin_mag} and the Elk template is "
            + ("spin-polarized" if base.magmt is not None else "unpolarized")
            + "; run Elk in the same spin regime for the template")
    geometry = base.geometry
    order, shift = _match_atoms(calculation, geometry)

    at = np.asarray(system.cell.at, dtype=float)
    rho = np.asarray(result.density, dtype=float)
    fields = rho if nspin_mag == 1 else np.stack([rho[0] + rho[1], rho[0] - rho[1]])
    c, mill, G = _grid_coefficients(fields, at)
    c = c * np.exp(-1j * G @ shift)[None, :]  # ours(r - shift): Elk's frame

    lmaxo, lmmaxo = params.lmaxo, params.lmmaxo
    natmtot = len(order)
    species_of = np.concatenate([[i] * n for i, n in enumerate(geometry.natoms)]).astype(int)
    rhomt = np.zeros_like(base.rhomt)
    magmt = np.zeros_like(base.magmt) if spinpol else None
    centres = np.asarray(geometry.positions, dtype=float) @ at
    transforms = {}
    becsum_of = {}
    for t, atoms in enumerate(calculation.augmentation.species_atoms):
        if result.becsum[t] is not None:
            for k, a in enumerate(atoms):
                becsum_of[a] = (t, np.asarray(result.becsum[t])[:, k])
    for ias in range(natmtot):
        s = species_of[ias]
        nr, nri = base.nrmt[s], params.nrmti[s]
        r = np.asarray(base.rsp[s], dtype=float)[:nr]
        f = _muffin_tin_expansion(c, G, centres[ias], lmaxo, r)
        t, becsum = becsum_of[int(order[ias])]
        pseudo = calculation.pseudos[t]
        d = _onecentre(pseudo, calculation.paw.species[t], becsum)
        r_paw = np.asarray(pseudo.r, dtype=float)
        support = np.nonzero(np.any(np.abs(d.reshape(-1, d.shape[-1])) > 1e-10, axis=0))[0]
        if len(support) and r_paw[support[-1]] > r[-1] + 1e-8:
            raise NotImplementedError(
                f"the PAW one-centre correction of {pseudo.element} reaches "
                f"{r_paw[support[-1]]:.3f} bohr and Elk's muffin tin ends at "
                f"{r[-1]:.3f}: the all-electron density there would have to be an "
                "interstitial function. Run Elk with a larger rmt for this species.")
        nlm = d.shape[1]
        lmax_rho = int(round(np.sqrt(nlm))) - 1
        if lmax_rho not in transforms:
            transforms[lmax_rho] = _qe_to_elk(lmax_rho)
        d = np.einsum("ab,sbr->sar", transforms[lmax_rho], d)
        if nspin_mag == 2:  # becsum channels are (up, down)
            d = np.stack([d[0] + d[1], d[0] - d[1]])
        n = min(nlm, lmmaxo)
        f[:, :n] += _onto(r_paw, d[:, :n], r)
        core = _onto(r_paw, np.asarray(pseudo.paw.ae_rho_core, dtype=float)[None], r)[0]
        f[0, 0] += core * np.sqrt(FOURPI)
        if r_match > 0.0:
            m = int(np.searchsorted(r, r_match))
            if 0 < m < nr and base.rhomt[0, m, ias] != 0.0:
                f[0, 0, :m] = base.rhomt[0, :m, ias] * f[0, 0, m] / base.rhomt[0, m, ias]
        f[:, (params.lmaxi + 1) ** 2:, :nri] = 0.0
        rhomt[:, :nr, ias] = f[0]
        if spinpol:
            magmt[:, :nr, ias, 0] = f[1]

    ng = np.array(base.ngridg)
    box = np.zeros((fields.shape[0],) + tuple(ng), dtype=complex)
    inside = np.all(np.abs(mill) <= (ng - 1) // 2, axis=1)
    box[(slice(None),) + tuple((mill[inside] % ng).T)] = c[:, inside]
    interstitial = np.real(np.fft.ifftn(box, axes=(1, 2, 3)) * np.prod(ng))
    state = dataclasses.replace(base, rhomt=rhomt, rhoir=interstitial[0])
    moments = None
    if spinpol:
        state = dataclasses.replace(state, magmt=magmt, magir=interstitial[1][..., None])
        moments = [float(np.sqrt(FOURPI) * np.trapezoid(
            magmt[0, :base.nrmt[s], ias, 0] * np.asarray(base.rsp[s])[:base.nrmt[s]] ** 2,
            np.asarray(base.rsp[s])[:base.nrmt[s]])) for ias, s in enumerate(species_of)]
        moments = np.round(np.asarray(moments), 3)
    symmetry, _ = crystal_symmetrizer(state, lmaxo, moments=moments)
    potential = kohn_sham_potential(state, params, symmetry)
    ry = {k: 2.0 * np.asarray(v) for k, v in potential.items()}
    state = dataclasses.replace(
        state, vclmt=ry["vclmt"], vclir=ry["vclir"], vxcmt=ry["vxcmt"],
        vxcir=ry["vxcir"], vsmt=ry["vsmt"], vsir=ry["vsir"])
    if spinpol:
        state = dataclasses.replace(state, bxcmt=ry["bxcmt"], bxcir=ry["bxcir"],
                                    bsmt=ry["bsmt"], bsir=ry["bsir"])
    return state, params


def write_elk_state_from(directory, result, calculation, template, r_match: float = 1.0e-4) -> Path:
    """Write ``directory/STATE.OUT`` from a PAW state, on ``template``'s meshes."""
    from defuqumat.io.elk_write import write_elk_state

    state, _ = all_electron_state(result, calculation, template, r_match=r_match)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return write_elk_state(directory / "STATE.OUT", state)
