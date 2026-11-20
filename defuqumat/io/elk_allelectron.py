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
builds Elk's Kohn-Sham potential from it (:mod:`defuqumat.io.elk_potential`,
without external fields: Elk adds its own ``bfieldc``/``bfcmt`` in ``genbs``
at the end of its first loop, so that loop runs without them),
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
* **Near each nucleus the density is Elk's** (:data:`CORE_REGION`). A scalar
  relativistic dataset's core is flat at the nucleus where Elk's Dirac 1s
  density diverges weakly, ``r^(2(gamma - 1))``, 6 to 9 per cent over the
  innermost points for Si; those few points carried Elk's first residual
  (0.066 Ha under LDA against 0.0052 with Elk's shape up to 1e-4 bohr). A GGA
  needs more: its potential near the nucleus depends on the cusp itself, so
  the template's density is taken as it is below 0.1/e = 0.037 bohr and joined
  to ours by 0.1 bohr.
* The interstitial density is the grid density with Elk's FFT box, its
  coefficients outside Elk's box dropped.

**Refused, each by name:** a dataset that is not PAW (no all-electron partial
waves to rebuild from), a one-centre correction reaching past Elk's
muffin-tin radius (the all-electron density there would not be a smooth
interstitial function; enlarge ``rmt``), a template of another cell, other
atoms or another spin regime, a DFT+U template (its ``vmatmt`` would be
written beside our density and used in Elk's first diagonalisation), and a
spin-orbit run (no such cell has been run through Elk). A noncollinear state is written with its
magnetization vector (``ndmag = 3``); the magnetization a fully relativistic
dataset's small component carries on its own (QE's ``pfunc_rel`` term) is
left out, and the small component's charge is in every component, as QE's
``pfunc`` has it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from defuqumat.system.elements import element_symbol

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
    # Our element is the dataset's, not the label's: a label is free text
    # ('O_up', 'Cup'), and read as a symbol it can name another element.
    names = [element_symbol(str(calculation.pseudos[t].element))
             for t in system.structure.types]
    elk_names = [_element(geometry.species[i])
                 for i, n in enumerate(geometry.natoms) for _ in range(n)]
    best = None
    for candidate in range(len(ours)):
        if names[candidate] != elk_names[0]:
            continue
        shift = elk[0] - ours[candidate]
        order = []
        for k, position in enumerate(elk):
            d = ours + shift - position
            d -= np.round(d)
            hits = np.nonzero(np.linalg.norm(d @ at, axis=1) < 1e-5)[0]
            if len(hits) != 1 or names[hits[0]] != elk_names[k]:
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
    """The element symbol of an Elk species file name, ``'O.in'`` or ``'Ni_afm.in'``.

    The stem is read by ``pw.x``'s label rule (:func:`element_symbol`): a tag
    after a digit, ``-`` or ``_`` is dropped, so ``'O_up'`` is oxygen.
    """
    return element_symbol(str(name).strip().strip("'").split(".")[0])


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


#: How the ``l = 0`` density near each nucleus is taken, by functional
#: (measured on Si2, P181): ``(r_join, scaled)``. Below ``r_join / e`` the
#: template's density, joined to ours by ``r_join`` with a ``C^2`` step in
#: the natural ``log r``, over a factor ``e`` and not a decade (audit
#: 2026-11-20 #6; the measured rows below are of this join); ``scaled``
#: rescales the template to ours at ``r_join`` (Elk's Dirac *shape* on our
#: density) instead of taking it as it is.
CORE_REGION = {
    # LDA: Elk's shape up to 1e-4 bohr (outright below 3.7e-5), first residual
    # 0.0052 Ha against 0.083 from scratch, 16 loops against 19; the template
    # outright below 0.037 bohr, (0.1, False), gives 0.0172 and 17.
    "lda": (1.0e-4, True),
    # A GGA carries the Laplacian of the density, and its potential near the
    # nucleus (-704 Ha at 1e-4 bohr on Si) depends on the Dirac cusp itself:
    # our shape matched below 1e-4 gave 17 Ha against 0.74 from scratch, Elk's
    # own core region (outright below 0.037 bohr, joined by 0.1) gives 0.73 and
    # 15 loops against 14.
    "gga": (0.1, False),
    # A spin-polarized GGA, measured on O2 stretched to 3.4 bohr under PBE: the
    # unpolarized choice above starts 31 times closer (0.12 Ha against 4.41 from
    # scratch) and then needs 49 loops against Elk's 32, where Elk's shape up to
    # 1e-4 bohr starts at 177 Ha and needs 31 (the PAW core everywhere: 203 Ha,
    # 35). The first residual does not predict the count here. One cell each.
    "gga-spin": (1.0e-4, True),
}


def _dirac_core(r, ours, template, r_join, scaled, decades: float = 1.0):
    """Elk's ``l = 0`` density at the nucleus, joined smoothly to ours (:data:`CORE_REGION`).

    A join at one point is continuous but not smooth, and a gradient-corrected
    potential turns the kink into a spike, so the join is a ``C^2`` step
    (``6t^5 - 15t^4 + 10t^3``) in the natural log of ``r``, from ``r_join /
    e**decades`` to ``r_join``. With the default ``decades = 1`` that is a
    factor ``e``, not ten: below ``r_join / e`` the result is the template
    alone (0.0357 bohr is the last such point of Elk's Si mesh at ``r_join =
    0.1``), and from ``r_join`` on it is ours. The P181 rows were measured with
    this join, so the name ``decades`` is kept and the log is not changed.
    """
    m = int(np.searchsorted(r, r_join))
    if not 0 < m < len(r) or template[m] == 0.0:
        return ours
    scaled = template * (ours[m] / template[m] if scaled else 1.0)
    t = np.clip((np.log(r) - np.log(r_join) + decades) / decades, 0.0, 1.0)
    w = 6.0 * t ** 5 - 15.0 * t ** 4 + 10.0 * t ** 3
    return (1.0 - w) * scaled + w * ours


def all_electron_state(result, calculation, template, core_region="auto", onecentre: bool = True):
    """``(state, params)``: ``template`` with our density and Elk's potential of it.

    ``template`` is an Elk run directory of the same crystal; ``state`` is an
    :class:`~defuqumat.io.elk.ElkState` ready for
    :func:`~defuqumat.io.elk_write.write_elk_state`. ``core_region`` is
    ``"auto"`` (by the template's functional, :data:`CORE_REGION`), ``None``
    (the PAW core everywhere) or ``(r_join, scaled)``. ``onecentre = False``
    leaves the one-centre term out, the control of P181 (the grid density and
    the core alone): it is three times further from Elk in Elk's first loop.
    """
    from defuqumat.io.elk import ElkState
    from defuqumat.io.elk_potential import (crystal_symmetrizer, kohn_sham_potential,
                                            read_elk_parameters, refuse_untranscribed)

    template = Path(template)
    base = ElkState.read(template, allow_noncollinear=True)
    params = read_elk_parameters(template)
    refuse_untranscribed(params, base.magmt is not None)
    if base.dftu != 0:
        # readstate.f90:251 reads vmatmt when the run and the file both have
        # dftu, and gndstate.f90 diagonalises with it (genevfsv) before
        # genvmatmt rebuilds it: a matrix of the template's density beside ours.
        raise NotImplementedError(
            f"the Elk template is a DFT+U state (dftu = {base.dftu}): its potential "
            "matrix vmatmt would be written beside our density, and Elk reads it "
            "and uses it in its first diagonalisation. That matrix belongs to the "
            "template's density, and the DFT+U potential matrix of ours is not "
            "computed here. Take the template from the same Elk input without "
            "dft+u; Elk run with dft+u then starts its matrix at zero, since the "
            "file has none (readstate.f90:251).")
    if core_region == "auto":
        gga = params.xctype[0] >= 20
        polarized = int(calculation.nspin_mag) > 1
        core_region = CORE_REGION["lda" if not gga else ("gga-spin" if polarized else "gga")]
    if not calculation.is_paw or any(getattr(p, "paw", None) is None for p in calculation.pseudos):
        raise NotImplementedError(
            "an Elk state is rebuilt from the all-electron partial waves of a PAW "
            "dataset, and this calculation has a species that is not PAW; a "
            "norm-conserving or ultrasoft dataset has no all-electron partial "
            "waves to rebuild the density inside the spheres from")
    system = calculation.system
    nspin_mag = int(calculation.nspin_mag)
    if system.lspinorb:
        raise NotImplementedError(
            "a spin-orbit run is not written for Elk: no such cell has been run "
            "through Elk. A nonmagnetic one is a spin-polarized ndmag = 3 state in "
            "Elk, which spin-orbit coupling requires there, against this run's one "
            "density component, and a magnetic one takes Elk's group with the spin "
            "rotation det(R) R only (crystal_symmetrizer(spinorb=True)); both are "
            "outstanding in PLAN.md P181.")
    spinpol = nspin_mag in (2, 4)
    ndmag = {1: 0, 2: 1, 4: 3}[nspin_mag]
    if spinpol != (base.magmt is not None) or (spinpol and base.ndmag != ndmag):
        held = ("unpolarized" if base.magmt is None
                else f"spin-polarized with ndmag = {base.ndmag}")
        raise ValueError(
            f"this calculation has nspin_mag = {nspin_mag} and the Elk template is "
            f"{held}; run Elk in the same spin regime for the template (ndmag = 1 "
            "for nspin = 2, a noncollinear ndmag = 3 for a magnetic spinor run)")
    geometry = base.geometry
    order, shift = _match_atoms(calculation, geometry)

    at = np.asarray(system.cell.at, dtype=float)
    rho = np.asarray(result.density, dtype=float)
    # (n,), (n, m_z) or (n, m_x, m_y, m_z): defuqumat's collinear pair is (up, down).
    fields = np.stack([rho[0] + rho[1], rho[0] - rho[1]]) if nspin_mag == 2 else rho
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
        if onecentre:
            f[:, :n] += _onto(r_paw, d[:, :n], r)
        core = _onto(r_paw, np.asarray(pseudo.paw.ae_rho_core, dtype=float)[None], r)[0]
        f[0, 0] += core * np.sqrt(FOURPI)
        if core_region is not None:
            joined = _dirac_core(r, f[0, 0], np.asarray(base.rhomt[0, :nr, ias]), *core_region)
            if spinpol:
                # The magnetization follows the charge's new shape near the
                # nucleus, so the polarization there stays ours: a spin GGA
                # takes the gradients of each spin density, and Elk's core
                # under our magnetization gave O2 under PBE a residual that
                # hovered at 0.1 Ha for 40 loops (P181).
                ratio = np.divide(joined, f[0, 0], out=np.ones_like(joined),
                                  where=np.abs(f[0, 0]) > 1e-300)
                f[1:, :, :] *= ratio[None, None, :]
            f[0, 0] = joined
        f[:, (params.lmaxi + 1) ** 2:, :nri] = 0.0
        rhomt[:, :nr, ias] = f[0]
        if spinpol:
            for k in range(ndmag):
                magmt[:, :nr, ias, k] = f[1 + k]

    ng = np.array(base.ngridg)
    box = np.zeros((fields.shape[0],) + tuple(ng), dtype=complex)
    inside = np.all(np.abs(mill) <= (ng - 1) // 2, axis=1)
    box[(slice(None),) + tuple((mill[inside] % ng).T)] = c[:, inside]
    interstitial = np.real(np.fft.ifftn(box, axes=(1, 2, 3)) * np.prod(ng))
    state = dataclasses.replace(base, rhomt=rhomt, rhoir=interstitial[0])
    moments = None
    if spinpol:
        state = dataclasses.replace(state, magmt=magmt,
                                    magir=np.moveaxis(interstitial[1:], 0, -1))
        moments = np.array([[np.sqrt(FOURPI) * np.trapezoid(
            magmt[0, :base.nrmt[s], ias, k] * np.asarray(base.rsp[s])[:base.nrmt[s]] ** 2,
            np.asarray(base.rsp[s])[:base.nrmt[s]]) for k in range(ndmag)]
            for ias, s in enumerate(species_of)])
        moments = np.round(moments[:, 0] if ndmag == 1 else moments, 3)
    if spinpol and params.xctype[0] >= 20:
        import warnings

        warnings.warn(
            "a spin-polarized state under a GGA: on O2 under PBE Elk took 31 loops "
            "from this state against 32 from scratch, and 49 with Elk's core region "
            "taken outright, although that one started 31 times closer; what sets the "
            "count is open (PLAN.md P181). The state is a valid start, not a faster "
            "one.", stacklevel=3)
    symmetry, _ = crystal_symmetrizer(state, lmaxo, moments=moments, params=params)
    potential = kohn_sham_potential(state, params, symmetry)
    ry = {k: 2.0 * np.asarray(v) for k, v in potential.items()}
    state = dataclasses.replace(
        state, vclmt=ry["vclmt"], vclir=ry["vclir"], vxcmt=ry["vxcmt"],
        vxcir=ry["vxcir"], vsmt=ry["vsmt"], vsir=ry["vsir"])
    if spinpol:
        state = dataclasses.replace(state, bxcmt=ry["bxcmt"], bxcir=ry["bxcir"],
                                    bsmt=ry["bsmt"], bsir=ry["bsir"])
    return state, params


def write_elk_state_from(directory, result, calculation, template, core_region="auto",
                         onecentre: bool = True) -> Path:
    """Write ``directory/STATE.OUT`` from a PAW state, on ``template``'s meshes."""
    from defuqumat.io.elk_write import write_elk_state

    state, _ = all_electron_state(result, calculation, template, core_region=core_region,
                                  onecentre=onecentre)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return write_elk_state(directory / "STATE.OUT", state)
