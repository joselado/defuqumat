"""Preset tips: the weight of each orbital from the real orbital of the apex atom.

``PLAN.md`` P179, ``plans/tip-orbitals.md`` section 12 and the review in
``plans/tip-orbitals-review/``.

**What the size of an orbital does.** In the flat vacuum a tip orbital of any
size, centred at the apex, couples as ``K_b D_b psi(r0)``: its size does not
change the map of that orbital (the mean-value property of the vacuum equation,
which is Chen's argument), and only sets the weight ``K_b``. So the size matters
where orbitals of different ``l`` are mixed, and for the absolute scale.

**Which weight is Bardeen's.** Bardeen's element is ``<chi|U_t|psi>``, with
``U_t`` the potential of the tip atom. For an atomic eigenstate ``(T_l + U_t) R =
e R``, so ``U_t R = (e - T_l) R`` and

    K_l(kappa) = int_0^r_c i_l(kappa r) (e - T_l) R(r) r^2 dr,

which is the tail amplitude on ``k_l(kappa r) Y_lm`` of a tip state at the
tunnelling energy whose inside is the atomic orbital, up to a factor that does
not depend on ``l`` (the addition theorem of ``exp(-kappa |r - r'|)/|r - r'|``).
It is computed on the pseudo-atomic orbital of the UPF file with the eigenvalue
the file stores (``pseudo_energy``), and **never on a Slater fit**: a Slater
function is not the eigenstate of any atomic potential, so ``U_t R`` is not
defined for it, and the plain overlap ``int R i_l r^2 dr`` carries a pole at
``kappa = sqrt(-e)`` that puts the 6s over 2p weight off by 3.5 to 8. Measured
on the pseudo-orbitals (``plans/tip-orbitals-review/kappa_scan.py``): O 2p at
``kappa = 0.57/bohr`` reads 1.2997, 1.3028, 1.3020 at radial cutoffs of 8, 12, 16
bohr, smooth through ``sqrt(-e) = 0.815``; Pt 6s reads 6.19, 6.50, 6.51.
:func:`bardeen_weight` itself, with its own second derivative on the whole mesh,
gives 1.3075, 1.3072, 1.3040 for O 2p at 8, 12, 16 bohr.

**Chen's weights are those of unnormalised angular functions**
(:mod:`defuqumat.transport.tip`), so the weight of ``R Y_b`` in Table I is
``K_l N_b / N_s`` with ``Y_b = N_b P_b``: ``sqrt(3)`` for each p, ``sqrt(15)``
for ``dxy``, ``dyz``, ``dxz``, ``sqrt(15)/2`` for ``dx2-y2`` and ``sqrt(5)/2``
for ``dz2``.

**The presets here are single families**, the ones where the literature fixes
which orbital carries the current without a measured share: ``"CO"`` the pi pair
of the oxygen (Gross et al., PRL 107, 086101 (2011); the probe-particle model of
Krejci et al., PRB 95, 045407 (2017), which models a CO tip as px and py), ``"Pt"``
and ``"Cu"`` the s orbital, ``"W"`` the ``dz2`` orbital (Chen's argument
for atomic resolution with tungsten), and ``"Pb"`` the 6p shell -- in the metal
the Fermi level lies in the 6p band, 6s being about 8.5 eV below (the shipped
``Pb.pbe-dn-rrkjus_psl.1.0.0.UPF``: 6s at -0.8849 Ry, 6p at -0.2571 Ry, 0.6279
Ry apart). A mixed tip (W with its s and
``dz2``, CO with its sigma part) needs the share of each orbital at ``E_F`` on
the apex, which is a PDOS of a tip model and is not here; :func:`from_upf` builds
one from shares you give.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = ["PRESETS", "preset", "from_upf", "bardeen_weight",
           "orbital_eigenvalue", "ANGULAR_NORM"]

#: ``N_b / N_s`` with ``Y_b = N_b P_b``: what turns the weight of a normalised
#: orbital into the weight of Chen's unnormalised one.
ANGULAR_NORM = {"s": 1.0, "px": np.sqrt(3.0), "py": np.sqrt(3.0),
                "pz": np.sqrt(3.0), "dxy": np.sqrt(15.0), "dyz": np.sqrt(15.0),
                "dxz": np.sqrt(15.0), "dx2-y2": np.sqrt(15.0) / 2.0,
                "dz2": np.sqrt(5.0) / 2.0}

#: Where the preset datasets live, the committed test pseudopotentials.
PSEUDO_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "pseudo"


def orbital_eigenvalue(path, label: str) -> float:
    """The eigenvalue in Ry of the pseudo-atomic orbital ``label`` of a UPF file.

    UPF v2 stores it as ``pseudo_energy`` on each ``PP_CHI``; a file without it
    (or with 0, as older files write) is refused, since the Bardeen weight needs
    it. A label that two orbitals share is refused too
    (:func:`_refuse_a_shared_label`).
    """
    text = Path(path).read_text(errors="replace")
    found = []
    for tag in re.finditer(r"<PP_CHI\.\d+([^>]*)>", text):
        attributes = dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', tag.group(1)))
        if attributes.get("label", "").strip().upper() == label.strip().upper():
            found.append(float(attributes.get("pseudo_energy", "0") or 0.0))
    _refuse_a_shared_label(path, label, len(found))
    if found and found[0] != 0.0:
        return found[0]
    raise ValueError(
        f"{Path(path).name} does not store the eigenvalue of its {label} orbital "
        "(pseudo_energy), which the Bardeen weight needs; use a pslibrary file")


@lru_cache(maxsize=32)
def _orbital(path: str, label: str):
    from defuqumat.pseudo.upf import read_upf

    data = read_upf(path)
    found = [orbital for orbital in data.orbitals
             if orbital.label.strip().upper() == label.strip().upper()]
    _refuse_a_shared_label(path, label, len(found))
    if found:
        r = np.asarray(data.r, dtype=float)
        chi = np.asarray(found[0].chi, dtype=float)
        return r, chi, int(found[0].l), orbital_eigenvalue(path, label)
    raise ValueError(f"{Path(path).name} has no orbital {label!r}")


def _refuse_a_shared_label(path, label: str, count: int) -> None:
    """Two orbitals under one label are the two ``j`` of a fully relativistic file.

    ``Bi.rel-pbe-dn-rrkjus_psl.1.0.0.UPF`` has two ``6P`` (``pseudo_energy``
    -0.4085 and -0.2726 Ry, ``j = 1/2`` and ``3/2``) and two ``5D``. Taking the
    first would give the tip the weight of one ``j`` component without saying
    so, and a tip whose ``p_1/2`` and ``p_3/2`` carry different weights is a
    spinor tip channel, which is not implemented (``PLAN.md`` P179, "What is
    outstanding").
    """
    if count > 1:
        raise NotImplementedError(
            f"{Path(path).name} has {count} orbitals labelled {label!r}, which is "
            "a fully relativistic dataset with one orbital per j: a tip whose "
            "j = l - 1/2 and j = l + 1/2 orbitals carry different weights is a "
            "spinor tip channel, which is not implemented. Use the scalar-"
            "relativistic dataset of the same element")


def bardeen_weight(path, label: str, kappa: float, cutoff: float = 16.0) -> float:
    """``K_l(kappa)`` of the pseudo-atomic orbital ``label``, see the module.

    ``(e - T_l) R r^2 = r (e chi + chi'' - l(l+1) chi / r^2)`` with ``chi = r R``
    as the file tabulates it, integrated against ``i_l(kappa r)`` up to
    ``cutoff`` bohr. The source ``U_t R`` holds 99.9 per cent of its weight
    inside 6 bohr for O 2p and 9.5 bohr for Pt 6s, and the result moves by 0.2
    per cent between 12 and 16 bohr for Pt 6s, so 16 is the default. The sign
    is the orbital's own; only the magnitude enters an incoherent channel.
    """
    from scipy.special import spherical_in

    r, chi, l, energy = _orbital(str(path), label)
    keep = (r > 1.0e-4) & (r <= float(cutoff))
    r, chi = r[keep], chi[keep]
    chi = chi * np.sign(chi[np.argmax(np.abs(chi))])
    second = np.gradient(np.gradient(chi, r), r)
    source = r * (energy * chi + second - l * (l + 1) * chi / r**2)
    integrand = spherical_in(l, float(kappa) * r) * source
    return float(-np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r)))


def from_upf(path, shares: dict, label: str | None = None):
    """A tip whose orbitals are the apex atom's, as incoherent channels.

    ``shares`` maps an orbital of Chen's table to ``(upf_label, n_b)``: the
    pseudo-atomic orbital of the file it is taken from (``"2P"``, ``"5D"``) and
    its share of the apex states at the tunnelling energy (a PDOS fraction). The
    weight of each channel is ``sqrt(n_b) |K_l(kappa)| N_b / N_s``, evaluated
    at the map's own ``kappa(E)``. The channels are incoherent because the
    relative phase of two orbitals of the apex is not in a PDOS.
    """
    from defuqumat.transport.tip import ORBITALS, TipOrbital

    path = str(path)
    channels, radial = [], []
    for name, (upf_label, share) in shares.items():
        if name not in ORBITALS:
            raise ValueError(f"unknown tip orbital {name!r}")
        _, _, l, _ = _orbital(path, upf_label)
        if l != ORBITALS[name]:
            raise ValueError(f"{upf_label} of {Path(path).name} has l = {l} and "
                             f"{name} needs l = {ORBITALS[name]}")
        if share < 0.0:
            raise ValueError(f"the share of {name} is negative: {share}")
        if share == 0.0:
            continue
        weight = float(np.sqrt(share) * ANGULAR_NORM[name])
        channels.append(((name, complex(weight)),))
        radial.append(((name, _radial_factor(path, upf_label)),))
    if not channels:
        raise ValueError("every share is zero, so the tip carries no current")
    return TipOrbital(channels=tuple(channels),
                      label=label or f"{Path(path).name}: {sorted(shares)}",
                      radial=tuple(radial))


def _radial_factor(path, upf_label):
    def factor(kappa, _path=path, _label=upf_label):
        return abs(bardeen_weight(_path, _label, float(kappa)))
    return factor


def _single_family(dataset: str, upf_label: str, names: tuple, label: str):
    def build():
        path = PSEUDO_DIR / dataset
        if not path.is_file():
            raise FileNotFoundError(
                f"the preset tip {label.split(':')[0]!r} needs {dataset} in "
                f"{PSEUDO_DIR}; build the tip with from_upf on another file")
        return from_upf(path, {name: (upf_label, 1.0) for name in names},
                        label=label)
    return build


#: name -> the function that builds the preset tip. Each is one family of
#: orbitals, so its size sets only the scale (see the module docstring).
PRESETS: dict = {
    "CO": _single_family("O.pbe-n-rrkjus_psl.1.0.0.UPF", "2P", ("px", "py"),
                         "CO: the O 2p pi pair, px and py"),
    "W": _single_family("W.pbe-spn-rrkjus_psl.1.0.0.UPF", "5D", ("dz2",),
                        "W: the 5d dz2 orbital"),
    "Pb": _single_family("Pb.pbe-dn-rrkjus_psl.1.0.0.UPF", "6P", ("px", "py", "pz"),
                         "Pb: the 6p shell"),
    "Pt": _single_family("Pt.pbe-n-kjpaw_psl.0.1.UPF", "6S", ("s",),
                         "Pt: the 6s orbital"),
    "Cu": _single_family("Cu.pbe-dn-rrkjus_psl.1.0.0.UPF", "4S", ("s",),
                         "Cu: the 4s orbital"),
}


def preset(name: str):
    """The preset tip ``name`` as a :class:`~defuqumat.transport.tip.TipOrbital`."""
    if name not in PRESETS:
        raise ValueError(f"no preset tip {name!r}; known: {sorted(PRESETS)}")
    return PRESETS[name]()
