"""The GIPAW phase: how a projector or a localized orbital is dressed by a uniform field.

Pickard and Mauri's gauge-including projector augmented waves (PRB 63, 245101 (2001)),
as Lu et al. use it for the nonlocal pseudopotential in a finite field (PRL 135, 196401
(2025), arXiv:2505.06585) and Wang et al. for their Gaussians (arXiv:2609.01053): a
function centred at ``tau`` is multiplied by

    exp(-i a(tau) . r - i f(r - tau)),   f(d) = a(d) . d / 2,   grad f = a_Lan - a_sym,

so that it sees the symmetric gauge about its own centre. The constant ``exp(-i a(tau) .
tau)`` is dropped (it cancels in ``|p~><p~|``), which leaves the phase ``-a(tau) . d -
a(d) . d / 2`` with ``d = r - tau`` the minimum image on a periodic box, the expression of
:meth:`defuqumat.magnetic.conventions.MagneticCell.gipaw_phase`. A magnetic translation
commutes with the nonlocal term only once every projector carries it, which is why both
routes put the bare nonlocal term nowhere.

Grid projectors (a sphere of box points per atom with the dressed ``beta``) are not here
yet; each route builds its own until they are folded back (the design's note).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from defuqumat.magnetic.conventions import MagneticCell, minimum_image

__all__ = ["gipaw_phase", "gipaw_factor", "minimum_image_displacements"]


def minimum_image_displacements(points, tau, box_lattice) -> np.ndarray:
    """``d = r - tau`` folded to the minimum image of the periodic box (host, bohr)."""
    return minimum_image(np.asarray(points, dtype=float) - np.asarray(tau, dtype=float), box_lattice)


def gipaw_phase(cell: MagneticCell, d, tau) -> jnp.ndarray:
    """``-a(tau) . d - a(d) . d / 2`` on ``d`` ``(..., 3)``, already the minimum image (bohr)."""
    return cell.gipaw_phase(d, tau)


def gipaw_factor(cell: MagneticCell, d, tau, dtype=None) -> jnp.ndarray:
    """``exp(i gipaw_phase)``, complex in the policy's dtype (the cell's precision by default)."""
    phase = gipaw_phase(cell, d, tau)
    dtype = cell.cell.precision.complex if dtype is None else dtype
    return jax.lax.complex(jnp.cos(phase), jnp.sin(phase)).astype(dtype)
