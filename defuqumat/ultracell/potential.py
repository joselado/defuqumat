"""The ultracell Kohn-Sham potential, and the subtraction that stops it doubling.

``PLAN.md`` P88. Three things happen here and only the last one is subtle.

**Hartree.** ``4 pi rho(G+Q) / |G+Q|^2`` over the ultracell box, which is
:meth:`~defuqumat.ultracell.grid.Ultracell.g2`. The ``G = 0, Q != 0`` elements are
**finite and kept**: they are the terms that screen a long-wavelength external
potential, which is the physics the whole method exists for, so dropping them
the way a zone-centre calculation drops ``G = 0`` would remove the answer. Only
``G = Q = 0`` goes, against the compensating background, exactly as
``hartree`` does in the unit cell and as ``gengclqu`` does in Elk
(``gclq(1) = 0``, and ``gengclgq`` then puts ``gclq(iq)`` into the ``G = 0``
element of every other ``Q``).

``q0cut`` (Elk's manual 5.100) is not implemented, and if it ever is, implement
the **code** and not the manual: the manual says the Green's function is zeroed
for every ``|G+Q| < q0cut``, while ``gengclqu.f90`` applies the cut-off only to
``gclq``, the ``G = 0`` element of each ``Q``. The two differ.

**Exchange-correlation, and why it is the supercell's own.** The box *is* a
supercell's FFT box: its reciprocal lattice is the supercell's, so the set
``{G + Q}`` it carries is a set of the supercell's own reciprocal vectors, and
``v_of_rho`` run on that set with the supercell's lattice is the supercell's
``v_of_rho`` with one difference, which is the sphere: the tiled unit-cell
dense sphere (``keep``, Elk's choice and the one that makes ``N = 1`` the unit
cell exactly) rather than a sphere in the supercell's own reciprocal space.
So :func:`ultracell_sphere` builds that set once, as a
:class:`~defuqumat.basis.gvectors.GVectors` on the box, and the potential is
then ``v_of_rho`` and nothing written for the ultracell. What that buys is
every term ``v_of_rho`` has, each taken over the *whole* modulation:

* the **gradient correction**, whose ``grad rho`` then carries the envelope's
  own gradient. Elk's ``potxcu`` calls ``potxc`` once per cell ``R`` on that
  cell's density, so its gradient misses the envelope's at order ``Q``; this
  one does not, because it is the gradient of the ultracell density on the
  ultracell's own reciprocal vectors.
* ``nosource``'s projection, ``(G+Q)((G+Q).B)/|G+Q|^2`` over the whole box,
  which couples every cell to every other one exactly as the supercell's does.
  The per-cell potential the refusal used to name was Elk's, not this code's.
* the Tran-Blaha coefficient ``c``, an average over the ultracell, which is
  the supercell's average.

At a tiled density every non-zero component sits at ``Q = 0``, where ``keep``
is exactly the unit cell's dense sphere, so the gradient and its divergence
are the unit cell's to round-off and the tiled null survives.

**What it costs**: the box ``GVectors`` holds ``N ngm`` integer triples, a
supercell SCF's ``ngm``, and the cartesian ``G + Q`` it computes on demand is
three reals per entry; a GGA adds four box transforms per potential build,
three for ``grad rho`` and one for the divergence, as it does in the unit cell.

**The subtraction, which is the first thing to get wrong.** The ultracell
Hamiltonian's diagonal is the *unit cell's* eigenvalue ``eps_{k+Q,n}``, and
that eigenvalue already contains the whole unit-cell potential -- the local
pseudopotential, the Hartree term and the exchange-correlation term at the
unmodulated density. So the potential that goes into the matrix is only the
*difference*,

    dV(r) = v_scf[rho_ULR](r) - tile(v_scf[rho_0](r)) + V_ext(r),

which is Elk's ``vblocalu``. ``vltot`` cancels identically between the two,
since the atoms do not move, so it never appears. Put the whole potential in
instead of the difference and every eigenvalue is doubled in its potential
part, which converges perfectly well to the wrong answer.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import jax

from defuqumat.basis.gvectors import GVectors
from defuqumat.scf.potential import v_of_rho
from defuqumat.system.cell import Cell
from defuqumat.ultracell.grid import Ultracell

__all__ = [
    "UltracellPotential",
    "UltracellSphere",
    "ultracell_potential",
    "ultracell_sphere",
    "delta_potential",
    "with_external_potential",
    "require_an_ultracell_functional",
]


#: ``v_of_rho`` compiled once per box shape, with ``source_free`` and
#: ``noncollinear_gga`` static as :mod:`defuqumat.scf.driver` has them.
_box_potential = jax.jit(v_of_rho, static_argnums=(8, 9))


class UltracellPotential:
    """Hartree plus exchange-correlation on the ultracell box, and its pieces.

    ``ehart`` and ``etxc`` are **ultracell** totals, ``N`` times a unit cell's
    at a tiled density; ``meta_c`` is the Tran-Blaha coefficient, zero for any
    other functional; ``e_sawtooth`` is the saw-tooth field's ``E_f`` over the
    ultracell, zero without ``tefield``, set by the driver, which holds the atom
    copies the ionic dipole needs.
    """

    __slots__ = ("v_scf", "ehart", "etxc", "meta_c", "e_sawtooth")

    def __init__(self, v_scf, ehart, etxc, meta_c=0.0, e_sawtooth=0.0):
        self.v_scf = v_scf
        self.ehart = ehart
        self.etxc = etxc
        self.meta_c = meta_c
        self.e_sawtooth = e_sawtooth


class UltracellSphere:
    """The box's reciprocal vectors as a ``GVectors``, and the supercell they belong to.

    Built once per run by :func:`ultracell_sphere`. ``gvectors`` is the tiled
    dense sphere ``keep`` written as ultracell Miller indices on the box, so
    its ``fft_index`` is a gather from and a scatter to the box; ``cell`` is
    the ultracell lattice, ``a^u_i = n_i a_i``, whose reciprocal vectors are
    ``b_i / n_i``, which is what makes ``gvectors.cartesian(cell)`` the
    cartesian ``G + Q``. ``rho_core_g`` is the tiled core charge on that set,
    taken from the unit cell's own ``rho_core_g`` rather than from a transform
    of its real-space copy, so that a GGA's core gradient at ``Q = 0`` is the
    unit cell's exactly.
    """

    __slots__ = ("gvectors", "cell", "rho_core_g")

    def __init__(self, gvectors, cell, rho_core_g=None):
        self.gvectors = gvectors
        self.cell = cell
        self.rho_core_g = rho_core_g


def ultracell_sphere(ultracell: Ultracell, cell: Cell, keep, dense=None,
                     rho_core_g=None) -> UltracellSphere:
    """The box ``GVectors`` and the ultracell lattice, for ``v_of_rho`` on the box.

    Args:
        keep: the boolean box mask, the tiled unit-cell dense sphere.
        dense: the unit cell's dense ``GVectors``, needed only with
            ``rho_core_g``, whose entries it labels.
        rho_core_g: the unit cell's core charge on its dense sphere, or ``None``.

    ``G = 0`` is put first, as :func:`~defuqumat.basis.gvectors.generate_gvectors`
    does and as consumers of a ``GVectors`` may assume; nothing else about the
    order matters, since every use is a gather and a scatter through
    ``fft_index``.
    """
    flat = np.flatnonzero(np.asarray(keep).reshape(-1))
    miller = ultracell.miller_at(flat)
    order = np.argsort(np.sum(miller.astype(np.float64) ** 2, axis=-1), kind="stable")
    miller = miller[order]
    gvectors = GVectors(
        miller=jnp.asarray(miller), grid=tuple(int(n) for n in ultracell.grid),
        ecut=0.0,
    )
    lattice = Cell(
        at=jnp.asarray(cell.at) * jnp.asarray(ultracell.shape, dtype=cell.at.dtype)[:, None],
        alat=float(cell.alat), precision=cell.precision,
    )
    core_g = None
    if rho_core_g is not None:
        # The core charge lives at ``Q = 0`` only, at box index ``n G``; every
        # other entry of the set is zero.
        box = np.zeros(ultracell.points, dtype=np.asarray(rho_core_g).dtype)
        at_q0 = ultracell.box_index(np.asarray(dense.miller), 0)
        box[at_q0] = np.asarray(rho_core_g)
        core_g = jnp.asarray(box[np.asarray(gvectors.fft_index)])
    return UltracellSphere(gvectors, lattice, core_g)


def require_an_ultracell_functional(functional, source_free: bool = False,
                                    nspin_mag: int = 1,
                                    augmented: bool = False) -> None:
    """Refuse what has not been measured on the box, each by name.

    A gradient-corrected functional, ``nosource`` and a potential-only
    meta-GGA all run through ``v_of_rho`` on the box (module docstring), the
    last with ``tau`` built from the ultracell states and mixed beside the
    density. What is still refused is a GGA or a meta-GGA on a noncollinear
    magnetization, whose local-frame rotation has not been checked against a
    supercell, and a meta-GGA on an ultrasoft or PAW dataset, whose one-centre
    ``tau`` the ultracell's per-copy terms do not carry.
    """
    if functional.is_meta and int(nspin_mag) != 1:
        raise NotImplementedError(
            f"the ultracell refuses the meta-GGA {functional.name} on a "
            f"magnetic cell (nspin_mag = {int(nspin_mag)}): tau is built on the "
            "box per collinear channel and measured against a supercell on an "
            "unpolarized cell only, and a spinor tau is a 2x2 matrix the "
            "ultracell does not build"
        )
    if functional.is_meta and augmented:
        raise NotImplementedError(
            f"the ultracell refuses the meta-GGA {functional.name} with an "
            "ultrasoft or PAW dataset: tau is built from the smooth states on "
            "the box, and the one-centre terms evaluated per atom copy are "
            "handed no meta_c, so the two halves would be different "
            "functionals. Use a norm-conserving dataset"
        )
    if functional.is_gradient and int(nspin_mag) == 4:
        raise NotImplementedError(
            f"the ultracell refuses the gradient-corrected functional "
            f"{functional.name} on a noncollinear magnetization: the gradient "
            "of the box density is taken over the whole modulation and is "
            "measured against a supercell for nspin = 1 and 2, while the "
            "local spin frame the noncollinear GGA is evaluated in has not "
            "been. Use an LDA, or a collinear run"
        )


def ultracell_potential(
    rho_r: jnp.ndarray,
    sphere: UltracellSphere,
    rho_core_tiled: jnp.ndarray | None,
    functional,
    quantization_axis=None,
    source_free: bool = False,
    noncollinear_gga: str = "kubler",
    tau: jnp.ndarray | None = None,
) -> UltracellPotential:
    """``v_of_rho`` on the ultracell box: the supercell's, on the tiled sphere.

    Args:
        rho_r: ``(nspin, *box)`` real density, normalised per unit cell -- so
            its integral over the ultracell is ``N`` times the electron count.
        sphere: :func:`ultracell_sphere`, built once for the run.
        rho_core_tiled: the unit cell's core charge tiled onto the box, or
            ``None``.
        quantization_axis, source_free, noncollinear_gga: the unit-cell
            calculation's own, passed through so that the box and the frozen
            eigenvalues are built from the same functional.
        tau: ``(nspin, *box)`` kinetic energy density in Ry, for a meta-GGA
            and only for one, from
            :func:`~defuqumat.ultracell.density.ultracell_kinetic_density`.

    The Hartree term is ``4 pi rho(G+Q)/|G+Q|^2`` with only ``G = Q = 0``
    dropped, and the energies are ultracell totals: ``v_of_rho`` integrates over
    the cell it is handed, which is the ultracell. Masking happens in the
    gather, on both the Hartree side and the gradient side, and the
    exchange-correlation term is evaluated on the unmasked grid density, which
    is what the unit cell does and therefore what the ``Q = 0`` subtraction
    needs.
    """
    potential = _box_potential(
        jnp.asarray(rho_r), sphere.gvectors, sphere.cell, rho_core_tiled,
        functional, sphere.rho_core_g, quantization_axis,
        None if tau is None else jnp.asarray(tau),
        bool(source_free), str(noncollinear_gga), None,
    )
    return UltracellPotential(
        v_scf=potential.v_scf,
        ehart=potential.ehart,
        etxc=potential.etxc,
        meta_c=potential.meta_c,
    )


def delta_potential(potential_box, potential_cell, ultracell: Ultracell, external=None):
    """``dV`` for the Hamiltonian: the ultracell potential minus the unit cell's.

    ``potential_cell`` is ``(nspin, *cell grid)``, the converged unit-cell
    ``v_scf`` the frozen eigenvalues already carry; ``external`` is an optional
    ``(*box,)`` or ``(nspin, *box)`` real field added on top, which is what an
    applied modulation is.
    """
    delta = potential_box - ultracell.tile(potential_cell)
    if external is None:
        return delta
    external = jnp.asarray(external)
    if external.ndim == 3:
        external = jnp.broadcast_to(external, delta.shape)
    return delta + external


def with_external_potential(calculation, v_ext_r):
    """A copy of ``calculation`` whose local potential carries ``v_ext_r``.

    The hook the ultracell's *reference* calculation needs and nothing else
    does: validating an ultracell run means doing the same physics in a real
    ``N``-cell supercell, and the modulation has to reach the supercell's SCF
    somehow. An external scalar potential belongs in ``vltot`` -- it is felt by
    both spin channels in full and it is not a function of the density, which
    is exactly what ``vltot`` already is.

    ``v_ext_r`` is on that calculation's own dense grid, in Ry.

    **Shallow, and deliberately so.** The copy shares every array with the
    original; only ``vltot`` is replaced. Anything that rebuilds ``vltot`` --
    ``at_positions``, ``at_cell``, ``at_strain`` -- drops the external
    potential, which is correct for a moving geometry and is why this is a
    helper here rather than a field on :class:`~defuqumat.scf.driver.Calculation`.
    """
    import copy

    v_ext_r = jnp.asarray(v_ext_r)
    if v_ext_r.shape != tuple(calculation.basis.dense.grid):
        raise ValueError(
            f"the external potential is on {v_ext_r.shape} and this "
            f"calculation's dense grid is {tuple(calculation.basis.dense.grid)}"
        )
    moved = copy.copy(calculation)
    moved.vltot = calculation.vltot + jnp.real(v_ext_r)
    return moved


def hartree_kernel(ultracell: Ultracell, cell: Cell, keep: np.ndarray):
    """``1/|G+Q|^2`` on the box, zero at ``G = Q = 0`` and outside ``keep``."""
    g2 = ultracell.g2(cell)
    inverse = np.where(g2 > 1e-12, 1.0 / np.where(g2 > 1e-12, g2, 1.0), 0.0)
    return jnp.asarray(np.where(keep, inverse, 0.0), dtype=ultracell.precision.real)
