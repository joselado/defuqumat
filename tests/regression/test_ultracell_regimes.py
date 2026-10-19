"""P88 beyond its first regimes: GGA, ``nosource``, a saw-tooth field, a field-converged reference.

Each regime here was refused by name until its term was written, and each is
measured the way ``test_ultracell.py`` measures the method: against a real
``N``-cell supercell run through this package's own SCF in the same regime,
under the same applied potential, which shares the unit-cell machinery and none
of the ultracell assembly. The tiled null comes first in each, because it is
exact (``dV`` at round-off, the unit cell's own total energy) where the
supercell comparison is a convergence in ``nbnd``.

What the four share is one change: the ultracell potential is ``v_of_rho`` on
the box's own reciprocal vectors (:func:`~defuqumat.ultracell.potential.ultracell_sphere`),
so the gradient correction and the ``nosource`` projection are taken over the
whole modulation, and the saw-tooth and a converged field are carried the way
``vltot`` is.
"""

import types
from unittest import mock

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from defuqumat import Calculator
from defuqumat.basis.builder import build_basis
from defuqumat.scf import fields as fields_module
from defuqumat.scf.driver import Calculation, run_scf
from defuqumat.ultracell import run_ultracell, with_external_potential
from tests.regression.test_ultracell import (
    AMPLITUDE,
    MATCHED_ECUT,
    _fourier,
    _modulation,
    _Overridden,
)

pytestmark = pytest.mark.regression


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """Every test here builds cells that share no shape; see ``test_ultracell.py``."""
    yield
    jax.clear_caches()


SILICON = """&control
 calculation='scf'{control}
/
&system
 ibrav=2, celldm(1)=10.20, nat={nat}, ntyp=1, ecutwfc={ecut:.1f},
 nosym=.true., noinv=.true.{system}
/
&electrons
 conv_thr=1.0d-12
/
{cell}ATOMIC_SPECIES
 Si 28.086 {pseudo}
ATOMIC_POSITIONS alat
{atoms}
K_POINTS automatic
 {k0} {k1} {k2} 0 0 0
{cards}"""


def _silicon(tmp_path, pseudo_dir, kgrid, shape=(1, 1, 1), ecut=MATCHED_ECUT,
             pseudo="Si.pz-vbc.UPF", control="", system="", tag="si", cards=""):
    """The two-atom cell, or its ``shape`` supercell with every atom copied.

    The supercell's lattice is ``n_i a_i`` exactly, so its reciprocal lattice is
    the ultracell's and the two densities compare Fourier component by Fourier
    component; at ``MATCHED_ECUT`` the two FFT boxes are the same box, which the
    energy's sign needs.
    """
    from defuqumat.system.cell import Cell

    alat = 10.20
    unit = np.asarray(Cell.from_ibrav(2, [alat, 0, 0, 0, 0, 0]).at_alat)
    tau = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    if tuple(shape) == (1, 1, 1):
        cell, positions = "", tau
    else:
        vectors = unit * np.asarray(shape)[:, None]
        cell = "CELL_PARAMETERS alat\n" + "\n".join(
            " %.12f %.12f %.12f" % tuple(v) for v in vectors) + "\n"
        shifts = np.stack(np.meshgrid(*[np.arange(n) for n in shape],
                                      indexing="ij"), axis=-1).reshape(-1, 3)
        positions = np.concatenate([tau + s @ unit for s in shifts])
    atoms = "\n".join(" Si %.12f %.12f %.12f" % tuple(p) for p in positions)
    text = SILICON.format(
        control=control, system=system, ecut=ecut, pseudo=pseudo, cards=cards,
        nat=len(positions), cell=cell, atoms=atoms,
        k0=kgrid[0], k1=kgrid[1], k2=kgrid[2],
    )
    if tuple(shape) != (1, 1, 1):
        text = text.replace("ibrav=2, celldm(1)=10.20", f"ibrav=0, celldm(1)={alat}")
    path = tmp_path / f"{tag}-{'x'.join(map(str, shape))}.in"
    path.write_text(text)
    return Calculator.from_file(path, pseudo_dir=pseudo_dir)


def _supercell_reference(supercell, shape, nbnd, conv_thr=1e-11, **options):
    """The supercell SCF under one period of ``AMPLITUDE cos`` along ``x``."""
    calculation = Calculation(supercell.system, supercell.pseudos)
    grid = tuple(calculation.basis.dense.grid)
    coordinates = np.stack(np.meshgrid(
        *[np.arange(m) / m * n for m, n in zip(grid, shape)], indexing="ij"),
        axis=-1)
    reference = run_scf(
        supercell.system, supercell.pseudos,
        calculation=with_external_potential(
            calculation, jnp.asarray(_modulation(shape)(coordinates))),
        conv_thr=conv_thr, nbnd=nbnd, **options,
    )
    assert reference.converged
    return reference, grid


def _tiled_null(calculator, scf, shape, kgrid, nbnd, conv_thr=1e-10,
                density_tol=1e-5, **options):
    """``N > 1`` with nothing applied: one iteration, ``dV`` at round-off, the same energy."""
    result = run_ultracell(
        calculator.system, calculator.pseudos, scf, shape, kgrid, nbnd=nbnd,
        conv_thr=conv_thr, states_conv_thr=1e-10, **options,
    )
    assert result.converged and result.iterations == 1
    dv = float(np.abs(np.asarray(result.delta_v)).max())
    tiled = np.asarray(result.ultracell.tile(jnp.asarray(scf.density)))
    density = float(np.abs(np.asarray(result.density) - tiled).max() / np.abs(tiled).max())
    energy = float(result.total_energy - scf.total_energy)
    print(f"  null: dV {dv:.2e}  density {density:.2e}  energy {energy:+.2e} Ry")
    assert dv < 1e-13
    assert density < density_tol
    return result, energy


def _energy_ladder(calculator, scf, shape, kgrid, reference, ladder, keep=None,
                   **options):
    """Total energy per cell at each ``nbnd``, minus the supercell's.

    ``keep``, a dict, receives the ultracell result at each ``nbnd``.
    """
    exact = reference.total_energy / int(np.prod(shape))
    gaps = {}
    for nbnd in ladder:
        result = run_ultracell(
            calculator.system, calculator.pseudos, scf, shape, kgrid,
            nbnd=nbnd, external=_modulation(shape), conv_thr=1e-11,
            states_conv_thr=1e-10, **options,
        )
        assert result.converged
        gaps[nbnd] = result.total_energy - exact
        if keep is not None:
            keep[nbnd] = result
        jax.clear_caches()
    print("  energy above the supercell: "
          + ", ".join(f"nbnd={n}: {g:+.3e}" for n, g in gaps.items()))
    return gaps


# -- the refusals that stay, and the ones that moved ---------------------------


def test_what_the_box_still_refuses(tmp_path, pseudo_dir):
    """What the box has not been measured in refuses; an unpolarized GGA, ``nosource`` and TB09 pass."""
    from defuqumat.ultracell.potential import require_an_ultracell_functional
    from defuqumat.xc.functional import get_functional

    with pytest.raises(NotImplementedError, match="meta-GGA .* magnetic cell"):
        require_an_ultracell_functional(get_functional("tb09"), nspin_mag=2)
    with pytest.raises(NotImplementedError, match="ultrasoft or PAW"):
        require_an_ultracell_functional(get_functional("tb09"), augmented=True)
    require_an_ultracell_functional(get_functional("tb09"))
    with pytest.raises(NotImplementedError, match="noncollinear magnetization"):
        require_an_ultracell_functional(get_functional("PBE"), nspin_mag=4)
    with pytest.raises(NotImplementedError, match="collinear spin-polarized"):
        require_an_ultracell_functional(get_functional("PBE"), nspin_mag=2)
    require_an_ultracell_functional(get_functional("PBE"))
    require_an_ultracell_functional(get_functional("PZ"), source_free=True,
                                    nspin_mag=4)


def test_a_berry_phase_field_is_still_refused(tmp_path, pseudo_dir):
    """``lelfield`` keeps its refusal now that ``tefield`` has lost its own."""
    from defuqumat.ultracell.driver import require_an_ultracell_regime

    calculator = _silicon(tmp_path, pseudo_dir, (2, 2, 2))
    basis = build_basis(calculator.system)
    require_an_ultracell_regime(calculator.system, calculator.pseudos, basis)
    with pytest.raises(NotImplementedError, match="lelfield"):
        require_an_ultracell_regime(
            _Overridden(calculator.system, berry_field=object()),
            calculator.pseudos, basis)


TEFIELD = "\n tefield=.true., dipfield={dipfield}"
SAWTOOTH = "\n edir=3, eamp=0.002, emaxpos=0.6, eopreg=0.2"


def test_a_saw_tooth_along_the_modulation_is_refused(tmp_path, pseudo_dir):
    """The box's saw-tooth would have the ultracell's period: refused, before any solve."""
    calculator = _silicon(tmp_path, pseudo_dir, (2, 1, 1),
                          control=TEFIELD.format(dipfield=".false."),
                          system=SAWTOOTH.replace("edir=3", "edir=1"))
    stand_in = types.SimpleNamespace(converged=True, magnetic_field=None)
    with pytest.raises(NotImplementedError, match="edir = 1 with 2 cells"):
        run_ultracell(calculator.system, calculator.pseudos, stand_in,
                      (2, 1, 1), (1, 1, 1), nbnd=8)


def test_a_penalty_converged_reference_is_still_refused(tmp_path, pseudo_dir):
    """A penalty's potential moves with ``m``, so it cannot be tiled as a fixed field."""
    calculator = _silicon(tmp_path, pseudo_dir, (2, 1, 1))
    for constraint in ("total", "atomic"):
        stand_in = types.SimpleNamespace(
            converged=True, field_scale=1.0,
            magnetic_field=types.SimpleNamespace(constraint=constraint))
        with pytest.raises(NotImplementedError, match="penalty constraint"):
            run_ultracell(calculator.system, calculator.pseudos, stand_in,
                          (2, 1, 1), (1, 1, 1), nbnd=8)


def test_a_per_atom_or_secant_feedback_reference_is_refused(tmp_path, pseudo_dir):
    """The box runs the cell-wide ``'fsm'`` under Elk's rule; the rest refuse.

    A per-atom scheme would need one field per atom copy, and the box carries
    the unit cell's spheres; the secant rule steps on a converged inner loop
    and is not measured on the box. ``'fsm'`` and ``reducebf`` are measured
    (``test_a_feedback_reference_converges_to_the_supercell_under_the_same_scheme``).
    """
    calculator = _silicon(tmp_path, pseudo_dir, (2, 1, 1))
    for constraint in ("atomic fsm", "atomic fsm direction"):
        stand_in = types.SimpleNamespace(
            converged=True, field_scale=1.0,
            magnetic_field=types.SimpleNamespace(constraint=constraint,
                                                 reducebf=1.0, fsm_update="elk"))
        with pytest.raises(NotImplementedError, match="per-atom feedback"):
            run_ultracell(calculator.system, calculator.pseudos, stand_in,
                          (2, 1, 1), (1, 1, 1), nbnd=8)
    stand_in = types.SimpleNamespace(
        converged=True, field_scale=1.0,
        magnetic_field=types.SimpleNamespace(constraint="fsm", reducebf=1.0,
                                             fsm_update="secant"))
    with pytest.raises(NotImplementedError, match="fsm_update = 'secant'"):
        run_ultracell(calculator.system, calculator.pseudos, stand_in,
                      (2, 1, 1), (1, 1, 1), nbnd=8)


def test_a_fading_reference_whose_field_is_still_on_is_refused(tmp_path, pseudo_dir):
    """``reducebf`` is measured where the unit cell's field has faded; above 1.5e-4 Ry it refuses.

    The stand-in carries 2e-4 Ry, between the guard and the 1e-3 Ry the
    guard used to sit at, so it fires only at the corrected threshold.
    """
    calculator = _silicon(tmp_path, pseudo_dir, (2, 1, 1))
    stand_in = types.SimpleNamespace(
        converged=True, field_scale=0.01,
        magnetic_field=types.SimpleNamespace(
            constraint="none", reducebf=0.9, fsm_update="elk",
            residual=lambda scale: scale * 0.02))
    with pytest.raises(NotImplementedError, match="has not faded"):
        run_ultracell(calculator.system, calculator.pseudos, stand_in,
                      (2, 1, 1), (1, 1, 1), nbnd=8)


def test_a_feedback_reference_with_a_fading_field_is_refused(tmp_path, pseudo_dir):
    """``'fsm'`` with ``reducebf < 1``, built by hand, refuses as the input builder does.

    The builder refuses the pair because the decay scales the applied field
    while the controller steps the stored one; a ``MagneticField`` built
    without the builder would otherwise pass every other guard (its residual
    is set to zero here so the fading guard cannot be the one that fires).
    """
    calculator = _silicon(tmp_path, pseudo_dir, (2, 1, 1))
    stand_in = types.SimpleNamespace(
        converged=True, field_scale=1e-6,
        magnetic_field=types.SimpleNamespace(
            constraint="fsm", reducebf=0.8, fsm_update="elk",
            residual=lambda scale: 0.0))
    with pytest.raises(NotImplementedError,
                       match="feedback scheme constrained_magnetization = 'fsm' "
                             "with reducebf = 0.8"):
        run_ultracell(calculator.system, calculator.pseudos, stand_in,
                      (2, 1, 1), (1, 1, 1), nbnd=8)


def test_a_spinor_reference_under_a_field_is_refused(tmp_path, pseudo_dir):
    """The converged field is measured at ``nspin = 2`` only; a spinor reference refuses."""
    calculator = _silicon(tmp_path, pseudo_dir, (2, 1, 1))
    stand_in = types.SimpleNamespace(
        converged=True, field_scale=1.0,
        magnetic_field=types.SimpleNamespace(constraint="none", reducebf=1.0))
    with pytest.raises(NotImplementedError, match="noncollinear ground state"):
        run_ultracell(_Overridden(calculator.system, nspin=4), calculator.pseudos,
                      stand_in, (2, 1, 1), (1, 1, 1), nbnd=8)


# -- I31: a gradient-corrected functional --------------------------------------


@pytest.mark.slow
def test_a_gga_ultracell_bounds_the_gga_supercell_from_above(tmp_path, pseudo_dir):
    """PBE: the tiled null, then the energy ladder against a PBE supercell.

    The same claim ``test_the_total_energy_bounds_the_supercells_from_above``
    makes for the LDA, and for the same reason it carries a sign: the bases are
    nested in ``nbnd`` and the two boxes are the same box at ``MATCHED_ECUT``.
    What it adds is the gradient of the box density, which carries the
    envelope's own gradient; a gradient taken cell by cell, as Elk's ``potxcu``
    takes it, is a different functional from the supercell's and the bound is
    not a statement about it.
    """
    shape, kgrid = (2, 1, 1), (1, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator = _silicon(tmp_path, pseudo_dir, folded, pseudo="Si.pbe-rrkj.UPF")
    scf = calculator.get_scf(conv_thr=1e-12, nbnd=12)
    assert scf.converged
    _, null_energy = _tiled_null(calculator, scf, shape, kgrid, nbnd=12)
    assert abs(null_energy) < 1e-10

    supercell = _silicon(tmp_path, pseudo_dir, kgrid, shape=shape,
                         pseudo="Si.pbe-rrkj.UPF")
    reference, _ = _supercell_reference(supercell, shape, nbnd=12)
    gaps = _energy_ladder(calculator, scf, shape, kgrid, reference, (12, 24, 48))
    assert gaps[48] < gaps[24] < gaps[12]
    assert all(g > 0.0 for g in gaps.values()), gaps
    assert gaps[48] < 1e-6, gaps


#: The partly-polarized hydrogen lattice of ``test_ultracell.py``, under PBE by
#: ``input_dft`` so that the cell, the grid and the pseudopotential are the ones
#: the LDA ladder there was measured on.
HYDROGEN = """&control
 calculation='scf'
/
&system
 ibrav=0, celldm(1)=5.5, nat={nat}, ntyp=1, ecutwfc=15.0,
 nosym=.true., noinv=.true.,
 nspin=2, starting_magnetization(1)=0.8,
 occupations='smearing', smearing='gaussian', degauss=0.02{system}
/
&electrons
 conv_thr=1.0d-11
/
CELL_PARAMETERS alat
{rows}
ATOMIC_SPECIES
 H 1.008 H.pz-vbc.UPF
ATOMIC_POSITIONS crystal
{atoms}
K_POINTS automatic
 {k0} {k1} {k2} 0 0 0
"""


def _hydrogen(tmp_path, pseudo_dir, kgrid, shape=(1, 1, 1), system="", tag="h"):
    rows = "\n".join(" %.10f %.10f %.10f" % tuple(v)
                     for v in np.diag(shape).astype(float))
    frac = np.stack(np.meshgrid(*[np.arange(n) / n for n in shape],
                                indexing="ij"), axis=-1).reshape(-1, 3)
    atoms = "\n".join(" H %.10f %.10f %.10f" % tuple(x) for x in frac)
    path = tmp_path / f"{tag}-{'x'.join(map(str, shape))}.in"
    path.write_text(HYDROGEN.format(nat=len(frac), rows=rows, atoms=atoms,
                                    system=system, k0=kgrid[0], k1=kgrid[1],
                                    k2=kgrid[2]))
    return Calculator.from_file(path, pseudo_dir=pseudo_dir)


def _density_ladder(calculator, scf, shape, kgrid, reference, grid, ladder,
                    conv_thr=1e-10, **options):
    """Charge and moment errors against the supercell, per ``nbnd``."""
    miller = np.stack(np.meshgrid(
        np.arange(-4, 5), np.arange(-2, 3), np.arange(-2, 3), indexing="ij"
    ), axis=-1).reshape(-1, 3)
    theirs = np.asarray(reference.density)
    their_charge = _fourier(theirs.sum(axis=0), grid, miller)
    their_moment = _fourier(theirs[0] - theirs[1], grid, miller)
    rows = []
    for nbnd in ladder:
        result = run_ultracell(
            calculator.system, calculator.pseudos, scf, shape, kgrid, nbnd=nbnd,
            external=_modulation(shape, 0), conv_thr=conv_thr,
            states_conv_thr=1e-8, **options,
        )
        assert result.converged
        box = result.ultracell.grid
        ours = np.asarray(result.density)
        charge = float(np.abs(_fourier(ours.sum(axis=0), box, miller)
                              - their_charge).max() / np.abs(their_charge).max())
        moment = float(np.abs(_fourier(ours[0] - ours[1], box, miller)
                              - their_moment).max() / np.abs(their_moment).max())
        free = result.total_energy + (result.field_energy or 0.0)
        rows.append((nbnd, charge, moment, free))
        jax.clear_caches()
    exact = (reference.total_energy + (reference.field_energy or 0.0)) / int(np.prod(shape))
    for nbnd, charge, moment, free in rows:
        print(f"  nbnd={nbnd}: charge {charge:.3e}  moment {moment:.3e}  "
              f"free energy - supercell {free - exact:+.3e} Ry")
    return rows, exact


@pytest.mark.slow
@pytest.mark.xfail(strict=True, raises=NotImplementedError,
                   reason="a polarized GGA ultracell is refused: three D22 runs "
                   "of this check did not close (the guard's message has the "
                   "numbers); lift the marker with the guard")
def test_a_polarized_gga_ultracell_converges_to_the_supercell(tmp_path, pseudo_dir):
    """PBE at ``nspin = 2``: the null, then charge and moment against a supercell.

    The two channels' gradients enter correlation together (QE's ``v2c_ud``),
    so this is the regime where a gradient built channel by channel, or cell by
    cell, would show first.
    """
    shape, kgrid = (2, 1, 1), (2, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    pbe = ",\n input_dft='PBE'"
    calculator = _hydrogen(tmp_path, pseudo_dir, folded, system=pbe)
    assert calculator.system.input_dft is not None
    # PBE on this dilute lattice stalls near 9e-10 Ry (measured: 100
    # iterations at 8.9e-10 against 1e-11), the gradient's noise where the
    # density is small, so both SCFs stop at 1e-9 and the ultracell at 1e-8.
    scf = calculator.get_scf(conv_thr=1e-9, nbnd=8)
    assert scf.converged and abs(scf.magnetization) > 0.1
    # The moment of this cell is soft, so the 1e-9 stop leaves the density at
    # 1.2e-4 of its maximum from the frozen states' own (measured); the energy,
    # stationary, is 4.1e-9 Ry.
    _, null_energy = _tiled_null(calculator, scf, shape, kgrid, nbnd=8,
                                 conv_thr=1e-8, density_tol=1e-3)
    assert abs(null_energy) < 1e-7

    supercell = _hydrogen(tmp_path, pseudo_dir, kgrid, shape=shape, system=pbe,
                          tag="h-sup")
    reference, grid = _supercell_reference(supercell, shape, nbnd=16,
                                           conv_thr=1e-9, max_iterations=300,
                                           mixing_beta=0.3)
    rows, _ = _density_ladder(calculator, scf, shape, kgrid, reference, grid,
                              (8, 16, 24), conv_thr=1e-8)
    charges = [r[1] for r in rows]
    moments = [r[2] for r in rows]
    assert charges == sorted(charges, reverse=True), charges
    assert moments == sorted(moments, reverse=True), moments
    assert charges[-1] < 5e-4 and moments[-1] < 5e-4, rows


# -- I32: DFT+U ----------------------------------------------------------------


HUBBARD = "HUBBARD {atomic}\nU Si-3p 2.0\n"


@pytest.mark.slow
def test_a_hubbard_ultracell_bounds_the_hubbard_supercell_from_above(
        tmp_path, pseudo_dir):
    """U on Si 3p, three cells: the tiled null, the energy ladder, and each copy's ``ns``.

    The energy cannot tell the copies apart: replacing every copy's ``ns`` by
    their mean still lies above the supercell and falls with ``nbnd``
    (+3.0e-7 Ry at 48 bands, measured), and flipping the sign of the phase in
    the projection onto a copy relabels copy ``c`` as copy ``-c`` in the
    occupations and in the matrix together, which leaves the energy
    bit-identical. So the per-copy statement is checked on the matrices
    themselves: copy ``c``'s ``ns`` from the ultracell against the supercell's
    ``ns`` on the atoms at ``tau + R_c``, which an average and a relabelling
    both fail, since the three copies sit at three different phases of the
    modulation and no operation of the cell maps ``R_1`` onto ``-R_1``.
    """
    shape, kgrid = (3, 1, 1), (1, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    # 14 Ry puts the unit cell on (18, 18, 18) and the three-cell supercell on
    # (54, 18, 18), the same box; ``MATCHED_ECUT`` gives (50, 18, 18) at three
    # cells, where the energy's sign would not be a statement about the method.
    calculator = _silicon(tmp_path, pseudo_dir, folded, ecut=14.0,
                          cards=HUBBARD, tag="u")
    assert calculator.system.hubbard is not None
    scf = calculator.get_scf(conv_thr=1e-12, nbnd=12)
    assert scf.converged and "hubbard" in scf.energy_terms
    null, null_energy = _tiled_null(calculator, scf, shape, kgrid, nbnd=12)
    assert abs(null_energy) < 1e-10
    # ``eth`` alone is first order in ``ns``, which the frozen states hold to
    # their own accuracy (3.9e-9 Ry measured) where the total is stationary.
    assert null.energy_terms["hubbard"] == pytest.approx(
        scf.energy_terms["hubbard"], abs=1e-8)
    unit_ns = np.asarray(scf.ns)
    null_ns = np.asarray(null.hubbard_ns)
    assert null_ns.shape == (3,) + unit_ns.shape
    null_gap = float(np.abs(null_ns - unit_ns[None]).max())
    print(f"  null: every copy's ns against the unit cell's {null_gap:.2e}")
    assert null_gap < 1e-6

    supercell = _silicon(tmp_path, pseudo_dir, kgrid, shape=shape, ecut=14.0,
                         cards=HUBBARD, tag="u-sup")
    reference, _ = _supercell_reference(supercell, shape, nbnd=18)
    kept = {}
    gaps = _energy_ladder(calculator, scf, shape, kgrid, reference, (12, 24, 48),
                          keep=kept)
    assert gaps[48] < gaps[24] < gaps[12]
    assert all(g > 0.0 for g in gaps.values()), gaps
    # 6.2e-8 measured; one ``ns`` averaged over the copies gives 3.0e-7.
    assert gaps[48] < 1.5e-7, gaps

    # The supercell's atoms are the unit cell's two at each shift ``s`` along
    # ``a_1``, in that order (``_silicon``), so the slots of shift ``s`` are
    # ``2 s`` and ``2 s + 1``; copy ``c`` is the shift ``R_c``.
    nat = unit_ns.shape[1]
    triples = np.asarray(kept[48].ultracell.q_triples)
    whole = np.asarray(reference.ns)
    exact = np.stack([whole[:, nat * int(t[0] % shape[0]):
                            nat * (int(t[0] % shape[0]) + 1)] for t in triples])
    copies = np.asarray(kept[48].hubbard_ns)
    assert copies.shape == exact.shape
    cells = len(triples)
    errors = np.array([[float(np.abs(copies[c] - exact[d]).max())
                        for d in range(cells)] for c in range(cells)])
    spread = min(float(np.abs(exact[c] - exact[d]).max())
                 for c in range(cells) for d in range(cells) if c != d)
    average = float(np.abs(copies.mean(axis=0)[None] - exact).max())
    print("  ns of copy c against the supercell's atoms at shift d:\n"
          + "\n".join("    " + "  ".join(f"{e:.2e}" for e in row) for row in errors))
    print(f"  closest pair of supercell shifts {spread:.2e}; "
          f"the copies' mean against each {average:.2e}")
    own = np.diag(errors)
    # Each copy's own atoms are the closest, by a factor that a relabelling
    # (an off-diagonal entry) or an average cannot reach.
    for c in range(cells):
        others = [errors[c, d] for d in range(cells) if d != c]
        assert own[c] < 0.2 * min(others), errors
    assert own.max() < 0.2 * spread, (own, spread)
    assert own.max() < 0.2 * average, (own, average)


# -- I33: a potential-only meta-GGA --------------------------------------------


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_a_tb09_ultracell_converges_to_the_tb09_supercell(tmp_path, pseudo_dir):
    """Tran-Blaha: ``tau`` from the ultracell states, against a TB09 supercell.

    The functional has no energy, so what is compared is the induced density,
    Fourier component by Fourier component, as the LDA density ladder of
    ``test_ultracell.py`` compares it. ``tau`` is built on the box from the
    ultracell states and mixed beside the density, and ``c``, the cell
    average Tran and Blaha's potential carries, is the ultracell's average,
    which is the supercell's.

    **At ``MATCHED_ECUT``, not at 12 Ry.** At 12 Ry the box is (30, 15, 15)
    and the supercell's own grid (32, 15, 15), and TB09, whose potential is
    built from ``tau / rho`` and ``|grad rho| / rho`` on the real-space grid,
    feels the two samplings where the LDA does not: measured on D22, the
    ladder stopped at 1.084e-2 and 1.080e-2 at ``nbnd = 48`` and 64, a floor.
    On the matched pair (36, 18, 18) it reads 1.243e-1, 2.816e-2, 8.144e-3 and
    6.218e-3 at ``nbnd = 12, 24, 48, 64`` (``notes/incompat/I33.md``), about
    twice the LDA ladder of P88 at the same rungs and still falling.
    """
    shape, kgrid = (2, 1, 1), (1, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    tb09 = ",\n input_dft='tb09'"
    calculator = _silicon(tmp_path, pseudo_dir, folded, ecut=MATCHED_ECUT,
                          system=tb09, tag="tb09")
    assert calculator.system.input_dft is not None
    scf = calculator.get_scf(conv_thr=1e-11, nbnd=8)
    assert scf.converged and scf.tau is not None
    _tiled_null(calculator, scf, shape, kgrid, nbnd=12)

    supercell = _silicon(tmp_path, pseudo_dir, kgrid, shape=shape,
                         ecut=MATCHED_ECUT, system=tb09, tag="tb09-sup")
    reference, grid = _supercell_reference(supercell, shape, nbnd=12)
    from defuqumat.basis.builder import build_basis as _build
    miller = np.asarray(_build(supercell.system).dense.miller)
    exact = _fourier(np.asarray(reference.density)[0], grid, miller)

    errors = {}
    # 64 is the top rung: at 80 on the 12 Ry cell (169 plane waves) one
    # k-point's frozen solve did not settle in 100 Davidson steps and the error
    # jumped to 5.4 (D22, the previous batch); 2 nbnd has to fit
    # (cegterg.f90:125).
    for nbnd in (12, 24, 48, 64):
        result = run_ultracell(
            calculator.system, calculator.pseudos, scf, shape, kgrid,
            nbnd=nbnd, external=_modulation(shape), conv_thr=1e-10,
            states_conv_thr=1e-10,
        )
        assert result.converged
        box = result.ultracell.grid
        ours = _fourier(np.asarray(result.density)[0], box, miller)
        flat = _fourier(np.asarray(result.ultracell.tile(
            jnp.asarray(scf.density)))[0], box, miller)
        errors[nbnd] = float(np.abs((ours - flat) - (exact - flat)).max()
                             / np.abs(exact - flat).max())
        jax.clear_caches()
    print("  induced density error: "
          + ", ".join(f"nbnd={n}: {e:.3e}" for n, e in errors.items()))
    assert errors[64] < errors[48] < errors[24] < errors[12]
    assert errors[64] < 0.01, errors


# -- I34: nosource -------------------------------------------------------------


NONCOLLINEAR = """&control
 calculation='scf'
/
&system
 ibrav=0, celldm(1)=5.5, nat={nat}, ntyp=1, ecutwfc=15.0,
 nosym=.true., noinv=.true.,
 noncolin=.true., starting_magnetization(1)=0.8,
 angle1(1)=54.735610, angle2(1)=45.0,
 occupations='smearing', smearing='gaussian', degauss=0.02,
 nosource=.true.
/
&electrons
 conv_thr=1.0d-11
 mixing_beta=0.3
/
CELL_PARAMETERS alat
{rows}
ATOMIC_SPECIES
 H 1.008 H.pz-vbc.UPF
ATOMIC_POSITIONS crystal
{atoms}
K_POINTS automatic
 {k0} {k1} {k2} 0 0 0
"""


def _source_free(tmp_path, pseudo_dir, kgrid, shape=(1, 1, 1), tag="nosource"):
    rows = "\n".join(" %.10f %.10f %.10f" % tuple(v)
                     for v in np.diag(shape).astype(float))
    frac = np.stack(np.meshgrid(*[np.arange(n) / n for n in shape],
                                indexing="ij"), axis=-1).reshape(-1, 3)
    atoms = "\n".join(" H %.10f %.10f %.10f" % tuple(x) for x in frac)
    path = tmp_path / f"{tag}-{'x'.join(map(str, shape))}.in"
    path.write_text(NONCOLLINEAR.format(nat=len(frac), rows=rows, atoms=atoms,
                                        k0=kgrid[0], k1=kgrid[1], k2=kgrid[2]))
    return Calculator.from_file(path, pseudo_dir=pseudo_dir)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:nosource projects")
def test_a_source_free_ultracell_converges_to_the_supercell(tmp_path, pseudo_dir):
    """``nosource``: the projection over the whole box against the supercell's.

    The moment starts along ``(1,1,1)/sqrt(3)`` and ``nosource``, which ties
    the field's spin components to the spatial ``G``, turns it onto ``x``; the
    modulation runs along ``x`` too, so ``(G+Q) . B_xc`` is non-zero at the
    modulation's own ``Q`` and the projection there couples the two cells --
    the term the refusal named, and the term a per-cell projection misses.

    **The check can tell the two apart** (D22, ``notes/incompat/I34.md``): at
    ``nbnd = 16`` the box projection reads 7.1e-5, 1.5e-4, 1.7e-4, 1.6e-4 for
    the charge and the three magnetization components, and the same run with
    the projection dropped at every ``Q != 0`` reads 1.1e-3 on the charge and
    2.5e-1 on ``m_x``, 1700 times the box's.
    """
    shape, kgrid = (2, 1, 1), (2, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator = _source_free(tmp_path, pseudo_dir, folded)
    assert calculator.system.nosource
    scf = calculator.get_scf(conv_thr=1e-12, nbnd=16)
    assert scf.converged
    _tiled_null(calculator, scf, shape, kgrid, nbnd=8, mixing_beta=0.3)

    supercell = _source_free(tmp_path, pseudo_dir, kgrid, shape=shape,
                             tag="nosource-sup")
    reference, grid_sup = _supercell_reference(
        supercell, shape, nbnd=32, conv_thr=1e-13, max_iterations=300)
    rho_sup = np.asarray(reference.density)

    miller = None
    errors = []
    for nbnd in (8, 16):
        result = run_ultracell(
            calculator.system, calculator.pseudos, scf, shape, kgrid,
            nbnd=nbnd, conv_thr=1e-10, states_conv_thr=1e-8,
            mixing_beta=0.3, max_iterations=200, external=_modulation(shape),
        )
        assert result.converged
        grid_u = tuple(result.ultracell.grid)
        if miller is None:
            axes = [np.fft.fftfreq(min(u, s)) * min(u, s)
                    for u, s in zip(grid_u, grid_sup)]
            miller = np.stack(np.meshgrid(*axes, indexing="ij"),
                              axis=-1).reshape(-1, 3).astype(int)
        rho_u = np.asarray(result.density)
        ours = [_fourier(rho_u[c], grid_u, miller) for c in range(4)]
        theirs = [_fourier(rho_sup[c], grid_sup, miller) for c in range(4)]
        # **Normalised by the charge for the charge and by the largest
        # magnetization component for all three of its components**, not each
        # by its own: ``nosource`` turns the moment of this cell onto ``x``
        # (the unit cell ends at m = (0.2326, 1e-5, -7e-6)), so ``m_y`` and
        # ``m_z`` are 3e-6 at their largest and a per-component ratio reads
        # an absolute 2.6e-7 as 9 per cent.
        scale_m = max(np.abs(theirs[c]).max() for c in (1, 2, 3))
        errors.append([
            float(np.abs(ours[c] - theirs[c]).max()
                  / (np.abs(theirs[0]).max() if c == 0 else scale_m))
            for c in range(4)
        ])
        print(f"  nbnd={nbnd}: " + ", ".join(f"{e:.3e}" for e in errors[-1]))
        jax.clear_caches()
    # The charge and the moment's own component fall with nbnd; the two
    # components the projection holds at zero sit at an absolute 2e-7 floor.
    for component in (0, 1):
        column = [row[component] for row in errors]
        assert column == sorted(column, reverse=True), (component, column)
    assert max(errors[-1]) < 1e-3, errors


# -- I35: a saw-tooth field ----------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("dipfield", [False, True])
def test_a_saw_tooth_ultracell_bounds_the_supercell_from_above(
        dipfield, tmp_path, pseudo_dir):
    """``tefield`` along ``a3``, the modulation along ``a1``: null, then the energy ladder.

    The field is across the modulation, which is the slab geometry the
    saw-tooth is for, and is the one case where the supercell's saw-tooth is
    the tiled unit cell's: ``b'_3 = b_3`` for a supercell doubled along ``a1``.
    ``dipfield`` makes the field a function of the density's dipole, which on
    the box is the ultracell's average and in the supercell the supercell's.

    **The ``dipfield`` arm is smeared (Gaussian, 0.02 Ry).** Bulk silicon has
    no vacuum for the ramp to sit in, so its dipole correction is large (0.54
    Ry of ``E_f`` against 0.025 without it) and it closes the gap: with fixed
    occupations the unit cell's fourth band tops out at 0.474 Ry over the
    folded k-points and the fifth bottoms out at 0.380 (D22, measured), so
    "four bands at every k-point" is not the lowest four of the folded set the
    ultracell fills, and the tiled null stalled at dr2 = 4.4e-3 for that reason
    alone. Smeared, the occupations are a function of the energies and the two
    fillings agree; a Gaussian keeps the free energy variational, so the bound
    still holds. Measured on D22 (``notes/incompat/I35.md``): +8.092e-5,
    +3.528e-6, +3.636e-7 Ry per cell without ``dipfield`` and +1.952e-5,
    +4.408e-6, +4.789e-7 with it, at ``nbnd = 12, 24, 48``. In the
    ``dipfield`` arm the energy is second order in the dipole's response and
    does not see it, so ``E_f`` per cell is asserted as well.
    """
    shape, kgrid = (2, 1, 1), (1, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    control = TEFIELD.format(dipfield=".true." if dipfield else ".false.")
    system = SAWTOOTH + (
        ",\n occupations='smearing', smearing='gaussian', degauss=0.02"
        if dipfield else "")
    calculator = _silicon(tmp_path, pseudo_dir, folded, control=control,
                          system=system, tag="saw")
    assert calculator.system.efield is not None
    scf = calculator.get_scf(conv_thr=1e-12, nbnd=12)
    assert scf.converged and "efield" in scf.energy_terms
    null, null_energy = _tiled_null(calculator, scf, shape, kgrid, nbnd=12)
    assert abs(null_energy) < 1e-10
    # ``E_f`` alone moves at first order with the frozen states' density
    # (7e-8 relative, 6.6e-10 Ry measured without ``dipfield``; 2.3e-6 and
    # 2.0e-6 Ry with it, whose ``D^2`` term is 20 times the bare ramp's
    # energy), where the total does not.
    assert null.energy_terms["efield"] == pytest.approx(
        scf.energy_terms["efield"], abs=1e-5 if dipfield else 1e-8)

    supercell = _silicon(tmp_path, pseudo_dir, kgrid, shape=shape,
                         control=control, system=system, tag="saw-sup")
    reference, _ = _supercell_reference(supercell, shape, nbnd=12)
    kept = {}
    gaps = _energy_ladder(calculator, scf, shape, kgrid, reference, (12, 24, 48),
                          keep=kept)
    assert gaps[48] < gaps[24] < gaps[12]
    assert all(g > 0.0 for g in gaps.values()), gaps
    assert gaps[48] < 1e-6, gaps
    if dipfield:
        # The energy cannot tell the box dipole from a frozen one: the dipole
        # moves by 3.4e-5 under this modulation, so holding the saw-tooth at
        # the tiled density's moves the gaps by 1e-8 and the induced density
        # by 2e-3 of itself. ``E_f`` per cell can, at first order in that
        # move: 3.6e-6 Ry from the supercell's at 48 bands with the box
        # dipole, 1.1e-4 and levelling off with the frozen one (D22, I35).
        field = reference.energy_terms["efield"] / int(np.prod(shape))
        errors = [abs(kept[n].energy_terms["efield"] - field) for n in (12, 24, 48)]
        print("  E_f per cell from the supercell's: "
              + ", ".join(f"{e:.3e}" for e in errors))
        assert errors == sorted(errors, reverse=True), errors
        assert errors[-1] < 1e-5, errors


# -- I36: a reference converged under a field ----------------------------------


@pytest.mark.slow
def test_a_field_converged_reference_converges_to_the_supercell_under_it(
        tmp_path, pseudo_dir):
    """``B_field`` on the unit cell, held on every copy: null, then the supercell.

    The frozen eigenvalues carry the converged field, the ultracell tiles it
    and subtracts it again with the reference potential, and its Zeeman energy
    is paired into ``deband`` so that it stays outside the total. The null
    asserts all three at once (one iteration, the unit cell's total and its
    field energy); the ladder asserts that the modulated state under the field
    is the supercell's under the same field.
    """
    shape, kgrid = (2, 1, 1), (2, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    field = ",\n B_field(3)=0.02"
    calculator = _hydrogen(tmp_path, pseudo_dir, folded, system=field,
                           tag="h-field")
    scf = calculator.get_scf(conv_thr=1e-11, nbnd=8)
    assert scf.converged and scf.magnetic_field is not None
    null, null_energy = _tiled_null(calculator, scf, shape, kgrid, nbnd=8)
    # **The stationary quantity is the total plus the field's energy**, the
    # functional the loop minimises; the total alone moves at first order with
    # the moment, by ``B dm``: -3.9e-8 Ry measured here, from a moment the
    # frozen states hold to 1.5e-6.
    free = (null.total_energy + null.field_energy) - (scf.total_energy + scf.field_energy)
    print(f"  null free energy {free:+.2e} Ry, total {null_energy:+.2e} Ry")
    assert abs(free) < 1e-10
    assert null.field_energy == pytest.approx(scf.field_energy, abs=1e-7)

    supercell = _hydrogen(tmp_path, pseudo_dir, kgrid, shape=shape,
                          system=field, tag="h-field-sup")
    # **Both sides are mixed adaptively.** Under the field the two-atom
    # supercell stalled at 2.3e-7 Ry after 100 Anderson iterations, at 3.7e-6
    # after 300 at beta 0.3, and did not close under 'robust' either, while
    # 'adaptive' converged it in 39 (D22); the ultracell loop stalled at dr2
    # 3.7e-7 to 2.9e-5 Ry after 800 Kerker-Anderson iterations at beta 0.2 at
    # every rung and converged in 70 to 71 adaptively, which takes no Kerker.
    # The soft direction is the moment's size: the modulation moves the mean
    # moment density from 0.00380 (tiled) to 0.00451.
    reference, grid = _supercell_reference(supercell, shape, nbnd=16,
                                           max_iterations=300,
                                           mixing_mode="adaptive")
    rows, exact = _density_ladder(calculator, scf, shape, kgrid, reference, grid,
                                  (8, 16, 24), mixing_mode="adaptive",
                                  kerker=False, max_iterations=600)
    # Measured (``notes/incompat/I36.md``): charge 6.5e-4, 2.0e-4, 1.4e-4,
    # moment 9.0e-4, 2.9e-4, 1.2e-4, and the free energy above the supercell's
    # by 9.1e-6, 2.2e-6, 8.8e-7 Ry per cell at nbnd = 8, 16, 24.
    frees = [r[3] for r in rows]
    assert frees == sorted(frees, reverse=True), frees
    assert min(frees) > exact, (frees, exact)
    charges = [r[1] for r in rows]
    moments = [r[2] for r in rows]
    assert charges == sorted(charges, reverse=True), charges
    assert moments == sorted(moments, reverse=True), moments
    assert charges[-1] < 5e-4 and moments[-1] < 5e-4, rows


# -- I36-feedback: the feedback field and reducebf on the box -------------------


FSM = (",\n constrained_magnetization='fsm', lambda=0.25,"
       " fixed_magnetization(3)={target}")
FADING = ",\n B_field(3)=0.02, reducebf={reducebf}"


def _field_of(result):
    """The field a converged run ended with, ``B z`` in Ry (scale included)."""
    field = result.magnetic_field
    return float(np.asarray(field.uniform).ravel()[0]) * float(result.field_scale)


def _moment_per_cell(density, cell_volume):
    """``int (rho_up - rho_down)`` over one cell; both densities are per cell."""
    rho = np.asarray(density)
    return float((rho[0] - rho[1]).mean()) * float(cell_volume)


def _driven_ladder(calculator, scf, shape, kgrid, reference, grid, ladder,
                   target, **options):
    """Densities, field, moment and the matched-moment energy against the supercell.

    ``E - B (M - M_target)`` per cell removes the moment's miss to first order:
    at matched moment the nested bases bound ``total_energy`` from above, and a
    miss ``dM`` inside ``FSM_TOLERANCE`` moves it by ``B dM``, which is above
    the ladder's steps. Without a target (``reducebf``) it is the total.
    """
    cells = int(np.prod(shape))
    miller = np.stack(np.meshgrid(
        np.arange(-4, 5), np.arange(-2, 3), np.arange(-2, 3), indexing="ij"
    ), axis=-1).reshape(-1, 3)
    theirs = np.asarray(reference.density)
    their_charge = _fourier(theirs.sum(axis=0), grid, miller)
    their_moment = _fourier(theirs[0] - theirs[1], grid, miller)
    volume = float(calculator.system.cell.volume)
    their_field = _field_of(reference)
    their_m = _moment_per_cell(theirs, volume)
    their_e = reference.total_energy / cells
    if target is not None:
        their_e -= their_field * (their_m - target)
    rows = []
    for nbnd in ladder:
        result = run_ultracell(
            calculator.system, calculator.pseudos, scf, shape, kgrid, nbnd=nbnd,
            external=_modulation(shape, 0), states_conv_thr=1e-8, **options,
        )
        assert result.converged, (nbnd, result.accuracy)
        box = result.ultracell.grid
        ours = np.asarray(result.density)
        charge = float(np.abs(_fourier(ours.sum(axis=0), box, miller)
                              - their_charge).max() / np.abs(their_charge).max())
        moment = float(np.abs(_fourier(ours[0] - ours[1], box, miller)
                              - their_moment).max() / np.abs(their_moment).max())
        b = _field_of(result)
        m = _moment_per_cell(ours, volume)
        e = result.total_energy
        if target is not None:
            e -= b * (m - target)
        rows.append(dict(nbnd=nbnd, charge=charge, moment=moment, field=b, m=m,
                         energy=e - their_e, iterations=result.iterations,
                         scale=float(result.field_scale)))
        print(f"  nbnd={nbnd}: charge {charge:.3e}  moment {moment:.3e}  "
              f"B {b:+.6e} (supercell {their_field:+.6e}, diff {b - their_field:+.2e})"
              f"  M {m:.6f} (supercell {their_m:.6f})  energy - supercell "
              f"{e - their_e:+.3e} Ry  scale {result.field_scale:.3e}  "
              f"in {result.iterations} iterations")
        jax.clear_caches()
    return rows, their_field, their_m


@pytest.mark.slow
@mock.patch.object(fields_module, "FSM_TOLERANCE", 1e-6)
def test_a_feedback_reference_converges_to_the_supercell_under_the_same_scheme(
        tmp_path, pseudo_dir):
    """``'fsm'`` on the unit cell, run again on the box: null, then the supercell.

    The supercell under the same scheme targets ``N`` times the unit cell's
    moment, and the box reads its moment per cell, so the targets match. What
    discriminates is the field: the supercell's moves away from the unit
    cell's because the modulation moves the moment, and the box's has to follow
    it rather than stay at the reference's.
    """
    shape, kgrid = (2, 1, 1), (2, 2, 2)
    cells = int(np.prod(shape))
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    # **The target sits just above the free moment, the gain below 2/chi0,
    # and the controller's tolerance at 1e-6.** The cell's moment is stiff
    # at the operating field and not near zero: 0.62343 free and 0.62510
    # under B = +0.01 Ry is a secant chi = 0.17 mu_B/Ry (0.19 between 0.01
    # and the 0.0148 the unit cell converges to), while near zero field the
    # modulated box responds with 0.42 to 0.47 on the positive side
    # (I36-feedback-ver/f-0.9.log, f-0.93.log) and 0.93 at B = -1e-3 to
    # -1.9e-3 (h1-neg.log), and B = -0.01 flips it onto the other branch;
    # targets 0.4 and 0.5 never converged (D22, d1.log, d2.log). The box's
    # first output at fixed input responds with chi0 near 7 mu_B/Ry, so
    # lambda = 2 threw the moment between +1 and -1 for 40 iterations
    # (d5.log); 0.25 converges (d6.log). And FSM_TOLERANCE = 1e-3 mu_B
    # leaves the field undetermined by 1e-3 / chi, about 6e-3 Ry with the
    # chi of 0.17 to 0.19 at the operating field, above the 4.1e-3 the
    # modulation moves it by: at lambda = 0.1 the unit cell stopped at
    # M = 0.62500, a miss of exactly the tolerance (d6.log). So the check is
    # taken with the tolerance at 1e-6 on all three runs, which is what
    # defines the state.
    target = 0.626
    calculator = _hydrogen(tmp_path, pseudo_dir, folded,
                           system=FSM.format(target=target), tag="h-fsm")
    scf = calculator.get_scf(conv_thr=1e-12, nbnd=8, max_iterations=1500)
    assert scf.converged and scf.magnetic_field is not None
    b_ref = _field_of(scf)
    m_ref = _moment_per_cell(scf.density, calculator.system.cell.volume)
    print(f"  unit cell: B {b_ref:+.6e} Ry, M {m_ref:.6f} (target {target}) "
          f"in {scf.iterations} iterations")
    assert abs(b_ref) > 1e-3, "the unit-cell controller was not exercised"

    null, _ = _tiled_null(calculator, scf, shape, kgrid, nbnd=8)
    assert _field_of(null) == pytest.approx(b_ref, abs=1e-14)

    supercell = _hydrogen(tmp_path, pseudo_dir, kgrid, shape=shape,
                          system=FSM.format(target=cells * target),
                          tag="h-fsm-sup")
    reference, grid = _supercell_reference(supercell, shape, nbnd=16,
                                           conv_thr=1e-12, max_iterations=1500,
                                           mixing_mode="adaptive")
    rows, b_sup, m_sup = _driven_ladder(
        calculator, scf, shape, kgrid, reference, grid, (8, 16, 24), target,
        conv_thr=1e-11, mixing_mode="adaptive", kerker=False,
        max_iterations=1500)
    print(f"  supercell: B {b_sup:+.6e} Ry, M per cell {m_sup:.6f}; "
          f"B_sup - B_ref {b_sup - b_ref:+.3e} Ry")

    # The alternative the lift replaces: the reference's field held fixed.
    import copy

    from defuqumat.scf.fields import MagneticField

    held = copy.copy(scf)
    field = scf.magnetic_field
    held.magnetic_field = MagneticField(
        regions=field.regions, uniform=field.uniform, atomic=field.atomic,
        targets=None, penalty=0.0)
    frozen = run_ultracell(
        calculator.system, calculator.pseudos, held, shape, kgrid, nbnd=24,
        external=_modulation(shape, 0), states_conv_thr=1e-8, conv_thr=1e-11,
        mixing_mode="adaptive", kerker=False, max_iterations=1500)
    m_frozen = _moment_per_cell(frozen.density, calculator.system.cell.volume)
    print(f"  frozen field at nbnd=24: M per cell {m_frozen:.6f} against the "
          f"target {target}")

    fields = [abs(r["field"] - b_sup) for r in rows]
    assert fields[-1] < fields[0], fields
    assert fields[-1] < 0.1 * abs(b_sup - b_ref), (fields, b_sup, b_ref)
    for key in ("charge", "moment"):
        errors = [r[key] for r in rows]
        assert errors == sorted(errors, reverse=True), (key, errors)
    energies = [r["energy"] for r in rows]
    assert energies == sorted(energies, reverse=True), energies
    assert min(energies) > 0.0, energies
    assert all(abs(r["m"] - target) < 1e-6 for r in rows), rows
    assert abs(m_frozen - target) > 5 * max(abs(r["m"] - target) for r in rows)


@pytest.mark.slow
def test_a_fading_field_reference_converges_to_the_supercell_under_it(
        tmp_path, pseudo_dir):
    """``reducebf`` on the unit cell, continued on the box: null, then the supercell.

    The box continues the decay from the scale the reference ended at; the
    supercell runs its own from full strength. Both end with the field faded,
    and the state they reach is the modulated field-free one, which the
    ladder compares.
    """
    shape, kgrid = (2, 1, 1), (2, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    reducebf = 0.8
    calculator = _hydrogen(tmp_path, pseudo_dir, folded,
                           system=FADING.format(reducebf=reducebf), tag="h-fade")
    scf = calculator.get_scf(conv_thr=1e-11, nbnd=8, max_iterations=300,
                             mixing_mode="adaptive")
    assert scf.converged and scf.magnetic_field is not None
    print(f"  unit cell: field_scale {scf.field_scale:.3e}, residual "
          f"{scf.magnetic_field.residual(scf.field_scale):.3e} Ry in "
          f"{scf.iterations} iterations")
    null, _ = _tiled_null(calculator, scf, shape, kgrid, nbnd=8)
    assert null.field_scale == scf.field_scale

    supercell = _hydrogen(tmp_path, pseudo_dir, kgrid, shape=shape,
                          system=FADING.format(reducebf=reducebf),
                          tag="h-fade-sup")
    reference, grid = _supercell_reference(supercell, shape, nbnd=16,
                                           max_iterations=300,
                                           mixing_mode="adaptive")
    print(f"  supercell: field_scale {reference.field_scale:.3e} in "
          f"{reference.iterations} iterations")
    rows, b_sup, m_sup = _driven_ladder(
        calculator, scf, shape, kgrid, reference, grid, (8, 16, 24), None,
        conv_thr=1e-10, mixing_mode="adaptive", kerker=False,
        max_iterations=600)
    energies = [r["energy"] for r in rows]
    assert energies == sorted(energies, reverse=True), energies
    assert min(energies) > 0.0, energies
    for key in ("charge", "moment"):
        errors = [r[key] for r in rows]
        assert errors == sorted(errors, reverse=True), (key, errors)
    assert all(r["scale"] < scf.field_scale for r in rows), rows
    # **The moment follows the decay the iteration counts predict.** The box
    # and the supercell stop after their own iteration counts, so their
    # leftover fields differ, and the moment per cell differs by chi times
    # that gap. The box's field is predicted from the reference's residual
    # and the box's own count (one decay after every iteration but the
    # converged one, so reducebf ** (iterations - 1): 4.154e-6 Ry from the
    # 1.181e-4 reference in 16), not read from the result, so a field held
    # frozen at the reference's residual would leave chi (R_ref - B_pred),
    # 5e-5 mu_B on this cell, against the 3e-6 the decayed box leaves
    # (D22, r5-check.log). chi = 0.45 mu_B/Ry per cell is the near-zero
    # response on the positive side, 0.42 to 0.47 in
    # I36-feedback-ver/f-0.9.log and f-0.93.log.
    chi = 0.45
    r_ref = float(scf.magnetic_field.residual(scf.field_scale))
    for r in rows:
        b_pred = r_ref * reducebf ** (r["iterations"] - 1)
        miss = (r["m"] - m_sup) - chi * (b_pred - b_sup)
        print(f"  nbnd={r['nbnd']}: predicted B {b_pred:.4e} Ry, moment miss "
              f"after chi dB {miss:+.3e} mu_B")
        assert abs(miss) < 1e-5, (r, b_pred, b_sup, m_sup, miss)
