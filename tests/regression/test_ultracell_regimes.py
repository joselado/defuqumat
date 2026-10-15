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

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from defuqumat import Calculator
from defuqumat.basis.builder import build_basis
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


def _tiled_null(calculator, scf, shape, kgrid, nbnd, conv_thr=1e-10, **options):
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
    assert density < 1e-5
    return result, energy


def _energy_ladder(calculator, scf, shape, kgrid, reference, ladder, **options):
    """Total energy per cell at each ``nbnd``, minus the supercell's."""
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
        jax.clear_caches()
    print("  energy above the supercell: "
          + ", ".join(f"nbnd={n}: {g:+.3e}" for n, g in gaps.items()))
    return gaps


# -- the refusals that stay, and the ones that moved ---------------------------


def test_what_the_box_still_refuses(tmp_path, pseudo_dir):
    """What the box has not been measured in refuses; GGA, ``nosource`` and TB09 pass."""
    from defuqumat.ultracell.potential import require_an_ultracell_functional
    from defuqumat.xc.functional import get_functional

    with pytest.raises(NotImplementedError, match="meta-GGA .* magnetic cell"):
        require_an_ultracell_functional(get_functional("tb09"), nspin_mag=2)
    with pytest.raises(NotImplementedError, match="ultrasoft or PAW"):
        require_an_ultracell_functional(get_functional("tb09"), augmented=True)
    require_an_ultracell_functional(get_functional("tb09"))
    with pytest.raises(NotImplementedError, match="noncollinear magnetization"):
        require_an_ultracell_functional(get_functional("PBE"), nspin_mag=4)
    require_an_ultracell_functional(get_functional("PBE"), nspin_mag=2)
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
    _, null_energy = _tiled_null(calculator, scf, shape, kgrid, nbnd=8,
                                 conv_thr=1e-8)
    assert abs(null_energy) < 1e-7

    supercell = _hydrogen(tmp_path, pseudo_dir, kgrid, shape=shape, system=pbe,
                          tag="h-sup")
    reference, grid = _supercell_reference(supercell, shape, nbnd=16,
                                           conv_thr=1e-9)
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
    """U on Si 3p, three cells: the tiled null, then the energy ladder.

    Three cells rather than two, because the projection onto copy ``c`` carries
    ``exp(i (k0+Q) . R_c)`` and at two cells ``R_1`` and ``-R_1`` are the same
    copy, so a sign error in the phase would be invisible; at three it puts the
    occupation of one copy on the other.
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

    supercell = _silicon(tmp_path, pseudo_dir, kgrid, shape=shape, ecut=14.0,
                         cards=HUBBARD, tag="u-sup")
    reference, _ = _supercell_reference(supercell, shape, nbnd=18)
    gaps = _energy_ladder(calculator, scf, shape, kgrid, reference, (12, 24, 48))
    assert gaps[48] < gaps[24] < gaps[12]
    assert all(g > 0.0 for g in gaps.values()), gaps
    assert gaps[48] < 1e-6, gaps


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
    """
    shape, kgrid = (2, 1, 1), (1, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    tb09 = ",\n input_dft='tb09'"
    calculator = _silicon(tmp_path, pseudo_dir, folded, ecut=12.0, system=tb09,
                          tag="tb09")
    assert calculator.system.input_dft is not None
    scf = calculator.get_scf(conv_thr=1e-11, nbnd=8)
    assert scf.converged and scf.tau is not None
    _tiled_null(calculator, scf, shape, kgrid, nbnd=12)

    supercell = _silicon(tmp_path, pseudo_dir, kgrid, shape=shape, ecut=12.0,
                         system=tb09, tag="tb09-sup")
    reference, grid = _supercell_reference(supercell, shape, nbnd=12)
    from defuqumat.basis.builder import build_basis as _build
    miller = np.asarray(_build(supercell.system).dense.miller)
    exact = _fourier(np.asarray(reference.density)[0], grid, miller)

    errors = {}
    # 80 is the top: at 96 the frozen solve has 169 plane waves for 96 bands
    # and Davidson needs 2 nbnd of room (cegterg.f90:125), which it refuses.
    for nbnd in (12, 24, 48, 80):
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
    assert errors[80] < errors[48] < errors[24] < errors[12]
    assert errors[80] < 0.01, errors


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
    """
    shape, kgrid = (2, 1, 1), (1, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    control = TEFIELD.format(dipfield=".true." if dipfield else ".false.")
    calculator = _silicon(tmp_path, pseudo_dir, folded, control=control,
                          system=SAWTOOTH, tag="saw")
    assert calculator.system.efield is not None
    scf = calculator.get_scf(conv_thr=1e-12, nbnd=12)
    assert scf.converged and "efield" in scf.energy_terms
    null, null_energy = _tiled_null(calculator, scf, shape, kgrid, nbnd=12)
    assert abs(null_energy) < 1e-10
    # ``E_f`` alone is linear in the density, so it moves at first order with
    # the frozen states' 7e-8 (6.6e-10 Ry measured) where the total does not.
    assert null.energy_terms["efield"] == pytest.approx(
        scf.energy_terms["efield"], abs=1e-8)

    supercell = _silicon(tmp_path, pseudo_dir, kgrid, shape=shape,
                         control=control, system=SAWTOOTH, tag="saw-sup")
    reference, _ = _supercell_reference(supercell, shape, nbnd=12)
    gaps = _energy_ladder(calculator, scf, shape, kgrid, reference, (12, 24, 48))
    assert gaps[48] < gaps[24] < gaps[12]
    assert all(g > 0.0 for g in gaps.values()), gaps
    assert gaps[48] < 1e-6, gaps


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
    reference, grid = _supercell_reference(supercell, shape, nbnd=16)
    rows, _ = _density_ladder(calculator, scf, shape, kgrid, reference, grid,
                              (8, 16, 24))
    charges = [r[1] for r in rows]
    moments = [r[2] for r in rows]
    assert charges == sorted(charges, reverse=True), charges
    assert moments == sorted(moments, reverse=True), moments
    assert charges[-1] < 5e-4 and moments[-1] < 5e-4, rows
