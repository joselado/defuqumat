"""Route U step 3: the projectability of graphene's atomic trials against the band window N.

The graphene cell of the acceptance (c = 15 A, 50 Ry, the full zero-field Hamiltonian on
graphene-scf.npz), the N1 = 16 loops of kappa2 = 0 and 1/12 (q = 1), one NSCF at NMAX bands, and
1 - p_j = 1 - sum_{m<N} |<psi_m|g_j>|^2 / |g_j|^2 for every N in the list, for p_z, the other
pseudo-atomic orbitals (2s, 2p) and hydrogenic d at several zona (r = 1). The guard of the
workflow refuses 1 - p > 1e-4.

    python mag_route_u3_project.py [NMAX]
"""
import sys
import time

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from defuqumat.magnetic.hybrid import atomic_trials, projected_hybrid
from defuqumat.scf.driver import Calculation
from defuqumat.system.kpoints import KPoints, for_spin
from defuqumat.topology.states import build_plane_wave_states
from defuqumat.wannier.pao import pao_projections
from defuqumat.wannier.projections import parse_projections
from defuqumat.workflows.landau_ultracell import ultracell_kpoints
from defuqumat.workflows.nscf import fixed_density_ethr, fixed_density_hamiltonians

sys.path.insert(0, "plans/orbital-field")
from mag_route_u3_graphene import DATA, ED_BOX, RY_EV, graphene  # noqa: E402

NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 96
calc = graphene()
system, pseudos = calc.system, calc.pseudos
points = ultracell_kpoints(1, 16, [0.0, 1.0 / 12])
kpoints = KPoints.from_crystal(points, np.ones(len(points)), system.cell)
system_k = eqx.tree_at(lambda s: s.kpoints, system, for_spin(kpoints, system.nspin))
t0 = time.perf_counter()
calculation = Calculation(system_k, pseudos).for_bands(NMAX)
density = jnp.asarray(np.load(DATA / "graphene-scf.npz")["density"])
hams = fixed_density_hamiltonians(calculation, density)
eig, wfc = calculation.diagonalize(hams, NMAX, None, fixed_density_ethr(1e-10, calculation.nelec))
eig = np.asarray(eig[0])
states = build_plane_wave_states(calculation, wfc[0])
print(f"NSCF {len(points)} k at {NMAX} bands: {time.perf_counter() - t0:.1f} s", flush=True)

crystal = np.asarray(system.structure.positions_crystal(system.cell), dtype=float)
paos = pao_projections(pseudos, system.structure, crystal)
sets = {"pz": [p for p in paos if (p.l, p.m) == (1, 1)],
        "s, px, py": [p for p in paos if (p.l, p.m) != (1, 1)]}
for zona in (0.5, 1.0, 2.0, 3.0):
    sets[f"d zona {zona:g}"] = parse_projections([f"C:d:zona={zona}"], list(system.structure.symbols),
                                                 crystal.T, np.asarray(system.cell.bg, dtype=float))
values = {name: atomic_trials(states, trials, pseudos, float(system.cell.volume))
          for name, trials in sets.items()}
for N in (8, 16, 24, 32, 48, 64, 80, 96):
    if N > NMAX:
        continue
    top = (eig[:, N - 1].min() - ED_BOX) * RY_EV
    line = f"N {N:3d} (band N at least {top:+.2f} eV above E_D):"
    for name, g in values.items():
        gauge = projected_hybrid(np.asarray(wfc[0]), g, N)
        line += f"  {name} {np.max(1 - gauge.projectability):.2e}"
    print(line, flush=True)
