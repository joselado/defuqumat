import sys, time
import numpy as np
from pathlib import Path
from defuqumat.io.elk import ElkState
from defuqumat.io.elk_potential import read_elk_parameters, coulomb_potential, exchange_correlation_potential, _setup
d = Path(sys.argv[1])
state = ElkState.read(d); params = read_elk_parameters(sys.argv[2] if len(sys.argv) > 2 else d)
print(params)
t = time.perf_counter(); setup = _setup(state, params); print("setup", time.perf_counter() - t)
t = time.perf_counter(); vclmt, vclir = coulomb_potential(state, params, setup); print("potcoul", time.perf_counter() - t)
ref_mt, ref_ir = state.vclmt / 2, state.vclir / 2
for ias in range(state.rhomt.shape[2]):
    s = setup.species_of[ias]; nr = len(setup.r[s])
    d0 = np.abs(vclmt[0, :nr, ias] - ref_mt[0, :nr, ias]); s0 = np.abs(ref_mt[0, :nr, ias])
    dl = np.abs(vclmt[1:, :nr, ias] - ref_mt[1:, :nr, ias]).max(); sl = np.abs(ref_mt[1:, :nr, ias]).max()
    print(f"atom {ias}: vclmt l=0 max rel {np.max(d0 / s0):.2e}  l>0 max abs {dl:.2e} (scale {sl:.2e})")
print(f"vclir max abs {np.abs(vclir - ref_ir).max():.2e} (scale {np.abs(ref_ir).max():.2e})")
t = time.perf_counter(); vxcmt, vxcir, bxcmt, bxcir = exchange_correlation_potential(state, params, None, setup); print("potxc", time.perf_counter() - t)
ref_mt, ref_ir = state.vxcmt / 2, state.vxcir / 2
for ias in range(state.rhomt.shape[2]):
    s = setup.species_of[ias]; nr = len(setup.r[s])
    by_l = [np.abs(vxcmt[l*l:(l+1)**2, :nr, ias] - ref_mt[l*l:(l+1)**2, :nr, ias]).max() for l in range(params.lmaxo + 1)]
    print(f"atom {ias}: vxcmt by l", " ".join(f"{x:.1e}" for x in by_l), f" scale {np.abs(ref_mt[0,:nr,ias]).max():.1f}")
print(f"vxcir max abs {np.abs(vxcir - ref_ir).max():.2e} (scale {np.abs(ref_ir).max():.2e})")
from defuqumat.io.elk_potential import crystal_symmetrizer
sym, nsym = crystal_symmetrizer(state, params.lmaxo)
print("crystal group:", nsym, "operations")
vxcmt_s, _, _, _ = exchange_correlation_potential(state, params, sym, setup)
ref_mt = state.vxcmt / 2
for ias in range(state.rhomt.shape[2]):
    s = setup.species_of[ias]; nr = len(setup.r[s])
    by_l = [np.abs(vxcmt_s[l*l:(l+1)**2, :nr, ias] - ref_mt[l*l:(l+1)**2, :nr, ias]).max() for l in range(params.lmaxo + 1)]
    print(f"atom {ias}: symmetrised vxcmt by l", " ".join(f"{x:.1e}" for x in by_l))
print("S(Elk's own vxcmt) - Elk's:", np.abs(sym(ref_mt) - ref_mt).max())
