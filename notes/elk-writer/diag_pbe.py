import sys, numpy as np
from pathlib import Path
exec(open("write_direct_pbe.py").read().split("template = Path")[0])
from defuqumat.io.elk import ElkState
from defuqumat.io.elk_allelectron import all_electron_state
conv = ElkState.read(Path("../si2-pbe/conv"))
r = conv.rsp[0][:conv.nrmt[0]]; y = np.sqrt(4*np.pi)
for rm in (0.0, 1e-4):
    st, params = all_electron_state(res, calc.calculation, Path("../template-pbe"), r_match=rm)
    print(f"--- r_match {rm}")
    for x in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.5):
        i = np.searchsorted(r, x)
        rho_rel = st.rhomt[0, i, 0] / conv.rhomt[0, i, 0] - 1
        print(f"r={r[i]:.1e} rho rel {rho_rel:+.2e}  vxc ours {st.vxcmt[0,i,0]/2/y:12.4f} conv {conv.vxcmt[0,i,0]/2/y:12.4f}  vcl diff {(st.vclmt[0,i,0]-conv.vclmt[0,i,0])/2/y:+.2e}")
    print("vxcmt l>0 max abs diff", np.abs(st.vxcmt[1:, :len(r), 0] - conv.vxcmt[1:, :len(r), 0]).max() / 2)
