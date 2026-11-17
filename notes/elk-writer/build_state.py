"""Write our Si2 all-electron density into an Elk run directory (hand-off ready)."""
import sys, shutil, dataclasses
from pathlib import Path
exec(open("ae_gate.py").read().split("report = {}")[0])
from defuqumat.io.elk_write import write_elk_state, HANDOFF_INPUT
variant = sys.argv[3]
out = Path(sys.argv[4]); out.mkdir(parents=True, exist_ok=True)
rhomt = np.zeros_like(init.rhomt)
match = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
for ia in range(2):
    rhomt[:, :nrmt, ia] = mt_density(ia, variant)
    if match > 0:
        # Elk's Dirac core near the nucleus: the template's l=0 shape, scaled to ours at r_match
        m = np.searchsorted(rsp, match)
        rhomt[0, :m, ia] = init.rhomt[0, :m, ia] * rhomt[0, m, ia] / init.rhomt[0, m, ia]
ng = np.array(init.ngridg)
fr = np.indices(ng).reshape(3, -1, order="F").T / ng      # Fortran order of the grid
pts = fr @ geo.avec
rhoir = (np.exp(1j * pts @ G.T) @ c).real.reshape(tuple(ng), order="F")
charge_mt = [float(np.sqrt(4*np.pi) * np.trapezoid(rhomt[0, :nrmt, ia] * rsp**2, rsp)) for ia in range(2)]
print("MT charges", charge_mt)
state = dataclasses.replace(init, rhomt=rhomt, rhoir=rhoir, directory=out)
write_elk_state(out / "STATE.OUT", state)
for f in ("GEOMETRY.OUT", "Si.in"):
    shutil.copy(ELK / "conv" / f, out / f)
base = (ELK / "conv" / "elk.in").read_text().split("sppath")[1]
(out / "elk.in").write_text(HANDOFF_INPUT + "\nsppath" + base)
print("written", out)
