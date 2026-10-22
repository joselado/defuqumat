"""I119: single band precision against the double state, per regime the guard refused.

python3 harness.py <case> [conv_thr ...] [--unguarded] [--no-floor]

For each conv_thr: run_scf with the band side in float32, compared with a double
run at conv_thr 1e-12: total energy, eigenvalues, density (integrated |drho|), and
the regime's order parameter where the result carries one (magnetization, ns).
--unguarded patches the refusal list out (to measure before lifting the guard);
--no-floor also puts ethr back to QE's 1e-13 and drops the conv_thr floor, the
single-precision protection switched off; --no-conv-floor drops only the conv_thr
floor, to read where the gap to the double state stops falling. --baseline adds a
double run at each conv_thr, which separates the SCF's own error from the precision's;
--continue runs a double SCF to 1e-11 from each single result.
"""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import defuqumat  # noqa: F401,E402  (x64)
from defuqumat.io.pwin import read_pw_input  # noqa: E402
from defuqumat.pseudo import read_upf  # noqa: E402
from defuqumat.scf import driver  # noqa: E402
from defuqumat.scf.driver import Calculation, run_scf  # noqa: E402
from defuqumat.system import build_system  # noqa: E402

QE = REPO / "tests/data/qe"
BENCH = REPO / "benchmarks"
PSEUDO = REPO / "tests/data/pseudo"

CASES = {
    "us": (BENCH / "si2-us-1k.in", []),
    "paw": (BENCH / "si2-paw-1k.in", []),
    "noncolin": (QE / "h-atom-noncolin.in", [("angle1(1) = 0.0", "angle1(1) = 60.0")]),
    "spiral": (QE / "h-chain-spiral-elk.in", []),
    "ldau": (QE / "alas-ldau.in", [("4 4 4 0 0 0", "2 2 2 0 0 0")]),
    "meta": (BENCH / "si-1k.in", [("ecutwfc=12.0", "ecutwfc=12.0, input_dft='tb09'")]),
    "si": (BENCH / "si-1k.in", []),
    "spiral-us": (QE / "o-chain-spiral-us.in", []),
    "spiral-paw": (QE / "o-chain-spiral-paw.in", []),
    "soc-us": (QE / "alas-epsilon-us-soc.in", [("4 4 4 1 1 1", "2 2 2 1 1 1")]),
}


def load(case):
    path, edits = CASES[case]
    text = path.read_text()
    for old, new in edits:
        assert old in text, (case, old)
        text = text.replace(old, new)
    with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False) as handle:
        handle.write(text)
    system = build_system(read_pw_input(handle.name))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return system, pseudos


def unguard(no_floor=False, no_conv_floor=False):
    def patched(self):
        self.ethr_floor = max(driver.ETHR_MIN, self.SINGLE_ETHR_FACTOR
                              * self.band_precision.eps * float(self.system.ecutwfc))
        self.conv_thr_floor = self.SINGLE_CONV_THR_FLOOR
        if no_floor:
            self.ethr_floor, self.conv_thr_floor = driver.ETHR_MIN, 0.0
        if no_conv_floor:
            self.conv_thr_floor = 0.0

    Calculation._require_band_precision_supported = patched


def observables(result):
    out = {"E": float(result.total_energy),
           "eig": np.asarray(result.eigenvalues, dtype=float),
           "occ": np.asarray(result.occupations, dtype=float),
           "rho": np.asarray(result.density, dtype=float)}
    out["charge"] = float(np.asarray(result.density[0], dtype=float).sum())
    out["becsum_dtype"] = ",".join(str(np.asarray(b).dtype) for b in result.becsum) or "-"
    if result.becsum:
        out["becsum"] = np.concatenate([np.asarray(b, dtype=float).ravel() for b in result.becsum])
    for name in ("magnetization", "absolute_magnetization", "magnetization_vector", "ns", "tau"):
        value = getattr(result, name, None)
        if value is not None:
            try:
                out[name] = np.asarray(value, dtype=float)
            except Exception:  # noqa: BLE001
                pass
    return out


def compare(a, b, system):
    omega = float(system.cell.volume)
    npts = a["rho"][0].size
    line = {"dE": a["E"] - b["E"],
            "deig": float(np.abs(a["eig"] - b["eig"]).max()),
            # bands that hold charge in the double state: an empty band is
            # converged only to Davidson's looser empty-band threshold
            "deig_occ": float(np.abs(a["eig"] - b["eig"])[b["occ"] > 1e-4 * b["occ"].max()].max()),
            "drho": float(np.abs(a["rho"] - b["rho"]).sum() * omega / npts)}
    line["dQ"] = (a["charge"] - b["charge"]) * omega / npts
    for key in ("magnetization", "absolute_magnetization", "magnetization_vector", "ns", "becsum"):
        if key in a and key in b:
            line["d" + key] = float(np.abs(np.atleast_1d(a[key] - b[key])).max())
    if "tau" in a and "tau" in b:
        line["dtau"] = float(np.abs(a["tau"] - b["tau"]).sum() * omega / npts)
    return line


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    case, thrs = args[0], [float(x) for x in args[1:]] or [1e-5, 1e-6, 1e-7]
    if any(flag in sys.argv for flag in ("--unguarded", "--no-floor", "--no-conv-floor")):
        unguard(no_floor="--no-floor" in sys.argv,
                no_conv_floor="--no-conv-floor" in sys.argv)
    system, pseudos = load(case)
    start = time.perf_counter()
    exact = run_scf(system, pseudos, conv_thr=1e-12, max_iterations=200)
    ref = observables(exact)
    print(f"{case}: double E = {ref['E']:.10f} Ry, {len(exact.history)} it, "
          f"converged={exact.converged}, {time.perf_counter() - start:.1f} s", flush=True)
    for thr in thrs:
        start = time.perf_counter()
        calc = Calculation(system, pseudos, band_precision="single")
        single = run_scf(system, pseudos, calculation=calc, conv_thr=thr, max_iterations=200)
        line = compare(observables(single), ref, system)
        if "--continue" in sys.argv:
            begin = time.perf_counter()
            finished = run_scf(system, pseudos, conv_thr=1e-11, max_iterations=200,
                               starting_from=single)
            cont = compare(observables(finished), ref, system)
            print(f"  continued in double to 1e-11: it={len(finished.history)} "
                  f"wfc={finished.wavefunctions.dtype} "
                  + " ".join(f"{k}={v:+.2e}" for k, v in cont.items())
                  + f"  ({time.perf_counter() - begin:.1f} s)", flush=True)
        if "--baseline" in sys.argv:
            loose = run_scf(system, pseudos, conv_thr=thr, max_iterations=200)
            base = compare(observables(loose), ref, system)
            print(f"  double conv_thr={thr:.0e} it={len(loose.history)} "
                  + " ".join(f"{k}={v:+.2e}" for k, v in base.items()), flush=True)
        print(f"  single conv_thr={thr:.0e} ethr_floor={calc.ethr_floor:.1e} "
              f"it={len(single.history)} converged={single.converged} "
              f"wfc={single.wavefunctions.dtype} rho={single.density.dtype} "
              f"becsum={observables(single)['becsum_dtype']} "
              + " ".join(f"{k}={v:+.2e}" for k, v in line.items())
              + f"  ({time.perf_counter() - start:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
