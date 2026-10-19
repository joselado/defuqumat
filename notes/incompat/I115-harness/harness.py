"""I115: the whole-cell transmission of an augmented ultracell against its image.

Run from a checkout root (slot.sh cds there). For each dataset of the committed
silicon sheet, a modulated two-cell ultracell at two states_conv_thr values:
the Gram matrix C^dagger S C at every k0 against the identity, and the
exit_region='volume' transmission against run_ultracell_stm, each with the
ultracell S and with S switched off (the naive lift).
"""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))

import defuqumat  # noqa: F401,E402
import defuqumat.workflows.ultracell as wu  # noqa: E402
from defuqumat import Calculator  # noqa: E402
from defuqumat.transport.substrate import volume_overlap  # noqa: E402
from defuqumat.ultracell.driver import run_ultracell  # noqa: E402
from defuqumat.workflows.ultracell import (  # noqa: E402
    run_ultracell_stm, run_ultracell_transport)

ROOT = Path.cwd()
PSEUDO = ROOT / "tests" / "data" / "pseudo"
CASE = ROOT / "tests" / "data" / "qe" / "ultracell-sheet-paw.in"
DATASETS = {"ultrasoft": "Si.pz-n-rrkjus_psl.0.1.UPF",
            "paw": "Si.pz-n-kjpaw_psl.0.1.UPF"}
SHAPE, KGRID = (2, 1, 1), (2, 4, 1)
AMPLITUDE = 0.05
THRESHOLDS = tuple(float(x) for x in (sys.argv[1:] or ["1e-8", "1e-11"]))


def modulation(x):
    return AMPLITUDE * np.cos(2 * np.pi * x[..., 0] / SHAPE[0])


def gram_errors(result, apply_s):
    states = result.states
    worst_off, worst_diag = 0.0, 0.0
    for ispin in range(states.blocks):
        for ik in range(states.nk0):
            block = np.asarray(states.block(ispin, ik))
            gram = volume_overlap(
                block, states.mask(ik), int(states.npol),
                overlap=None if apply_s is None
                else (lambda p, i=ik: apply_s(p, i)))
            eye = np.eye(gram.shape[0])
            worst_off = max(worst_off,
                            float(np.abs((gram - eye) * (1 - eye)).max()))
            worst_diag = max(worst_diag,
                             float(np.abs(np.diag(gram) - 1).max()))
    return worst_off, worst_diag


def main():
    out = Path(tempfile.mkdtemp())
    lines = []
    for dataset, name in DATASETS.items():
        text = CASE.read_text().replace(DATASETS["paw"], name)
        path = out / f"sheet-{dataset}.in"
        path.write_text(text)
        calculator = Calculator.from_file(path, pseudo_dir=PSEUDO)
        t = time.time()
        scf = calculator.get_scf()
        assert scf.converged
        print(f"{dataset}: scf {time.time() - t:.1f} s, "
              f"E_F {float(scf.fermi_energy):.8f}", flush=True)
        energy = float(scf.fermi_energy)
        for states_thr in THRESHOLDS:
            t = time.time()
            result = run_ultracell(
                calculator.system, calculator.pseudos, scf, SHAPE, KGRID,
                nbnd=12, external=modulation, conv_thr=1e-10,
                states_conv_thr=states_thr)
            assert result.converged
            t_u = time.time() - t
            apply_s = wu._ultracell_overlap_operator(
                calculator.system, calculator.pseudos, result.states, "volume")
            on = gram_errors(result, apply_s)
            off = gram_errors(result, None)

            geometry = dict(height=0.85, axis=2, shape=(12, 4))
            image = np.asarray(run_ultracell_stm(
                calculator.system, calculator.pseudos, result, energy=energy,
                width=0.05, **geometry).values)
            t = time.time()
            ours = run_ultracell_transport(
                calculator.system, calculator.pseudos, result,
                exit_height=0.15, exit_axis=2, energies=energy,
                broadening=0.05, exit_region="volume", **geometry)
            t_v = time.time() - t
            original = wu._ultracell_overlap_operator
            wu._ultracell_overlap_operator = lambda *a, **k: None
            try:
                naive = run_ultracell_transport(
                    calculator.system, calculator.pseudos, result,
                    exit_height=0.15, exit_axis=2, energies=energy,
                    broadening=0.05, exit_region="volume", **geometry)
            finally:
                wu._ultracell_overlap_operator = original
            scale = image.max()
            d_on = float(np.abs(np.asarray(ours.image) - image).max() / scale)
            d_off = float(np.abs(np.asarray(naive.image) - image).max() / scale)
            line = (
                f"{dataset} states_conv_thr={states_thr:g}: ultracell "
                f"{t_u:.1f} s, volume transport {t_v:.1f} s | Gram with S: "
                f"offdiag {on[0]:.3e} diag {on[1]:.3e} | Gram S off: offdiag "
                f"{off[0]:.3e} diag {off[1]:.3e} | image rel diff with S "
                f"{d_on:.3e}, S off {d_off:.3e} | least_eig "
                f"{ours.least_eigenvalue:.6e} (S off "
                f"{naive.least_eigenvalue:.6e}) image max {scale:.4e}")
            print(line, flush=True)
            lines.append(line)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
