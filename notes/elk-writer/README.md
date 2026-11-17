# The PAW all-electron start for Elk (P181)

The harness behind `PLAN.md` P181. `si2-pz.elk.in` is the Elk input (Elk 11.0.2,
`xctype = 2`); run it once with `tasks 0` into `conv/` and once with `tasks 9006`
(elkpy patch 0024, the initial state) into `init/`, both with `Si.in` from Elk's
species directory and `GEOMETRY.OUT` copied into `init/`.

- `ae_gate.py <dir> <ecutwfc>` rebuilds the all-electron density from defuqumat's
  PAW state on Elk's meshes and compares it with `conv/` beside `init/`.
- `build_state.py <dir> <ecutwfc> <full|trunc|core|pseudo> <out> [r_match]` writes
  that density into an Elk run directory with the hand-off input; `r_match` takes
  Elk's Dirac core shape from the template below that radius.
- `run_chain.sh <run dir>` (with `ELK` and `TEMPLATE` set) runs the hand-off loop
  and then the SCF, and prints the loop count and the first residuals.
