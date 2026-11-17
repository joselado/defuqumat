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

The runs of the end-to-end tables (P181, "End to end, with the potential on file"):

- `write_direct.py <template> <out>` and `write_direct_pbe.py`: Si2 under PZ and PBE, written with
  `write_elk_state_from` on a template; `run_variant.sh <writer> <template> <out> <epspot>` writes and
  runs Elk `task = 1`.
- `write_o2.py`, `write_o2nc.py`: the stretched O2, collinear and tilted noncollinear; the Elk inputs
  of the reference runs are `runs/o2-lsda.elk.in`, `runs/o2-noncollinear.elk.in` (and
  `runs/si2-pbe.elk.in`, `runs/h-spin-pbe.elk.in` for the PBE checks).
- `check_pot.py <elk run dir>`: `potcoul` and `potxc` against the potentials Elk wrote.
- `diag_pbe.py`: the PBE potential near the nucleus against Elk's, with and without Elk's core.
- `guide_snippet.py`: the snippet of `docs/guide/elk-seed.md`, executed on this workstation.

The template of each run is Elk's initial state (task 9006) with the converged run's `INFO.OUT`,
`GEOMETRY.OUT` and `elk.in`. The scripts carry the scratch paths of the session in their command
lines only; the inputs they need are here.
