"""The robust-mixing test bed (``plans/robust-mixing.md`` section 5, R0).

One cell per failure class, each with the arms it is run under. A path is
relative to the repository root, or a bare name for an input written by
``make_inputs.py`` into the generated directory.
"""

#: name -> (input, class, arms). An arm is (mixing_mode, extra options).
CELLS = {
    "al-v32": ("al-v32", "sloshing, slab", ["anderson", "ldos"]),
    "co-c10": ("tools/vacuum_sweep/co/co-c10.in", "film, magnetic, MV", ["anderson", "ldos"]),
    "fe-unstable-k4": ("fe-unstable-k4", "magnetic state", ["anderson", "tf"]),
    "co-helix4-nosoc": ("tests/data/qe/co-helix4-nosoc.in", "limit cycle", ["anderson", "tf"]),
    "ni-kind1-force": ("tests/data/qe/ni-kind1-force.in", "DFT+U minima", ["anderson", "tf"]),
    "al8-rattled": ("al8-rattled", "nonlinearity", ["anderson", "tf"]),
    "si64-1k-ecut30": ("benchmarks/si64-1k-ecut30.in", "large cell", ["anderson"]),
    "si-1k": ("benchmarks/si-1k.in", "control", ["anderson"]),
    "al2-metal": ("tests/data/qe/al2-metal.in", "control", ["anderson"]),
    "graphene": ("tests/data/qe/graphene-monolayer.in", "control, monolayer", ["anderson"]),
}

#: Every run gets this many iterations, whatever the input says.
MAX_ITERATIONS = 200
