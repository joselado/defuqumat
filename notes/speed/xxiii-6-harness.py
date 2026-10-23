"""OPEN.md Part XXIII item 6, the walked third derivative: bit-identity, warm timing,
memory and the HLO count, in one harness for one slot.sh slot.

    slot.sh <worktree> . <log> /abs/path/xxiii-6-harness.py <input.in> <kbatch> <outdir> <repeats> [count]

Runs the SCF, ``electrostriction`` on a host store (the walked route) once, records the
arguments of ``walked_susceptibility_derivative`` and calls it again ``repeats`` times
(warm), saving the columns and ``depsilon_dstrain`` as .npy for a byte comparison between
two commits. With ``count`` it also lowers the compiled chunk pass and counts its HLO.
"""
import dataclasses
import os
import re
import resource
import sys
import time
import warnings

import jax
import numpy as np

path, kbatch, outdir, repeats = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
count = len(sys.argv) > 5 and sys.argv[5] == "count"
os.makedirs(outdir, exist_ok=True)

from defuqumat import Calculator
import defuqumat.response.chunked_third as ct
from defuqumat.response.electrostriction import electrostriction

record = {}
seen_passes = {}
original_third_passes = ct._third_passes


def spy_passes(*args, **kwargs):
    passes = original_third_passes(*args, **kwargs)
    key = id(passes)
    if key in seen_passes:
        return seen_passes[key]

    def wrap(name, fn):
        def call(*a):
            record.setdefault(name, a)
            out = fn(*a)
            if name == "global":
                record.setdefault("global_out", out)
            return out
        call.lower = fn.lower
        return call

    wrapped = {name: wrap(name, fn) for name, fn in passes.items()}
    seen_passes[key] = wrapped
    return wrapped


ct._third_passes = spy_passes
original = ct.walked_susceptibility_derivative
calls = []


def spy_walk(*args, **kwargs):
    calls.append((args, kwargs))
    t = time.time()
    out = original(*args, **kwargs)
    calls[-1] = calls[-1] + (time.time() - t,)
    return out


ct.walked_susceptibility_derivative = spy_walk

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    calculator = Calculator.from_file(path, pseudo_dir="tests/data/pseudo",
                                      k_batch=kbatch, conv_thr=1e-10, announce=False)
    result = calculator.get_scf()
calculation = calculator.calculation
hosted = dataclasses.replace(result, wavefunctions=np.asarray(result.wavefunctions))
print("nk", hosted.wavefunctions.shape, "k_batch", calculation.k_batch, flush=True)
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
t0 = time.time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res = electrostriction(calculation, hosted)
print(f"electrostriction first call {time.time() - t0:.1f} s, first walk {calls[0][2]:.1f} s", flush=True)
np.save(f"{outdir}/depsilon.npy", np.asarray(res.depsilon_dstrain))
args, kwargs = calls[0][0], calls[0][1]
columns = original(*args, **kwargs)
np.save(f"{outdir}/columns.npy", np.stack([np.asarray(c) for c in columns]))
rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"ru_maxrss MiB: before electrostriction {rss0:.0f}, after the walk {rss1:.0f}", flush=True)

if "global_out" in record:
    g, dg = record["global_out"]
    nbytes = lambda tree: sum(np.asarray(l).nbytes for l in jax.tree_util.tree_leaves(tree))
    ntangent = len(args[5])
    print(f"ntangent {ntangent}; per tangent resident (v, ddd, volume + tangents): "
          f"{(nbytes(g) + nbytes(dg)) / 1e6:.3f} MB; v shape {np.asarray(g[1]).shape}", flush=True)

times = []
for i in range(repeats):
    t = time.time()
    original(*args, **kwargs)
    times.append(time.time() - t)
    print(f"warm walk {i}: {times[-1]:.2f} s", flush=True)
print(f"warm walk median {np.median(times):.2f} s of {len(times)} samples; all {np.round(times, 2).tolist()}", flush=True)
print(f"ru_maxrss MiB end {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}", flush=True)

if count:
    passes = next(iter(seen_passes.values()))
    for name in ("forward", "global", "chunk"):
        a = record[name]
        lowered = passes[name].lower(*a)
        compiled = lowered.compile()
        text = compiled.as_text()
        ma = compiled.memory_analysis()
        ops = re.findall(r"= \S+ ([a-z\-]+)\(", text)
        from collections import Counter
        c = Counter(ops)
        print(f"HLO {name}: instructions {len(ops)}, while {c['while']}, sine {c['sine']}, cosine {c['cosine']}, "
              f"exponential {c['exponential']}, fft {c['fft']}, custom-call {c['custom-call']}, "
              f"dot {c['dot']}, fusion {c['fusion']}", flush=True)
        print(f"  memory_analysis {name}: temp {ma.temp_size_in_bytes / 1e6:.1f} MB, args {ma.argument_size_in_bytes / 1e6:.1f} MB, "
              f"out {ma.output_size_in_bytes / 1e6:.1f} MB", flush=True)
        open(f"{outdir}/hlo_{name}.txt", "w").write(text)
