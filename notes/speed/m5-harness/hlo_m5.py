"""OPEN.md Part III M5 check (2): temp bytes of the compiled ``_every_k`` and its largest buffers.

    DEFUQUMAT_CACHE_DIR=off python3 notes/speed/m5-harness/hlo_m5.py <input> <out.json>

Compiles (never runs) the jitted unit under ``davidson_eigensolver_all`` the way
``tools/gpu/davidson_memory.py`` does, reads ``memory_analysis().temp_size_in_bytes``, and
lists the ten largest array shapes that appear in the compiled HLO text, by bytes, with the
opcode of one instruction producing each (the largest intermediate is read off the text and
not off the einsum string). The preconditioner's candidates are the shapes ending in
``npwx`` x ``nkb`` (or ``nkb_group``): look for ``[npwx, nkb]`` against ``[npwx, nat, nh]``.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

import defuqumat.solvers as solvers
import defuqumat.solvers.davidson as davidson
from defuqumat import Calculator

path, out = sys.argv[1], Path(sys.argv[2])
real = davidson.davidson_eigensolver_all
captured = {}


class _Done(Exception):
    pass


def spy(hamiltonian, bands, psi0=None, ethr=None, **kw):
    unit = davidson._every_k
    lowered = unit.lower(
        hamiltonian, bands, psi0, ethr, None,
        kw.get("david", davidson.DAVID_NDIM),
        kw.get("max_iterations", davidson.MAX_ITERATIONS),
        kw.get("k_batch", "default"), robust=False, return_finite=True,
        return_steps=True,
        **({} if kw.get("indices") is None else {"indices": kw["indices"]}),
    )
    compiled = lowered.compile()
    analysis = compiled.memory_analysis()
    text = compiled.as_text()
    sizes = {"c128": 16, "c64": 8, "f64": 8, "f32": 4, "s32": 4, "s64": 8, "pred": 1}
    seen = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        for dtype, dims in re.findall(r"\b(c128|c64|f64|f32|s32|s64)\[([0-9,]*)\]", line.split("=", 1)[0]):
            shape = tuple(int(d) for d in dims.split(",") if d)
            nbytes = int(np.prod(shape, dtype=np.int64)) * sizes[dtype] if shape else sizes[dtype]
            opcode = line.split("=", 1)[1].strip().split("(")[0].split()[-1] if "(" in line else "?"
            seen.setdefault((dtype, shape), (nbytes, opcode))
    top = sorted(seen.items(), key=lambda item: -item[1][0])[:10]
    captured.update(
        temp_bytes=int(analysis.temp_size_in_bytes),
        argument_bytes=int(analysis.argument_size_in_bytes),
        output_bytes=int(analysis.output_size_in_bytes),
        nk=int(hamiltonian.nk), nbnd=int(bands),
        largest_hlo_shapes=[dict(dtype=k[0], shape=list(k[1]), bytes=v[0], opcode=v[1])
                            for k, v in top],
    )
    raise _Done


solvers.EIGENSOLVERS["davidson"] = spy
solvers.EIGENSOLVERS["david"] = spy
try:
    calculator = Calculator.from_file(path, pseudo_dir="tests/data/pseudo", david=4, k_batch=1)
    calculation = calculator.calculation
    captured["npwx"] = int(calculation.basis.planewaves.npwx)
    calculator.get_scf(max_iterations=1)
except _Done:
    pass
out.write_text(json.dumps(captured, indent=1))
print(json.dumps(captured, indent=1))
