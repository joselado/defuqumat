"""Compare the levels of two route U probe logs (mag_route_u.py output), per nbnd block.

Usage: python mag_route_u_compare.py A.log B.log [nlow]
Prints, for every nbnd present in both, max over the magnetic k of |levels_A - levels_B|
over the lowest nlow levels (default 24), and the largest signed difference A - B.
"""
import re
import sys

import numpy as np


def parse(path):
    out, nbnd, rows, buf = {}, None, [], None
    for line in open(path):
        m = re.match(r"== nbnd (\d+)", line)
        if m:
            if nbnd is not None and rows:
                out[nbnd] = rows
            nbnd, rows = int(m.group(1)), []
            continue
        if line.startswith("k (") and "levels[" in line:
            buf = line.split("Ry", 1)[1]
            if "]" in buf:
                rows.append(np.array(buf.strip(" [\n]").split(), float))
                buf = None
            continue
        if buf is not None:
            buf += line
            if "]" in line:
                rows.append(np.array(buf.replace("[", " ").replace("]", " ").split(), float))
                buf = None
    if nbnd is not None and rows:
        out[nbnd] = rows
    return out


a, b = parse(sys.argv[1]), parse(sys.argv[2])
nlow = int(sys.argv[3]) if len(sys.argv) > 3 else 24
for nbnd in sorted(set(a) & set(b)):
    la = np.array([r[:nlow] for r in a[nbnd]])
    lb = np.array([r[:nlow] for r in b[nbnd]])
    d = la - lb
    print(f"nbnd {nbnd}: lowest {nlow} levels over {len(la)} k: max |A - B| {np.max(np.abs(d)):.3e} Ry, "
          f"max A - B {d.max():.3e}, min A - B {d.min():.3e}")
