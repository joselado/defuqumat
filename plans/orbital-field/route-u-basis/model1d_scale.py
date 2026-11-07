import numpy as np, sys
src = open("model1d.py").read().split('if __name__ == "__main__":')[0]
ns = {}; exec(src, ns)
ns["B"] = 0.544   # l_B = 1.356 spacings along b1hat (graphene 1/10, spacing a sin 60)
import types
m = types.SimpleNamespace(**ns)
def line(kind, N, J, s, win):
    g = dict(ns); g["B"] = 0.544
    return eval("line", g)(kind, N, J, s, win)
print("B 0.544 (l_B 1.356 spacings); errors/B")
for s in (0.5, 0.7, 1.0, 1.356):
    for J in (1, 2, 3):
        eps, dmin, herm = line("gauss", 8, J, s, None)
        ref = (2*np.arange(4)+1)*0.544
        print(f"gauss s={s} J={J}: {' '.join(f'{e:+.2e}' for e in (eps[:4]-ref)/0.544)}  d_min {dmin:.1e}")
