import numpy as np, sys
sys.argv = ["x"]
import importlib.util
spec = importlib.util.spec_from_file_location("m", "model1d.py"); m = importlib.util.module_from_spec(spec)
src = open("model1d.py").read().split('if __name__ == "__main__":')[0]
exec(src, m.__dict__)
print("--- exact line, wider Gaussians")
for s in (0.7, 1.0):
    for J in (1, 2, 3):
        m.report(f"gauss s={s} N=8 J={J}", *m.line("gauss", 8, J, s, None))
print("--- box N1=10 with window (3,4) on projected Gaussians, s=0.5 and 0.35")
