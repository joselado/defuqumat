"""1D model with the hexagonal cell's longitudinal gauge term: H = (p + alpha x)^2 + B^2 x^2,
alpha = 2 fcoef = pi p/q (per cell^2), B = 0.544 (q=10) scaled as 10/q; spacing = a sin60.
Basis: real Hermite-Gaussians (J per cell) or the same with a per-cell chirp exp(-i alpha (x-X)^2/2)
or with the global chirp exp(-i alpha x^2/2); exact line."""
import numpy as np, sys
src = open("model1d.py").read().split('if __name__ == "__main__":')[0]
ns = {}; exec(src, ns)
grid, bandlimit = ns["grid"], ns["bandlimit"]
from numpy.polynomial.hermite import hermval
M, Xmax = ns["M"], ns["Xmax"]

def run(q, J, s, chirp, N=8):
    B = 0.544 * 10.0 / q
    alpha = np.pi / q          # 2 fcoef, p = 1
    x, p = grid(2 * Xmax + 12)
    fs = []
    for j in range(J):
        c = np.zeros(j + 1); c[j] = 1.0
        f = hermval(x / s, c) * np.exp(-x ** 2 / (2 * s ** 2))
        f = bandlimit(f.astype(complex), p, N)
        fs.append(f / np.sqrt(np.sum(np.abs(f) ** 2) / M))
    basis = []
    for X in range(-Xmax, Xmax + 1):
        for f in fs:
            g = np.roll(f, X * M)
            if chirp == "cell":
                g = g * np.exp(-0.5j * alpha * (x - X) ** 2)
            elif chirp == "global":
                g = g * np.exp(-0.5j * alpha * x ** 2)
            basis.append(g)
    A = np.array(basis)
    def op(g):   # (p + alpha x)^2 g + B^2 x^2 g
        pg = np.fft.ifft(p * np.fft.fft(g))
        pig = pg + alpha * x * g
        return np.fft.ifft(p * np.fft.fft(pig)) + alpha * x * pig + B ** 2 * x ** 2 * g
    HA = np.array([op(g) for g in A])
    S = A.conj() @ A.T / M; H = A.conj() @ HA.T / M
    eps, dmin, herm = ns["solve"](H, S)
    ref = (2 * np.arange(4) + 1) * B
    return (eps[:4] - ref) / B, dmin, herm

for q in (10, 50):
    for chirp in (None, "cell", "global"):
        for s in (0.5, 1.356 * np.sqrt(q / 10)):
            for J in (1, 2, 3):
                err, dmin, herm = run(q, J, s, chirp)
                print(f"q {q:3d} chirp {str(chirp):6s} s {s:.3f} J {J}: {' '.join(f'{e:+.2e}' for e in err)}  d_min {dmin:.1e} herm {herm:.1e}", flush=True)
