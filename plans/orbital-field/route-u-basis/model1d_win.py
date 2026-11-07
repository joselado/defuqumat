import numpy as np
src = open("model1d_gauge.py").read().split("for q in (10, 50):")[0]
ns = {}; exec(src, ns)
# add a window (3,4) about each cell to the basis of run(): monkeypatch via a copy
import types
def run_w(q, J, s, chirp, win, N=8):
    B = 0.544 * 10.0 / q; alpha = np.pi / q
    x, p = ns["ns"]["grid"](2 * ns["ns"]["Xmax"] + 12)
    from numpy.polynomial.hermite import hermval
    M = ns["ns"]["M"]; Xmax = ns["ns"]["Xmax"]
    fs = []
    for j in range(J):
        c = np.zeros(j + 1); c[j] = 1.0
        f = hermval(x / s, c) * np.exp(-x ** 2 / (2 * s ** 2))
        f = ns["ns"]["bandlimit"](f.astype(complex), p, N) * ns["ns"]["window"](x, *win)
        fs.append(f / np.sqrt(np.sum(np.abs(f) ** 2) / M))
    basis = []
    for X in range(-Xmax, Xmax + 1):
        for f in fs:
            g = np.roll(f, X * M)
            if chirp == "cell": g = g * np.exp(-0.5j * alpha * (x - X) ** 2)
            basis.append(g)
    A = np.array(basis)
    def op(g):
        pg = np.fft.ifft(p * np.fft.fft(g)); pig = pg + alpha * x * g
        return np.fft.ifft(p * np.fft.fft(pig)) + alpha * x * pig + B ** 2 * x ** 2 * g
    HA = np.array([op(g) for g in A]); S = A.conj() @ A.T / M; H = A.conj() @ HA.T / M
    eps, dmin, herm = ns["ns"]["solve"](H, S); ref = (2 * np.arange(4) + 1) * B
    return (eps[:4] - ref) / B, dmin
for s in (0.5, 1.356):
    for J in (1, 2):
        err, dmin = run_w(10, J, s, "cell", (3.0, 4.0))
        print(f"q 10 window (3,4) cell chirp s {s} J {J}: {' '.join(f'{e:+.2e}' for e in err)} d_min {dmin:.1e}")
