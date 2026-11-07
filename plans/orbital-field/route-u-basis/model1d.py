"""1D model of route U's basis on the free Landau problem.

At fixed y-momentum the free Landau problem in the Landau gauge is the oscillator
H = p^2 + B^2 x^2 (Ry-like units, levels (2n+1) B), x in cells (a = 1).  graphene at
1/10: B_ph a^2 = 0.0335723 Ry * (4.6487 bohr)^2 = 0.7256, l_B = 1.17 cells.

Bases, each a lattice of functions f(x - X), X integer:
  sinc  : the hybrid Wannier functions of the lowest N free bands (band n is
          |p| in [(n-1) pi, n pi]), i.e. (sin(n pi x) - sin((n-1) pi x)) / (pi x).
  wsinc : the same times a C-inf window (inner, outer) in cells (u1's remedy).
  gauss : Hermite-Gaussians H_j(x/s) exp(-x^2/2 s^2), j < J, band-limited to the lowest N
          free bands (the projection P_N g of candidate (a)); the Loewdin step changes the
          basis inside the span and not the span, so it is left out here.

Matrix elements two ways:
  line : exact on a long line (the span alone).
  box  : the code's rule: functions periodized over N1 cells, bra at cell 0, ket rolled by
         Delta, x the sawtooth in (-N1/2, N1/2] about the bra, blocks |Delta| <= w =
         N1//2 - 1 only.
"""
import sys
import numpy as np

B = 0.7256
M = 64                      # points per cell
Xmax = 14                   # lattice sites -Xmax..Xmax


def grid(ncell):
    n = ncell * M
    x = (np.arange(n) - n // 2) / M
    p = 2 * np.pi * np.fft.fftfreq(n, d=1.0 / M)
    return x, p


def bandlimit(f, p, N):
    return np.fft.ifft(np.fft.fft(f) * (np.abs(p) < N * np.pi))


def window(x, inner, outer):
    ax = np.abs(x)
    t = np.clip((ax - inner) / (outer - inner), 1e-12, 1 - 1e-12)
    s = np.exp(-1 / t) / (np.exp(-1 / t) + np.exp(-1 / (1 - t)))
    out = 1.0 - s
    out[ax <= inner] = 1.0
    out[ax >= outer] = 0.0
    return out


def functions(kind, x, p, N, J, s, win):
    fs = []
    if kind in ("sinc", "wsinc"):
        xs = np.where(x == 0, 1e-300, x)
        for n in range(1, N + 1):
            f = (np.sin(n * np.pi * xs) - np.sin((n - 1) * np.pi * xs)) / (np.pi * xs)
            f[x == 0] = 1.0
            if kind == "wsinc":
                f = f * window(x, *win)
            fs.append(f.astype(complex))
    else:
        from numpy.polynomial.hermite import hermval
        for j in range(J):
            c = np.zeros(j + 1); c[j] = 1.0
            f = hermval(x / s, c) * np.exp(-x ** 2 / (2 * s ** 2))
            fs.append(bandlimit(f.astype(complex), p, N))
    fs = [f / np.sqrt(np.sum(np.abs(f) ** 2) / M) for f in fs]
    return fs


def solve(h, s_mat):
    d, u = np.linalg.eigh(s_mat)
    keep = d > 1e-10 * d.max()
    t = u[:, keep] / np.sqrt(d[keep])
    hh = t.conj().T @ h @ t
    herm = np.max(np.abs(hh - hh.conj().T)) / np.max(np.abs(hh))
    return np.linalg.eigvalsh((hh + hh.conj().T) / 2), d.min() / d.max(), herm


def line(kind, N, J, s, win):
    x, p = grid(2 * Xmax + 12)
    fs = functions(kind, x, p, N, J, s, win)
    basis, pbasis = [], []
    for X in range(-Xmax, Xmax + 1):
        for f in fs:
            g = np.roll(f, X * M)
            basis.append(g)
            pbasis.append(np.fft.ifft(p ** 2 * np.fft.fft(g)))
    A = np.array(basis); PA = np.array(pbasis)
    S = A.conj() @ A.T / M
    H = (A.conj() @ PA.T + B ** 2 * (A.conj() * x ** 2) @ A.T) / M
    return solve(H, S)


def box(kind, N, J, s, win, N1):
    x, p = grid(N1)             # sawtooth about the bra, (-N1/2, N1/2]
    fs = functions(kind, x, p, N, J, s, win)   # periodized by construction on the box
    w = N1 // 2 - 1
    nf = len(fs)
    pf = [np.fft.ifft(p ** 2 * np.fft.fft(f)) for f in fs]
    sites = list(range(-Xmax, Xmax + 1))
    nb = len(sites) * nf
    H = np.zeros((nb, nb), complex); S = np.zeros((nb, nb), complex)
    for i, X in enumerate(sites):
        for k, Xp in enumerate(sites):
            D = Xp - X
            if abs(D) > w:
                continue
            for a in range(nf):
                for b in range(nf):
                    ket = np.roll(fs[b], D * M); pket = np.roll(pf[b], D * M)
                    bra = fs[a].conj()
                    S[i * nf + a, k * nf + b] = np.sum(bra * ket) / M
                    H[i * nf + a, k * nf + b] = (np.sum(bra * pket)
                                                 + B ** 2 * np.sum(bra * (X + x) ** 2 * ket)) / M
    return solve(H, S)


def report(tag, eps, dmin, herm):
    ref = (2 * np.arange(4) + 1) * B
    err = eps[:4] - ref
    print(f"{tag:42s} err(n=0..3)/B {' '.join(f'{e / B:+.2e}' for e in err)}  "
          f"d_min {dmin:.1e} herm {herm:.1e}", flush=True)


if __name__ == "__main__":
    print(f"B {B}, l_B {1 / np.sqrt(B):.3f} cells; errors (E - (2n+1)B)/B, negative = below exact")
    print("--- exact line: the span alone")
    for N in (1, 2, 4):
        report(f"sinc N={N}", *line("sinc", N, 0, 0, None))
    for win in ((2.0, 2.5), (4.0, 5.0), (6.0, 7.5)):
        for N in (2, 4):
            report(f"wsinc N={N} window {win}", *line("wsinc", N, 0, 0, win))
    for s in (0.35, 0.5):
        for N in (4, 8):
            for J in (1, 2, 3, 4):
                report(f"gauss s={s} N={N} J={J}", *line("gauss", N, J, s, None))
    print("--- box rule: the code's matrix elements (periodized, sawtooth, |Delta|<=N1//2-1)")
    for N1 in (10, 20):
        for N in (2, 4):
            report(f"sinc N={N} box N1={N1}", *box("sinc", N, 0, 0, None, N1))
    for N1, win in ((10, (2.0, 2.5)), (20, (4.0, 5.0)), (28, (6.0, 7.5))):
        for N in (2, 4):
            report(f"wsinc N={N} window {win} box N1={N1}", *box("wsinc", N, 0, 0, win, N1))
    for N1 in (10, 20):
        for N in (4, 8):
            for J in (1, 2, 3, 4):
                report(f"gauss s=0.5 N={N} J={J} box N1={N1}", *box("gauss", N, J, 0.5, None, N1))
