"""Finite-difference orbital magnetization -(E(+B)-E(-B))/2B on the Peierls torus against
the modern-theory k-space formula, Haldane model, Chern (M=0) and trivial (M large) cases.
Units hbar = e = 1, Phi0 = 2 pi, lengths in units of a, energies in t1.
Moment per cell: FD = -(E_mag(+) - E_mag(-)) A_cell / (4 pi) (p = 1, any q);
k-space: (1/Nk) sum_k sum_{n occ} sum_{m != n} Im[v^x_nm v^y_mn] (E_m + E_n - 2 mu)/(E_n - E_m)^2.
"""
import numpy as np
import sys

a1 = np.array([1.0, 0.0]); a2 = np.array([0.5, np.sqrt(3) / 2])
tau = {0: np.zeros(2), 1: (a1 + a2) / 3}
acell = abs(a1[0] * a2[1] - a1[1] * a2[0])
t1, t2, phi = 1.0, 0.15, 0.3 * np.pi


def haldane(mass):
    hr = {}
    def add(R, m, n, v):
        hr.setdefault(R, np.zeros((2, 2), complex))[m, n] += v
    for R in [(0, 0), (-1, 0), (0, -1)]:
        add(R, 0, 1, t1); add((-R[0], -R[1]), 1, 0, t1)
    add((0, 0), 0, 0, mass); add((0, 0), 1, 1, -mass)
    # NNN: A -> A along +a1, +a2-a1?, ... the three vectors b = a1, a2 - a1, -a2 with phase +phi for A
    for R in [(1, 0), (-1, 1), (0, -1)]:
        add(R, 0, 0, t2 * np.exp(1j * phi)); add((-R[0], -R[1]), 0, 0, t2 * np.exp(-1j * phi))
        add(R, 1, 1, t2 * np.exp(-1j * phi)); add((-R[0], -R[1]), 1, 1, t2 * np.exp(1j * phi))
    return hr


def torus(hr, q, p, kfrac):
    A1, A2 = q * a1, a2
    omega = A1[0] * A2[1] - A1[1] * A2[0]
    b1 = np.array([A2[1], -A2[0]]) / omega; b2 = np.array([-A1[1], A1[0]]) / omega
    k = 2 * np.pi * (kfrac[0] * b1 + kfrac[1] * b2)
    A = lambda r: 2 * np.pi * p * (b1 @ r) * b2
    H = np.zeros((2 * q, 2 * q), complex)
    for j in range(q):
        for (n1, n2), block in hr.items():
            jp, m1 = (j + n1) % q, (j + n1) // q
            P = m1 * A1 + n2 * A2
            for s in range(2):
                for t_ in range(2):
                    tv = block[s, t_]
                    if tv == 0: continue
                    r_mu = j * a1 + tau[s]; c_nu = jp * a1 + tau[t_]; r_nu = c_nu + P
                    m = 0.5 * (r_mu + r_nu)
                    phase = -(A(m) @ (r_mu - r_nu)) - (A(P) @ c_nu) + k @ P
                    H[2 * j + s, 2 * jp + t_] += tv * np.exp(1j * phase)
    return H


def energy(hr, q, p, nk, nocc_per_cell=1):
    e = 0.0; gaps = []
    for i in range(nk[0]):
        for j in range(nk[1]):
            H = torus(hr, q, p, ((i + 0.5) / nk[0], (j + 0.5) / nk[1]))
            assert abs(H - H.conj().T).max() < 1e-12
            w = np.linalg.eigvalsh(H)
            n = nocc_per_cell * q
            e += w[:n].sum(); gaps.append((w[n - 1], w[n]))
    return e / (nk[0] * nk[1]), np.array(gaps)


def bloch(hr, k):
    """H(k) in the atomic gauge (phases carry R + tau_n - tau_m) and its k-derivatives."""
    H = np.zeros((2, 2), complex); dH = np.zeros((2, 2, 2), complex)
    for (n1, n2), block in hr.items():
        R = n1 * a1 + n2 * a2
        for m in range(2):
            for n in range(2):
                d = R + tau[n] - tau[m]
                ph = np.exp(1j * (k @ d))
                H[m, n] += block[m, n] * ph
                dH[:, m, n] += block[m, n] * ph * 1j * d
    return H, dH


def kspace(hr, mu, nk=400):
    b1 = 2 * np.pi * np.array([a2[1], -a2[0]]) / acell; b2 = 2 * np.pi * np.array([-a1[1], a1[0]]) / acell
    tot = 0.0; chern = 0.0
    for i in range(nk):
        for j in range(nk):
            k = (i + 0.5) / nk * b1 + (j + 0.5) / nk * b2
            H, dH = bloch(hr, k)
            w, v = np.linalg.eigh(H)
            vx = v.conj().T @ dH[0] @ v; vy = v.conj().T @ dH[1] @ v
            n, m = 0, 1
            cross = (vx[n, m] * vy[m, n]).imag
            tot += cross * (w[m] + w[n] - 2 * mu) / (w[n] - w[m]) ** 2
            chern += -2 * cross / (w[n] - w[m]) ** 2
    return tot / nk**2, chern / nk**2 * (2 * np.pi) ** 2 / acell / (2 * np.pi)  # Chern = (1/2pi) int Omega d2k


if __name__ == "__main__":
    for label, mass in [("Chern (M=0)", 0.0), ("trivial (M=1.5 x 3 sqrt3 t2)", 1.5 * 3 * np.sqrt(3) * t2)]:
        hr = haldane(mass)
        # zero-field gap edges for mu
        ws = []
        for i in range(60):
            for j in range(60):
                k = (i / 60) * 2 * np.pi * np.array([a2[1], -a2[0]]) / acell + (j / 60) * 2 * np.pi * np.array([-a1[1], a1[0]]) / acell
                ws.append(np.linalg.eigvalsh(bloch(hr, k)[0]))
        ws = np.array(ws); ev, ec = ws[:, 0].max(), ws[:, 1].min(); mu = 0.5 * (ev + ec)
        mk, C = kspace(hr, mu)
        print(f"\n{label}: gap [{ev:.4f}, {ec:.4f}], mu_mid={mu:.4f}, Chern={C:.3f}, "
              f"k-space M(mu_mid)={mk:.6f}, dM/dmu = -C/(2 pi) -> {-C/(2*np.pi):.4f}")
        for q in [25, 50, 100, 200, 400]:
            nk = (3, 3)
            ep, gp = energy(hr, q, +1, nk); em, gm = energy(hr, q, -1, nk)
            fd = -(ep - em) * acell / (4 * np.pi)
            mu_p = 0.5 * (gp[:, 0].max() + gp[:, 1].min()); mu_m = 0.5 * (gm[:, 0].max() + gm[:, 1].min())
            B = 2 * np.pi / (q * acell)
            print(f"  q={q:4d} B={B:.5f}  FD={fd:+.6f}  FD-M={fd - mk:+.2e}  (FD-M)*q={ (fd-mk)*q:+.3e}  "
                  f"(FD-M)*q^2={(fd-mk)*q*q:+.3e}  mid-gap at +B {mu_p:+.4f} at -B {mu_m:+.4f}")
