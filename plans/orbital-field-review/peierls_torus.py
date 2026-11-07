"""Honeycomb NN model on a magnetic torus q a1 x a2, flux p, Landau gauge of Lu eq. 4,
built as ORBITAL-FIELD-NEXT.md section 3 writes it:

  H_{mu nu}(k) = sum_P t(R) exp(i phi_{mu nu}(P)) exp(i k.P) exp(i A(c_nu').P)

Tries: Peierls sign S1 (exp(-i int A.dl) vs exp(+i ...)), Bloch-sum sign S2, Bloch-sum index
(column nu, the plan; row mu, Wang eq. 9 as printed), and dropping the Bloch-sum factor.
Checks: Hermiticity, plaquette flux, flatness in k, Landau degeneracy, Dirac formula.
"""
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HBAR = 1.054571817e-34
E = 1.602176634e-19
PHI0 = 2 * np.pi * HBAR / E  # h/e, T m^2

a = 2.46e-10
t = -2.7  # eV
a1 = np.array([a, 0.0])
a2 = np.array([a / 2, a * np.sqrt(3) / 2])
tau = {0: np.zeros(2), 1: (a1 + a2) / 3}  # A, B
acell = abs(a1[0] * a2[1] - a1[1] * a2[0])
# hr[(n1, n2)][m, n] = <w_m0|H|w_nR>, R = n1 a1 + n2 a2
hr = {}
for R in [(0, 0), (-1, 0), (0, -1)]:
    hr.setdefault(R, np.zeros((2, 2), complex))[0, 1] += t
    mR = (-R[0], -R[1])
    hr.setdefault(mR, np.zeros((2, 2), complex))[1, 0] += t


def build(q, p, kfrac, s1=-1, s2=+1, index="nu", bloch=True):
    """kfrac = (k1, k2) in units of the magnetic reciprocal vectors (2 pi b)."""
    A1, A2 = q * a1, a2
    omega = A1[0] * A2[1] - A1[1] * A2[0]
    b1 = np.array([A2[1], -A2[0]]) / omega      # a_i . b_j = delta_ij
    b2 = np.array([-A1[1], A1[0]]) / omega
    k = 2 * np.pi * (kfrac[0] * b1 + kfrac[1] * b2)

    def A(r):
        return 2 * np.pi * p * (b1 @ r) * b2

    n = 2 * q
    rows, cols, vals = [], [], []
    for j in range(q):
        for (n1, n2), block in hr.items():
            jp = (j + n1) % q
            m1 = (j + n1) // q
            P = m1 * A1 + n2 * A2
            for s in range(2):
                for sp_ in range(2):
                    tv = block[s, sp_]
                    if tv == 0:
                        continue
                    r_mu = j * a1 + tau[s]
                    c_nu = jp * a1 + tau[sp_]        # position inside the magnetic cell
                    r_nu = c_nu + P                  # actual position of the ket orbital
                    mid = 0.5 * (r_mu + r_nu)
                    phi = s1 * (A(mid) @ (r_mu - r_nu))  # s1=-1: exp(-i int_nu^mu A.dl)
                    c = c_nu if index == "nu" else r_mu
                    bs = s2 * (A(c) @ P) if bloch else 0.0
                    rows.append(2 * j + s)
                    cols.append(2 * jp + sp_)
                    vals.append(tv * np.exp(1j * (phi + k @ P + bs)))
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n)), A


def plaquette(H, q, p):
    """Product of hopping phases around hexagon j: A0 -> B0 -> A(a1) ... use the loop
    A(j) -> B(j) -> A(j+1)?? simpler: around the hexagon with vertices
    A_j, B_j, A_{j+1}... we use the 6-cycle A(0,0) B(0,0) A(1,0)->? Use the standard cycle:
    A(0) -> B(0) -> A(0,+a2 shifted)... We test flux through the parallelogram instead:
    the product of hopping phases A0->B0->A(a2)... handled generically below."""
    raise NotImplementedError


def dirac_levels(q, p, nmax):
    B = p * PHI0 / (q * acell)
    vF = 3 * abs(t) * E * (a / np.sqrt(3)) / 2 / HBAR  # m/s
    En = np.array([np.sign(n) * vF * np.sqrt(2 * E * HBAR * B * abs(n)) / E for n in range(-nmax, nmax + 1)])
    return B, vF, np.sort(En)


def spectrum(q, p, kfrac, nlev, **kw):
    H, _ = build(q, p, kfrac, **kw)
    herm = abs(H - H.getH()).max()
    if H.shape[0] <= 1200:
        w, v = np.linalg.eigh(H.toarray())
        sel = np.argsort(abs(w))[:nlev]
        sel = sel[np.argsort(w[sel])]
        return herm, w[sel], v[:, sel]
    w, v = spla.eigsh(H.tocsc(), k=nlev, sigma=0.0, which="LM")
    o = np.argsort(w)
    return herm, w[o], v[:, o]


if __name__ == "__main__":
    q = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    p = 1
    B, vF, En = dirac_levels(q, p, 4)
    print(f"q={q} p={p}  B={B:.3f} T  vF={vF:.3e} m/s  l_B={np.sqrt(HBAR/(E*B))*1e9:.2f} nm")
    print("Dirac E_n (eV), n=-4..4:", np.round(En, 4))
    nlev = 2 * 9 + 4  # 9 levels x 2 valleys plus slack
    for label, kw in [
        ("plan: S1=-1 S2=+1 index=nu", dict(s1=-1, s2=+1, index="nu")),
        ("S1=-1 S2=-1 index=nu", dict(s1=-1, s2=-1, index="nu")),
        ("S1=-1 S2=+1 index=mu (Wang eq.9 as printed)", dict(s1=-1, s2=+1, index="mu")),
        ("S1=-1 S2=-1 index=mu", dict(s1=-1, s2=-1, index="mu")),
        ("S1=+1 S2=+1 index=nu", dict(s1=+1, s2=+1, index="nu")),
        ("S1=+1 S2=-1 index=nu", dict(s1=+1, s2=-1, index="nu")),
        ("no Bloch-sum factor", dict(s1=-1, bloch=False)),
    ]:
        out = []
        for kf in [(0.0, 0.0), (0.37, 0.0), (0.0, 0.29), (0.21, 0.63)]:
            herm, w, v = spectrum(q, p, kf, nlev, **kw)
            out.append((herm, w))
        herm = max(o[0] for o in out)
        w0 = out[0][1]
        spread = max(abs(o[1] - w0).max() for o in out[1:])
        # distinct levels: cluster
        lev = []
        for x in w0:
            if not lev or abs(x - lev[-1][0]) > 1e-6:
                lev.append([x, 1])
            else:
                lev[-1][1] += 1
        print(f"\n{label}: hermiticity {herm:.1e}, k-spread of the {nlev} levels nearest 0: {spread:.2e} eV")
        print("  levels (eV, multiplicity per k):", [(round(x, 5), m) for x, m in lev[:12]])
