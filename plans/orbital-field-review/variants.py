import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, sys
from peierls_torus import a1, a2, tau, hr, dirac_levels, HBAR, E, PHI0, acell

def build(q, p, kfrac, dressing):
    A1, A2 = q * a1, a2
    omega = A1[0] * A2[1] - A1[1] * A2[0]
    b1 = np.array([A2[1], -A2[0]]) / omega
    b2 = np.array([-A1[1], A1[0]]) / omega
    k = 2 * np.pi * (kfrac[0] * b1 + kfrac[1] * b2)
    A = lambda r: 2 * np.pi * p * (b1 @ r) * b2
    f = lambda d: 0.5 * (A(d) @ d)
    rows, cols, vals = [], [], []
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
                    if dressing == "peierls":      # straight line, Landau gauge; Bloch factor -A(P).c_nu
                        phi = -(A(m) @ (r_mu - r_nu)); bs = -(A(P) @ c_nu)
                    elif dressing == "peierls+plan":  # the plan: straight line with +A(c_nu).P
                        phi = -(A(m) @ (r_mu - r_nu)); bs = +(A(c_nu) @ P)
                    elif dressing == "gipaw":        # London/GIPAW midpoint phase; Bloch factor +A(c_nu).P (Wang eq. 4)
                        phi = (A(r_mu) @ m + f(m - r_mu)) - (A(r_nu) @ m + f(m - r_nu)); bs = +(A(c_nu) @ P)
                    elif dressing == "gipaw+mu":     # Wang eq. 9 as printed: row index
                        phi = (A(r_mu) @ m + f(m - r_mu)) - (A(r_nu) @ m + f(m - r_nu)); bs = +(A(r_mu) @ P)
                    rows.append(2*j+s); cols.append(2*jp+t_); vals.append(tv*np.exp(1j*(phi + k@P + bs)))
    return sp.csr_matrix((vals, (rows, cols)), shape=(2*q, 2*q))

def levels(H, nlev):
    if H.shape[0] <= 1600:
        w, v = np.linalg.eigh(H.toarray()); sel = np.argsort(abs(w))[:nlev]; sel = sel[np.argsort(w[sel])]
        return w[sel], v[:, sel]
    w, v = spla.eigsh(H.tocsc(), k=nlev, sigma=0.0137, which="LM"); o = np.argsort(w); return w[o], v[:, o]

if __name__ == "__main__":
    q = int(sys.argv[1]); p = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    B, vF, En = dirac_levels(q, p, 6)
    print(f"q={q} p={p} B={B:.3f} T; Dirac n=0..6: {np.round(En[6:], 4)}")
    nlev = 2 * p * 13
    for d in ["peierls", "peierls+plan", "gipaw", "gipaw+mu"]:
        ws = []; herm = 0
        for kf in [(0, 0), (0.37, 0), (0, 0.29), (0.21, 0.63)]:
            H = build(q, p, kf, d); herm = max(herm, abs(H - H.getH()).max())
            w, v = levels(H, nlev); ws.append(w)
        spread = max(abs(w - ws[0]).max() for w in ws[1:])
        w0 = ws[0]; lev = []
        for x in w0:
            if not lev or abs(x - lev[-1][0]) > 1e-7: lev.append([x, 1])
            else: lev[-1][1] += 1
        pos = [(x, m) for x, m in lev if x > 1e-6][:6]
        print(f"\n[{d}] hermiticity {herm:.1e}  k-spread {spread:.1e} eV")
        print("   zero level multiplicity:", [m for x, m in lev if abs(x) < 1e-6])
        print("   E_n>0 (eV, mult):", [(round(x, 4), m) for x, m in pos])
        print("   E_n/Dirac_n       :", [round(x / En[6 + i + 1], 4) for i, (x, m) in enumerate(pos)])
        if d == "peierls":
            # sublattice weight of the zero-mode subspace: trace of P_A over it (rule D4 invariant)
            z = v[:, abs(w) < 1e-6]; PA = np.zeros(2*q); PA[0::2] = 1
            print("   zero modes: tr P_A =", round(float(np.einsum('ia,i,ia->', z.conj(), PA, z).real), 6),
                  " individual A weights:", np.round((abs(z)**2 * PA[:, None]).sum(0), 4))
