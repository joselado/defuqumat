import numpy as np
from fd_morb import *
for label, mass in [("Chern (M=0)", 0.0), ("trivial (M = 1.5 x 3 sqrt3 t2 sin phi)", 1.5 * 3 * np.sqrt(3) * t2 * np.sin(phi))]:
    hr = haldane(mass)
    ws = []
    b1 = 2*np.pi*np.array([a2[1], -a2[0]])/acell; b2 = 2*np.pi*np.array([-a1[1], a1[0]])/acell
    for i in range(90):
        for j in range(90):
            ws.append(np.linalg.eigvalsh(bloch(hr, (i/90)*b1 + (j/90)*b2)[0]))
    ws = np.array(ws); ev, ec = ws[:, 0].max(), ws[:, 1].min(); mu = 0.5*(ev+ec)
    mk, C = kspace(hr, mu, nk=300)
    print(f"\n{label}: gap [{ev:.4f},{ec:.4f}] mu_mid={mu:.4f} Chern={C:.3f} k-space M(mu_mid)={mk:.6f} slope -C/2pi={-C/(2*np.pi):.4f}")
    for q, nk in [(25, (3, 3)), (25, (6, 6)), (50, (3, 3)), (100, (3, 3)), (200, (3, 3)), (400, (2, 2))]:
        ep, gp = energy(hr, q, +1, nk); em, gm = energy(hr, q, -1, nk)
        fd = -(ep - em) * acell / (4*np.pi)
        mu_p = 0.5*(gp[:,0].max()+gp[:,1].min()); mu_m = 0.5*(gm[:,0].max()+gm[:,1].min())
        B = 2*np.pi/(q*acell)
        print(f"  q={q:4d} nk={nk} B={B:.5f} FD={fd:+.6f} FD/M={fd/mk:+.5f} (FD-M)*q={(fd-mk)*q:+.3e} (FD-M)*q^2={(fd-mk)*q*q:+.3e} "
              f"mid-gap +B {mu_p:+.4f} -B {mu_m:+.4f}  M(mu_mid(+B))-M(mu_mid)={-C/(2*np.pi)*(mu_p-mu):+.2e}")
