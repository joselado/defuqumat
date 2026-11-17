import sys, numpy as np, numba
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from defuqumat.io.qe_density import read_rhog
import xml.etree.ElementTree as ET
from pathlib import Path

@numba.njit(cache=True)
def hpsort_eps(n, ra, ind, eps):
    # literal, 1-indexed (index 0 unused)
    if ind[1] == 0:
        for i in range(1, n + 1):
            ind[i] = i
    if n < 2:
        return
    l = n // 2 + 1
    ir = n
    while True:
        if l > 1:
            l = l - 1
            rra = ra[l]; iind = ind[l]
        else:
            rra = ra[ir]; iind = ind[ir]
            ra[ir] = ra[1]; ind[ir] = ind[1]
            ir = ir - 1
            if ir == 1:
                ra[1] = rra; ind[1] = iind
                break
        i = l
        j = l + l
        while j <= ir:
            if j < ir:
                if abs(ra[j] - ra[j + 1]) >= eps:
                    if ra[j] < ra[j + 1]:
                        j = j + 1
                else:
                    if ind[j] < ind[j + 1]:
                        j = j + 1
            if abs(rra - ra[j]) >= eps:
                if rra < ra[j]:
                    ra[i] = ra[j]; ind[i] = ind[j]
                    i = j; j = j + j
                else:
                    j = ir + 1
            else:
                if iind < ind[j]:
                    ra[i] = ra[j]; ind[i] = ind[j]
                    i = j; j = j + j
                else:
                    j = ir + 1
        ra[i] = rra; ind[i] = iind

@numba.njit(cache=True)
def enumerate_g(bg, gcutm, ni, nj, nk, gamma_only):
    # bg: (3,3) with bg[:, a] = b_a, units 2pi/alat (Fortran bg(1:3, a))
    cap = (2*ni+1)*(2*nj+1)*(2*nk+1)
    mill = np.zeros((cap, 3), np.int64)
    g2 = np.zeros(cap + 1)
    ngm = 0
    istart = 0 if gamma_only else -ni
    tt = np.zeros(2*nk+1)
    for i in range(istart, ni + 1):
        jstart = 0 if (gamma_only and i == 0) else -nj
        tx0 = i*bg[0,0]; tx1 = i*bg[1,0]; tx2 = i*bg[2,0]
        for j in range(jstart, nj + 1):
            kstart = 0 if (gamma_only and i == 0 and j == 0) else -nk
            ty0 = tx0 + j*bg[0,1]; ty1 = tx1 + j*bg[1,1]; ty2 = tx2 + j*bg[2,1]
            for k in range(kstart, nk + 1):
                t0 = ty0 + k*bg[0,2]; t1 = ty1 + k*bg[1,2]; t2 = ty2 + k*bg[2,2]
                tt[k-kstart] = t0*t0 + t1*t1 + t2*t2
            for k in range(kstart, nk + 1):
                v = tt[k-kstart]
                if v <= gcutm:
                    ngm += 1
                    g2[ngm] = v if v > 1e-8 else 0.0
                    mill[ngm-1, 0] = i; mill[ngm-1, 1] = j; mill[ngm-1, 2] = k
    return mill[:ngm], g2[:ngm+1].copy(), ngm

def qe_order(at_alat, alat, ecutrho, nr, gamma_only):
    bg = np.linalg.inv(at_alat).T  # rows b_a in 2pi/alat; at_alat rows a_i in alat
    bgF = bg.T.copy()              # bgF[:, a] = b_a
    tpiba = 2*np.pi/alat
    gcutm = ecutrho / tpiba**2
    ni, nj, nk = [(n-1)//2 for n in nr]
    mill, g2, ngm = enumerate_g(bgF, gcutm, ni, nj, nk, gamma_only)
    ind = np.zeros(ngm + 1, np.int64)
    hpsort_eps(ngm, g2, ind, 1e-8)
    return mill[ind[1:] - 1]

for name in sys.argv[1:]:
    d = Path(name)
    gamma, b, miller, rho = read_rhog(d / "charge-density.dat")
    out = ET.parse(d / "data-file-schema.xml").getroot().find("output")
    st = out.find("atomic_structure"); alat = float(st.get("alat"))
    at = np.stack([np.array(st.find(f"cell/a{i}").text.split(), float) for i in (1,2,3)])
    fft = out.find("basis_set/fft_grid")
    nr = [int(fft.get(x)) for x in ("nr1","nr2","nr3")]
    ecutrho = float(out.find("basis_set/ecutrho").text) * 2
    mine = qe_order(at/alat, alat, ecutrho, nr, gamma)
    same = mine.shape == miller.shape and np.array_equal(mine, miller)
    # defuqumat's own lexsort order for comparison
    bg = np.linalg.inv(at/alat).T
    g2 = np.sum((miller @ bg)**2, axis=1)
    lex = miller[np.lexsort((miller[:,2], miller[:,1], miller[:,0], np.round(g2, 12)))]
    print(f"{d.name:14s} gamma={gamma} ngm file {len(miller)} mine {len(mine)} identical={same} "
          f"lexsort-identical={np.array_equal(lex, miller)} b-check={np.abs(b - bg*2*np.pi/alat).max():.1e}")
