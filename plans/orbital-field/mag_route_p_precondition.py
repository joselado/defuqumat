"""Which preconditioner makes Davidson converge on the magnetic box: u, h, or both averaged."""
import sys, time
import numpy as np
import jax
import jax.numpy as jnp
from defuqumat import Calculator
from defuqumat.units import RY_TO_EV, TPI
from defuqumat.magnetic.blochgrid import MagneticGridHamiltonian, landau_problem, start_block
from defuqumat.solvers.davidson import davidson_eigensolver_all


def g_psi(r, d, e, box):
    x = d.reshape(box)[None] - e.reshape(-1, 1, 1, 1)
    den = 0.5 * (1.0 + x + jnp.sqrt(1.0 + (x - 1.0) ** 2))
    return jnp.fft.ifftn(jnp.fft.fftn(r, axes=(1, 2, 3)) / den, axes=(1, 2, 3))


def true_diagonals(self, ik):
    """<m|pi^2|m> of a box plane wave in the u and in the h representation (twist ignored)."""
    real = self.precision.real
    kappa = self.kpoints[ik]
    p = float(self.p)
    c = [TPI * (jnp.asarray(np.rint(np.fft.fftfreq(n, 1.0 / n)), dtype=real) + kappa[a]) for a, n in enumerate(self.box)]
    shape = [(-1, 1, 1), (1, -1, 1), (1, 1, -1)]
    c = [v.reshape(s) for v, s in zip(c, shape)]
    g = self.metric
    mean, square = jnp.pi * p, (TPI * p) ** 2 / 3.0
    # u: pi_2 = c2 + A2(xi1); h: pi_1 = c1 - 2 pi p xi2 (mean -pi p)
    du = (g[0, 0] * c[0] ** 2 + 2 * g[0, 1] * c[0] * (c[1] + mean) + g[1, 1] * (c[1] ** 2 + 2 * mean * c[1] + square)
          + g[2, 2] * c[2] ** 2 + 2 * g[0, 2] * c[0] * c[2] + 2 * g[1, 2] * (c[1] + mean) * c[2])
    dh = (g[1, 1] * c[1] ** 2 + 2 * g[0, 1] * c[1] * (c[0] - mean) + g[0, 0] * (c[0] ** 2 - 2 * mean * c[0] + square)
          + g[2, 2] * c[2] ** 2 + 2 * g[0, 2] * (c[0] - mean) * c[2] + 2 * g[1, 2] * c[1] * c[2])
    return du.reshape(-1), dh.reshape(-1)


class Variant(MagneticGridHamiltonian):
    def precondition(self, residual, energies, ik, diagonal=None, overlap_diagonal=None):
        mode = self.preconditioner
        if diagonal is None:
            diagonal = self.diagonal(ik)
        box = self.box
        r = residual.reshape((-1,) + box)
        real = self.precision.real
        n1, n2, _ = box
        xi1 = jnp.arange(n1, dtype=real) / n1
        xi2 = jnp.arange(n2, dtype=real) / n2
        ph = self._phase(TPI * self.p * xi1[:, None] * xi2[None, :])[None, :, :, None]
        if mode in ("ud", "uhd"):
            du, dh = true_diagonals(self, ik)
            mv = jnp.mean(self.potential_box)
            pu = g_psi(r, du + mv, energies, box)
            if mode == "ud":
                out = pu
            else:
                out = 0.5 * (pu + jnp.conj(ph) * g_psi(ph * r, dh + mv, energies, box))
        elif mode == "u":
            out = g_psi(r, diagonal, energies, box)
        elif mode == "h":
            out = jnp.conj(ph) * g_psi(ph * r, diagonal, energies, box)
        elif mode == "uh":
            out = 0.5 * (g_psi(r, diagonal, energies, box) + jnp.conj(ph) * g_psi(ph * r, diagonal, energies, box))
        elif mode == "identity":
            out = r
        return out.reshape(residual.shape).astype(residual.dtype)


calc = Calculator.from_keywords(lattice="hexagonal", a=2.46, c=15.0,
    atoms=[("C", (0, 0, 0)), ("C", (1 / 3, 2 / 3, 0))], pseudos={"C": "C.pbe-hgh.UPF"},
    pseudo_dir="tests/data/pseudo", ecut=50 * RY_TO_EV, kmesh=(1, 1, 1))
cell = calc.system.cell
q = int(sys.argv[1]); grid = tuple(int(x) for x in sys.argv[2].split(","))
nbnd = int(sys.argv[3]); david = int(sys.argv[4]); p = int(sys.argv[5]) if len(sys.argv) > 5 else 1
ethr = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-12
modes = sys.argv[8].split(",") if len(sys.argv) > 8 else ("u", "h", "uh")
pot = None
if len(sys.argv) > 6 and sys.argv[6] == "graphene":
    pot = np.load("plans/orbital-field/graphene-scf.npz")["potential"][0]
for mode in modes:
    base = MagneticGridHamiltonian.build(cell, p, q, 50.0, [[0.13, 0.27, 0.0]], pot, grid=grid)
    h = Variant(**{f: getattr(base, f) for f in ("potential_box", "kpoints", "metric", "p", "q", "grid", "ecutwfc",
                                                  "band_batch", "precision")}, preconditioner=mode)
    s = start_block(h, 0, nbnd)
    t0 = time.perf_counter()
    e, v, st, un = davidson_eigensolver_all(h, nbnd, psi0=s[None], ethr=ethr, david=david,
                                            max_iterations=200, indices=np.array([0]), return_steps=True)
    print(mode, "p", p, "q", q, "ethr", ethr, "david", david, "steps", int(st[0]), "unsettled", int(un[0]), f"{time.perf_counter()-t0:.1f}s",
          np.round(np.asarray(e[0])[:6], 10).tolist(), flush=True)
