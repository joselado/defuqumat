"""The field kernels of route U on an analytic Landau state: is the operator right?

The hexagonal a(r) = c1 (b1 . r) b2 has a part along b1, c1 (b1 . b2) x b1hat with x = b1hat . r,
which is the gradient of f = c1 (b1 . b2) x^2 / 2; the rest is B x yhat with B = c1 |b1 x b2|
= B_ph. So exp(-i f) exp(-B x^2 / 2) is the lowest Landau state at zero momentum along a2 (E =
B_ph) and x times it the next one (E = 3 B_ph). They depend on xi1 only, so the box is N1 cells
along a1 with a fine grid there and four points on the other two axes. The kernels
_bra_side/_ket_block of ultracell_field.py are fed these functions directly (every box
point is an entry), with h0 = "box", no window, shift 0, kref 0, both forms, and the field
reversed (scale -1) as the control that must not give B_ph.

Usage: python mag_route_u_landau_check.py [q] [N1] [n1_per_cell]
"""
import sys

import jax.numpy as jnp
import numpy as np

from defuqumat.magnetic.ultracell_field import MagneticUltracell, _bra_side, _ket_block
from defuqumat.system.cell import Cell

q = int(sys.argv[1]) if len(sys.argv) > 1 else 10
n1 = int(sys.argv[2]) if len(sys.argv) > 2 else 14
per_cell = int(sys.argv[3]) if len(sys.argv) > 3 else 48
lattice = np.load("plans/orbital-field/graphene-scf.npz")["lattice"]
cell = Cell.from_vectors(lattice)
cell_grid = (per_cell, 4, 4)
for scale in (1.0, -1.0):
    ultra = MagneticUltracell.build(cell, (1, q), [[]], [0.0], cell_grid, nbnd=1, n1=n1, w=0,
                                    potential=np.zeros(cell_grid), h0="box", scale=scale)
    b = np.linalg.inv(np.asarray(cell.at)).T
    area = float(ultra.magnetic.area)
    c1 = float(ultra.c1)
    b_ph = 2 * np.pi / q / area                    # the field the basis is built for (scale 1)
    xi = np.asarray(ultra.xi)
    x = xi / np.linalg.norm(b[0])                  # bohr along b1hat, centred on cell 0
    f = (2 * np.pi / q) * (b[0] @ b[1]) * x ** 2 / 2   # the gauge of scale = +1
    g0 = np.exp(-1j * f) * np.exp(-b_ph * x ** 2 / 2)
    states = np.stack([g0, x * g0])                # (2, n1_box)
    grid = ultra.grid
    u = np.broadcast_to(states[:, :, None, None], (2,) + grid).astype(complex)
    npts = int(np.prod(grid))
    coeffs = np.fft.fftn(u, axes=(1, 2, 3)).reshape(2, npts) / npts
    index = jnp.arange(npts)
    values = jnp.asarray(coeffs)
    window = jnp.ones_like(ultra.xi)
    kref = jnp.asarray(0.0)
    u_bra, h_bra, p_bra = _bra_side(values, values, index, window, ultra.gg, ultra.g_b2,
                                    ultra.potential, kref, ultra.b22, grid=grid, box_h0=True,
                                    windowed=False)
    for form in ("ket", "symmetric"):
        h, s, _ = _ket_block(u_bra, h_bra, p_bra, values, index, jnp.asarray(0), window, ultra.g_b2,
                          ultra.xi, kref, ultra.c1, ultra.b12, ultra.b22, grid=grid,
                          windowed=False, symmetric=form == "symmetric")
        h, s = np.asarray(h), np.asarray(s)
        e0 = (h[0, 0] / s[0, 0]).real
        e1 = (h[1, 1] / s[1, 1]).real
        off = abs(h[0, 1]) / abs(h[0, 0])
        herm = np.max(np.abs(h - h.conj().T)) / np.max(np.abs(h))
        print(f"q {q} N1 {n1} grid/cell {per_cell} scale {scale:+.0f} form {form:9s}: B_ph {b_ph:.10f} Ry; "
              f"E0/B_ph - 1 = {e0 / b_ph - 1:+.3e}, E1/(3 B_ph) - 1 = {e1 / (3 * b_ph) - 1:+.3e}, "
              f"|H01|/|H00| {off:.2e}, Hermiticity {herm:.2e}, edge weight "
              f"{np.abs(g0[np.abs(xi) > n1 / 2 - 0.5]).max() ** 2:.1e}", flush=True)
