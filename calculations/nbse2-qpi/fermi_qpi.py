"""The Fermi surface beside the QPI map, on the same axes in 1/bohr.

The Fermi surface is the spectral weight at E, sum_n eta/pi / ((E - e_nk)^2 + eta^2),
from the eigenvalues qpi.py stored on the whole N x N grid, tiled over the
neighbouring zones. The QPI panels are |delta rho(q)| of one impurity site.

Usage:  python3 fermi_qpi.py <input.in> <out.npz> <figure.png> [site] [eta] [offset index]
                             [log | linear]
"""
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from defuqumat import Calculator
from defuqumat.workflows.transport import whole_grid

source, npz, figure = sys.argv[1:4]
site = sys.argv[4] if len(sys.argv) > 4 else "Nb"
eta = float(sys.argv[5]) if len(sys.argv) > 5 else 0.005
which = int(sys.argv[6]) if len(sys.argv) > 6 else 1
scale = sys.argv[7] if len(sys.argv) > 7 else "log"

data = np.load(npz)
bg = data["bg"]
eigenvalues = data["eigenvalues"][0]                       # (nk, nbnd), Ry
n = int(round(np.sqrt(eigenvalues.shape[0])))
energy = data["energies"][which]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    system = Calculator.from_file(source, pseudo_dir="tests/data/pseudo",
                                  announce=False).system
crystal = np.asarray(whole_grid(system, (n, n, 1)).crystal(system.cell))[:, :2]
index = np.rint(np.mod(crystal, 1.0) * n).astype(int) % n
weight = (eta / np.pi / ((energy - eigenvalues) ** 2 + eta ** 2)).sum(axis=1)
grid = np.zeros((n, n))
grid[index[:, 0], index[:, 1]] = weight

# three zones each way, so the pockets around every K and Gamma of the window show
tiles = 3
steps = np.arange(tiles * n) - (tiles * n) // 2           # crystal coordinate x n
tiled = grid[np.ix_(steps % n, steps % n)]
f1, f2 = np.meshgrid(steps / n, steps / n, indexing="ij")
k = f1[..., None] * bg[0] + f2[..., None] * bg[1]

corners = np.array([(1, 1), (-1, 2), (-2, 1), (-1, -1), (1, -2), (2, -1)]) / 3.0 @ bg
corners = corners[np.argsort(np.arctan2(corners[:, 1], corners[:, 0]))]
hexagon = np.vstack([corners, corners[:1]])
reach = 1.6 * np.linalg.norm(corners[0])

size = data[f"{site}_eta{eta:g}_ldos_q"].shape[-1]
p = np.fft.fftshift(np.fft.fftfreq(size, d=1.0 / size)) / n
p1, p2 = np.meshgrid(p, p, indexing="ij")
q = p1[..., None] * bg[0] + p2[..., None] * bg[1]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
axes[0].pcolormesh(k[..., 0], k[..., 1], tiled, shading="gouraud", cmap="magma",
                   vmin=0, vmax=np.percentile(tiled, 99.5))
axes[0].set_title(f"Fermi surface: A(k, E), eta = {eta:g} Ry")
for ax, reader, title in zip(axes[1:], ("ldos", "landauer"),
                             ("QPI, LDOS at the tip", "QPI, tip to exit plane")):
    image = np.abs(data[f"{site}_eta{eta:g}_{reader}_q"][which]).copy()
    image[size // 2, size // 2] = 0.0
    if scale == "log":
        ax.pcolormesh(q[..., 0], q[..., 1], np.log10(image / image.max() + 1e-6),
                      shading="gouraud", cmap="magma", vmin=-3, vmax=0)
        ax.set_title(f"{title}, log10 |delta rho(q)|")
    else:
        ax.pcolormesh(q[..., 0], q[..., 1], image / image.max(),
                      shading="gouraud", cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"{title}, |delta rho(q)| / max")
for ax in axes:
    ax.plot(hexagon[:, 0], hexagon[:, 1], "w-", lw=0.7)
    ax.set(xlim=(-reach, reach), ylim=(-reach, reach), aspect="equal",
           xlabel="1/bohr")
axes[0].set_ylabel("1/bohr")
layers = list(system.structure.symbols).count("Nb")
fig.suptitle(f"NbSe2, {layers} layer{'s' if layers > 1 else ''}, impurity on "
             f"{site}, E - E_F = {data['offsets'][which]:+.3f} Ry "
             f"(left: k, right: q, same axes)")
fig.savefig(figure, dpi=110, bbox_inches="tight")
print("wrote", figure)
