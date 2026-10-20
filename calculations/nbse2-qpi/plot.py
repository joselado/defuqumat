"""Plot qpi.py's output: |delta rho(q)| of both readers, one row per case.

Usage:  python3 plot.py <out.npz> <figure.png> [offset index, default 1 = E_F]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

data = np.load(sys.argv[1])
which = int(sys.argv[3]) if len(sys.argv) > 3 else 1
bg = data["bg"]
cases = sorted({k.rsplit("_", 2)[0] for k in data.files if k.endswith("_ldos_q")})
first = data[f"{cases[0]}_ldos_q"]
size = first.shape[-1]
n = int(round(np.sqrt(data["eigenvalues"].shape[1])))
m = size // n
p = np.fft.fftshift(np.fft.fftfreq(size, d=1.0 / size)) / n
f1, f2 = np.meshgrid(p, p, indexing="ij")
q = f1[..., None] * bg[0] + f2[..., None] * bg[1]
# the first zone of the hexagonal lattice: its corners are the K points
corners = np.array([(1, 1), (-1, 2), (-2, 1), (-1, -1), (1, -2), (2, -1)]) / 3.0 @ bg
corners = corners[np.argsort(np.arctan2(corners[:, 1], corners[:, 0]))]
hexagon = np.vstack([corners, corners[:1]])
reach = 1.6 * np.linalg.norm(corners[0])

fig, axes = plt.subplots(len(cases), 2, figsize=(9, 4.2 * len(cases)), squeeze=False)
for row, case in enumerate(cases):
    for col, reader in enumerate(("ldos", "landauer")):
        image = np.abs(data[f"{case}_{reader}_q"][which]).copy()
        image[size // 2, size // 2] = 0.0
        ax = axes[row, col]
        ax.pcolormesh(q[..., 0], q[..., 1], np.log10(image / image.max() + 1e-6),
                      shading="gouraud", cmap="magma", vmin=-3, vmax=0)
        ax.plot(hexagon[:, 0], hexagon[:, 1], "w-", lw=0.7)
        ax.set(xlim=(-reach, reach), ylim=(-reach, reach), aspect="equal")
        wrap = data[f"{case}_wrap"][which, 0]
        ax.set_title(f"{case}, {reader}, wrap {wrap:.3f}", fontsize=9)
fig.suptitle(f"1H-NbSe2, E - E_F = {data['offsets'][which]:+.3f} Ry, log10 |delta rho(q)|")
fig.savefig(sys.argv[2], dpi=110, bbox_inches="tight")
print("wrote", sys.argv[2])
