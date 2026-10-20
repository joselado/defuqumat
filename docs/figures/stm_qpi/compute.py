"""Graphene with one scatterer on a carbon atom: QPI at the tip, two readers."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

graphene = Calculator.from_file("graphene.in", pseudo_dir=PSEUDO)
fermi = graphene.get_scf().fermi_energy
qpi = graphene.get_qpi(1.0, 0.62, impurity=0, sigma=1.0, energies=[fermi + 0.08],
                       readers=("ldos", "landauer"), exit_height=0.38,
                       shape=(4, 4), broadening=0.06, grid=(24, 24, 1), nbnd=12)
print(f"T-matrix          {qpi.t_matrix[0, 0]:.4f} Ry")
print(f"wraparound        {qpi.wraparound[0, 0]:.4f}")
print(f"contour k-points  {qpi.contour_points[0, 0]:.1f}")


def c3_residue(values):
    """max |f - C3 f| / max |f| on the q grid, C3 about the scatterer."""
    n = values.shape[0]
    c = n // 2
    p = np.arange(n) - c
    p1, p2 = np.meshgrid(p, p, indexing="ij")
    turned = np.stack([p1, p2], axis=-1) @ np.array([[-1, 1], [-1, 0]])
    image = values[(turned[..., 0] + c) % n, (turned[..., 1] + c) % n]
    return np.abs(values - image).max() / np.abs(values).max()


for reader in ("ldos", "landauer"):
    print(f"C3 residue {reader:9s}{c3_residue(np.abs(qpi.transforms[reader][0])):.1e}")

save(ldos=np.abs(qpi.transforms["ldos"][0]),
     landauer=np.abs(qpi.transforms["landauer"][0]),
     map_ldos=qpi.maps["ldos"][0], map_landauer=qpi.maps["landauer"][0],
     bg=np.asarray(graphene.system.cell.bg)[:2, :2],
     divisions=np.array(qpi.divisions), shape=np.array(qpi.shape),
     t_matrix=qpi.t_matrix, wraparound=qpi.wraparound,
     contour=qpi.contour_points)
