"""A square sheet of hydrogen with one scatterer on an atom: QPI, two readers."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

sheet = Calculator.from_file("h-sheet.in", pseudo_dir=PSEUDO)
fermi = sheet.get_scf().fermi_energy
qpi = sheet.get_qpi(0.5, 0.7, impurity=0, sigma=1.0, energies=[fermi],
                    readers=("ldos", "landauer"), exit_height=0.3,
                    shape=(4, 4), broadening=0.03, grid=(32, 32, 1))
print(f"T-matrix          {qpi.t_matrix[0, 0]:.4f} Ry")
print(f"wraparound        {qpi.wraparound[0, 0]:.4f}")
print(f"contour k-points  {qpi.contour_points[0, 0]:.0f}")

# the 2k_F peaks along the axes against those along the diagonals
bg = np.asarray(sheet.system.cell.bg)[:2, :2]
n, m = qpi.divisions, qpi.shape
p = [np.fft.fftshift(np.fft.fftfreq(n[i] * m[i], d=1 / (n[i] * m[i]))) / n[i] for i in (0, 1)]
f1, f2 = np.meshgrid(*p, indexing="ij")
q = f1[..., None] * bg[0] + f2[..., None] * bg[1]
size = np.linalg.norm(q, axis=-1)
angle = np.degrees(np.arctan2(q[..., 1], q[..., 0])) % 90
ring = (size > 0.3) & (size < 1.0)
ratio = {}
for reader in ("ldos", "landauer"):
    image = qpi.qpi[reader][0]
    axial = image[ring & ((angle < 5) | (angle > 85))].max()
    diagonal = image[ring & (np.abs(angle - 45) < 5)].max()
    ratio[reader] = axial / diagonal
    print(f"{reader:9s} axial / diagonal 2k_F peak {ratio[reader]:.3f}")

save(ldos=qpi.qpi["ldos"][0], landauer=qpi.qpi["landauer"][0],
     qx=q[..., 0], qy=q[..., 1], bg=bg, t_matrix=qpi.t_matrix,
     wraparound=qpi.wraparound, contour=qpi.contour_points,
     ratio_ldos=ratio["ldos"], ratio_landauer=ratio["landauer"])
