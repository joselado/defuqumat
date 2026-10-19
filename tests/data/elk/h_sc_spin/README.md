# `h_sc_spin`, simple cubic hydrogen with a moment

The spin-polarized counterpart of `../h_sc`: one hydrogen atom at the origin of
a simple cubic cell, stretched to $a = 5.0$ bohr so the one electron polarizes
partially, and run collinear (`spinpol = .true.`, so `ndmag = 1`). It is the
ground truth for reading the magnetic records of `STATE.OUT` and for seeding an
`nspin = 2` run from them (`notes/incompat/I69.md`).

| | |
|---|---|
| Elk | 11.0.2, the D22 build `/l/ladovj1/review/harmonics-elk/bin/elk`, one thread |
| Structure | simple cubic, $a = 5.0$ bohr, one H at the origin, `tshift = .false.` |
| k-mesh | $4\times4\times4$ |
| Smearing | Gaussian (`stype = 0`), `swidth = 0.01` Ha, the 0.02 Ry of `scf.in` |
| Seed field | `bfieldc = (0, 0, 0.05)`, `reducebf = 0.9`, `mixtype = 1` |
| Tasks | 0, 33 (`RHO3D.OUT`) and 73 (`MAG3D.OUT`), `np3d = 16 16 16` |

`elk.in` is the input as run; the lattice constant was chosen from a scan of
4, 5 and 6 bohr. At 4 bohr the moment is $-0.044$, at 6 bohr the run cycles
between $-0.8$ and $-1.0$ for 200 iterations without converging, and at 5 bohr
it converges in 81 iterations to the numbers below. With `reducebf = 0.5` the
field was gone in a dozen iterations and the mixer landed on the unpolarized
fixed point at 5 and 6 bohr. At the last iteration the field is
$0.05 \times 0.9^{81} = 9.8\times10^{-6}$ in Elk's units, so the state is the
spontaneous moment to that accuracy. Its sign, negative, is the one the seed
field picks, since Elk couples it as $+c_b\,\sigma\cdot\mathbf{B}$.

## Numbers this run produces

| | |
|---|---|
| total moment | $-0.3219731543$ |
| muffin-tin moment | $-0.1726933960$ |
| interstitial moment | $-0.1492797583$ |
| muffin-tin charge | $0.4342551995$ |
| interstitial charge | $0.5657448005$ |
| total energy | $-0.48424807526364$ Ha |

`MAG3D.OUT` holds $(x, y, z, m_x, m_y, m_z)$ per point; for a collinear run
`vecplot` puts the moment in the third column and zeros in the other two.

Regenerate with `./regenerate.sh path/to/elk`. It deletes everything Elk writes
that is not committed.
