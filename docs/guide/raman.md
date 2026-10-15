# Raman and infrared spectra

In this page we compute the vibrational spectra of a crystal at $\Gamma$, the
lines that a Raman spectrometer and an infrared spectrometer record. We start
with the Raman tensor of each atom, then we project it on the phonon modes to get
the activity of each line, and finally we add the long-range electric field of a
polar crystal, which splits its optical mode into a transverse and a longitudinal
one.

## The Raman tensor

What changes in a crystal when an atom moves? The electrons follow the atom, so
the dielectric tensor changes. The Raman tensor of an atom is this change:

$$
\frac{\partial \varepsilon_{ij}}{\partial \tau_{a,k}} ,
$$

where $\tau_{a,k}$ is the position of atom $a$ along the cartesian direction
$k$. It is a third derivative of the energy, two times with respect to an
electric field and one time with respect to a position, and it is the quantity
from which a Raman intensity is built. We take it as a derivative of the energy
itself, so no displaced cells and no second SCF are needed. Here we take
zincblende AlAs, and we compute it with `calc.get_raman_tensors()`:

```python
import numpy as np
from defuqumat import Calculator

alas = Calculator.from_file("docs/figures/raman_spectrum/alas.in",
                            pseudo_dir="tests/data/pseudo")
raman = alas.get_raman_tensors()
print(raman.raman.shape)                  # (atom, direction of the move, i, j)
print(f"Al moved along x, eps_yz: {raman.raman[0, 0, 1, 2]:+.6f} 1/bohr")
print(f"As moved along x, eps_yz: {raman.raman[1, 0, 1, 2]:+.6f} 1/bohr")
print(f"Al moved along x, eps_xx: {raman.raman[0, 0, 0, 0]:+.1e} 1/bohr")
print(f"sum over the atoms: {raman.sum_rule_relative:.1e} of the largest entry")
```

```
(2, 3, 3, 3)
Al moved along x, eps_yz: -3.137197 1/bohr
As moved along x, eps_yz: +3.138091 1/bohr
Al moved along x, eps_xx: +5.6e-16 1/bohr
sum over the atoms: 2.8e-04 of the largest entry
```

`raman.raman` has one $3\times3$ tensor for each atom and each direction of the
move, in inverse bohr, and `raman.epsilon` is the dielectric tensor it was taken
at. AlAs has the point group $\bar 4 3m$, where a tensor of rank three has only
one independent component, the one with three different indices. So moving Al
along $x$ changes only $\varepsilon_{yz}$, by $-3.137$ per bohr, and every other
entry is zero to $10^{-16}$. The code averages the tensors over the operations
of the crystal, so these zeros are exact by construction. The sum over the atoms
is not imposed by this average. Moving the whole crystal cannot change $\varepsilon$, so the
tensors of the two atoms should add up to zero. Here they cancel to
$2.8\times10^{-4}$ of their size, which is the error of the plane-wave basis. The
code reports this sum and does not force it to zero.

## Modes and their activities

An experiment does not see the tensor of one atom, it sees a mode, which is a
frequency, an intensity and, for Raman, a depolarization ratio. We now project
the Raman tensors and the Born effective charges $Z^*$ on the eigendisplacement
$z^\nu$ of each phonon mode $\nu$:

$$
R^{\nu}_{ij} = \sum_{a,k} \frac{\partial \chi_{ij}}{\partial \tau_{a,k}}\, z^{\nu}_{a,k},
\qquad
p^{\nu}_i = \sum_{a,k} Z^{*}_{a,ik}\, z^{\nu}_{a,k} .
$$

$R^\nu$ is the Raman tensor of the mode, with $\chi = (\varepsilon - 1)/4\pi$,
and $p^\nu$ is its dipole. For a powder only two invariants of $R^\nu$ survive
the average over orientations, the trace $\alpha = \mathrm{tr}\,R^\nu/3$ and the
anisotropy $\beta^2$, and the Raman activity and the depolarization ratio are

$$
I^\nu_{\rm Raman} = 45\alpha^2 + 7\beta^2,
\qquad
\rho^\nu = \frac{3\beta^2}{45\alpha^2 + 4\beta^2},
$$

while the infrared activity is $|p^\nu|^2$. So a mode is Raman active if it
changes the polarizability, and it is infrared active if it carries a dipole.
These are two separate questions. We compute the whole spectrum with
`calc.get_vibrational_spectrum()`, which computes the Raman tensors, the Born
charges and the phonons at $\Gamma$ in one call:

```python
spectrum = alas.get_vibrational_spectrum()
print(spectrum.table())
```

```
# mode   [cm-1]    [THz]      IR          Raman   depol.fact
    1      2.76    0.0827    0.3600         0.0000    0.7500
    2      2.76    0.0827    0.3600         0.0000    0.7500
    3      2.76    0.0827    0.3600         0.0000    0.7500
    4    353.25   10.5902    5.9216       452.3244    0.7500
    5    353.25   10.5902    5.9216       452.3244    0.7500
    6    353.25   10.5902    5.9216       452.3244    0.7500
```

The infrared activity is in (D/Å)$^2$/amu and the Raman activity in
Å$^4$/amu. The same numbers are in `spectrum.frequencies`,
`spectrum.infrared`, `spectrum.raman_activity` and `spectrum.depolarisation`.
The optical triplet of AlAs at 353.25 cm$^{-1}$ is active in both channels, and
its depolarization ratio is 3/4, because its Raman tensor has no trace. The
acoustic triplet should be at zero frequency with no activity. It comes out at
2.76 cm$^{-1}$ with an infrared activity of 0.36, because the acoustic sum rule
and the charge neutrality of $Z^*$ are not imposed, so these numbers measure how
far the basis is from converged. Let us now take silicon:

```python
silicon = Calculator.from_file("docs/figures/raman_spectrum/si.in",
                               pseudo_dir="tests/data/pseudo")
si = silicon.get_vibrational_spectrum()
print(si.table())
```

```
# mode   [cm-1]    [THz]      IR          Raman   depol.fact
    1      5.68    0.1703    2.3509         0.0000    0.5590
    2      5.68    0.1703    2.3509         0.0000    0.6508
    3      5.68    0.1703    2.3509         0.0000    0.6292
    4    519.21   15.5654    0.0000      9750.5394    0.7500
    5    519.21   15.5654    0.0000      9750.5394    0.7500
    6    519.21   15.5654    0.0000      9750.5394    0.7500
```

The optical triplet of silicon is at 519.21 cm$^{-1}$, against 520 cm$^{-1}$
measured, which is the line every Raman spectrometer is calibrated on. It is
Raman active and infrared silent. The reason is symmetry: one operation of the
crystal maps one silicon atom onto the other, so the two atoms have the same
$Z^*$, and the optical mode moves them against each other, so its dipole is zero.
This is why silicon has no one-phonon infrared absorption. The acoustic triplet again
carries the residue of the sum rules, here 5.68 cm$^{-1}$ and an infrared
activity of 2.35, because the two equal $Z^*$ of silicon are not exactly zero on
this basis.

Note the depolarization ratios of the acoustic triplet, 0.5590, 0.6508 and
0.6292. A degenerate multiplet has no preferred basis, so the code can return
any rotation of its three modes, and the number of one mode inside the multiplet
depends on that rotation. The sum over the multiplet does not, because both
invariants are quadratic in $R^\nu$. So when you compare two runs or two codes,
you compare the sums, which `spectrum.by_manifold()` gives:

```python
for freq, raman_sum, ir_sum in si.by_manifold():
    print(f"{freq:7.2f} cm^-1   Raman {raman_sum:10.4f}   IR {ir_sum:.1e}")
```

```
   5.68 cm^-1   Raman     0.0000   IR 7.1e+00
 519.21 cm^-1   Raman 29251.6182   IR 2.0e-30
```

Each row is one multiplet, with its mean frequency and the activities summed over
its modes. The infrared activity of the optical triplet of silicon is
$2\times10^{-30}$, which is zero to round-off.

## LO-TO splitting and the static dielectric constant

We now add the long-range electric field. In AlAs the two atoms carry opposite
charges, so the optical mode that moves them against each other creates a
macroscopic electric field across the crystal. This field costs energy, and how
much depends on the direction $\hat{\mathbf q}$ from which we approach $\Gamma$,
so the dynamical matrix gets a non-analytic term of rank one:

$$
D_{a\alpha,b\beta} \;\to\; D_{a\alpha,b\beta}
  + \frac{4\pi e^2}{\Omega}\,
    \frac{(\hat{\mathbf q}\cdot Z^{*}_a)_\alpha\,(\hat{\mathbf q}\cdot Z^{*}_b)_\beta}
         {\hat{\mathbf q}\cdot\varepsilon^\infty\cdot\hat{\mathbf q}} .
$$

The term pushes up the mode that is longitudinal along $\hat{\mathbf q}$ and
leaves the two transverse ones alone, so a polar crystal has two optical
frequencies at $\Gamma$, $\omega_{\rm TO}$ and $\omega_{\rm LO}$. In silicon
$Z^*$ is zero by symmetry, so the term is zero and there is one line. The two
frequencies are what infrared reflectivity and Raman scattering measure, and the
band between them is where a polar crystal reflects almost all the light. We get it with
`loto_direction`, the direction $\hat{\mathbf q}$ in cartesian coordinates:

```python
lo = alas.get_vibrational_spectrum(loto_direction=(1, 0, 0), neutralize=True)
print(np.round(lo.frequencies, 1))        # cm^-1, LO along x
print(np.round(lo.transverse_frequencies, 1))
to_freq, lo_freq = lo.transverse_frequencies.max(), lo.frequencies.max()
print(f"splitting {lo_freq - to_freq:.1f} cm^-1")
```

```
[  2.8   2.8   2.8 353.3 353.3 391.4]
[  2.8   2.8   2.8 353.3 353.3 353.3]
splitting 38.1 cm^-1
```

With a direction, `lo.frequencies` are the modes with the field and
`lo.transverse_frequencies` are the modes without it. The triplet splits into
two TO modes at 353.3 cm$^{-1}$ and one LO mode at 391.4 cm$^{-1}$, so the
splitting is 38.1 cm$^{-1}$, against 361, 402 and 41 cm$^{-1}$ measured.

`neutralize=True` is important here. A crystal that is moved as a whole creates
no field, so the Born charges must add up to zero, $\sum_a Z^*_a = 0$. A finite
basis misses this a little, and then the cell has a net charge. The non-analytic
term sees this charge and pushes up a longitudinal acoustic mode. We see it if we
leave the charges as they are:

```python
raw = alas.get_vibrational_spectrum(loto_direction=(1, 0, 0))
print(np.round(raw.frequencies, 1))       # without neutralize
```

```
[  2.8   2.8  33.9 353.3 353.3 383.6]
```

One acoustic mode is now at 33.9 cm$^{-1}$ and the LO mode is 7.8 cm$^{-1}$
lower, at 383.6 cm$^{-1}$, which are both wrong. `neutralize=True` shares the
missing charge equally over the atoms before anything else is computed. So for a
polar crystal you should always give it with `loto_direction`.

The same charges also screen a static field. Each polar mode is an oscillator
that the field drives, so the static dielectric constant is the electronic one
plus one term for each mode:

$$
\varepsilon^0_{ij} = \varepsilon^\infty_{ij}
  + \frac{4\pi e^2}{\Omega}\sum_\nu \frac{p^\nu_i\,p^\nu_j}{\omega_\nu^2} ,
$$

with the mode dipole $p^\nu$ from above. $\varepsilon^\infty$ is what an optical
measurement sees, far above the phonon frequencies, and $\varepsilon^0$ is what a
capacitance measures. Together with the two frequencies they obey the
Lyddane-Sachs-Teller relation,
$\varepsilon^0/\varepsilon^\infty = (\omega_{\rm LO}/\omega_{\rm TO})^2$:

```python
eps_inf = lo.epsilon[0, 0]
eps_0 = lo.static_permittivity[0, 0]
print(f"eps_inf {eps_inf:.3f}   eps_0 {eps_0:.3f}   ionic part {lo.ionic_permittivity[0, 0]:.3f}")
print(f"eps_0 / eps_inf   {eps_0 / eps_inf:.5f}")
print(f"(w_LO / w_TO)^2   {(lo_freq / to_freq) ** 2:.5f}")
```

```
eps_inf 13.004   eps_0 15.961   ionic part 2.957
eps_0 / eps_inf   1.22742
(w_LO / w_TO)^2   1.22742
```

`lo.static_permittivity` is $\varepsilon^0$ and `lo.ionic_permittivity` is the
part of it that comes from the polar modes, both as $3\times3$ tensors. The
relation holds to every printed digit. For a crystal with one polar mode this
relation is exact, so here it checks that the LO frequency and $\varepsilon^0$
are built from the same charges and the same constants. It does not tell us that
either number is converged. The two dielectric constants are not converged. This
cell has a cutoff of 10 Ry and a $4\times4\times4$ mesh that contains $\Gamma$, and it
gives 13.00 and 15.96, against 8.16 and 10.06 measured. Their ratio is much
better, 1.227 against 1.233 from the two measured constants, because the error
that pushes one constant up pushes the other up too.

![The Raman and infrared spectra of silicon and AlAs at Gamma, with the TO and LO frequencies of AlAs.](img/raman_spectrum.png#only-light)
![The Raman and infrared spectra of silicon and AlAs at Gamma, with the TO and LO frequencies of AlAs.](img/raman_spectrum-dark.png#only-dark)

The figure shows the two spectra as a spectrometer would record them. Each mode
is a Lorentzian with a half-width of 8 cm$^{-1}$, which stands for the resolution
of the instrument and is not computed, and we leave out the acoustic modes below
20 cm$^{-1}$. In the Raman panel each curve is scaled to its own peak, since the
activity of silicon, 9750.5 per mode, is about 22 times that of AlAs, 452.3. We see one
line in each crystal, silicon at 519 cm$^{-1}$ and AlAs at 353 cm$^{-1}$. In the
infrared panel both crystals are on one scale. AlAs absorbs at its TO frequency,
353 cm$^{-1}$. The dashed line is its LO frequency, 391 cm$^{-1}$, which the
absorption does not show. It is the upper edge of the band between
$\omega_{\rm TO}$ and $\omega_{\rm LO}$ where the crystal reflects the light,
and a reflectivity measurement sees it. Silicon is a flat line at zero, which we lift by 0.02 so that it can be seen. The AlAs spectra of this
figure are computed with `neutralize=True`, and this changes the infrared
activity of the optical mode from 5.92 to 7.58 per mode, so the sum rule of the
charges matters for the intensities as well as for the LO frequency. The figure
takes 62 s on one core.

??? example "Code of this figure (docs/figures/raman_spectrum/)"

    ```python title="compute.py"
    --8<-- "raman_spectrum/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "raman_spectrum/plot.py"
    ```

!!! note "Limits"
    The Raman tensors and the spectra need an insulator. A metal has no
    $\varepsilon^\infty$, so they give an error.

    The phonons and the Raman tensors give an error for two spin channels, for
    spinors (noncollinear magnetism and spin-orbit coupling), for DFT+U, for spin
    spirals and for a potential-only meta-GGA. A ground state converged under a
    magnetic field or with a constrained moment also gives an error, so you run
    the ground state again without them.

    With `nosym`, a shifted k-mesh gives an error, because a shifted mesh does
    not have the symmetry of the crystal. Use an unshifted mesh, as here, or let
    the code use the symmetry.

    The second-order susceptibility $\chi^{(2)}$ and the electro-optic tensor
    give an error. They need one term that the code does not compute, and this
    term is 42 per cent of the answer.

    The LO-TO splitting is a quantity at $\Gamma$. The code does not compute the
    phonon dispersion through it, which needs the non-analytic term at every
    $\mathbf q$ and an interpolation over the zone.

The notebook for this page is
[26, Raman and infrared spectra](https://github.com/joselado/defuqumat/blob/master/notebooks/26_raman_and_infrared_spectra.ipynb).
