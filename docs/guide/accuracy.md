# Accuracy

In this page we show how close the numbers of defuqumat are to those of Quantum
ESPRESSO and of Elk, and how we check the quantities that neither code
computes. This is the one page of the guide with comparisons between codes, and
every agreement in it is the largest difference we measured on the case named
beside it.

## Checking one number yourself

How do we know that a total energy is right? When Quantum ESPRESSO computes the
same quantity, we run the same input file, with the same pseudopotential, in
both codes and compare the numbers. The two codes then solve the same
Kohn-Sham equations in the same plane-wave basis, so they must agree up to the
convergence threshold of the SCF, and a difference larger than that is an error
in one of them. Here we take silicon with one atom moved off its site, so that
there is a force, and compare it with the output of `pw.x` 7.5 for the same
input, which is in the repository:

```python
from defuqumat import Calculator
from defuqumat.io import read_qe_output

calc = Calculator.from_file("docs/figures/accuracy_silicon/si-force.in",
                            pseudo_dir="tests/data/pseudo")
scf = calc.get_scf()
forces = calc.get_forces()
pwx = read_qe_output("tests/data/qe/reference.out.si2-nc-force")

print(f"total energy   {scf.total_energy:.8f} Ry,      pw.x {pwx.total_energy:.8f} Ry")
print(f"force on Si 1  {forces.forces[0, 0]:.8f} Ry/bohr,   pw.x {pwx.forces[0, 0]:.8f} Ry/bohr")
print(f"differences    {abs(scf.total_energy - pwx.total_energy):.1e} Ry and "
      f"{abs(forces.forces[0, 0] - pwx.forces[0, 0]):.1e} Ry/bohr")
```

```
total energy   -15.78740371 Ry,      pw.x -15.78740371 Ry
force on Si 1  0.06039670 Ry/bohr,   pw.x 0.06039673 Ry/bohr
differences    1.4e-09 Ry and 3.3e-08 Ry/bohr
```

The total energies are the same to all eight decimals that `pw.x` prints, and
the difference of 1.4e-9 Ry is at the level of `conv_thr = 1e-10`, which both
runs use. The force differs in the eighth decimal. Note that we obtain the force
as the derivative of the total energy, while `pw.x` sums its terms one by one,
so this is a check of the derivative too. `read_qe_output` reads any `pw.x`
output file, and the files `tests/data/qe/reference.out.*` are the `pw.x` 7.5
outputs of the inputs next to them, so you can repeat this check on the other
cases of the tables below.

The convergence threshold matters for such a comparison. The test-suite of
Quantum ESPRESSO runs at the default `conv_thr = 1e-6`, so its energies are
known only to about 1e-8 Ry, and in that case we cannot see an agreement better
than that. For this reason most of the reference outputs in the repository were
made with `conv_thr = 1e-10` or tighter.

## Against Quantum ESPRESSO

We now list what we compared against Quantum ESPRESSO, which means `pw.x` for
the ground state and its derivatives, `ph.x` for the linear response, and
`projwfc.x` and `dynmat.x` for the post-processing. We start with the total
energy, the quantity every other one is built on:

| total energy | agreement |
|---|---|
| norm-conserving, LDA | about 1e-9 Ry |
| ultrasoft and PAW | at most 3e-9 Ry |
| PBE, revPBE and PBEsol | at most 6e-9 Ry |
| metals, with every smearing and with tetrahedra | 2.5e-8 Ry |
| collinear spin, `nspin = 2` | 1.2e-9 Ry |
| oxygen atom at `tot_magnetization = 2` | 4.9e-9 Ry |
| noncollinear magnetism | 2.8e-9 Ry |
| spin-orbit coupling | at most 1.3e-8 Ry |
| DFT+U | at most 6.7e-9 Ry |
| DFT+U+J on antiferromagnetic FeO, $J = 1$ eV | 4.2e-9 Ry |
| DFT+U on spinors, BN and iron with spin-orbit coupling | 7.7e-9 and 1.4e-9 Ry |
| a slab in a field with the dipole correction, CO and the Ni slab with CO | 4e-10 and 7e-11 Ry |
| a finite field in a crystal by the Berry phase, eight-atom silicon | 1.2e-8 Ry |

These are small cells, with two to eight atoms. We also ran cells of ten atoms,
one for each kind of physics, so that the comparison covers more than one or
two k-points and bands: silicon with the three kinds of pseudopotential,
aluminium with smearing and with tetrahedra, a hydrogen chain with collinear and
noncollinear spin, graphite with Grimme's D2, nickel with DFT+U and bismuth with
spin-orbit coupling. Their total energies agree to between 3.2e-10 and 2.2e-8
Ry, and the bands of the nine cells without spin-orbit coupling agree to between
4.8e-5 and 2.9e-4 eV.

The band energies agree to 0.0002 eV. This is limited by `pw.x`, which prints
four decimals in eV, and which interpolates the local potential from a table
in $q$ where we integrate it directly, which moves the levels by a few tenths
of a meV. Let us now take the derivatives of the energy:

| derivative | agreement |
|---|---|
| forces | at most 2e-5 Ry/bohr |
| forces, noncollinear hydrogen chain | 8.9e-7 Ry/bohr |
| forces, platinum with spin-orbit coupling, ultrasoft and PAW | 7.5e-6 and 7.3e-7 Ry/bohr |
| stress, 13 cases | at most 2.7e-7 Ry/bohr$^3$ |
| stress, spin-orbit cases | at most 1.2e-6 Ry/bohr$^3$ |
| stress, bcc iron with its moment in the plane, PBE | 1.6e-7 Ry/bohr$^3$ (0.019 kbar of 152.71) |
| relaxed geometry | 1e-6 bohr |
| relaxed geometry, ten silicon atoms | 6.8e-6 bohr |
| Berry-phase dipole in a finite field | 8.6e-6 |

The dipole in a finite field agrees better than `pw.x` itself is converged
there, since the value of `pw.x` still moves by 1.6e-5 in its last iteration.
We now come to the linear response, against `ph.x`:

| response | agreement |
|---|---|
| dielectric constant, norm-conserving, ultrasoft and PAW | at most 1.2e-4 |
| Born effective charges, norm-conserving silicon | every digit `ph.x` prints |
| Born effective charges, ultrasoft and PAW silicon | 1.0e-5 and 9e-6 |
| Born effective charges of AlAs | $+2.101065$ and $-2.165827$, against $+2.10106$ and $-2.16581$ |
| dielectric constant and Born charges of triplet O$_2$, `nspin = 2` | $1.11091441$ against $1.110915996$, $Z^*$ $0.13372$ against $0.13367$ |
| phonons at $\Gamma$, silicon, norm-conserving | 0.05 cm$^{-1}$ |
| phonons at $\Gamma$, silicon, ultrasoft and PAW | 0.020 and 0.017 cm$^{-1}$ |
| phonons at $\Gamma$, aluminium, a metal | 0.002 cm$^{-1}$ |
| phonons at $L$ and $X$, silicon | 0.054 and 0.038 cm$^{-1}$ |
| phonons at $\mathbf q = (1/4, 0, 0)$, aluminium | 0.0004 cm$^{-1}$ |
| electron-phonon linewidths and $\lambda$, aluminium | every digit `ph.x` prints |

The phonons of silicon have the same floor of about 0.05 cm$^{-1}$ at every
$\mathbf q$, and its reason is the same table interpolation as for the bands.
The last group is the post-processing:

| post-processing | agreement |
|---|---|
| projections on atomic orbitals, against `projwfc.x`, seven cases | 6.9e-4 |
| Löwdin charges, against `projwfc.x` | 4.7e-5 electrons |
| projections resolved by $j$, platinum with ultrasoft and PAW | 6.3e-6, and the curves to 0.16 per cent of the peak |
| Raman and infrared spectra, LO-TO splitting, against `dynmat.x` | every digit `dynmat.x` prints |
| Berry-phase polarization, `lberry`, AlAs and SiC | string by string, every digit printed |
| orbital magnetization, `lorbm`, an iodine atom | $-1.1744384$ against $-1.1744357$ |

The orbital magnetization differs by 2.7e-6, and this is the difference between
two densities converged separately, since a change of `conv_thr` between 1e-6
and 1e-12 moves our value by 2e-7 to 5e-6.

## Against Elk

Let us now take the quantities that Elk computes and `pw.x` does not, or not in
the same form. Here we cannot expect agreement to many digits, because Elk is an
all-electron code with a LAPW basis, while we use pseudopotentials and plane
waves. So the two codes do not solve the same equations, and a difference of a
few per cent, or of 0.1 to 0.4 eV on a peak position, is what the comparison
can resolve. What the comparison does show is that both codes compute the same
physical quantity, with the same conventions and the same units:

| quantity | case | defuqumat | Elk |
|---|---|---|---|
| magnon energy | fcc nickel, $\mathbf q = (0.1, 0.1, 0)$ | 115.04 meV | 115.93 meV |
| effective mass, $\Gamma_{1v}$ | silicon, PBE | $1.16262\,m_e$ | $1.16238\,m_e$ |
| effective mass, $\Gamma_{2'c}$ | silicon, PBE | $0.170440\,m_e$ | 0.36 per cent apart |
| exciton peak, bootstrap kernel | LiF | 14.05 eV | 13.67 eV |
| SHG resonance position | AlAs | 2.152 eV | 2.163 eV |
| SHG peak height | AlAs | 25.90 a.u. | 25.70 a.u. |
| static dielectric constant, real time against independent particles | silicon | 15.11 | 15.19 |
| nesting, $N(0)/D(E_F)^2$ | aluminium | | 1.5 per cent apart |
| density read from an Elk run | simple-cubic hydrogen | | 4.5e-11 at each of 4096 points |

The magnon of Elk is 117.92 meV in its own example, which applies a small
magnetic field, and 115.93 meV is this value without the 1.99 meV of Zeeman
gap. The magnetic moment of the same run is $0.6178\,\mu_B$ against
$0.6152\,\mu_B$. For the exciton of LiF, the difference of 0.37 eV comes from
the pseudopotential, which has no Li semicore states, and not from the k-mesh,
since a run on the shifted k-mesh of Elk moves the height of the peak and not
its position. The last row is not a comparison of two calculations: we read the
density from the files of an Elk run, and it agrees with the density that Elk
prints on its own grid to the precision of that file.

## Against an identity

What do we compare with when neither code computes a quantity? We check it
against a condition that the right answer must satisfy, and that does not use
the code that computed it. This can be an exact value, a second route to the
same number, or a finite difference of a quantity that is already checked:

| quantity | check | agreement |
|---|---|---|
| Chern number | an integer on any k-mesh | exact |
| $\mathbb Z_2$ invariant | Wannier-centre flow against the parity product | the same integer |
| Berry curvature map | the plaquette flux of Fukui, Hatsugai and Suzuki, AlAs | 1.45e-4 per plaquette |
| velocity operator | finite difference of $H(\mathbf k)$ | 1.8e-9 |
| spin spiral | a four-cell noncollinear supercell, norm-conserving | 3e-12 Ry |
| spin spiral, ultrasoft and PAW | the same supercell, an oxygen chain | 3.5e-12 and 2.1e-11 Ry |
| Dzyaloshinskii-Moriya energy of a spiral | a four-cell supercell, a Ni-I chain | 9e-12 Ry |
| `K_POINTS gamma` storage | an explicit $k = 0$ on the whole sphere | 1e-14 Ry, forces 3e-16 Ry/bohr |
| noncollinear run without spin-orbit coupling | the collinear run of the same cell, hydrogen atom | 3e-12 Ry |
| anisotropy with the coupling switched off | must be zero, tetragonal cobalt | 2e-10 meV |
| strain response | finite difference of converged strained runs | 1.9e-4, 4.6e-4 and 4.7e-4 (NC, US, PAW) |
| spin-polarized $\chi_0$ | finite difference of the density, triplet O$_2$ | 1.1e-6 |
| absorption spectrum, $\epsilon_M(0)$ | the static dielectric constant of the linear-response route, silicon | 1.3e-2 on 22 |
| LO-TO splitting | the Lyddane-Sachs-Teller relation, AlAs | 5.0e-11 |
| angular momenta $\langle L\rangle$, $\langle S\rangle$ | $\langle L\rangle = 0$ without spin-orbit coupling, silicon | 1.7e-16 |
| tunnelling transport | two routes, real space and reciprocal space | 1e-12 |
| ultracell | the unit cell when nothing is modulated, PAW | 8.5e-13 Ry |
| ultracell | a real supercell, spinor PAW silicon | $+6.37\times10^{-7}$ Ry at 96 bands, and falling |
| nesting function | free electrons, $N(q) \propto 1/q$ below $2k_F$ | 1e-4 |

The strongest checks in this table are the ones where a wrong code would give
a different number and not a slightly worse one. A Chern number that is not an
integer, or a spiral that does not reproduce its supercell, says that something
is missing. The absorption spectrum agrees to 1.3e-2 on a value of 22, and this
residue is the finite number of empty bands in the sum over states, which the
linear-response route does not need. The ultracell sits above the supercell by
an amount that falls with the number of bands, from $4.80\times10^{-5}$ Ry at 24
bands to $6.37\times10^{-7}$ Ry at 96, so this is a convergence and not an
error.

## Where the agreement is poor

We now list the comparisons that do not agree well, with the reason when we know
it:

- **The piezoelectric tensor of AlAs** converges slowly with the k-mesh. With
  the stress route, $e_{14}$ is $-0.763786$, $-0.687475$, $-0.672897$ and
  $-0.669907$ C/m$^2$ on the $4\times4\times4$, $6\times6\times6$,
  $8\times8\times8$ and $10\times10\times10$ meshes. The finite difference of
  the Berry-phase polarization, the route of Elk, gives $-0.661386$ and
  $-0.661964$ on the first two. So at $4\times4\times4$ the two routes are 15
  per cent apart, and they come together as the mesh grows.
- **The magnetic structure factors of bcc iron** agree with Elk to 5.8 per cent
  at the first reflection and 17 per cent at the second, and worse at larger
  $|\mathbf H|$. A higher cutoff does not change this, so it is the
  pseudopotential, which smooths the 3d shell where the moment is.
- **The static SHG susceptibility of AlAs** is $-3.21$ a.u. here and $-3.50$ in
  Elk, 8 per cent apart, while the resonance and the peak height agree to 0.5
  and 0.8 per cent.
- **The nesting function of aluminium** differs from Elk by 4.7 per cent in its
  raw value, which is the all-electron Fermi surface of Elk. Normalized by the
  DOS at the Fermi level, it agrees to 1.5 per cent.

In the past one more case was on this list, ten bismuth atoms with spin-orbit
coupling, which was 1.9e-4 Ry from `pw.x`. The difference came from the way a
negative density is treated in the gradient correction, and since the correction
the two codes agree to 2.2e-8 Ry on $-1477.737$ Ry.

!!! note "Limits"
    Every number on this page was computed in double precision. Single precision
    is a mode for speed on a GPU, and we make no claim about its accuracy.

    An agreement can be no better than the convergence of the two runs. A
    reference output made at the default `conv_thr = 1e-6` of `pw.x` gives the
    total energy to about 1e-8 Ry, and its forces and its terms of the energy
    to much less. So when you compare with your own `pw.x` run, you set
    `conv_thr` to 1e-10 or tighter in both codes.

The reference outputs of `pw.x`, `ph.x` and `projwfc.x` are in
[tests/data/qe/](https://github.com/joselado/defuqumat/tree/master/tests/data/qe),
with the input file of each next to it, and the outputs of Elk are in
[tests/data/elk/](https://github.com/joselado/defuqumat/tree/master/tests/data/elk).
