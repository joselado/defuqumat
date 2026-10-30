# Are the DFT fields compressible as quantics tensor trains?

Started 2026-10-30. The question is whether the objects a plane-wave code carries (the
density, the local potential, the Kohn-Sham states) can be stored and learned as a
quantics tensor train (QTT) with fewer numbers than the plane-wave basis already uses,
and whether tensor cross interpolation (TCI) finds that train from few samples. The TCI
is `qutecipy` (`~/Documents/programs/claude/qutecipy`, commit `3eecc32`, pure NumPy,
`crossinterpolate2`). The scripts and the raw numbers are in
`calculations/qtt-compression/`.

The bottom line so far. In real space every field has a train whose rank saturates as
the grid is refined, the best bit ordering is grouped (all bits of x, then y, then z), and
TCI learns that train with the error asked. Whether the train beats the plane waves is
decided mostly by one thing: whether the cell is a repetition of a smaller one. In bulk silicon
(2, 8 and a perfect 64-atom cell) the plane waves are 8 to 80 times smaller at matched
accuracy, because the sphere of a repeated cell is mostly zero and the plane waves pay
only for the small cell. Breaking the repetition fills the sphere: one vacancy in 64
atoms takes the matched count of the density from 1110 to 40282, and then the gap falls
to 2 to 7 at a tolerance of 1e-4 (the vacancy, the displaced cell and the Al slab), and to
about 1 at 1e-2, where the vacancy potential is the one field whose train is half the
plane-wave count. Storing the perfect crystal as plane waves and the vacancy's change as a
train gives the vacancy potential at 1e-2 in 1448 numbers, 12 times fewer than its plane
waves. For the density that decomposition does not help, because the change a vacancy
makes to it stays at about 2 per cent of the maximum out to the far side of a 64-atom
cell. In G space a train counted in real numbers is no better than the real-space train
for a real field, and TCI there still misses part of the sphere.

## How the fields are put on a quantics grid

Each field is a finite Fourier series on its G sphere, so we evaluate it exactly on a
grid of 2^R points per crystal axis, by placing its sphere coefficients into a
zero-padded 2^R box and doing one inverse FFT (`synth` in `qtt_probe.py`). This is not
an interpolation: the values are the field itself, and R can be raised at no loss. A
grid coordinate is written as R bits per axis, 3R bits in all, and the 3R bits are the
sites of the tensor train, each of dimension 2.

Two trains are built for each field, ordering and tolerance:

- **TT-SVD**: the reference train, from an SVD sweep over the full array truncated at a
  relative Frobenius tolerance (`ttsvd_ranks`). It is the cost function of the ordering
  search. Its measured maximum error, relative to max |f|, lies between 1.0 and 2.2 times
  the tolerance in every case checked (`results/matched_pw_*.json`).
- **TCI**: `crossinterpolate2` with `normalizeerror=True`, started at the maximum of |f|.
  The full train is then contracted and compared with the full array, so the error
  quoted is the measured maximum error relative to max |f|, not TCI's own estimate. The
  sample count is the number of distinct grid points TCI asked for.

Note that the two tolerances are in different norms (Frobenius for TT-SVD, maximum for
TCI), so a rank comparison between the two at the same number is approximate.

The parameter count of a train is the sum of its core sizes, in real numbers for a real
field and in complex numbers for a state. The plane-wave reference is the number of
coefficients needed at the same accuracy, the **matched count**: we keep the largest
|c_G| first and find, by bisection, the smallest number kept whose field has a maximum
error at most the tolerance times max |f| (`matched_pw.py`). For a real field the
coefficients are Hermitian, so its matched count is also roughly its number of real
parameters; for a state both sides count complex numbers.

Two controls go through the same pipeline. White noise on the grid does not compress: the
rank saturates at the matrix bound (64 for fused, 128 for interleaved at R=5) at every
tolerance, and TCI samples the whole grid. A field with random coefficients on the same
G sphere as the density separates what comes from the band limit from what comes from
the physics.

## Cells

| cell | input | atoms | ecutwfc (Ry) | FFT grid | ngm | npw at the k-point |
|---|---|---|---|---|---|---|
| Si, fcc primitive | `benchmarks/si-1k.in` | 2 | 12 | 16^3 | 1459 | 180 |
| Si, simple cubic | `benchmarks/si8-1k.in` | 8 | 12 | 24^3 | 6031 | 738 |

Both are LDA, norm-conserving (`Si.pz-vbc.UPF`), one k-point at (0.25, 0.25, 0.25) in
units of 2pi/a. The state shown is band 1, as its cell-periodic part u_k(r); band 4 of the
2-atom cell behaves the same (`results/orderings_R4-5-6-7_tol1e-04.json`). The potential
is `SCFResult.potential`, the total local potential (local pseudopotential, Hartree and
exchange-correlation); the nonlocal projectors are not in it.

The 8-atom cell is the conventional cubic cell of the same crystal, so it is exactly four
copies of the fcc cell. In reciprocal space this means that only the G vectors with h, k,
l all even or all odd carry weight: 1273 of the 6031 density coefficients are nonzero, the
same 1273 as in the 2-atom cell, and 180 of the 738 coefficients of the state. So the
8-atom cell is a test of the repetition, not of a larger crystal.

## Which bit ordering

Let us ask first which order of the 3R bits gives the smallest train. Five fixed orders
were compared with TT-SVD at tolerance 1e-4 for R=4 to 7, on the 2-atom cell. At R=7
(2^21 grid points):

| field | interleaved xyz, coarse to fine | interleaved, fine to coarse | grouped x, y, z, coarse to fine | grouped, x reversed |
|---|---|---|---|---|
| density | r=89, 54738 | r=88, 53868 | r=46, 15818 | r=46, 15102 |
| potential | r=87, 67726 | r=86, 66334 | r=51, 20112 | r=51, 19206 |
| u_k band 1 | r=72, 39558 | r=71, 38608 | r=28, 6200 | r=28, 6054 |
| random, same sphere | r=203, 235066 | r=201, 232798 | r=89, 43394 | r=89, 42712 |

(r is the maximum bond dimension, the second number the parameter count.) A zig-zag
interleaving (x y z, then z y x) is the same as plain interleaving to within 1 per cent.

The grouped order wins, and its advantage grows with R: for the density the interleaved
train is 1.65 times larger at R=4 and 3.46 times at R=7, for the state 4.1 times at R=4
and 6.4 times at R=7. The reason for this is the Fourier structure of the field. A plane
wave e^{2 pi i h x} is a product over the bits of x, so within one axis the rank across
a cut is at most the number of distinct Miller components h along that axis, and across
the cut between two axes it is at most a count of distinct pairs of components. In the
interleaved order every cut separates coarse from fine in all three directions at once,
and every G vector of the sphere is its own rank-one term, so the bound is the whole
sphere. The measured profile shows this: for the density at R=5 the grouped ranks are
`[2, 4, 8, 13, 13, 25, 40, 41, 24, 13, 11, 7, 4, 2]`, with the peak in the middle of the
y block, where the cut separates (x, coarse y) from (fine y, z).

A greedy search over all pairwise swaps of the 3R sites (`orderings.py ... greedy`),
started from both the interleaved and the grouped order, ends on the grouped order or its
x-reversed twin for the density and band 1 at R=4 and R=5, with the parameter count
within 1.2 per cent of grouped. So among single-bit orderings on these cells, grouped is
the optimum the search finds. The fused scheme (one site of dimension 8 per scale)
merges the three bits of each scale of the interleaved order, so it is cheaper than
interleaved (24192 against 32998 parameters for the density at R=5, 1e-4) but cannot
reach grouped (9668), because its cuts are a subset of interleaved's.

## Rank against grid resolution

A band-limited field should have a rank that stops growing once the grid resolves its
shortest wavelength, and then the parameter count grows only linearly in R. This is what
we see, grouped order, tolerance 1e-4, 2-atom cell:

| R | grid points | density r, params | u_k band 1 r, params | random sphere r, params |
|---|---|---|---|---|
| 4 | 4096 | 38, 5684 | 26, 2196 | 59, 9528 |
| 5 | 32768 | 41, 9668 | 28, 3798 | 60, 17656 |
| 6 | 262144 | 45, 13382 | 28, 5064 | 85, 31540 |
| 7 | 2097152 | 46, 15818 | 28, 6200 | 89, 43394 |

The state is saturated at rank 28 from R=5 on, and the density at about 46. Note that the
compression against this grid, 133x for the density and 338x for the state at R=7, is
not a property of the physics: it comes from the band limit and grows without end with
R, so it says nothing about whether the QTT is a better basis than the plane waves.

## TCI in real space, against the plane waves

TCI with the grouped order, at R=7, both cells. The last column compares the reference
train with the matched plane-wave count:

| cell | field | tol | TT-SVD r, params | TCI r, params | TCI samples | TCI max error | matched plane waves (of all) | TT-SVD / matched |
|---|---|---|---|---|---|---|---|---|
| Si2 | density | 1e-2 | 21, 3860 | 26, 5102 | 15217 | 3.6e-2 | 458 (1459) | 8.4 |
| Si2 | density | 1e-4 | 46, 15818 | 53, 20056 | 43064 | 3.2e-4 | 1061 (1459) | 14.9 |
| Si2 | density | 1e-6 | 72, 33184 | 79, 38278 | 70319 | 2.3e-6 | 1270 (1459) | 26.1 |
| Si2 | u_k band 1 | 1e-4 | 28, 6200 | 31, 6914 | 13487 | 1.5e-4 | 180 (180) | 34.4 |
| Si2 | random sphere | 1e-4 | 89, 43394 | 96, 46664 | 123913 | 3.0e-4 | | |
| Si8 | density | 1e-2 | 30, 7666 | 37, 10240 | 21470 | 4.2e-2 | 518 (6031) | 14.8 |
| Si8 | density | 1e-4 | 59, 23226 | 69, 29462 | 44855 | 2.9e-4 | 1084 (6031) | 21.4 |
| Si8 | density | 1e-6 | 89, 43446 | 104, 51702 | 72961 | 1.6e-6 | 1271 (6031) | 34.2 |
| Si8 | u_k band 1 | 1e-4 | 37, 10264 | 40, 11470 | 17895 | 2.5e-4 | 179 (738) | 57.3 |
| Si8 | random sphere | 1e-4 | 151, 113582 | 163, 125766 | 273693 | 2.7e-4 | | |

R=6 is in `results/tci_si*_grouped.json` and reads the same way. Three things to read in
this table.

TCI finds the train. Its bond dimension is 5 to 25 per cent above the reference one, and
its measured maximum error lands within a factor 5 of the tolerance asked. Its sample
count follows the rank and not the cell: 43000 to 45000 samples for the density at 1e-4
in both cells. That is 2 per cent of the 2^21 grid, but the grid a plane-wave code pays
for is its FFT box, 4096 points for Si2 and 13824 for Si8, which already holds the field
exactly; against it TCI asks for 10 and 3 times more points. So the small sample fraction
is the same band-limit effect as the compression against the grid, not a gain.

The plane waves are smaller at matched accuracy, by a factor 8 to 60. On a small crystal
at this cutoff the G sphere is already the compressed form of the field, so a QTT of it
does not win.

The gap does not close with the cell. From 2 to 8 atoms the matched count of the density
stays at about 1070 and that of the state at 180, because the 8-atom cell carries the
same nonzero coefficients as the 2-atom one, while the train grows by 1.5 and 1.7. The
real-space train pays for the repetition and the plane waves do not: the four copies are
related by the fcc translations (a/2)(1,1,0) and its permutations, which are not a
repetition of the cell along the cubic axes, so the train cannot drop its coarse bits. The
physical density is still 2.7 (Si2) and 4.9 (Si8) times cheaper than a random field on
the same sphere, which is the decay of its Fourier coefficients: most of the sphere falls
under the tolerance, which the matched count shows directly (1061 of 1459 at 1e-4).

## QTT of the plane-wave coefficients

We will now see whether the coefficients themselves compress better than the field. The
coefficient c(h) is placed at index h + N/2 of a box of 2^R per axis, with R the smallest
that holds the sphere (16^3 for both fields of Si2 and the state of Si8, 32^3 for the
Si8 density), and the box is unfolded into bits as in real space (`gspace.py`). The
tolerance of the train is chosen by bisection so that the **field** it gives back has
the asked maximum error, measured on a 2^7 grid. At field tolerance 1e-4, parameters in
complex numbers:

| cell | field | box | nonzero coefficients | TT-SVD interleaved r, params | TT-SVD grouped r, params | TCI grouped, field error |
|---|---|---|---|---|---|---|
| Si2 | density | 4096 | 1273 | 36, 6504 | 30, 4506 | 3.8e-3 |
| Si2 | u_k band 1 | 4096 | 180 | 16, 1576 | 14, 1338 (exact) | 2.2e-1 |
| Si8 | density | 32768 | 1273 | 35, 8080 | 36, 8276 | 2.8e-2 |
| Si8 | u_k band 1 | 4096 | 180 | 20, 1976 | 21, 2508 | 4.7e-3 |

The G-space train looks smaller than the real-space one (4506 against 15818 for the Si2
density), but its 4506 parameters are complex numbers with no pairing between them, so
in real numbers it is 9012, while a real field's coefficients satisfy c(-G) = c(G)* and
`ngm` of them are `ngm` real numbers. For the states both sides are complex. Counted this
way the train is still 7 to 14 times the number of nonzero coefficients,
and for Si2 it has more entries than the 4096-entry box it represents. The repetition
does cost less here: the Si8 density box is 8 times the Si2 one and its train only 1.8
times larger, because the all-even or all-odd rule is a statement about the lowest bit of
each Miller index.

TCI fails in G space. It returns trains of rank 1 to 25 whose field errors are 4e-3 to
0.6, where 1e-4 was asked, and the failure is worst at the loose tolerances: at 1e-2 it
stops at rank 1 to 8 with a field error of 0.36 to 0.61. You can think of the coefficient
box as a sparse tensor with one large peak at G=0; TCI starts there, its pivots never
reach the rest of the sphere, and its own error estimate, made on the pivots it has, does
not see what it missed. The TT-SVD trains, which see the whole box, show that the train
exists; what is missing is a way to find it from samples, perhaps a global pivot search
or initial pivots spread over the sphere.

## Larger cells, on Triton

We will now see whether a cell that is not a repetition changes the picture. Four cells
ran as one array on Triton (`qtt-cells.sbatch`, job 21308707, `batch-csl`, node `csl39`,
8 cores each, 8 to 15 minutes and 1.9 to 2.5 GB each by `seff`), through `qtt_cell.py`,
which does every measurement above in one run and allows a different number of bits per
axis:

- **Si64**: the 2x2x2 conventional cubic cell, a = 20.40 bohr, a power-of-two repetition of
  Si8 along every axis. This is the control.
- **Si63 vacancy**: the same cell with the atom at the origin removed, unrelaxed.
- **Si64 displaced**: every atom moved by a Gaussian random vector, 0.05 A rms per
  component (seed 0). The repetition is broken everywhere, weakly.
- **Al slab**: `benchmarks/al-slab.in`, five layers and vacuum, 5.3 x 5.3 x 31 bohr.

The Si cells are at `ecutwfc=12`, at Gamma, with a 0.1 eV Gaussian smearing (the vacancy
puts states in the gap); their FFT grid is 45^3, `ngm` 47497 and `npw` 6031. The real-space
grid is 2^8 per axis for Si (dR=2; dR=3 would be 2^27 points) and 2^7 x 2^7 x 2^10 for the
slab. The raw files are `results/triton-21308707/`.

At each tolerance: the matched plane-wave count, the best grouped real-space train (rank,
parameters) and its ratio to the plane waves, the G-space train by TT-SVD with its field
error bisected to the target, and its ratio **in real numbers** (twice its parameters for a
real field, as explained in the G-space section), then real-space TCI (samples, measured
maximum error) and the field error of G-space TCI started from the 32 largest
coefficients:

| cell | field | tol | matched plane waves | real-space train r, params | ratio | G-space train params | ratio (real numbers) | real-space TCI samples, error | G-space TCI error |
|---|---|---|---|---|---|---|---|---|---|
| Si64 | density | 1e-02 | 531 | 32, 8668 | 16.3 | 4942 | 18.6 | 26437, 3.6e-02 | 6.2e-02 |
| Si64 | density | 1e-04 | 1110 | 60, 24890 | 22.4 | 9346 | 16.8 | 49348, 3.5e-04 | 3.5e-02 |
| Si64 | potential | 1e-02 | 412 | 23, 4752 | 11.5 | 1804 | 8.8 | 9044, 1.9e-02 | 1.2e-02 |
| Si64 | potential | 1e-04 | 4272 | 63, 34134 | 8.0 | 26892 | 12.6 | 69401, 3.9e-04 | 1.8e-03 |
| Si64 | u_k band 1 | 1e-02 | 134 | 21, 4194 | 31.3 | 1556 | 11.6 | 13457, 2.0e-02 | 2.2e-01 |
| Si64 | u_k band 1 | 1e-04 | 514 | 33, 10510 | 20.4 | 3476 | 6.8 | 42730, 4.9e-04 | 2.0e-01 |
| Si64 | u_k band 128 (top) | 1e-02 | 120 | 22, 4482 | 37.4 | 1562 | 13.0 | 14168, 3.0e-02 | 1.4e-02 |
| Si64 | u_k band 128 (top) | 1e-04 | 179 | 36, 14070 | 78.6 | 4268 | 23.8 | 47548, 4.8e-04 | 1.5e-04 |
| Si63 vacancy | density | 1e-02 | 21955 | 47, 22042 | 1.0 | 22124 | 2.0 | 113320, 4.5e-02 | 5.7e-01 |
| Si63 vacancy | density | 1e-04 | 40282 | 121, 127684 | 3.2 | 78438 | 3.9 | 477484, 3.1e-04 | 4.1e-02 |
| Si63 vacancy | potential | 1e-02 | 17345 | 29, 7978 | 0.5 | 4224 | 0.5 | 15779, 3.8e-02 | 6.9e-01 |
| Si63 vacancy | potential | 1e-04 | 38502 | 117, 124328 | 3.2 | 69522 | 3.6 | 359954, 5.5e-04 | 4.2e-03 |
| Si63 vacancy | u_k band 1 | 1e-02 | 5634 | 31, 10476 | 1.9 | 9424 | 1.7 | 55613, 2.8e-02 | 3.7e-01 |
| Si63 vacancy | u_k band 1 | 1e-04 | 6025 | 98, 73038 | 12.1 | 30018 | 5.0 | 238177, 3.3e-04 | 4.9e-03 |
| Si63 vacancy | u_k band 125 (top) | 1e-02 | 4789 | 60, 30892 | 6.5 | 12360 | 2.6 | 101884, 3.4e-02 | 6.4e-02 |
| Si63 vacancy | u_k band 125 (top) | 1e-04 | 5975 | 112, 88280 | 14.8 | 29072 | 4.9 | 232675, 2.4e-04 | 5.9e-16 |
| Si64 displaced | density | 1e-02 | 12170 | 73, 45298 | 3.7 | 35944 | 5.9 | 177503, 3.5e-02 | 2.4e-01 |
| Si64 displaced | density | 1e-04 | 37431 | 176, 225736 | 6.0 | 111816 | 6.0 | 778437, 3.3e-04 | 7.4e-03 |
| Si64 displaced | potential | 1e-02 | 12699 | 63, 40158 | 3.2 | 21250 | 3.3 | 78802, 3.2e-02 | 1.1e-01 |
| Si64 displaced | potential | 1e-04 | 36614 | 186, 244062 | 6.7 | 109112 | 6.0 | 601998, 4.6e-04 | 1.0e-03 |
| Si64 displaced | u_k band 1 | 1e-02 | 4573 | 57, 30340 | 6.6 | 17838 | 3.9 | 128740, 4.3e-02 | 2.4e-01 |
| Si64 displaced | u_k band 1 | 1e-04 | 6016 | 116, 90656 | 15.1 | 32094 | 5.3 | 247196, 2.5e-04 | 1.0e-15 |
| Si64 displaced | u_k band 128 (top) | 1e-02 | 3156 | 58, 27222 | 8.6 | 11104 | 3.5 | 62932, 4.5e-02 | 4.4e-02 |
| Si64 displaced | u_k band 128 (top) | 1e-04 | 5946 | 110, 85020 | 14.3 | 28226 | 4.7 | 271799, 2.8e-04 | 6.7e-16 |
| Al slab | density | 1e-02 | 1931 | 14, 2432 | 1.3 | 1750 | 1.8 | 7977, 3.8e-02 | 1.2e-01 |
| Al slab | density | 1e-04 | 3680 | 28, 7446 | 2.0 | 3768 | 2.0 | 20190, 2.7e-04 | 2.6e-03 |
| Al slab | potential | 1e-02 | 2999 | 15, 2954 | 1.0 | 1754 | 1.2 | 9610, 3.6e-02 | 2.9e-01 |
| Al slab | potential | 1e-04 | 4676 | 41, 12272 | 2.6 | 5538 | 2.4 | 24847, 3.0e-04 | 1.6e-03 |
| Al slab | u_k band 1 | 1e-02 | 423 | 10, 1324 | 3.1 | 590 | 1.4 | 4356, 1.9e-02 | 4.2e-01 |
| Al slab | u_k band 1 | 1e-04 | 594 | 15, 2966 | 5.0 | 1050 | 1.8 | 7649, 2.2e-04 | 2.2e-04 |
| Al slab | u_k band 6 (top) | 1e-02 | 402 | 10, 1358 | 3.4 | 710 | 1.8 | 3764, 2.6e-02 | 9.1e-02 |
| Al slab | u_k band 6 (top) | 1e-04 | 567 | 15, 2938 | 5.2 | 1050 | 1.9 | 6012, 1.2e-04 | 1.7e-15 |

Five things to read in this table.

The repetition is free for the train and free for the plane waves. Si64 has the same
matched density count as Si2 and Si8 (1110 against 1061 and 1084), and its real-space
train (24890) is close to that of Si8 (23226): the extra coarse bits cost almost nothing.
So the perfect crystal gives the same gap as before, 8 to 80.

Breaking the repetition fills the sphere, and the train grows less than the plane waves.
The vacancy takes the matched density count from 1110 to 40282, a factor 36, and the
train from 24890 to 127684, a factor 5. The gap at 1e-4 is then 3.2 for the vacancy, 6.0
for the displaced cell and 2.0 for the slab. The displaced cell is the harder one for the
train (density rank 176 against 121) while its matched count is about the same, meaning
that a weak disorder everywhere costs the train more than one strong local defect.

At 1e-2 the two are level, and the vacancy potential is a win. Its train has 7978 numbers
in real space and 8448 in G space against 17345 plane waves. One caveat on the real-space
number: its tolerance is in the Frobenius norm, and on Si2 that truncation gave a maximum
error 1.0 to 2.2 times the tolerance, so the match is approximate; the G-space number is
bisected on the maximum error of the field and has no such caveat. The vacancy density at
1e-2 is level (22042 against 21955), and the slab density and potential are within 1.3.

The states compress worse than the fields built from them. At 1e-4 a single u_k costs 5 to
15 times its plane waves in real space and 2 to 5 in G space in the three cells that are
not repetitions (20 to 80 and 7 to 24 in Si64), more than the fields built from them. A G-space train of a state is complex on both sides, so here G
space does help, by a factor 2 to 3.

TCI in real space works at this size and TCI in G space does not. Real-space TCI meets the
tolerance within a factor 5 in every row, from 3764 to 778437 samples, which is 0.02 to 5
per cent of the 2^24 grid and up to 9 times the 45^3 FFT box. Starting G-space TCI from
the 32 largest coefficients helps sometimes (Si64 potential at 1e-2: 0.92 to 0.012) and
not reliably: the vacancy density at 1e-4 still has a field error of 4e-2.

## The change a vacancy makes

The vacancy result suggests a decomposition: the perfect crystal as plane waves of the
small cell (1110 coefficients for the density), plus the change rho_vac - rho_perfect as a
train, if the change is local. Both 64-atom cells have the same G set, so the change is
exact on the grid (`diff_field.py`). It is compressed to an error of tol times max
rho_vac, the target the whole field had, at dR=1 (2^21 points):

| field | tol | change as a train r, params | change in plane waves | whole field, train at dR=1 | whole field, plane waves |
|---|---|---|---|---|---|
| density | 1e-2 | 56, 26244 | 21539 | 47, 19336 | 21955 |
| density | 1e-4 | 144, 133900 | 40181 | 119, 107100 | 40282 |
| potential | 1e-2 | 8, 1036 | 16725 | 28, 6618 | 17345 |
| potential | 1e-4 | 96, 66270 | 38745 | 114, 101906 | 38502 |

So the change costs as much as the whole field, except for the potential at 1e-2, where it
is a rank-8 train of 1036 numbers. There the decomposition works: the perfect crystal's
potential at 1e-2 is 412 plane waves (the Si64 row above), so the vacancy potential is
412 + 1036 = 1448 numbers, against 17345 plane waves and 6618 for a train of the whole
field, a factor 12 below the plane waves. Note that the two parts each carry an error of
1e-2 of their own maximum, so the total error is up to about 2e-2 of max |V|; this has not
been measured on the sum. For the density the decomposition does not help at any tolerance
measured. The reason for this is that the change is not local in
this cell. Measured from the vacant site, max |drho| / max rho is 0.81 within 2 bohr and
0.71 from 2 to 4 bohr, then falls to 2 to 4 per cent from 4 to 10 bohr, and stays at 1.5
to 1.8 per cent out to the farthest point of the cell (14 to 18 bohr); the change
integrates to -4.00 electrons, as it should. The potential is the field whose
tail falls under 1e-2 here, and the density is not. In a 64-atom cell at Gamma, the vacancy
disturbs the density of the whole cell at the per cent level, and a tolerance of 1e-2 or
tighter has to resolve that everywhere. The decomposition needs a cell where this tail
falls under the tolerance, which is a question about the screening and the k-point
sampling, not about the train.

## What this does not yet say

- The cells are unrelaxed and sampled at Gamma. A relaxed vacancy, a denser k-mesh and a
  larger cell would change how far the change reaches, and that is what decides whether
  the decomposition can work.
- The tolerance is a relative maximum error on the field, not an energy. What 1e-4 on the
  density costs in the total energy (through the Hartree and exchange-correlation terms)
  has not been computed.
- All cells use `ecutwfc=12`, a low cutoff. A higher cutoff adds coefficients that are
  small, which the matched count already discounts; this has not been checked.
- Only one k-point and one band at a time. The occupied manifold as one array-valued
  train with shared pivots (`crossinterpolate2_array`) has not been tried.
- Only single-bit orderings were searched, from two starts on Si2; on the larger cells
  only the six grouped axis orders and interleaved were compared. The cell axes are the
  crystal axes.
- The real-space trains are truncated in the Frobenius norm and compared with plane waves
  matched in the maximum norm, which is an approximate match (factor 1.0 to 2.2 on Si2).

## Next steps, roughly in order of what they decide

- A 216-atom cell (3x3x3) or 512-atom cell (4x4x4) with a vacancy, to see whether the tail
  of the change falls under 1e-2 far from the defect, and whether the gap keeps closing
  with the cell. The 4x4x4 cell is a power-of-two repetition of Si8, which keeps the
  perfect reference cheap for the train.
- The real-space trains truncated to a maximum error (bisection, as in G space), to remove
  the norm caveat from the 1e-2 comparison.
- The potential, which compresses best, at tolerances between 1e-2 and 1e-4, to find
  where the crossover lies.
- The Hartree energy of the compressed density against the exact one, to turn the
  tolerance into an energy error.
- A G-space TCI that finds the whole sphere: the global pivot finder, or pivots from a
  coarse plane-wave solution.
- Array-valued TCI over the occupied bands at one k-point.

## How to rerun

From the repository root, with `qutecipy` at `~/Documents/programs/claude/qutecipy` (or
`QUTECIPY_PATH` set). `QTT_INPUT` selects the cell for every script, default
`benchmarks/si-1k.in`:

```bash
# orderings, TT-SVD only, R = 4..7, tolerance 1e-4 (about 2 minutes)
python3 calculations/qtt-compression/orderings.py 4,5,6,7 1e-4
# greedy ordering search at R = 4, 5
python3 calculations/qtt-compression/orderings.py 4,5 1e-4 greedy
# real-space TCI with the grouped order
QTT_INPUT=benchmarks/si8-1k.in python3 calculations/qtt-compression/tci_grouped.py 6,7 out.json
# matched plane-wave counts and the TT-SVD maximum error, at R = 7
python3 calculations/qtt-compression/matched_pw.py 7 out.json
# QTT and TCI of the coefficient box
python3 calculations/qtt-compression/gspace.py out.json
```

The larger cells, from the repository root (`qtt_cell.py --help`; on Triton through
`qtt-cells.sbatch`, whose header has the submit line):

```bash
python3 calculations/qtt-compression/qtt_cell.py si63-vac out/ --dR 1,2 --tci-dR 2
python3 calculations/qtt-compression/diff_field.py out/ si63-vac si64 out/diff.json --dR 1
```

Each run of the older scripts does its own SCF (a few seconds for these cells), and every TCI call above takes
under 15 seconds on four cores. `qtt_probe.py` alone runs the first comparison, all five
fields with interleaved and fused orders at R=5 (`results/si_R45.json`).
