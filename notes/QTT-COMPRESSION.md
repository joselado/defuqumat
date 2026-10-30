# Are the DFT fields compressible as quantics tensor trains?

Started 2026-10-30. The question is whether the objects a plane-wave code carries (the
density, the local potential, the Kohn-Sham states) can be stored and learned as a
quantics tensor train (QTT) with fewer numbers than the plane-wave basis already uses,
and whether tensor cross interpolation (TCI) finds that train from few samples. The TCI
is `qutecipy` (`~/Documents/programs/claude/qutecipy`, commit `3eecc32`, pure NumPy,
`crossinterpolate2`). The scripts and the raw numbers are in
`calculations/qtt-compression/`.

The bottom line so far, on two small silicon cells. In real space every field has a
train whose rank saturates as the grid is refined, the best bit ordering is grouped (all
bits of x, then y, then z), and TCI learns that train with the error asked. But at
matched accuracy the plane waves are 15 to 60 times smaller than the train, and the gap
does not close from the 2-atom to the 8-atom cell, because the 8-atom cubic cell is four
copies of the 2-atom fcc cell. In G space, a train of the coefficient box is exact but
larger than the box itself at these sizes, and TCI fails there outright. Nothing here
compresses better than the plane waves yet; the open question is a large cell that is
not a repetition of a small one.

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

The G-space train is smaller than the real-space one (4506 complex against 15818 real
for the Si2 density), but it is still 3.5 to 14 times the number of nonzero coefficients,
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

## What this does not yet say

- The 8-atom cell is a repetition, so the cell-size trend is not measured yet. The
  meaningful larger cell has something non-periodic in it: a vacancy, a slab with
  vacuum, a moire.
- The tolerance is a relative maximum error on the field, not an energy. What 1e-4 on the
  density costs in the total energy (through the Hartree and exchange-correlation terms)
  has not been computed.
- Both cells use `ecutwfc=12`, a low cutoff. A higher cutoff adds coefficients that are
  small, which the matched count already discounts, so the plane-wave side should not
  lose much from it; this has not been checked.
- Only one k-point and one band at a time. The occupied manifold as one array-valued
  train with shared pivots (`crossinterpolate2_array`) is the natural object for a
  density built from states, and has not been tried.
- Only single-bit orderings were searched, from two starts. The cell axes are the crystal
  axes; for the fcc primitive cell they are not orthogonal.

## Next steps, roughly in order of what they decide

- A 64-atom silicon cell with one vacancy (from `benchmarks/si64-1k.in`): the first
  cell where the matched count grows with the cell, so the first real test of the trend.
- `benchmarks/al-slab.in`: a slab with vacuum, where the density is exponentially small
  over a large part of the cell, and where the relative tolerance has to be checked in
  the vacuum separately.
- TCI in G space with initial pivots spread over the sphere, or with the global pivot
  finder, to see whether the trains TT-SVD finds can be learned from samples.
- `benchmarks/si8-1k-ecut30.in`: the cutoff dependence at fixed cell.
- The Hartree energy of the compressed density against the exact one, to turn the
  tolerance into an energy error.
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

Each run does its own SCF (a few seconds for these cells), and every TCI call above takes
under 15 seconds on four cores. `qtt_probe.py` alone runs the first comparison, all five
fields with interleaved and fused orders at R=5 (`results/si_R45.json`).
