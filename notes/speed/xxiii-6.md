# speed/xxiii-6: OPEN.md Part XXIII item 6, the walked third derivative's loop order

Status: **partial, do not land yet.** The change is made and bit-identical on one ultrasoft cell
at two chunk sizes; the second cell, the Raman path and the 27 k-point timing were launched on
D22 and the machine went unreachable (ssh "Connection timed out", ping 100 per cent loss, also
through kosh) from about 07:50 until the end of my time, so their jobs are unread.

Branch `speed/xxiii-6`, commit `970355e5` (code and harness) on master `fbe14c1b`; the unit test
`tests/unit/test_walked_third_order.py` and this record go in the next commit. D22 worktrees
`/l/ladovj1/w-speed-xxiii-6` (new) and `/l/ladovj1/w-speed-xxiii-6-old` (`fbe14c1b`), logs and
outputs in `/l/ladovj1/review/speed/xxiii-6/`.

## The change (fix 1 only)

`walked_susceptibility_derivative` walks the chunks outer and the tangents inner in the forward
walk and in the chunk walk. A chunk's `field._arguments(rows, live)`, its `_stacked_rows`
(`dpsi` in the forward walk; `bare`, `dpsi`, `stored` in the chunk walk) are built once per chunk
per pass. Each tangent's forward sums still add in `field.chunks` order and its `energy` and
`denergy` still add in that order, so each column keeps its accumulation. The global step runs
tangent by tangent between the two walks, as before. Fix (2) (returning the moved `vltot + v` and
`deeq` from the global step) is untouched; it is last-bit, not bit-identical.

## Step 1: the count, before the change (old code, `fbe14c1b`)

`passes['chunk'].lower(*args).compile().as_text()` on `alas-piezo.in` with `nosym`, `noinv`,
`ecutwfc = 25`, `ecutrho = 200` and a single k-point (`1 1 1`, the HLO does not depend on nk),
`k_batch = 1`, memory mode, CPU; harness `notes/speed/xxiii-6-harness.py` with `count`, log
`count-old.log`:

| pass | instructions | `while` | sine | cosine | exponential | fft | dot |
|---|---|---|---|---|---|---|---|
| forward | 6686 | 0 | 10 | 10 | 2 | 24 | 107 |
| global | 51535 | 0 | 65 | 57 | 14 | 60 | 192 |
| chunk | 35162 | 0 | 65 | 59 | 11 | 66 | 426 |

There are no `while` loops in the compiled text (the augmentation is unrolled at this cell), so I
counted the radial transforms by their `sine`/`cosine` ops. One `_move` plus its tangent is
about 10 sines in the forward pass; the chunk pass moves the geometry three times (the residual's
`jvp`, the ultrasoft tail's `jvp`, the separable terms' `jvp`) and carries 65, so XLA did **not**
collapse the repeated rebuilds to one. Fix (1) is therefore not the whole lever: it removes the
per-tangent argument and upload cost, and the per-chunk rebuild inside the chunk pass stays for
fix (2). Temp sizes from `memory_analysis()`: forward 56.8 MB, global 106.9 MB, chunk 107.0 MB.

## Step 3: memory

With the chunk walk chunk-outer, the global step's outputs of every tangent are resident at once:
per tangent `v` and `dv` (nspin_mag x dense grid, float64), `ddd` and `dddd` (PAW only, `None`
otherwise), the volume and its tangent, plus the forward sums and their tangents (`becsum`-sized
and the three response sums, small). Added peak: (ntangent - 1) x 2 x nspin_mag x ngrid_dense x 8
bytes, plus the same for the PAW one-centre blocks. Measured on `alas-piezo-tiny` (dense grid
15^3, ntangent 6): 0.054 MB a tangent (0.32 MB for all six); on the `alas-piezo.in` cell (36^3,
ntangent 6): 0.747 MB a tangent (4.5 MB for all six). The process high-water (`ru_maxrss`) on
`alas-piezo-tiny` after the walk: kb1 old 2564 MiB, new 2565 MiB; kb3 old 2520, new 2519. So the
added residency is invisible against the 2.5 GiB the field stores and compiled code hold; it
scales with ntangent x dense grid and would matter only on a slab (a 200^3 dense grid is 64 MB a
field, 6 x 2 x 64 MB = 0.77 GB).

## Step 4: bit-identity and warm timing

Harness: SCF (`conv_thr=1e-10`), `electrostriction(calculation, hosted, elastic=False)` on a host
store (the walked route; `elastic=True` on an ultrasoft cell takes the whole-k route by
`_walk_lacks`), then the recorded `walked_susceptibility_derivative` call repeated three times.
Old and new ran at the same time on slot cores 4, 6, 8, 10 (the efficiency cores, all alike), one
thread, `cmp` on the saved `.npy`.

`alas-piezo-tiny.in` (ultrasoft AlAs, nosym, 8 k-points, 6 strain tangents):

| k_batch (chunks) | columns, byte for byte | `depsilon_dstrain`, byte for byte | warm walk, old, median of 3 | warm walk, new, median of 3 |
|---|---|---|---|---|
| 1 (8 chunks) | identical | identical | 27.91 s (27.91, 27.95, 26.39) | 26.96 s (26.94, 26.96, 26.96) |
| 3 (3 chunks, the last padded) | identical | identical | 9.95 s (9.70, 9.95, 10.02) | 9.67 s (9.57, 9.67, 9.67) |

That is 3.4 and 2.8 per cent of the walked call, one run of each arm side by side, so a small
gain of the size of the old arm's own spread at k_batch 1. At eight k-points the argument build is
a small share of the walk; the gain is expected to grow with the chunk count and the upload size.
The first call of the old code at k_batch 3 was 113.6 s against 111.4 s new (compile included, not
a measurement).

## Unit test

`tests/unit/test_walked_third_order.py` (two tests, 1 s, not slow): the compiled passes and the
store helpers are replaced by arithmetic with a call log. It asserts that `field._arguments` is
called `2 x nchunk` times (not `2 x nchunk x ntangent`), `_stacked_rows` `4 x nchunk` times, that
each tangent meets the chunks in order in both walks and that a chunk's calls are contiguous, and
that the global step receives each tangent's whole sums and the column is the expected
combination. Run on the old code on D22 (`unit-old.log`): **2 failed**; on the new code locally:
2 passed.

## Not done, owed

- Bit-identity on a second cell: `alas-piezo-tiny-paw.in` at k_batch 1 and 3 (jobs `paw8-{old,new}-kb{1,3}`
  launched, unread), and the Raman path (`raman_tensors` on `si-us-nosym.in`, k_batch 3, `KIND=raman`,
  jobs `raman-{old,new}-kb3`, unread). The 27 k-point timing (`alas-piezo-tiny-333.in`, k_batch 3,
  `k27-{old,new}-kb3`, unread). Re-read the logs in `/l/ladovj1/review/speed/xxiii-6/` when D22
  is back; compare with `cmp out-<tag>-old-*/columns.npy out-<tag>-new-*/columns.npy`. The inputs
  `alas-piezo-nosym-111.in` and `alas-piezo-tiny-333.in` are in that directory, not committed.
- The 8 k-point timing is at k_batch 1 and 3 on a 8-k cell, not on `alas-piezo.in` (64 k-points
  nosym at ecut 25 is out of reach of the 6 GB slot).
- The card timing the entry asks for (three passes timed apart at 8 and 27 k-points) is not taken:
  the recipe forbids the card.
- Fix (2) is untouched.
- Slow tests the change touches, for you to run: `tests/regression/test_streamed_third.py` (all
  of it, slow) and `test_piezoelectric_augmented.py`.

## Text for OPEN.md (apply only after the owed bit-identity reads identical)

Heading bracket: **[fix 1 done 2026-10-23, `<sha>`; fix 2 open]**

Paragraph at the top of the entry's body:

**Fix (1) done 2026-10-23, `<sha>`.** `walked_susceptibility_derivative` walks the chunks outer and the tangents inner in the forward and in the chunk walk, so a chunk's arguments, upload and stacked stores are built once per chunk per pass; each tangent keeps its chunk order, and the columns and `depsilon_dstrain` of `alas-piezo-tiny.in` (ultrasoft, 6 tangents, 8 k-points) are byte for byte the old ones at k_batch 1 and 3. The count asked for first: on `alas-piezo.in` (nosym, one k-point a chunk) `passes['chunk']` compiles to 65 sines and 59 cosines against 10 and 10 in the forward pass, which moves the geometry once where the chunk pass moves it three times, so XLA did not merge the rebuilds and (1) is not the whole lever. Warm walked call on `alas-piezo-tiny.in`, median of 3: 27.91 to 26.96 s at k_batch 1 and 9.95 to 9.67 s at k_batch 3. The added residency is the global step's `v`, `dv` (and `ddd`, `dddd` for PAW) of every tangent, 0.75 MB a tangent on the 36^3 grid, invisible in the process high-water (2564 against 2565 MiB). Fix (2) is untouched; the card timing is still owed.

PERFORMANCE.md: nothing yet; the two medians above are of 3 warm calls each but come from one run of each arm side by side on the 8-k cell, so I would wait for the 27-point pair.
