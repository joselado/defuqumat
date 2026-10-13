# Rewriting the README for a condensed matter physicist

The question this plan answers is what the README should say to a condensed matter
physicist who has not seen the code, and what it says now that such a reader does not
need. The reader wants to know what can be computed, by which formula, with which call,
and whether the number can be trusted; that is what stays. What goes is how the code gets
there: which datasets a quantity runs on, which mixer converges it, how a wavefunction is
stored, how a pool is laid out, what another code's routine does, and how the test suite
is run. None of it is deleted from the project, since every cut below names where the
content already lives or where it moves to.

The README was 861 lines, and the catalogue of quantities (515 lines) and the table
against Quantum ESPRESSO and Elk (144 lines) were three quarters of it. The rewrite
(2026-10-13) left it at 742 lines, the catalogue at 470 and the table at 87: the table
lost its footnotes as planned, while the catalogue lost less than the 300-line estimate
below, because what each bullet keeps, the formula, its meaning, what measures it, the
call and the notebook, is most of what each bullet was, and the implementation asides
were a sentence or a clause per bullet rather than half of it.

## What the reader is, and the three rules that follow

- The reader is a condensed matter physicist. An entry says what the quantity is, writes
  its formula, says what measures it, and names the call and the notebook. In that order.
- An entry does not say which pseudopotential dataset, spin regime or solver it runs on,
  unless the condition is physics the reader has to know (a Chern number needs a gap, a
  spiral with spin-orbit coupling breaks the generalized Bloch theorem). The full
  conditions are the amber boxes of `docs/features.tex`, and the README points there once.
- An entry does not carry a benchmark against `pw.x` or Elk. Those are `PERFORMANCE.md`
  and the "Accuracy summary" of the guide, and the README keeps the one accuracy table of
  "Is it right?".

## Section by section

**Banner and opening paragraph.** Keep. Cut the sentence on every derivative quantity
being a derivative of the total energy rather than a hand-derived formula, which is how the
code works; it is the opening of `docs/features.tex` already. Keep the clause that every
quantity Quantum ESPRESSO also computes was compared against it on the same input, since
that is the trust statement.

**Installing.** Keep. Add the `[notebooks]` extra in one line, since the plots in "A band
structure" need matplotlib.

**A first calculation.** Keep as it is. Add one sentence on where the reader's own
pseudopotential files go, since the example points at `tests/data/pseudo` and a reader
with a crystal of their own has to know that `pseudo_dir` is the answer.

**A band structure.** Keep as it is.

**One calculation on many cores.** Shorten to the launch block and two sentences: the
cores go to the k-points as `pw.x -nk` does, the ground state, forces, stress, relaxations
and spin spirals run this way, and everything else runs in one process. The reasoning on
why threads do not pay, the speedup percentages on six cores and on 128, the SIGTERM and
checkpoint detail all go; `PERFORMANCE.md` and the performance section of the guide carry
them (check that the two figures, 80 to 91 per cent and 20.3x on 54 cores, are in
`PERFORMANCE.md` before cutting, and add them if not). From 24 lines to about 10.

**What you can compute.** This is the main rewrite and it keeps its shape: one functional
at the top, nine groups, each opening with the equation of its headline quantity, and one
bullet per quantity with its formula, its meaning, its call and its notebook. What is
stripped from every bullet:

- the list of datasets and spin regimes a quantity runs on ("on norm-conserving, ultrasoft
  and PAW datasets, unpolarized, collinear and spin-orbit alike"), which appears in eleven
  bullets;
- the refusal conditions that are implementation rather than physics ("an ultrasoft or PAW
  metal is refused", "runs on a fully-relativistic norm-conserving dataset at
  `soc_scale = 0`"), which go to the amber boxes;
- the benchmark asides: the cobalt helix at 15 iterations against `pw.x`'s 119, and the
  instruction to set `noncollinear_gga = 'kubler'` for a comparison on a canted state;
- the solver and mixer vocabulary: the "Getting a hard calculation to converge" paragraph
  (20 lines) becomes about five lines, saying what the reader can state (a magnetic texture
  atom by atom, a constrained moment, a continuation from a converged run across a change
  of spin regime) with the three notebooks, and the catalogue of mixers, preconditioners,
  residual solvers and checkpoint variables goes to the guide, where it already is;
- the second route to the same quantity where it is an implementation choice rather than a
  different physical object: the third harmonic keeps one call, the real-time entries one
  each, and the clause on the projectors' curvature goes.

Specific entries that change beyond that: the noncollinear gradient correction bullet
shrinks to one sentence of physics inside the magnetism group (a gradient correction that
is smooth where the moment vanishes and has no quantization axis) and loses its iteration
counts; the high-harmonic bullet loses its sentence on which regimes take the same route;
the long-range modulation group keeps its equation and the three sentences that say what
the method is and that it is a variational bound, and loses the rest.

**Which of these Quantum ESPRESSO and Elk also compute.** The table stays, because
whether a quantity is a reimplementation or an extension is a question the reader asks
before choosing a code, and because the project's rules (`CLAUDE.md`, the `phase-audit`
skill and `deliverables.md`) make a row in this table a phase deliverable. Two changes:

- The nineteen footnotes (56 lines) leave the README. `docs/reference-codes.md` already
  presents itself as the evidence for every note, so the README keeps the superscripts and
  one sentence pointing at that file. The file stops at note 17 today, so notes 18
  (electron-phonon coupling in Elk) and 19 (real-time propagation in Elk and TDDFPT) are
  added to it, from the README's own text, before the README loses them.
- The "How to ask for it" column is cut back to one call per row; four rows (absorption,
  second harmonic, real time, third harmonic) have grown a second and third call with
  their own keyword arguments, which is reference-entry material.

The alternative, moving the whole table to `docs/`, would need `CLAUDE.md` lines 46, 93
and 129, `deliverables.md` item 2, the `phase-audit` skill's section 2 and the opening of
`docs/reference-codes.md` edited to point at the new place, and is not what this plan does
unless chosen.

**Is it right?** Keep the accuracy table and the paragraph on quantities neither code
computes, pinned by an identity. Cut the `pytest` block and the sentence on the QE
test-suite directory and the committed references, which are developer instructions; one
line under "Where to read more" says that the tests compare against Quantum ESPRESSO's
reference outputs and how to run them.

**What it refuses, and what it does not do.** Keep the first paragraph. The "Not yet"
list is stale and is rewritten from the guide rather than edited: it says a real-time
propagation that updates the potential is not done, while the catalogue three screens
above, the table's real-time row and five modules under `defuqumat/realtime/` have
`potential = 'hxc'`. The phonon dispersion and exact exchange claims are checked against
the guide and `PLAN.md` before they stay. The gamma-point paragraph is rewritten to what
the reader sees (a `K_POINTS gamma` input runs, and for an ultrasoft or PAW dataset, a run
with symmetry or a spinor run it uses the whole sphere and says so, with the same numbers)
and loses the storage explanation. The sentence on being 2.2 times slower than `pw.x` per
iteration stays, since it is what a reader sizing a calculation needs.

**Where to read more, License.** Keep.

## The check that runs on every line that survives

The README has drifted once already (the "Not yet" line), so the rewrite does not
paraphrase from the old README: every refusal, every number and every notebook link that
survives is checked against `docs/features.tex`, `PLAN.md` or `ls notebooks/`, and the
accuracy table against the guide's "Accuracy summary". The equations are checked against
the guide's for the same quantity, since GitHub renders them and a wrong sign is public.

## Order of work

1. Add notes 18 and 19 to `docs/reference-codes.md`, and the pools figures to
   `PERFORMANCE.md` if they are not there, so nothing is lost before the README shrinks.
2. Write the new README in one pass.
3. Run the check above on it.
4. Run the `phase-audit` skill's table step once, to confirm the audit still finds every
   row.
5. Commit, README and the two record files together.
