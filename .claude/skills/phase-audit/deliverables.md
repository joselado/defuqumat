# What a finished phase leaves behind, in full

*Moved out of `CLAUDE.md` verbatim on 2026-10-08.*


A phase is not done when the code runs. It is done when all five of these exist, and each
of the five has gone stale silently at least once, so each is checked rather than assumed.

**1. A number.** A concrete figure against `pw.x`, against Elk, or against an identity
that shares no machinery with the assembly — and, where a term is missing, a *measurement*
of what it is worth and a refusal by name. `PLAN.md` §3 is where it goes.

**2. A row in the README's feature table.** `README.md` carries every implemented feature,
with the input variable or entry point that reaches it and **two tick columns, `QE` and
`Elk`, saying whether either established code computes that quantity at all**. `(✓)` with
a numbered note means partly; **blank in both is the mark of a quantity neither code has**,
which is what tells a reader whether this is a reimplementation or an extension.

- **The table names quantities, not routines**, and **the rows are physics, not knobs.** A
  row is a thing someone would want to compute; smearing belongs inside the row about
  metals, not in a row of its own. A new feature adds *one* row; its variants, schemes and
  internal terms go in `PLAN.md`.
- **Each tick is a claim about someone else's source.** Before ticking `QE`, find the
  routine in the vendored tree; before ticking `Elk`, find it in the task list of
  `docs/elk_manual.txt` (§5.127) or in `vendor/elk/src/` in the user's `elkpy` checkout;
  before leaving both blank, grep both. Elk is the easier of the two to get wrong — its
  `z2*.f90` files are complex-matrix helpers with nothing to do with the Z2 invariant.
- **The entry-point column is a claim about *this* code.** `grep` the variable in
  `io/pwin.py` and `system/builder.py` and check the function is exported. A row naming
  something nothing parses is worse than no row.

**3. A section in the user guide.** The user guide is the website built from
`docs/guide/*.md` (Material for MkDocs, https://joselado.github.io/defuqumat/, with a PDF built
by CI); `docs/features.tex` was retired on 2026-10-15 and lives in git history only, and
nothing goes into it again. The plan and the page shape are `plans/user-guide-web.md`, the
voice is the last section of `.claude/rules/user-docs-style.md`, and the figures are
`docs/figures/<id>/` (`docs/figures/README.md`). A section is four things and the last two get
skipped:

- **what it computes**, as an equation where there is one — this is a physics document,
  not an API listing;
- **the entry point**, the `Calculator.get_*` call, checked by `grep` rather than remembered;
- **a snippet that has been run**, with its printed output pasted from that run (on D22, see
  the user's memory and `docs/d22-notes.local.md`). Checking a name exists is not enough: an
  audit that only checked `dir()` passed six broken snippets. Execute it. A figure, computed
  by its `compute.py` and drawn by its `plot.py`, where one says something;
- **what it refuses**, in the page's `!!! note "Limits"` admonition. The refusals are the
  promise that a run which starts is a run whose physics is there, and that promise is only
  usable if its edges are written down.

**Do not re-document standard `pw.x` variables** — `ecutwfc`, `ibrav`, `nbnd` mean what
they mean in QE and the guide says so once. Document this code's own knobs (`mbj_c`,
`spiral_q`, `LOCAL_MAGNETIC_FIELDS`) and the ones that gate a feature. **The audit that
catches drift is a set difference, not a read-through**: `tests/unit/test_facade_documentation.py`
checks that every `Calculator.get_*` is named on some page, and `tools/check_guide.py` checks
the rest (includes, images, nav). `mkdocs build --strict` must pass.

**4. A notebook.** See the section below.

**5. A timing against the code it was taken from.** See "Performance".
