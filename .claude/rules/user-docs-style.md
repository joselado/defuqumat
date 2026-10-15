---
paths:
  - "README.md"
  - "docs/**"
  - "notebooks/**"
  - "performance/README.md"
---

*Set by the user on 2026-10-13, after the README was rewritten in this style (the
first rewrite they approved). It is the voice for every user-facing text in this
project, and it wins over the global voice file (long chained sentences) wherever
the two disagree. The long-term goal is that all user-facing text reads this way:
any user-facing text that is written new, or rewritten for any reason, is written
in this style.*

## The style for user-facing text: simple English, written by a physicist

**Who reads it.** A condensed matter physicist who knows the physics and some
Python. **Who writes it.** A physicist whose English is good but not native: plain
words, short sentences, nothing idiomatic. The physics stays exact; only the
language gets simpler.

**What it covers.** `README.md`, the user guide `docs/guide/` and anything else under `docs/`
meant for a user, the prose cells of the notebooks and `notebooks/README.md`,
`performance/README.md`. It does **not** cover `PLAN.md`, `OPEN.md`, `notes/`,
`PERFORMANCE.md`, code, comments or docstrings: those are the record, and they stay
exact, technical and numeric as `CLAUDE.md` says.

### Sentences
- One fact per sentence. Most sentences are 8 to 20 words.
- No chains of clauses joined by colons, semicolons or "which is". Start a new
  sentence instead.
- Small connectors carry the logic: "so", "also", "then", "for example", "because",
  "this is", "here". Example: "Also `conv_thr` has the same meaning, so the SCF
  stops at the same point as in `pw.x`."
- A definition starts with "This is ..." or "It says how ...", not an appositive
  hung on the end of a long sentence.
- A question can open a paragraph: "What does a scanning-tunnelling microscope see?"
- No contractions, no em-dashes, no exclamation marks, no emojis.

### Words
- Common words over literary ones: "normal" not "ordinary", "if you do not give X"
  not "X defaults to", "shows step by step" not "walks through", "gives an error"
  not "is refused" (in running prose; "refuses" is fine as a section word),
  "creates" not "induces", "is zero" not "vanishes", "pushes up" not "raises".
- No idioms or phrasal flourishes: "point at", "read off", "carry on", "on the way
  out", "put back", "worth its cost".
- No possessive chains on things: "the header of the pseudopotential file", not
  "the dataset's header"; "the D2 pair sum of Grimme" is fine, as is the plain
  "Grimme's D2" in a table cell.
- Physics vocabulary stays exactly as it is: SCF, namelist, card, pseudopotential,
  Berry curvature, clamped-ion, Stoner continuum, k-mesh, variational. The reader
  knows these words; simplifying them would be wrong.
- "We" for what the code and the authors do ("we compared", "we solve"), "you" for
  the reader's choice ("you choose", "you need"), "the code" when "we" would be
  unclear.

### What stays
- Every formula, every call, every input variable, every number, every link and
  every table row. A rewrite in this style changes the language, never the content.
- Formulas are introduced by a short sentence that ends in a colon and followed by
  one or two sentences that say what they mean.
- In a catalogue (a bulleted list of features), each bullet starts with the bold
  name, then short sentences, then the call and the notebook at the end.

### Before and after
Before: "The pseudopotentials are read from the names on the `ATOMIC_SPECIES` card,
in UPF format, from `pseudo_dir`, which defaults to the input file's own directory;
a crystal of your own is an input file you write and the UPF files you fetched put
in that directory."

After: "The pseudopotentials are UPF files. The code reads their names from the
`ATOMIC_SPECIES` card and looks for them in `pseudo_dir`. If you do not give
`pseudo_dir`, the code looks in the folder of the input file. So, to compute your
own crystal, you write an input file and put the UPF files in that folder."

`README.md` as of 2026-10-13 is the reference example of the whole style.

### The user guide speaks as "we" (the user's decision, 2026-10-15)

For the user guide website (`docs/guide/*.md`) the user asked for first person and simple
English "like 'here we will compute...', similar to how it is in pyqula user guide". The
model is `documentation/guide/` of the pyqula repository (`bands-and-dos.md`,
`mean-field.md`). What that changes from the rule above, for the guide only:

- A page opens with "In this page we compute ...", a section with what we do next ("We
  now ...", "Let us now ...", "Here we take ..."), and the code is introduced by "We
  compute it with `calc.get_x()`". "We" is the default subject, "you" is the reader's own
  choice, never "I".
- Sentences stay simple but are not choppy: two facts may be joined with ", so", "and",
  "which is", "meaning that", or an appositive ("the density of states, the number of
  states at each energy"). A long run of very short sentences is what to avoid now. No
  semicolons and no long colon chains, as before.
- Everything under "What stays" above still holds: every formula, call, number, link and
  printed output is kept exactly.
