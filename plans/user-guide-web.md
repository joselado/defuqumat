# Plan: a user guide as a web page, in the shape of dmrgpy's

Repository: /u/40/ladovj1/data/Documents/programs/claude/defuqumat. Written 2026-10-15 on
master at `0fb818d`, at the user's request: "a user guide, as a webpage on github, like what
was done for dmrgpy ... the user guide should focus on a cond mat physicist, written by a
non-native english speaker with simple english. what matters is the equation, what the physics
means, and how to compute it, internal working of the code are not important ... it should
have plots (bands, dos, etc) and code snippets". Also: "reference to the user guide should be
at top of readme, right after description".

**Status: plan, no code.** Decisions 1 to 3 of section 9 were taken by the user on
2026-10-15: `features.tex` is retired fully, the root is `docs/guide/` plus `docs/figures/`,
and `features.pdf` is deleted in the last step together with the `.tex`. Decision 4 is asked at
step 3.

## 0. The question, and the answer in one paragraph

What should a condensed matter physicist who has never seen defuqumat be able to do after
reading one page of the guide? Know what the quantity is, see its equation, know what it means
and what experiment measures it, and run the five lines that compute it on a real crystal, with
a plot that shows what comes out. That is the whole page. What a page does not say is how the
code gets there: no mixers, no pytrees, no `npwx`, no Fortran routine names, no phase numbers,
no validation tables. The guide is a static website built with Material for MkDocs from
Markdown pages under `docs/guide/`, deployed to GitHub Pages by a workflow, with one PDF built
from the same pages. This is exactly the shape of `joselado/dmrgpy` (its `mkdocs.yml`,
`.github/workflows/guide.yml`, `docs/guide/`, `docs/figures/` and `docs/build_guide_pdf.py`
were read on 2026-10-15 and are the template), and the plan copies it rather than designing
a second one. Every figure comes from a script that is committed next to its data, so the
code shown under a plot is the code that drew it, and the website build never runs DFT.

## 1. What dmrgpy did, and what we copy

The pieces, with the file each lives in:

- `mkdocs.yml` at the repository root: `docs_dir: docs/guide`, `site_dir: site`,
  `strict: true`, theme `material` with tabs, sections, `toc.follow`, `content.code.copy`,
  a light and a dark palette; extensions `admonition`, `tables`, `toc` with permalinks,
  `pymdownx.arithmatex` (generic, MathJax 3), `pymdownx.highlight`, `pymdownx.superfences`,
  `pymdownx.details`, `pymdownx.snippets` with `base_path: [docs/figures]` and
  `check_paths: true`; a `nav` listing every page.
- `docs/guide/*.md`: one page per topic, `index.md` as Home, `img/` for the PNGs,
  `javascripts/mathjax.js` for the MathJax configuration.
- `docs/figures/<id>/`: one folder per figure with `compute.py` (runs the code, writes
  `data.npz`), `plot.py` (reads only `data.npz`, writes `docs/guide/img/<id>.png` and
  `<id>-dark.png`, does not import the package) and the committed `data.npz`.
- `docs/figures/common.py` (`PSEUDO`, `save`, `load`, with the git commit, host and wall
  time stored in the npz), `style.py` (the matplotlib style and `main(draw)` that writes both
  PNG files), `run.sh <id>` (one BLAS thread, a memory cap, writes `DONE` or `FAILED`),
  `plot.py <id>`, `README.md`. Since 2026-10-15 (the user asked for shorter scripts) both
  scripts run inside their folder with `docs/figures` on the path, so a `compute.py` or
  `plot.py` holds no path code and reads as a plain example.
- Each figure on a page is the pair
  `![caption](img/<id>.png#only-light)` / `![caption](img/<id>-dark.png#only-dark)`, a
  paragraph reading the figure, and a collapsed
  `??? example "Code of this figure (docs/figures/<id>/)"` that includes `compute.py` and
  `plot.py` by snippet.
- `.github/workflows/guide.yml`: on push to master touching the guide paths, `pip install
  mkdocs==1.6.1 mkdocs-material==9.7.7`, `mkdocs build`, then pandoc plus xelatex running
  `docs/build_guide_pdf.py site/user_guide.pdf` with `continue-on-error`, then
  `upload-pages-artifact` and `deploy-pages`. Pages is set to `build_type: workflow`.
- `docs/build_guide_pdf.py`: joins the pages in `nav` order, turns admonitions into quotes,
  replaces the collapsed code blocks by a pointer to the folder, keeps the light figure of
  each pair, and runs pandoc.

Two additions that are defuqumat's and not dmrgpy's: every figure folder carries the `pw.x`
input file it runs (`*.in`), and `pseudo_dir` in those inputs points at `tests/data/pseudo`
relative to the repository root, so the snippet on the page runs as written from a checkout.
Both `mkdocs` 1.6.1 and `material` import in the base anaconda environment here, so the site
builds locally before anything is pushed.

## 2. The reader, and the four rules that follow

- The reader is a condensed matter physicist with some Python. A page is written in the
  style of `.claude/rules/user-docs-style.md`, which already matches `docs/**` and so binds
  `docs/guide/` without an edit: short sentences, one fact each, common words, no idioms, the
  physics vocabulary exact.
- A section is: one sentence saying what we compute, the equation, two or three sentences on
  what it means and what measures it (ARPES, STM, neutrons, Raman, transport), the snippet,
  the figure, one paragraph that reads the figure, a short `!!! note "Limits"` admonition in
  simple English where the code refuses something, and the notebook link. In that order.
- A page never names a mixer, a solver, an array layout, a `PLAN.md` phase, a QE routine or a
  benchmark number against `pw.x`. "We compared this against Quantum ESPRESSO on the same
  input" is one sentence on Home, and the accuracy table of the README is linked, not copied.
- A figure is drawn by its `plot.py` from its `data.npz`, never copied from a notebook's
  `_files/` folder. The code under the plot has to be the code that made it.

## 3. What happens to `docs/features.tex` and `docs/features.pdf`

The user's reading is that `features.pdf` "seems to be just internal workings". It is in fact
the current user guide (`CLAUDE.md`: "how a user runs a feature, and what it refuses"), 9262
lines of LaTeX with an executed snippet and an amber box per feature, and it is deliverable 3
of every phase. It reads as internals because it carries the refusal conditions, the dataset
and spin-regime conditions and the validation numbers in the main text. The web guide replaces
it, and the question is only whether the `.tex` stays for a while.

If the PDF goes and the `.tex` stays, there are two user guides and they drift from the next
phase on. So the recommendation is to **retire both**, with the amber boxes moved: each
refusal becomes one or two sentences in the page's "Limits" admonition. The box environment
in `features.tex` is called `refuses`, not amber, and `grep -c 'begin{refuses}'
docs/features.tex` counts 61 of them on 2026-10-15; that is the size of the move in step 10.
The "Accuracy summary" section of `features.tex` (line 9207), the full table of what each
feature was compared against and to how many digits, is the one user-facing content that has
no other home: the README table is five rows and `PLAN.md` §3 is the record, not a user text.
It moves to its own page, `accuracy.md` under More, which is the one page exempt from the rule
against benchmark numbers. `features.tex` is kept in git history and nothing is lost. The
CI-built `user_guide.pdf` is the replacement for the committed PDF.

Every reference has to move, or the next phase writes into `features.tex` again:

| file | lines | change |
|---|---|---|
| `CLAUDE.md` | 21, 26, 56, 101, 139 | "the user guide" is `docs/guide/` on the website; deliverable 3 is a section on a guide page with a figure or a snippet and a Limits note |
| `.claude/skills/phase-audit/deliverables.md` | 32 | same |
| `.claude/skills/phase-audit/SKILL.md` | 30, 67, 83, 137 | the audit greps `docs/guide/*.md` for the `get_*` name; the build check is `mkdocs build --strict` |
| `.claude/rules/scope.md` | 15 | refusals are in the Limits admonitions of `docs/guide/` |
| `.claude/rules/user-docs-style.md` | 23 | name `docs/guide/` instead of `features.tex` |
| `README.md` | 136, 738, 776 to 779 | point at the website; see section 4 |
| `.gitignore` | 28 to 34 | drop the xelatex products; add `site/` and `docs/user_guide.pdf` |

## 4. The README

The user asked that the reference to the guide be at the top, right after the description. So
the first paragraph under the banner stays, and the next line is "The user guide is at
<https://joselado.github.io/defuqumat/>. It is also available as a
[PDF](https://joselado.github.io/defuqumat/user_guide.pdf)." as dmrgpy's README has it. This
top link is added as soon as the site is deployed (step 4). The three later references (lines
136, 738, 776 to 779) keep pointing at the committed `features.pdf` until the web guide covers
everything, which is what decision 3 asks for, and move in step 10: line 136 to the Limits
notes, line 738 to `accuracy.md`, lines 776 to 779 to the website. The feature table stays,
since it is the checklist of section 6, and gains one row for `get_ultracell_transport`, which
is named in the catalogue (line 629) but has no row. Nothing else in the README changes.

## 5. The pages

About 22 pages, grouped as the notebook index groups the notebooks. Each page names the
notebooks it draws on, and a figure is listed as `page_figure`. Sizes are a guess at the number
of sections, not of lines.

**Home** (`index.md`): what the code is in three sentences, install, a first calculation
(silicon, `Calculator.from_file`, `get_scf`, the energy), one band-structure figure as the
front image, how the guide is organized, the one sentence on comparison with QE and Elk, the
link to the notebooks and to the README accuracy table.

**Getting started**
- *Your own crystal* (`own-crystal.md`, notebook 25): the input file, the cell, the species
  and where the UPF files go, the k-mesh, the cutoff and the k-mesh convergence tests.
  Figures: `own-crystal_convergence` (energy against `ecutwfc` and against the mesh).
- *Choosing the functional and the pseudopotential* (`functionals.md`, notebooks 04, 05, 24):
  LDA, PBE, revPBE, PBEsol, Tran-Blaha; norm-conserving, ultrasoft, PAW in one paragraph each
  on what changes for the user. Figure: `functionals_gaps` (silicon's gap by functional).

**Ground state**
- *Total energy and the density* (`ground-state.md`, notebooks 02, 17): the Kohn-Sham
  functional, the SCF, `conv_thr`, the energy term by term, the density on the grid, when to
  use `mixing_mode='robust'`. Figure: `ground-state_density` (silicon's bond charge).
- *Band structure* (`bands.md`, notebooks 02, 08, 19): the k-path, `get_bands`, the gap, band
  velocities. Figure: `bands_silicon`.
- *Density of states* (`dos.md`, notebooks 06, 16): smearing and tetrahedra, `get_nscf`,
  `get_dos`, `get_pdos`, Löwdin charges and fat bands. Figures: `dos_silicon_aluminium`,
  `dos_pdos`.
- *Effective masses* (`effective-mass.md`, notebook 29): `get_effective_mass`. Figure:
  `effective-mass_curvature`.

**Structure**
- *Forces and relaxation* (`forces.md`, notebooks 09, 22): `get_forces`, `get_relax`,
  Grimme's D2. Figure: `forces_relaxation` (energy and maximum force per step), and
  `forces_vdw` (bilayer graphene binding curve).
- *Stress and the cell* (`stress.md`, notebooks 15, 23, 21): `get_stress`, the equation of
  state, `get_relax(variable_cell=True)` at a pressure, elastic constants. Figure:
  `stress_eos`.
- *Electric fields* (`electric-field.md`, notebooks 53, 54, 34): a slab in a field with the
  dipole correction, the Berry-phase polarization, a crystal in a finite field. Figure:
  `electric-field_dipole` (dipole against field with the slope as the polarizability).

**Magnetism**
- *Collinear and noncollinear magnetism* (`magnetism.md`, notebooks 07, 11, 43, 18): `nspin=2`,
  the exchange splitting, `noncolin`, site moments, constrained moments and fields,
  `STARTING_MOMENTS` and `with_moments`, `with_spin`. Figures: `magnetism_nickel_dos`
  (spin-resolved DOS), `magnetism_texture`.
- *Spin-orbit coupling* (`spin-orbit.md`, notebooks 08, 29, 39): `lspinorb`, the gap it opens,
  site `<L>`, `<S>`, `<J>`, orbital magnetization. Figure: `spin-orbit_bismuthene`.
- *Anisotropy and the easy axis* (`anisotropy.md`, notebooks 36, 47): the force theorem,
  `get_anisotropy`, `get_torque`, `get_relaxed_orientation`, `get_scf(rotate_moments=True)`.
  Figure: `anisotropy_energy_vs_angle`.
- *Spin spirals and exchange* (`spirals.md`, notebooks 12, 14, 48): the generalized Bloch
  theorem, `get_spiral_scan` and the Heisenberg constants, `get_spiral_relaxation`, the
  Dzyaloshinskii-Moriya energy. Figure: `spirals_eq`.
- *Magnons* (`magnons.md`, notebook 38): the transverse susceptibility, the Stoner continuum,
  `get_magnon_dispersion`, `get_spin_susceptibility`. Figure: `magnons_dispersion`.
- *DFT+U* (`dft-plus-u.md`, notebook 13): the occupation penalty, the double counting,
  `HUBBARD`, `hubbard_slater`, tensor moments. Figure: `dft-plus-u_feo` (the DOS and the occupations; it was `dft-plus-u_feo_gap` until the run showed
  FeO metallic in this symmetric cell at either U).

**Response and spectra**
- *Dielectric constant, Born charges and phonons* (`phonons.md`, notebooks 19, 20, 49): the
  second derivatives, `get_dielectric_tensor`, `get_born_charges`, `get_phonons`,
  `get_phonons_at_q`, `get_electron_phonon`. Figures: `phonons_silicon_q` (the branches),
  `phonons_linewidth`.
- *Raman and infrared* (`raman.md`, notebook 26): `get_raman_tensors`,
  `get_vibrational_spectrum`, LO-TO. Figure: `raman_spectrum`.
- *Optical spectra* (`optics.md`, notebooks 27, 30): `get_absorption` with the bootstrap
  kernel, `get_optical_conductivity`, the Kerr angle. Figures: `optics_absorption`,
  `optics_kerr`.
- *Nonlinear optics* (`nonlinear.md`, notebooks 32, 33, 51, 50): shift current, SHG, the
  third harmonic, high harmonics, `get_realtime`. Figures: `nonlinear_shg`,
  `nonlinear_hhg`.
- *Strain and polarization couplings* (`piezo.md`, notebooks 28, 21, 35): piezoelectric
  tensor, electrostriction, the elasto-optic tensor, the magnetoelectric tensor. Figure:
  `piezo_components`.

**Topology** (`topology.md`, notebook 10): Berry curvature, `get_chern`, `get_z2`,
`get_z2_3d`, Wannier-centre flow. Figure: `topology_berry_map`.

**Microscopy and scattering**
- *STM and tunnelling* (`stm.md`, notebooks 40, 41): `get_stm`, `get_sts`,
  `get_vertical_transport`, `get_momentum_transport`. Figure: `stm_graphite`.
- *Diffraction and nesting* (`scattering.md`, notebooks 37, 31): `get_structure_factors`,
  `get_nesting`. Figure: `scattering_forbidden_reflection`.

**Long-range modulations** (`ultracell.md`, notebooks 44, 45, 46): `get_ultracell`, its
energy, `get_ultracell_stm`, `get_ultracell_sts`, `get_ultracell_transport`, seeded spin
waves. Figure: `ultracell_sdw`.

**More**
- *Starting from an all-electron ground state* (`elk-seed.md`, notebook 42).
- *Performance and many cores* (`performance.md`, notebook 03, `performance/README.md`): the
  launch block, `k_batch`, `memory_mode`, GPUs in a paragraph, pointing at `PERFORMANCE.md`.
- *Accuracy* (`accuracy.md`): the "Accuracy summary" of `features.tex` as a table, what each
  feature was compared against and to how many digits. The one page that carries numbers
  against `pw.x` and Elk.
- *Where the notebooks are* is a line on Home, not a page.

The methods that are not a feature of their own go on the page of the feature they serve:
`with_cell`, `with_positions` and `with_kpoints` on `own-crystal.md`, `from_text` on Home,
`get_force_theorem` and `get_first_order_soc` on `anisotropy.md`,
`get_piezoelectric_kmesh_ladder` on `piezo.md`, `get_realtime_dielectric`,
`get_harmonic_orders` and `get_nonlinear_spectrum` on `nonlinear.md`.

That is 26 pages and 36 figures. Phases 8 and 9 of section 8 may merge `effective-mass.md`
into `bands.md` and `elk-seed.md` into `own-crystal.md` if they come out at one section each.

## 6. Coverage is a set difference

The checklist is the README feature table (lines 646 to 716, 71 rows) plus the methods of
`Calculator` (`grep 'def get_' defuqumat/calculator.py`: 64 `get_*`, five `with_*`, two
`from_*` on 2026-10-15). Every row and every method appears on exactly one page as a call in a
snippet or a catalogue bullet; section 5 assigns them by hand and the script of step 9 is what
says whether the assignment is complete. The mapping of the rows to the pages above:

| README rows | page |
|---|---|
| total energy, robust mixing | `ground-state.md` |
| bands, band velocities | `bands.md` |
| DOS, PDOS | `dos.md` |
| effective mass | `effective-mass.md` |
| all-electron start | `elk-seed.md` |
| pseudopotentials, functionals, Tran-Blaha | `functionals.md` |
| DFT+U, tensor moments, AMF, Yukawa Slater | `dft-plus-u.md` |
| van der Waals, forces, relaxation | `forces.md` |
| `tefield`, `dipfield`, `lelfield`, polarization | `electric-field.md` |
| variable cell, stress, strain response, elastic constants | `stress.md` |
| electrostriction, elasto-optic, piezoelectric, magnetoelectric | `piezo.md` |
| collinear, noncollinear, site moments, constrained, local fields | `magnetism.md` |
| spin-orbit, angular momenta, orbital magnetization | `spin-orbit.md` |
| spirals, spiral relaxation, `E(q)`, DM energy | `spirals.md` |
| anisotropy (five rows), torque, orientation, exchange torque | `anisotropy.md` |
| magnons, susceptibility | `magnons.md` |
| dielectric, Born, phonons at Gamma and q, electron-phonon | `phonons.md` |
| Raman tensors, Raman and IR spectra, LO-TO | `raman.md` |
| absorption, optical conductivity | `optics.md` |
| shift current, SHG, real time, third harmonic | `nonlinear.md` |
| Chern, Z2 | `topology.md` |
| STM, STS, vertical transport, momentum transport | `stm.md` |
| nesting, structure factors | `scattering.md` |
| ultracell, its energy, its STM | `ultracell.md` |

The check is a script, `tools/check_guide.py`: it reads `defuqumat/calculator.py` for the
`get_*` and `with_*` names, reads every `docs/guide/*.md`, and prints the methods that appear
on no page and the pages that name a method that does not exist. It runs in the `phase-audit`
skill in place of the `features.tex` grep, and as a fast test so the gate catches a new
`get_*` with no page.

## 7. Figures: cost, and where they run

The website build never runs DFT: `data.npz` is committed and CI runs only `mkdocs build` and
pandoc. Each `compute.py` is run once here or on D22 under `run.sh`, with one BLAS thread and
the memory cap, never beside a test run (the `test-runs` skill's rule for this shared
machine). The budget per figure is an estimate (only notebook 17's 282 s on D22 was read in
this session); `common.save` replaces it with measured times as the figures land:

- Under a minute, run here: silicon bands, DOS, density, convergence tests, forces and
  relaxation of a small cell, the equation of state, the Berry curvature map, the dielectric
  constant and the phonons at Gamma, the structure factors, the effective mass.
- One to ten minutes, run here one at a time: nickel's spin-resolved DOS, bismuthene with
  spin-orbit coupling, the spiral `E(q)`, the anisotropy curve, FeO with `U`, the Raman
  spectrum, the absorption spectrum, SHG, the STM image, the phonon branches at `q`, the
  electron-phonon linewidths, the dipole against field.
- Over ten minutes, run on D22: the magnon dispersion, the ultracell spin density wave, the
  high-harmonic spectrum, the Kerr angle, the vdW binding curve of bilayer graphene, the
  relaxed orientation of cobalt.

Each `compute.py` records its wall time in `data.npz` through `common.save`, and the figure's
README line says the time and the machine, so the budget above is replaced by measurements as
the figures land. A figure whose compute is over an hour is cut to a smaller cell or a coarser
mesh and the page says so in one sentence, since the point of the figure is the physics and
not the convergence.

## 8. Steps, in order

1. **Infrastructure.** `mkdocs.yml`, `docs/guide/index.md` with a placeholder,
   `docs/guide/javascripts/mathjax.js`, `docs/figures/{common.py,style.py,run.sh,README.md}`,
   `docs/build_guide_pdf.py` (dmrgpy's, with the paths changed), `.github/workflows/guide.yml`,
   `.gitignore` entries. `mkdocs build --strict` passes locally. Nothing pushed yet.
2. **Home, Your own crystal, Ground state, Band structure, Density of states**, with their
   six figures. This is the first deployable guide and it is the one to review the voice on.
3. **Ask, then enable Pages and push** (section 9, decision 4): the repository is public and
   `has_pages` is false, so enabling Pages with `build_type: workflow` is a settings change,
   and the first push of `guide.yml` deploys the site. Both go to the user with the exact
   command before they happen.
4. **README**: the link at the top only; the three later references wait for step 10.
5. **Structure** (three pages, five figures).
6. **Magnetism** (six pages, seven figures), the long-running figures sent to D22 first so they
   compute while the pages are written.
7. **Response and spectra** (five pages, nine figures).
8. **Topology, microscopy and scattering, the ultracell** (four pages, four figures).
9. **More**: all-electron start, performance, accuracy. `tools/check_guide.py` and its test.
10. **Retire `features.tex` and `features.pdf`**, move every one of the 61 `refuses` boxes
    that is not yet a Limits note (grep for `begin{refuses}`), move the three README
    references, and edit the seven rule and skill files of section 3 in the same commit, so no
    session writes into the `.tex` again. Build the PDF once locally with pandoc to see that
    `build_guide_pdf.py` handles every page.

Until step 10 the current rules still send a new phase's guide entry into `features.tex`. If
a phase lands in between, its entry is written as a guide page instead, and a one-sentence
freeze note in `CLAUDE.md` says so; the user decides whether to add that note now.

Steps 2 and 5 to 9 are each one commit per page or per group, with the figures' `data.npz`
committed beside the scripts.

## 9. Decisions for the user

1. Retire `features.tex` fully in step 10 (recommended), or keep it read-only until every amber
   box has moved and delete it later.
2. The root: `docs/guide/` plus `docs/figures/` as in dmrgpy (recommended), or a new top-level
   `guide/` folder.
3. When the committed `features.pdf` is deleted: in the first infrastructure commit, or in the
   last commit with the `.tex`.
4. Enabling GitHub Pages and the first deploying push: asked for at step 3 with the exact
   commands.
