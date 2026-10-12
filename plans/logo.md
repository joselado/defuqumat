# The logo: the magnetization density of a CrI3 monolayer, rendered in Blender

Written 2026-10-12 on `master` as a plan, reviewed by a fable subagent the same day, run
the same day; the first two parts are the plan as it was carried out, and "What was
measured" is the record. The deliverables are `docs/logo/defuqumat-icon.png` (1024 px, no
wordmark, with 256 and 64 px copies) and `docs/logo/defuqumat-banner.png` (2400 by 600,
the wordmark in pearl type carrying a plane wave, across a strip of the honeycomb, on
GitHub's dark grey, which the banner carries itself so that one picture serves both of
GitHub's themes), the input `docs/logo/cri3.in`, the SCF log `docs/logo/data/scf.log`,
and under `tools/logo/` the six files `compute.py`, `logo.py`, `background.py`,
`contrast.py`, `run_scf_d22.sh` and `symmetry_ab.sh`, whose defaults are the values that
made the pictures. The README shows the banner, and the title page of
`docs/features.tex` the 256 px icon. The saved density (62 MB) and the meshes it cuts
are regenerable and stay out of git.

## What was verified on this workstation before starting

- Blender 5.2.2 LTS is at `/u/40/ladovj1/unix/apps/blender-5.2.2-linux-x64/blender`, with a
  bundled Python 3.13.13 that has no JAX, so the physics runs in the anaconda interpreter and
  hands the geometry to Blender through files. Blender 2.79 is also present and is not used.
- Headless rendering works here: `blender -b --factory-startup --python t.py` rendered a
  128 px still in 0.5 s with Cycles on the CPU and in 1.5 s with EEVEE (there is a display,
  `DISPLAY=:0`, which EEVEE needs). There is no CUDA on this machine, so Cycles is CPU-only
  here; the only card is D22's A2000, shared with the owner's own renders.
- `bpy.ops.wm.ply_import` exists on 5.2, so a mesh goes in as PLY; scikit-image 0.26
  (marching cubes) is in the anaconda env and not in D22's venv, so the SCF ran on D22 and
  the meshes were cut here.
- The pattern followed is the seminar film script,
  `~/video_work/india_seminar/v3_rendered_scripts/film.py`: the scene is built from code as
  a deterministic function of its inputs, run as `blender -b --factory-startup --python
  logo.py -- <verb> [options]`, EEVEE for drafts and Cycles for the final, fonts by path.
- The name has no expansion recorded anywhere in the repository, so the wordmark is the
  name alone, lowercase as in `pyproject.toml`.

## The concept, decided 2026-10-12

The rule of this project is that a claim is a number, and the logo follows it: the picture
is a quantity the code computed, from an input it ran, rather than a stock crystal drawn by
hand. The owner chose the material from a list of ten (silicon was turned down as too
plain): **the CrI3 monolayer, the honeycomb ferromagnet, computed noncollinearly with
spin-orbit coupling, drawn as an isosurface of its magnetization density with the local
magnetization on top as arrows.** The icon is one CrI6 octahedron: the Cr's `t2g` shell as
an orange isosurface of `|m|` at 10 per cent of its maximum (a rounded cube with its dimples
toward the six ligands, which is the cubic anisotropy `x^2y^2 + y^2z^2 + z^2x^2` of the
`t2g` sum), the Cr moment as one red arrow, and on each iodine a translucent blue cloud,
the isosurface at 0.4 per cent of the maximum restricted to where `m` is antiparallel to the
Cr moment, with a small blue arrow of the field there. The banner is the honeycomb ring of
six such octahedra beside the wordmark.

- **Why noncollinear.** In a collinear run every arrow would be parallel by construction.
  With spin-orbit coupling the iodine `p` states carry a small induced moment, antiparallel
  to the Cr and tilted off the axis, so the arrows and the clouds have something to say.
- **What the fable review changed** (2026-10-12, before any code): the band count in the
  cost line (84 spinor bands for 70 electrons, not "about 160"); the surface the plan first
  named, `|m|` at 20 to 60 per cent, which never reaches an iodine and on which every arrow
  is parallel and of equal length, replaced by the two-surface picture above; the fine grid
  the mesh step first sized (a 3x3 tiling of a doubly refined dense grid is tens of GB),
  replaced by a crop to the layer, marching cubes on one cell and translation of the mesh;
  the atom arrows taken from `SCFResult.site_moments` instead of an integration of my own
  (both are printed, and agree to 0.03 Bohr magnetons on Cr); and the symmetry A/B below,
  since the magnetic group with time-reversed operations plus spin-orbit coupling had no
  committed check against `pw.x`.
- **Structure.** One layer of bulk CrI3 (`P-31m`): `a = 6.867` angstrom (`celldm(1) =
  12.9768`), the two Cr at `(1/3, 2/3)` and `(2/3, 1/3)` of the `z = 1/2` plane, the six I
  on the `6k` positions with `x = 0.35`, 1.56 angstrom above and below the Cr plane, and
  `c = 18.0` angstrom (`celldm(3) = 2.62123`), 14.9 angstrom of vacuum between iodine
  planes. The distances the code's own parser gives: every I bridges two Cr at 2.725
  angstrom, Cr to Cr 3.965. (The review's `x = 0.347988` gives 2.730 exactly; the 0.005
  angstrom is below anything a picture shows and the run kept 0.35.)
- **Datasets.** `Cr.rel-pbe-spn-rrkjus_psl.1.0.0.UPF` (ultrasoft, fully relativistic, 14
  valence electrons) and `I.rel-pbe-n-rrkjus_psl.1.0.0.UPF` (ultrasoft, 7), both already in
  `tests/data/pseudo/`; 70 electrons in the cell. Cutoffs 50 and 460 Ry, the Cr header's 48
  and 457 rounded up, since an augmentation charge under-resolved on the dense grid rings
  around the Cr and the ringing would show on the surface.
- **Regime.** `noncolin`, `lspinorb`, `starting_magnetization(1) = 0.21` on Cr with
  `angle1 = 0` (a seed of 2.9 Bohr magnetons along z), the iodine unpolarised; `mv`
  smearing at `degauss = 0.02` as every spin-orbit input in `tests/data/qe/` has (the gap
  is above 1 eV); a `4x4x1` mesh, which the magnetic group (12 operations, 6 of them with
  time reversal) reduces to 4 k-points; PBE, no `U`. The symmetry forbids the
  antiferromagnetic collapse (inversion maps Cr1 onto Cr2 with `t_rev = 0`), pins the
  in-plane total moment and the Cr site moments to exactly zero and exactly along z, and
  lets each iodine tilt only within its own vertical mirror plane.
- **The array.** At `nspin = 4` `SCFResult.density` is `(4, n1, n2, n3)` with channels
  `(n, m_x, m_y, m_z)` on the dense grid, augmentation included, in electrons per bohr
  cubed; the check `sum(rho[0]) * V / N = 70` passed to all printed digits.

## Pipeline, as run

1. `tools/logo/compute.py scf` on D22's CPU (`tools/logo/run_scf_d22.sh`: cores 2 to 9,
   `JAX_PLATFORMS=cpu`, an 8 GB scope cap, one k-point in flight), which parses the input,
   prints the distances and the size estimate, runs `Calculator.get_scf(k_batch=1,
   verbose=True)`, checks the integrals and saves the four channels with the cell and the
   atoms to `cri3.npz`.
2. `tools/logo/compute.py mesh` here: each channel refined 2x by zero-padding in reciprocal
   space (exact for a density band-limited to the `ecutrho` sphere), cropped to 6 angstrom
   either side of the Cr plane (the largest `|m|` outside the crop is 3e-6 of the maximum),
   marching cubes on `|m|` at 10 per cent and at 0.4 per cent (the latter keeping only the
   faces where `cos(m, z) < 0`), the mesh translated 3x3, one PLY per surface with vertex
   colours from `cos(m, z)`; arrows on a thinned subset of the lobe vertices, on a 1.4
   angstrom grid where `|m|` is above 0.4 per cent of its maximum, and one per atom from
   `site_moments`; atoms and Cr to I bonds in a JSON. 18 s.
3. `tools/logo/logo.py` in Blender: both surfaces imported (the importer's copy of the
   file's normals stripped, since Cycles shades inward custom normals black), clipped to a
   disc, the clouds kept only within 0.45 units of a drawn iodine and the lobes of the icon
   within 0.4 of its Cr; atoms, bonds, arrows as one mesh per group with a vertex colour from
   `cos(m, z)`, the wordmark as a flat emission colour; orthographic cameras whose frames
   are computed from the clip radius (icon: one Cr, 6.5 bohr, elevation 55 degrees; banner:
   a hexagon centre, 11.5 bohr, 50 degrees); Standard view transform, transparent film.
4. The finals: Cycles on this CPU, 256 samples, 1024 px and 2400 by 600; the icon
   downsampled to 256 and 64 px with pillow. Two choices made after the check on a white and
   on GitHub's dark background (the owner's to overrule, every one a dial): the icon carries
   no wordmark, since at 64 px the name under it is noise and the banner and the guide's
   title carry it; and the banner has a light-text twin, since the dark grey wordmark
   vanishes on the dark theme.

## What was measured

- **The SCF on D22** (`docs/logo/data/scf.log`): 14 iterations, 2110 s wall including the
  compile (about 150 s an iteration, four k-points, 84 spinor bands, 29651 plane waves per
  component, a 90x90x240 dense grid, 3.6 GB resident), total energy -550.53124190 Ry,
  `magnetization_vector = (0, 0, 6.0051)`, absolute magnetization 7.4096, Cr site moments
  2.99567 each, exactly along z; I site moments 0.06544 antiparallel, tilted by 1.25
  degrees (`theta = 178.75`) radially in their mirror planes, `phi` 120 degrees apart. The
  moment carried by the spheres sums to 5.60 of the 6.005; the rest is interstitial.
- **The magnetization density**: maximum `|m|` 0.634 electrons per bohr cubed at the Cr;
  the antiparallel region peaks at 1.26 per cent of that, so the clouds live at 0.4 per
  cent of the maximum, where 43 per cent of the points above the level are antiparallel.
  Along the Cr to I bond `|m|` falls to 0.4 per cent of its maximum at mid-bond and
  changes sign at 0.6 of the bond. The tilt of `m` from z is at most 5.8 degrees where
  `|m|` exceeds 1 per cent of its maximum (mean 1.2) and 18 degrees where it exceeds 0.3
  per cent, so on the lobe surface the arrows are parallel to within a degree, as the
  review said, and the magnetic content of the picture is the sign on the iodine.
- **The symmetry A/B** (`tools/logo/symmetry_ab.sh`, the same cell at 30/240 Ry on a
  `3x3x1` mesh): the symmetric arm, 3 k-points, converged in 17 iterations and 1148 s with
  Cr 3.00671 along z and I 0.06285 antiparallel, tilt 1.5 degrees, 120 degrees apart; the
  unsymmetrised arm, 9 k-points, also 17 iterations. **The two agree**: total energies
  -549.80220364 Ry both, 8.2e-10 Ry apart; `magnetization_vector` (0, 0, 6.05768) and
  absolute magnetization 7.44880 both; Cr site moments 1.8e-6 and 2.2e-6 Bohr magnetons
  apart, iodine ones 2.2e-8 to 2.8e-8; and on the 64x64x180 dense grid the channels differ
  pointwise by at most 2.6e-6 in `n` (1.1e-6 of its maximum), 2.1e-7 and 2.7e-7 in `m_x`
  and `m_y` (4.4e-6 and 5.2e-6 of theirs), 9.5e-7 in `m_z` (1.6e-6), with the tilt of `m`
  from z where `|m|` exceeds 2 per cent of its maximum identical in both arms (7.452
  degrees at most, 1.157 on average). So the magnetic group with its six time-reversed
  operations, applied with spin-orbit coupling on, symmetrises the magnetization to the
  unsymmetrised answer, at the level of the SCF threshold; the production run's 4 k-points
  stand. (`ab.log` and the two `.npz` are in `/l/ladovj1/review/logo/` on D22.)
- **Rendering** on this CPU (fourteen cores, Cycles): a 512 px draft at 64 samples 8 s,
  the 2048 by 512 banner 19 s; the finals at 256 samples, the 1024 px icon 87.8 s (98.4 with
  the wordmark it no longer carries), the 600 px icon 27.1 s, the 2400 by 600 banner 52 s without and 324 s with the depth
  of field and the pearl type of the second pass (the icon 86.7 s there); a 16-sample
  EEVEE draft of either about 2 s. The mesh step 18 s.

## The second pass, 2026-10-12, from the owner's four remarks on the first

The first pictures drew the remarks that the letters should be more artistic, that the
image and the letters should blend, that the arrows were too round, and that the
noncollinearity could not be seen; with them came the wordmark's idea, that the name reads
as its words. What changed:

- **The wordmark is the name alone** (the owner, third pass, after two forms of the
  name read as its words; `--wordmark acronym` keeps the inline form), and the arrows are
  matte (roughness 0.65, no metallic term), since the glossy ones read as mirrors.
- **The wordmark read both ways, in the second pass.** The line is "density functional quantum materials"
  set in one row on one baseline, each word's first letters (`de`, `fu`, `qu`, `mat`) at
  the full size and the rest of the word at half of it in italic, so the big letters read
  `defuqumat` through the line. (A first form put the completions above and below the
  big letters; the owner asked for one line.) The big letters are EB Garamond regular (the
  owner's pick from a sheet of Garamond regular, Garamond bold and Besley; GFS
  Baskerville has no Latin lowercase), the completions EB Garamond italic at 0.50 of the big size, a gap of 0.22 between words.
  Blender's text objects have no advance width, so each completion's `x` is the evaluated
  bounding box of a throwaway text object holding the prefix.
- **The letters are in the scene.** Pearl type (`Coat Weight` 1, roughness 0.22, a bevel of
  0.018 of the size, extruded 0.07), lit by the scene's lights, in a plane 1.2 extents
  nearer the camera than the honeycomb, and the banner is now a strip of the honeycomb
  (centred on the patch middle, a disc of 18 bohr, the frame 2 extents wide) with the
  wordmark across it; the camera's depth of field (f/2.4, focused on the wordmark's plane)
  softens the honeycomb behind. f/0.6 was tried first and turned it into a colour wash.
- **The arrows are slim**: shaft radius 0.011 and head 0.030 Blender units (0.022 and 0.055
  before), the head 0.34 of the length, 24 segments.
- **The noncollinearity is drawn where it is.** On this cell `m` is along z to a degree or
  two wherever `|m|` is above a few per cent of its maximum (Cr by symmetry, the iodine's
  1.25 degrees), so no arrow on an atom or a lobe can show it. Where `m` turns is the band
  between 0.05 and 2 per cent of the maximum around each iodine, where it changes sign:
  measured on the saved density, the tilt from z has a mean of 1.1 degrees above 3 per
  cent, 3.1 degrees (max 18) between 0.3 and 1 per cent, 7.9 degrees (90th percentile 17,
  max 66) between 0.1 and 0.3 per cent, and 16 degrees (max 90) between 0.05 and 0.1 per
  cent. The field arrows are therefore sampled on a 1 angstrom grid in that band (1319 in
  the 3x3 patch, mean tilt 7.4 degrees, 54 per cent antiparallel), drawn at one length
  (a direction field) and only where the tilt exceeds 15 degrees (23 inside the icon's
  disc, thinned to 0.22 units apart on the banner), coloured blue through grey to red by
  `cos(m, z)`. The honest caption is that the arrows show the direction of a magnetization
  of a fraction of a per cent of the Cr peak.
- **The dark-background twin of the banner is gone**: the pearl letters are lit, not
  coloured, so one banner serves both themes, and the README is back to one image. (This
  claim was false on the light theme, measured in the fourth pass below, and the twin is
  back.)

## The fourth pass, 2026-10-12, legibility and the moire

The owner's remark on the third pass was that the letters are hard to see, white on a
white background, and should be bigger and carry a motive related to the calculation,
"maybe a plane wave"; the plan was presented first and the three choices were the
owner's (dark letters with a moire motif; two twisted triangular lattices; 0.85 of the
height in Garamond bold). The fine dials (period, twist, dot width, relief) were picked
from EEVEE contact sheets and two Cycles drafts.

- **Why the letters vanished, measured.** Three causes stacked. The film is transparent,
  so GitHub's light theme supplies white wherever the honeycomb does not cover the strip,
  and the scene covered 56 per cent of it. The letters were pearl with an albedo of 0.93
  under a bright world, so they rendered near white. And the depth of field at f/2.4
  turned the honeycomb behind them into a light wash: composited on white, the median
  luminance of the strip was 0.77. The second pass's claim that lit pearl letters serve
  both themes is therefore false on the light one, and the record of that pass says so.
- **The check the earlier passes lacked**, `tools/logo/contrast.py`: the banner
  composited on white and on GitHub's dark grey (`#0d1117`), the letters taken from a
  mask render of the wordmark alone (`logo.py --mask-only`, EEVEE, 3 s), the surround a
  ring 1.5 per cent of the height (9 px) wide around them, and the figures the WCAG
  ratio of the two medians, (L1 + 0.05) / (L2 + 0.05), and the share of letter pixels
  at least a ratio of 3 (the WCAG level for large text) from the ring's median. The old
  banner read **2.03 on light** (no letter pixel reached 3) and 5.49 on dark, which is the
  owner's remark as a number: it was a dark-theme banner. The same check on the first
  draft of the new letters gave 5.86 on light and 1.25 on dark, the flip, which is why
  the twin is back rather than a single banner.
- **The letters.** The name alone in EB Garamond bold at 0.85 of the banner height (the
  word spans rows 55 to 543 of 600 and 83 per cent of the width; at 0.62 of the height
  the regular spanned about half), graphite (albedo 0.03) with the pearl's coat and
  bevel, 1.2 extents in front of the honeycomb as before, and the depth of field at f/5
  so what is behind stays structured. The vertical offset of the plain wordmark went
  from -0.06 to 0 of the height, since at 0.85 the old offset left 91 px above the f and
  21 below the q.
- **The moire.** Two triangular lattices of dots at a twist of 4 degrees, each the sum
  of three cosines 120 degrees apart (3 at a lattice site) rotated by half the twist
  either way, a dot where the sum passes 3 minus twice the dot width (0.7) through a
  smoothstep, the image the mean of the two; the lattice constant is 0.08 of the letter
  size (41 px on the 2400 px banner, about five periods across the x-height), so the
  beat `a / (2 sin(twist / 2))` is 14.3 lattice constants, 590 px, 3.4 beats across
  the word, which is what reads as the pairing and splitting of the dots along it. The
  image (2048 px across, packed into the scene) is mapped by the wordmark root's object
  coordinates onto the word's box once, so one pattern runs through all nine letters
  rather than restarting at each glyph; it mixes graphite with a lighter graphite (0.22)
  as the base colour and drives a bump of strength 0.08. The first draft, dots of width
  0.6 with a bump of 0.3 at a constant of 0.11, rendered as studs rather than as a
  moire; the smooth sum (width 1.0) at 0.11 read as soft blobs; the finer, sharper
  lattice is the pick. The twin for the dark theme is the same material with pearl
  (0.93) and mid-grey (0.45) dots, `--text-colour light`.
- **The finals** (Cycles, 256 samples, this CPU): the icon 79.5 s and **pixel-identical
  to the committed one** (every channel 0 of 255 apart; the PNG bytes differ, so a bare
  `render` shows the icon as modified in `git status`, and it was restored from HEAD), so
  the icon, its downsamples and `features.pdf` are untouched by this pass; the
  light-theme banner 240.5 s, the dark-theme twin 251.7 s. The contrast of the finals,
  each on its own theme: the graphite letters **3.93 on light** (77 per cent of letter
  pixels at 3 or above, the rest where a letter crosses a blue cloud or a Cr lobe), the
  pearl ones **7.17 on dark** (99 per cent). The ring's median on light is 0.42 in the
  Cycles final against 0.65 in the EEVEE draft of the same scene, so a contrast taken on
  a draft overstates the final; which part of the scene the two engines shade apart was
  not isolated. Each fails the other theme
  (1.09 and 2.01), which the README's `<picture>` element with a
  `prefers-color-scheme: dark` source avoids. The EEVEE drafts behind the choices: 3.5 s
  at 600 px, 8 s at 2400 px; a Cycles draft at 48 samples 69 s.

## The fifth pass, 2026-10-12, the moire out and a plane wave in

The owner's remark on the fourth pass was to remove the moire and put "something you
believe would look nice", so the choice of motif was mine, and it went back to the first
idea, a plane wave through the letters. The lattice code and its four dials are gone
(`git show 369a02c:tools/logo/logo.py` has them); `--text-style wave` replaces
`moire`, and the size, weight, graphite, twin and contrast check of the fourth pass
stand.

- **The wave.** One cosine, (1 + cos(k . r)) / 2, mapped once onto the word's box by the
  wordmark root's object coordinates as the moire was, with the wavelength 0.25 of the
  letter size (128 px on the 2400 px banner, about thirteen crests across the word and
  two across a letter), propagating 60 degrees from the baseline so the crests cross
  the strokes diagonally, and raised to the sixth power so a crest is a narrow line on
  graphite rather than half the length; the crest colour is a pale blue (0.50, 0.60,
  0.92) on the graphite letters and a mid blue (0.30, 0.42, 0.85) on the pearl twin,
  with a bump of 0.1 so the crest catches the light. The dials are `--wave-length`,
  `--wave-angle`, `--wave-sharp` and `--wave-bump`.
- **What was tried and turned down**, each an EEVEE draft at 2400 px (8 s). A
  wavelength near the letter size (0.6 to 1.4 of it, at 15 to 90 degrees) gives each
  letter a single gradient and the word no wave, and a wavelength of a full letter size
  along the baseline shades whole letters differently, which reads as a mistake. The
  plain cosine (power 1) with any blue crest spends half its length on the crest and
  turns the letters pale blue (the deep blue and power 1 drafts were judged by eye, not
  measured). Measured on the EEVEE drafts, the pale crest at power 3 read 3.74 on white
  where the moire draft had read 5.86, and the letters read as brushed metal; the power
  of 6 keeps them graphite (5.24 on the draft, letters' median luminance 0.085 against
  0.139 at power 3). A wavelength of 0.12 at 60 degrees (fine stripes, like a grating)
  was the other candidate, and to my eye it read as a texture rather than as a wave.
- **A trap on the way.** The first drafts all looked alike whatever the dials, and a
  render of the letters alone with white crests showed why: the wave was there, one
  period per letter, but a deep blue crest (0.10, 0.16, 0.48) against graphite under
  this lighting reads as shading, not as a pattern. The way to tell a texture from the
  lighting is a draft with the crest at white and the bump at zero.
- **The finals** (Cycles, 256 samples, this CPU): the icon 78.7 s and again
  pixel-identical to the committed one (restored from HEAD, as in the fourth pass); the
  light-theme banner 256.9 s, the dark-theme twin 253.3 s. The contrast, each on its
  own theme and against the fourth pass's letter mask (the geometry is unchanged): the
  graphite letters **3.53 on light** (56 per cent of letter pixels at 3 or above, down
  from 3.93 and 77 per cent with the moire, the crests being the brighter pixels), the
  pearl ones **7.07 on dark** (98 per cent). The other theme fails as before (1.02 and
  1.98), which the README's `<picture>` element avoided until the sixth pass below put
  the dark grey into the banner itself.

## The sixth pass, 2026-10-12, one banner that carries its background

The owner saw the fifth pass's two finals composited on their themes and kept the dark
one, so one banner now serves both of GitHub's themes by carrying its background: the
pearl letters with the wave, rendered on the transparent film as before, and
`tools/logo/background.py` puts GitHub's dark grey `#0d1117` behind them (`logo.py
render` runs it on the banner it writes; `--background` is the colour and `none` keeps
the film). The graphite letters stay a dial (`--text-colour dark`); the light-theme
banner, its twin file and the README's `<picture>` element are gone, and the README is
one `<img>` again.

- **Which space the background goes on in.** A browser blends a transparent PNG onto
  the page in display (sRGB) space, pixel by pixel, and that is what the owner saw and
  approved; a blend in linear light is the right sum of two lights, and on this banner
  the two differ: 481135 of the 1440000 pixels, a third, are partly transparent (the
  depth-of-field halos around the honeycomb and the soft edges of the clouds), and the
  linear blend moves them by up to 64 of 255 (mean 13) from the approved picture. So
  the blend is the display-space one, in pillow in the anaconda interpreter rather than
  in Blender's compositor, which blends in linear light. The committed banner is the
  fifth pass's transparent twin blended this way, no new render; downsampled to the
  1200 by 300 of the composite the owner saw, it is at most 13 of 255 and on average
  0.2 apart, which is the resampling filter.
- **The check**: **7.07 on both themes** (98 per cent of letter pixels at 3 or above,
  10th percentile 4.00), the fifth pass's dark-theme figure, now also the light-theme
  one, since the theme no longer reaches the letters. The file is 8-bit RGB, 1.05 MB
  against the 6.5 MB of the 16-bit transparent twin.
- **A bare `render` reproduces it**: run into a scratch directory after the edit
  (Cycles, 256 samples, this CPU; the icon 78.0 s, the banner 255.5 s plus the blend),
  the icon is pixel-identical to the committed one and the banner has 15 of its
  1440000 pixels 1 of 255 apart, the rest equal, which is the render's own last-bit
  noise under the denoiser rather than the blend. The committed files are the ones
  made from the fifth pass's render, not this check's.

## What can go wrong, and did

- The field arrows sampled near the Cr are parallel clutter and buried the lobe surface
  (the first real drafts); they are now restricted to the antiparallel region.
- The isosurface of `|m|` at 15 to 40 per cent is small (the Cr d shell is about one bohr),
  so a honeycomb-wide frame shows it as beads; the icon zooms on one octahedron instead.
- A surface imported from PLY renders black in Cycles (not in EEVEE): the importer stores
  the file's normals as the `custom_normal` attribute, and `Mesh.normals_split_custom_clear`
  no longer exists on 5.2; removing the attribute fixes it.
- A Blender camera's `matrix_world` is not evaluated in a scene built this way until a
  depsgraph update, so anything placed in the camera's frame is assembled from its location
  and rotation instead.
- An orthographic camera's frame is the `ortho_scale` along the larger side; the banner's
  height is a quarter of it.

## What was turned down

Silicon's bond charge, the FCC Brillouin zone with the bands inside it and a plane wave
through the lattice were the first three pictures; the nine materials that lost to CrI3 were
NiI2 (the spiral in `calculations/nii2-spiral/`), Mn3Sn, CrSBr, alpha-RuCl3, La2CuO4,
MnBi2Te4, CsV3Sb5, 1T-TaS2 and twisted bilayer graphene at 21.8 degrees.
