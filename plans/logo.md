# The logo: the magnetization density of a CrI3 monolayer, rendered in Blender

Written 2026-10-12 on `master` as a plan, reviewed by a fable subagent the same day, run
the same day; the first two parts are the plan as it was carried out, and "What was
measured" is the record. The deliverables are `docs/logo/defuqumat-icon.png` (1024 px, no
wordmark, with 256 and 64 px copies) and `docs/logo/defuqumat-banner.png` (2400 by 600,
the wordmark in pearl type across a strip of the honeycomb), the input `docs/logo/cri3.in`, the SCF log `docs/logo/data/scf.log`,
and under `tools/logo/` the four files `compute.py`, `logo.py`, `run_scf_d22.sh` and
`symmetry_ab.sh`, whose defaults are the values that made the pictures. The README shows
the banner, and the title page of `docs/features.tex` the 256 px icon. The saved density (62 MB) and the meshes it cuts are
regenerable and stay out of git.

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
  coloured, so one banner serves both themes, and the README is back to one image.

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
