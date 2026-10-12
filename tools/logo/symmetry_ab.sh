#!/bin/bash
# The symmetry A/B the plan's review asked for (plans/logo.md): the CrI3 cell of
# docs/logo/cri3.in at a reduced cutoff and a 3x3x1 mesh, once with the magnetic
# group on (3 k-points, six of the twelve operations time-reversed) and once with
# nosym (9 k-points), then the two densities compared point by point. The
# combination "magnetic group with t_rev operations plus spin-orbit coupling" has
# no committed check against pw.x, and the picture rests on how the
# magnetization is symmetrised; a route identity is testable at any cutoff, so
# this runs at 30/240 Ry. Launched on D22 beside the production run, on the
# cores it does not use:
#   ssh -f -o BatchMode=yes ladovj1@D22-0161.org.aalto.fi \
#       'nohup bash /l/ladovj1/review/logo/symmetry_ab.sh > /dev/null 2>&1 &' < /dev/null
# and waited on with `grep -q LOGO_AB_EXIT ab.log`.
set -u
RUN=/l/ladovj1/review/logo
CODE=/l/ladovj1/w-logo
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
export PYTHONPATH=$CODE DEFUQUMAT_CACHE_DIR=/l/ladovj1/jaxcache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$RUN" || exit 1
rm -f ab.log ab-sym.npz ab-nosym.npz
sed -e 's/ecutwfc = 50.0, ecutrho = 460.0/ecutwfc = 30.0, ecutrho = 240.0/' \
    -e 's/^ 4 4 1 0 0 0/ 3 3 1 0 0 0/' cri3.in > ab-sym.in
sed -e 's/nat = 8, ntyp = 2,/nat = 8, ntyp = 2, nosym = .true.,/' ab-sym.in > ab-nosym.in
grep -c 'ecutwfc = 30.0' ab-sym.in ab-nosym.in >> ab.log
grep -c 'nosym' ab-sym.in ab-nosym.in >> ab.log
for arm in sym nosym; do
    echo "=== arm $arm $(date +%T)" >> ab.log
    systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 \
        taskset -c 10-15 /l/ladovj1/venv/bin/python -u "$CODE/tools/logo/compute.py" scf \
            --input "$RUN/ab-$arm.in" --pseudo-dir "$CODE/tests/data/pseudo" \
            --out "$RUN/ab-$arm.npz" >> ab.log 2>&1
    echo "=== arm $arm exit=$? $(date +%T)" >> ab.log
done
taskset -c 10-15 /l/ladovj1/venv/bin/python -u - >> ab.log 2>&1 <<'EOF'
import numpy as np
a, b = np.load("ab-sym.npz"), np.load("ab-nosym.npz")
print("total energy  sym %.8f  nosym %.8f  diff %.2e Ry" % (a["total_energy"], b["total_energy"], a["total_energy"] - b["total_energy"]))
print("iterations", int(a["iterations"]), int(b["iterations"]))
print("magnetization_vector sym", a["magnetization_vector"].round(5), "nosym", b["magnetization_vector"].round(5))
print("absolute_magnetization sym %.5f nosym %.5f" % (a["absolute_magnetization"], b["absolute_magnetization"]))
for s, ma, mb in zip(a["symbols"], a["site_moments"], b["site_moments"]):
    print("  %-3s sym %s nosym %s diff %.2e" % (s, ma.round(4), mb.round(4), np.abs(ma - mb).max()))
ra, rb = a["density"], b["density"]
print("density shapes", ra.shape, rb.shape)
if ra.shape == rb.shape:
    for c, name in enumerate(("n", "m_x", "m_y", "m_z")):
        d = np.abs(ra[c] - rb[c]).max(); scale = np.abs(ra[c]).max()
        print("  %-3s max|diff| %.3e  max|field| %.3e  ratio %.2e" % (name, d, scale, d / max(scale, 1e-30)))
    mag_a = np.linalg.norm(ra[1:], axis=0); mag_b = np.linalg.norm(rb[1:], axis=0)
    print("  |m| max|diff| %.3e of max %.3e" % (np.abs(mag_a - mag_b).max(), mag_a.max()))
    # the tilt of m away from z where |m| matters: the quantity the arrows draw
    for lab, r in (("sym", ra), ("nosym", rb)):
        mag = np.linalg.norm(r[1:], axis=0); mask = mag > 0.02 * mag.max()
        tilt = np.degrees(np.arccos(np.clip(np.abs(r[3][mask]) / mag[mask], -1, 1)))
        print("  tilt from z where |m| > 2%%: %-5s max %.3f deg, mean %.3f deg" % (lab, tilt.max(), tilt.mean()))
EOF
echo "LOGO_AB_EXIT $(date +%T)" >> ab.log
