export const meta = {
  name: 'incompat-feedback',
  description: 'The user decisions of 2026-10-16: feedback field as solved state in Newton (I73), feedback schemes on the ultracell (I36), pin the I81 message',
  phases: [
    { title: 'Implement', detail: 'one agent per piece, own branch from incompat-t2, checks on D22' },
    { title: 'Verify', detail: 'adversarial re-check on D22' },
  ],
}
const S = '/tmp/claude-3291340/-u-40-ladovj1-data-Documents-programs-claude-defuqumat/ff012999-7b6a-4839-ab20-314366d0cc6a/scratchpad'
const RECIPE = `${S}/d22-recipe-3.md`
const BASE = 'e39c74f7'
const SLOWRULE = `SLOW TESTS ARE NEVER RUN (the user's rule): no pytest -m slow, no -m 'slow or not slow', no bare pytest on a file, no tools/run_regression.sh. Every pytest run carries -m 'not slow' or names one test not marked slow. A check that only a slow test reaches is measured with a harness script doing the same computation through slot.sh, and the slow test is written, marked slow and listed for the user.`
const UNITS = [
  { name: 'I73-feedback', task: `The user decided (2026-10-16) that the Newton-Krylov (residual) SCF solver should carry the feedback field of a fixed-spin-moment constraint as part of its solved state, the way the DFT+U occupations ns are carried, so that the refusal of 'fsm', 'atomic fsm', 'atomic fsm direction' and reducebf < 1 in scf/residual.py:make_residual is lifted. Read notes/incompat/I73.md first (its end has the earlier implementer's analysis: run_scf changes the field only through field.feedback() under FEEDBACK and field_scale *= reducebf), then the residual solver (scf/residual.py, the ns block as the model), the mixing loop's feedback update in scf/driver.py, and QE's PW/src for how the feedback field is updated per iteration (add_bfield.f90, the lambda/fsm update). Decide what the fixed point is: at convergence the feedback field is what makes the constrained moment equal its target, so the extra residual component is (moment - target), or the field update itself; reducebf < 1 scales the field down per iteration and its fixed point is the field at zero, say which you implement and why, and refuse reducebf if it has no well-defined fixed point. Check: the mixing loop's converged answer on the same cell with the same constraint (fe-fsm.in style, the atomic fsm cases of QE's test-suite pw_noncolin), total energy, moment and field, to conv_thr 1e-11 or better, and against pw.x where a committed reference exercises the constraint (tests/data/qe/reference.out.pw_noncolin-*). The integrator's guard for a spiral under a field in make_residual (I72 with I73) stays.` },
  { name: 'I36-feedback', task: `The user decided (2026-10-16) that the ultracell should run a feedback scheme (fsm feedback field, reducebf) on the box itself, lifting the part of I36 that is still refused. Read notes/incompat/I36.md first (what landed, what is refused and why), then defuqumat/ultracell/driver.py (run_ultracell and its guards) and how run_scf carries the final field after reducebf or a fixed spin moment. Implement the feedback update inside the ultracell loop so the field is converged on the box rather than carried frozen from the reference. Check: the explicit supercell SCF of the same modulation under the same constraint (the nested-basis identity of P88), energy per cell, moment and field, with the band count ladder the earlier ultracell items used (I31, I32 records). Keep any regime you do not measure refused by name.` },
  { name: 'I81-pin', task: `The user decided (2026-10-16) that the I81 refusal message is pinned by a test. Read notes/incompat/I81.md and the refusal of an ultrasoft or PAW dataset at q != 0 in defuqumat/response/phononq.py:require_a_two_sphere_regime, and tests/unit/test_phonon_at_q_augmented.py. The verifier found the branch's tests cannot tell the new message from the old one. Write a fast test that matches the parts of the new message the old one did not have (the four absent terms, addusdynmat's int2/int5, the supercell pointer for an insulator) and check it fails against the old message (git show 126d6ec:defuqumat/response/phononq.py). This is small: no D22 run is needed if the test builds no SCF; otherwise run it on D22.` },
]
const IMPL = { type: 'object', properties: {
  unit: { type: 'string' }, status: { type: 'string', enum: ['done', 'partial', 'blocked'] }, branch: { type: 'string' }, commit: { type: 'string' },
  d22_worktree: { type: 'string' }, number: { type: 'string' }, tests: { type: 'string' }, slow_tests_for_user: { type: 'string' },
  limits_and_readme: { type: 'string' }, d22_notes: { type: 'string' }, summary: { type: 'string' } },
  required: ['unit', 'status', 'branch', 'number', 'summary'] }
const VERDICT = { type: 'object', properties: { unit: { type: 'string' }, confirmed: { type: 'boolean' }, findings: { type: 'string' }, d22_notes: { type: 'string' } }, required: ['unit', 'confirmed', 'findings'] }

function implPrompt(u, fix) {
  return `You are implementing one piece of defuqumat's incompatibility work, a decision the user took after the T2 batch. Piece: ${u.name}.
${u.task}

Setup: you are in your own git worktree. Run \`git checkout -b incompat/${u.name} ${BASE}\` (${BASE} is the head of branch incompat-t2, which holds I1 to I85 as merged; if the branch exists, check it out, and if it is checked out in another worktree, work there). Do not push, do not touch master or incompat-t2.
Follow CLAUDE.md (JAX, differentiable, dtype policy, registries, the trap list; run the trap-check skill's list on your diff). If the refusal has no test, write one that fires it first. Every computation goes to D22 by ${RECIPE} (read it fully; the shared slot queue; ship with a bundle, range ${BASE}..incompat/${u.name}; a fresh D22 worktree /l/ladovj1/w-inc-${u.name}, checking \`ls -d /l/ladovj1/w-inc-${u.name}*\` first). ${SLOWRULE}
Stop rule: at most three D22 runs of the check or about two hours of your time; past that commit with the refusal in place and a message quoting what you measured, and mark partial. A lifted refusal without a number is not acceptable.
Write or update notes/incompat/<ID>.md: the number and against what, the guide Limits sentence to change (docs/guide/*.md, simple-English style of .claude/rules/user-docs-style.md) and README cell, the slow tests for the user. Do not edit PLAN.md, README.md, docs/guide/ or notes/INCOMPATIBILITIES*.md.
${fix ? `THIS IS A FIX ROUND. The verifier's findings:\n${fix}\nAddress them on the same branch, re-run the check, commit, update the record; if you cannot, put the refusal back and mark partial.` : ''}
Return the structured result.`
}
function verifyPrompt(u, r) {
  return `You are the adversarial verifier of ${u.name} in defuqumat, branch incompat/${u.name} at ${r.commit || '(head)'}. The task was:\n${u.task}\nThe implementer reported: ${JSON.stringify(r)}\nTry to show it is wrong. Work in your own worktree, never commit to the branch. Computations go to D22 by ${RECIPE} in a fresh worktree /l/ladovj1/w-inc-${u.name}-ver* (check ls first). ${SLOWRULE}
(a) the new test or your harness of it FAILS with the touched modules taken from ${BASE} (git checkout ${BASE} -- <modules>; restore with git checkout HEAD -- .); (b) read any reference constant at its cited line in the file itself; (c) change something the check should not depend on (another cell, another constraint target, another band count) and see it hold, and ask whether a wrong implementation would have read differently; (d) the neighbouring refusals still fire, and no regime is newly let through unmeasured; (e) read the -rs skip list; (f) the diff against CLAUDE.md's trap list. Return confirmed (true only if the done claim survives), findings concrete enough for a fix round, d22_notes.`
}
phase('Implement')
const res = await parallel(UNITS.map(u => async () => {
  let r = null, v = null
  for (let round = 0; round < 2; round++) {
    r = await agent(implPrompt(u, round ? v.findings : ''), { label: `impl:${u.name}${round ? ':fix' : ''}`, phase: 'Implement', schema: IMPL, isolation: 'worktree', agentType: 'general-purpose' })
    if (!r || r.status !== 'done') break
    v = await agent(verifyPrompt(u, r), { label: `verify:${u.name}${round ? ':2' : ''}`, phase: 'Verify', schema: VERDICT, isolation: 'worktree', agentType: 'general-purpose' })
    if (!v || v.confirmed) break
    log(`${u.name}: refuted${round ? ', left as is' : ', one fix round'}`)
  }
  return { unit: u.name, impl: r, verdict: v }
}))
return res
