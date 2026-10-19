export const meta = {
  name: 'incompat-t3',
  description: 'Implement tier T3 of notes/INCOMPATIBILITIES.md (I86-I122), checks on D22 CPU, verify, integrate onto incompat-t3',
  phases: [
    { title: 'Implement', detail: 'one agent per item or family, own worktree and branch incompat/<unit>, check on D22' },
    { title: 'Verify', detail: 'adversarial re-check of each done unit on D22' },
    { title: 'Integrate', detail: 'merge confirmed branches onto incompat-t3, fast gate on D22, record' },
  ],
}

const RECIPE = '/u/40/ladovj1/data/Documents/programs/claude/defuqumat/notes/incompat/workflow/d22-recipe-t3.md'
const LIST = 'notes/INCOMPATIBILITIES.md'
const BASE = '6c5b3a36'
const RECORD = 'notes/INCOMPATIBILITIES-T3.md'

const GROUPS = {
  chi0: ['I94', 'I95', 'I96'],
  sourcefree: ['I107', 'I108', 'I109'],
  rotate: ['I112', 'I113'],
  ultraseed: ['I116', 'I117'],
}
const RESUMED = new Set()
const DEPS = {}
const SKIP = {}

const items = args.items
const byId = Object.fromEntries(items.map(x => [x.id, x]))
const groupOf = {}
for (const [g, ids] of Object.entries(GROUPS)) ids.forEach(id => { groupOf[id] = g })
const units = []
const seen = new Set()
for (const it of items) {
  if (SKIP[it.id]) { log(`${it.id} left out: ${SKIP[it.id]}`); continue }
  const g = groupOf[it.id]
  if (g) {
    if (seen.has(g)) continue
    seen.add(g)
    units.push({ name: g, ids: GROUPS[g] })
  } else units.push({ name: it.id, ids: [it.id] })
}
const unitOfId = {}
units.forEach(u => u.ids.forEach(id => { unitOfId[id] = u.name }))
log(`${items.length} items in ${units.length} units: ${units.map(u => u.name).join(', ')}`)

function semaphore(n) {
  let free = n
  const waiting = []
  return {
    async take() { if (free > 0) { free--; return } await new Promise(r => waiting.push(r)) },
    give() { const w = waiting.shift(); if (w) w(); else free++ },
  }
}
const implSem = semaphore(9)
const resolvers = {}
const outcome = {}
units.forEach(u => { outcome[u.name] = new Promise(r => { resolvers[u.name] = r }) })

const IMPL = {
  type: 'object',
  properties: {
    unit: { type: 'string' },
    status: { type: 'string', enum: ['done', 'partial', 'blocked'] },
    branch: { type: 'string' },
    commit: { type: 'string' },
    d22_worktree: { type: 'string' },
    items: { type: 'array', items: { type: 'object', properties: {
      id: { type: 'string' }, status: { type: 'string', enum: ['done', 'partial', 'blocked'] },
      number: { type: 'string' }, check: { type: 'string' }, test: { type: 'string' },
      slow_tests_for_user: { type: 'string' } },
      required: ['id', 'status', 'number', 'check'] } },
    d22_notes: { type: 'string' },
    summary: { type: 'string' },
  },
  required: ['unit', 'status', 'branch', 'items', 'summary'],
}
const VERDICT = {
  type: 'object',
  properties: {
    unit: { type: 'string' },
    confirmed_ids: { type: 'array', items: { type: 'string' } },
    refuted_ids: { type: 'array', items: { type: 'string' } },
    findings: { type: 'string' },
    d22_notes: { type: 'string' },
  },
  required: ['unit', 'confirmed_ids', 'refuted_ids', 'findings'],
}

const SLOWRULE = `SLOW TESTS ARE NEVER RUN (the user's rule, 2026-10-16, binding on you): no pytest -m slow, no -m 'slow or not slow', no bare pytest on a file (it runs the slow tests in it), no tools/run_regression.sh. Every pytest run carries -m 'not slow' or names a single test that is not marked slow. A check that only a slow test reaches is measured with a harness script doing the same computation (run through slot.sh), and the slow test is written, marked slow, and listed for the user to run.`

function implPrompt(u, depNote, fix) {
  const list = u.ids.map(id => `${id} (${byId[id].tier}): ${byId[id].title}`).join('\n')
  return `You are implementing one unit of the incompatibility work in defuqumat (tier T3: an approximate identity, or one that needs a convergence). The user is not watching this run; a decision that would be theirs is taken the conservative way (keep the refusal, record the question) and written in the record. Unit: ${u.name}. Items:
${list}

Each item's full entry (guard, test asserting the refusal, missing term, the check, QE/Elk facts) is in ${LIST} under its "### <ID>." heading: read it first. Guard line numbers are of an older commit; grep the function name. The entries were written from a read of the refusals, not of the code around them, and sizings made that way have been wrong many times here, so read the guard's surroundings before trusting the stated missing term. The previous batches' records, notes/INCOMPATIBILITIES-NIGHT.md and notes/INCOMPATIBILITIES-T2.md, and the per-item records notes/incompat/I*.md show what a finished record looks like and what traps that batch hit (a refusal that did not fire on a shifted grid; a test that patched a method the new code no longer calls).
${depNote}
Setup: you are in your own git worktree, based on master ${BASE} (which already holds I1 to I85 as merged).${RESUMED.has(u.name) ? ` RESUMED: an earlier implementer of ${u.name} was cut off; its commits are on branch incompat/${u.name}, made on the OLD base 0feb4ee. Check that branch out (if git refuses because it is checked out in another worktree, find it with git worktree list and work in THAT worktree, reading git status there for uncommitted work), then \`git merge master\` (no rebase) and resolve conflicts against what I1 to I48 changed. Read its commits and any notes/incompat/${u.name}.md before continuing; its old D22 runs are under /l/ladovj1/review/incompat/${u.name}/ and were made on the old base, so its numbers must be re-measured on the merged head before they count.` : ` Run \`git checkout -b incompat/${u.name}\` (or check it out if it exists).`} Work only on that branch. Do not push. Do not touch master.

What to do, per item, in rank order:
1. If the entry says "Test: none", first write a test that fires the guard and see it fire before touching the guard.
2. Implement the missing term, following CLAUDE.md (JAX, differentiable, dtype policy, map_k/sum_k, registries, the trap list; run the trap-check skill's list against your diff before committing).
3. Turn the refusal into a check: a test that asserts the number of the entry's check (the committed reference file and line, or the identity). These are T3 items, so the check is approximate or needs a convergence: name the convergence parameter (nbnd, k-mesh, finite-difference step, propagation time, degauss), show the discrepancy at no fewer than two values of it and that it falls the way the approximation predicts, and set the test tolerance from the converged number, not from a single point that happens to pass. An agreement that would also have been read with the missing term absent is not a check; say what the discrepancy is with the term switched off. Where the entry offers a T4 route (a new pw.x or ph.x run), the serial binaries on D22 are allowed through slot.sh, and a new reference output is committed with its input under tests/data/qe/. A new test above about five seconds that is not a direct pw.x number is marked slow. The guard stays for any regime the fix does not cover, and its sibling refusals must still fire (a test for each).
4. Run the check on D22, CPU only, by the recipe in ${RECIPE} (read it fully; every computation goes there through the shared slot queue; nothing heavy on this workstation, which is shared). Ship your branch with a bundle as it says (range ${BASE}..incompat/${u.name}) and use a D22 worktree /l/ladovj1/w-inc-${u.name}${RESUMED.has(u.name) ? '-t2' : ''} (a new -v2 etc. for a new version, never move a worktree under a running job).
${SLOWRULE}
5. Commit. Write notes/incompat/<ID>.md per item: the number measured and against what (file:line or identity, with the sample count where it is a median), what was missing (the term as found in the code, saying where the entry's stated term was wrong), the exact sentence of the user-guide page's Limits note (docs/guide/*.md, name the page) to delete or replace and the replacement (in the simple-English style of .claude/rules/user-docs-style.md), the README feature-table row and cell to change, and the slow tests the user should run. Do NOT edit PLAN.md, README.md, docs/guide/, notebooks or ${LIST}; the integrator does that.
Stop rule: per item, at most three D22 runs of the check or about 90 minutes of your own time. Past that, commit with the refusal in place and a sharpened message quoting what you measured, write the record, and mark the item partial or blocked. A partial never lifts the guard. A lifted refusal without a number is not acceptable. A refusal that turns out to be physics rather than a missing term (as I29 was) is recorded as blocked with the argument, and the message says so.
${fix ? `\nTHIS IS A FIX ROUND. A verifier refuted part of this unit's earlier work, on the same branch incompat/${u.name}. Its findings:\n${fix}\nAddress them on the same branch: if \`git checkout incompat/${u.name}\` refuses because the branch is checked out in another worktree, find that worktree with \`git worktree list\` and work there instead. Re-run the check, commit, and update the records. If you cannot, put the refusal back for the refuted items and mark them partial.\n` : ''}
Return the structured result: unit, overall status, branch, final commit sha, D22 worktree, per item {id, status, number (the measured figure and its reference, or the discrepancy), check, test (file::name), slow_tests_for_user}, d22_notes (anything learned about running on D22 that is not in docs/d22-notes.local.md, else empty), and a short summary.`
}

function verifyPrompt(u, r) {
  return `You are the adversarial verifier of one unit of the incompatibility work in defuqumat. Unit ${u.name}, branch incompat/${u.name} at ${r.commit || '(its head)'}, D22 worktree ${r.d22_worktree || '/l/ladovj1/w-inc-' + u.name}. The implementer reported:
${JSON.stringify(r.items, null, 1)}
Summary: ${r.summary}

Your job is to try to show each claimed "done" item is wrong. You are in a worktree of your own; never edit the branch incompat/${u.name} and never commit. Do the revert of (a) in your own worktree (git checkout --detach incompat/${u.name}, then git checkout ${BASE} -- <modules>) or in a D22 worktree followed by git checkout -- . there. Computations go to D22 by ${RECIPE}; make a new D22 worktree /l/ladovj1/w-inc-${u.name}-ver rather than one the implementer might still move.
${SLOWRULE} Where the check is a slow test, reproduce its computation with your own harness script.
Re-running the implementer's own command only shows again that it passes, so instead:
(a) check the new test (or your harness of it) FAILS with the implementation reverted (the touched modules checked out from ${BASE}), so it is not a tautology;
(b) read the asserted constant at the cited line of the reference file itself, not in the test;
(c) for an identity, change something the identity should be independent of (swap the arms, change the mesh or the cell) and see it still holds; and ask whether the alternative hypothesis would have read differently;
(d) run the refusal tests of the neighbouring regimes that the touched guard must still refuse, and look for a regime the lifted guard now lets through that nothing covers (last batch: a shifted grid passed a uniformity test);
(e) read the -rs skip list of your own log (a missing DEFUQUMAT_QE_ROOT reads as a pass);
(f) read the diff against CLAUDE.md's trap list (abs at a forced zero, np.asarray in a differentiated path, hardcoded dtype, a hand-rolled vmap over k, wedge symmetrisation, rule D4, a caller-built k-set without for_spin), and check that a test that monkeypatches a method still patches the one the new code calls;
(g) these are T3 items, whose check is approximate or needs a convergence: measure at one more value of the convergence parameter that the implementer did not use and confirm the discrepancy keeps falling the way the approximation predicts, and re-measure yourself the discrepancy with the new term switched off that the implementer quoted, rather than taking it from the record; a tolerance set from a single point that happens to pass is a refutation.
Partial or blocked items: confirm only that the guard still fires and the record states a measured discrepancy or an argument.
Return confirmed_ids (done items that survived), refuted_ids (with the reason in findings, concrete enough for a fix round), findings, and d22_notes.`
}

async function waitDeps(u) {
  const needed = u.ids.flatMap(id => DEPS[id] || [])
  const pre = [...new Set(needed.map(id => unitOfId[id]))].filter(n => n && n !== u.name)
  if (!pre.length) return ''
  const outs = await Promise.all(pre.map(n => outcome[n]))
  const ok = needed.every(id => outs.some(o => o && o.confirmed.includes(id)))
  if (!ok) return null
  return `\nPrerequisites ${needed.join(', ')} were confirmed on branches ${pre.map(n => 'incompat/' + n).join(', ')}: merge them into your branch first (git merge, no rebase).\n`
}

async function runUnit(u) {
  const depNote = await waitDeps(u)
  if (depNote === null) {
    const o = { unit: u.name, status: 'skipped', reason: 'prerequisite not confirmed', confirmed: [], refuted: [], impl: null, verdicts: [] }
    resolvers[u.name](o)
    return o
  }
  let impl = null
  const verdicts = []
  let confirmed = [], refuted = []
  for (let round = 0; round < 2; round++) {
    await implSem.take()
    try {
      impl = await agent(implPrompt(u, depNote, round ? verdicts[verdicts.length - 1].findings : ''), {
        label: `impl:${u.name}${round ? ':fix' : ''}`, phase: 'Implement', schema: IMPL, isolation: 'worktree', agentType: 'general-purpose' })
    } finally { implSem.give() }
    if (!impl) break
    const done = impl.items.filter(i => i.status === 'done').map(i => i.id)
    const v = await agent(verifyPrompt(u, impl), { label: `verify:${u.name}${round ? ':2' : ''}`, phase: 'Verify', schema: VERDICT, isolation: 'worktree', agentType: 'general-purpose' })
    if (!v) break
    verdicts.push(v)
    confirmed = v.confirmed_ids.filter(id => done.includes(id))
    refuted = v.refuted_ids
    if (!refuted.length) break
    log(`${u.name}: refuted ${refuted.join(', ')}${round ? ', left as is' : ', one fix round'}`)
  }
  const o = { unit: u.name, status: impl ? impl.status : 'died', branch: impl && impl.branch, commit: impl && impl.commit, confirmed, refuted, impl, verdicts }
  resolvers[u.name](o)
  log(`${u.name}: ${o.status}, confirmed ${confirmed.length}/${u.ids.length}`)
  return o
}

phase('Implement')
const FIRST_BATCH = 16
const runs = units.map(u => runUnit(u).catch(e => {
  const o = { unit: u.name, status: 'error', error: String(e), confirmed: [], refuted: [] }
  resolvers[u.name](o)
  return o
}))

function slimOf(results) {
  return results.map(r => ({ unit: r.unit, status: r.status, branch: r.branch, commit: r.commit, confirmed: r.confirmed, refuted: r.refuted,
    items: r.impl ? r.impl.items : [], d22_notes: [r.impl && r.impl.d22_notes, ...(r.verdicts || []).map(v => v.d22_notes)].filter(Boolean).join(' | '),
    findings: (r.verdicts || []).map(v => v.findings).join(' | ').slice(0, 1500), reason: r.reason || r.error || '' }))
}

function integPrompt(slim, first, pending) {
  return `You are the integrator of the incompatibility work in defuqumat, tier T3 (I86 to I122). This is integration ${first ? 'ONE (the first ' + slim.length + ' units in rank order)' : 'TWO (the remaining units)'}. The outcome of each unit in this batch:
${JSON.stringify(slim)}
${pending.length ? 'Units still running and NOT in this batch: ' + pending.join(', ') : ''}

Work in your own worktree only; do not touch the main checkout or master, and do not push. Follow how notes/INCOMPATIBILITIES-T2.md and PLAN.md P155 did the previous batch.
0. ${first ? `Create branch incompat-t3 from master (${BASE}) and check it out.` : 'Check out the existing branch incompat-t3 (if it is checked out in another worktree, find it with git worktree list and work there).'} ${first ? 'Write' : 'Update'} ${RECORD} FIRST, with one line per unit of this batch and its reported outcome, and commit it; update and commit it after every later step, so that if you die the branch and the record are readable.
1. Merge, one by one in rank order, every branch with confirmed items (git merge --no-ff incompat/<unit>), committing after each; check each branch tip matches the reported sha. Resolve conflicts by keeping both sides' terms (most are neighbouring if-branches in one guard, and sibling tests that assert the other's refusal must be dropped or retargeted); if a conflict is not plain, abort that merge and record it as unmerged rather than spend long on it. After each conflicted merge, run the touched unit test files locally with -m 'not slow' only if they need no SCF, otherwise on D22. For a unit with confirmed and unconfirmed items, merge only if the unconfirmed items' guards still fire on it; otherwise take only its record files. From partial, blocked, refuted and skipped units take only the record files (git checkout incompat/<unit> -- notes/incompat/), when the branch exists.
2. Apply each landed item's record: the guide Limits sentence and README cell it names (one commit; user-facing text in the style of .claude/rules/user-docs-style.md, and tools/check_guide.py must pass), move the landed entries of ${LIST} into its "Closed since" section with the number, mark the partial and blocked ones in place, and ${first ? 'write one PLAN.md §3 entry (next phase number after the last in PLAN.md)' : 'extend the PLAN.md §3 entry integration one wrote'} with a table: ID, number, check, branch.
3. Run the fast gate on D22 by ${RECIPE}, NOT through slot.sh: bundle incompat-t3 (range ${BASE}..incompat-t3), fetch it on D22 as ${first ? 'incompat-t3' : '+incompat-t3:incompat-t3'}, a fresh worktree /l/ladovj1/w-inc-t3${first ? '' : 'b'} (a further -c, -d for a re-gate), and run tools/test-fast.sh there pinned to eight cores disjoint from the slot pool (taskset -c 0-3,12-15), with DEFUQUMAT_QE_ROOT=/l/ladovj1/qe-7.5/qe-7.5, /l/ladovj1/venv/bin first on PATH, -rs, in a systemd-run --user --scope with MemoryMax=16G, launched from a script with nohup and disown, polled at 60 s or more. ${SLOWRULE} Known not to be yours: the two tests/unit/test_check_guide.py failures (No module named 'yaml', the D22 venv's); any other failure is real. On a failure, attribute the failing file to the branch that touched it, git revert -m 1 that merge, undo its ledger, Limits and PLAN rows, record it as unmerged, re-gate; at most twice.
4. Finish ${RECORD}: per ID the outcome (landed with number / partial with the measured discrepancy / blocked and why / refuted and why / skipped), the gate result with its counts, the slow test files the merges touch or add, listed for the user to run, the follow-up deliverables that remain (guide prose, notebooks, timings), every "d22 notes" item from the reports (for the user to append to docs/d22-notes.local.md), and the D22 worktrees w-inc-* that are removable (remove none). Commit.
Return a short report: counts landed/partial/blocked/refuted/skipped, the gate result, the head of incompat-t3, the slow files for the user, and anything the user must decide.`
}

const firstNames = units.slice(0, FIRST_BATCH).map(u => u.name)
const restNames = units.slice(FIRST_BATCH).map(u => u.name)
const integOne = (async () => {
  const res = await Promise.all(runs.slice(0, FIRST_BATCH))
  return agent(integPrompt(slimOf(res), true, restNames), { label: 'integrate:1', phase: 'Integrate', isolation: 'worktree', agentType: 'general-purpose' })
})()
const restRes = await Promise.all(runs.slice(FIRST_BATCH))
const one = await integOne
let two = null
if (restRes.length) {
  phase('Integrate')
  two = await agent(integPrompt(slimOf(restRes), false, []), { label: 'integrate:2', phase: 'Integrate', isolation: 'worktree', agentType: 'general-purpose' })
}
const all = slimOf(await Promise.all(runs))
return { units: all.map(s => ({ unit: s.unit, status: s.status, confirmed: s.confirmed, refuted: s.refuted })), integ: [one, two] }
