export const meta = {
  name: 'incompat-night',
  description: 'Implement the 141 ranked incompatibilities of notes/INCOMPATIBILITIES.md, checks on D22 CPU, verify, integrate onto incompat-night',
  phases: [
    { title: 'Implement', detail: 'one agent per item or family, own worktree and branch incompat/<unit>, check on D22' },
    { title: 'Verify', detail: 'adversarial re-check of each done unit on D22' },
    { title: 'Integrate', detail: 'merge confirmed branches onto incompat-night, gate on D22, night record' },
  ],
}

const S = '/tmp/claude-3291340/-u-40-ladovj1-data-Documents-programs-claude-defuqumat/1e2f8321-329c-4ea5-8cb0-e39c316739dd/scratchpad'
const RECIPE = `${S}/d22-recipe.md`
const LIST = 'notes/INCOMPATIBILITIES.md'

// Families decided by Fable: one piece of code each, so one agent each.
const GROUPS = {
  strain: ['I10', 'I11'],
  finitefield: ['I14', 'I15', 'I56', 'I57'],
  spiral: ['I20', 'I21', 'I22', 'I23', 'I24', 'I25', 'I26', 'I27', 'I28', 'I29', 'I30'],
  ultracell: ['I31', 'I32', 'I33', 'I34', 'I35', 'I36'],
  wedge: ['I40', 'I41', 'I42', 'I43', 'I44', 'I45', 'I46', 'I47', 'I48'],
  magnon: ['I58', 'I59', 'I60', 'I61', 'I62'],
}
// Prerequisites: a dependent unit starts only once its prerequisite is CONFIRMED.
const RESUMED = new Set(['spiral', 'ultracell', 'wedge', 'I51', 'I52', 'I53'])
const DEPS = { I79: ['I2'], I82: ['I2'], I83: ['I5'], I84: ['I79'], I85: ['I15'] }

const items = args.items
const byId = Object.fromEntries(items.map(x => [x.id, x]))
const groupOf = {}
for (const [g, ids] of Object.entries(GROUPS)) ids.forEach(id => { groupOf[id] = g })
const units = []
const seen = new Set()
for (const it of items) {
  const g = groupOf[it.id]
  if (g) {
    if (seen.has(g)) continue
    seen.add(g)
    units.push({ name: g, ids: GROUPS[g], tier: GROUPS[g].map(i => byId[i].tier).sort()[0] })
  } else {
    units.push({ name: it.id, ids: [it.id], tier: it.tier })
  }
}
// The user's decision (2026-10-16 morning): this session does the first third only, I1 to I48.
const CUT = units.findIndex(u => u.name === 'wedge') + 1
log(`this session: ${units.slice(0, CUT).map(u => u.name).join(', ')}; left for another session: ${units.slice(CUT).map(u => u.name).join(', ')}`)
units.splice(CUT)
const unitOfId = {}
units.forEach(u => u.ids.forEach(id => { unitOfId[id] = u.name }))
log(`${items.length} items in ${units.length} units`)

// A semaphore so that at most 9 implementers run at once (D22 has 6 queue slots).
function semaphore(n) {
  let free = n
  const waiting = []
  return {
    async take() { if (free > 0) { free--; return } await new Promise(r => waiting.push(r)) },
    give() { const w = waiting.shift(); if (w) w(); else free++ },
  }
}
const implSem = semaphore(9)

// Per-unit outcome promises, so dependents can wait for prerequisites.
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
      number: { type: 'string' }, check: { type: 'string' }, test: { type: 'string' } },
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

function implPrompt(u, depNote, fix) {
  const list = u.ids.map(id => `${id} (${byId[id].tier}): ${byId[id].title}`).join('\n')
  return `You are implementing one unit of the overnight incompatibility work in defuqumat. The user is asleep; decisions that would be theirs were made by Fable and are below. Unit: ${u.name}. Items:
${list}

Each item's full entry (guard, test asserting the refusal, missing term, the check, QE/Elk facts) is in ${LIST} under its "### <ID>." heading: read it first. The entries were written from a read of the refusals, not of the code around them; the project records that sizings made that way have been wrong many times, so read the guard's surroundings before trusting the stated missing term.
${depNote}${RESUMED.has(u.name) ? `\nRESUMED: an earlier implementer of this unit was cut off by a session limit. Its work is on branch incompat/${u.name}, probably checked out in a leftover worktree: find it with \`git worktree list\` and work in THAT worktree, continuing from its commits and any uncommitted changes there (read git log and git status first), and its D22 runs under /l/ladovj1/review/incompat/${u.name}/.\n` : ''}
Setup: you are in your own git worktree. First run \`git checkout -b incompat/${u.name}\` (or check it out if it exists). Work only on that branch. Do not push. Do not touch master.

What to do, per item, in rank order:
1. If the entry says "Test: none", first write a test that fires the guard and see it fire (on D22; locally only if that test needs no SCF at all, since this workstation is shared) before touching the guard.
2. Implement the missing term, following CLAUDE.md (JAX, differentiable, dtype policy, map_k/sum_k, registries, the trap list; run the trap-check skill's list against your diff before committing).
3. Turn the refusal into a check: a test that asserts the number of the entry's check (the committed reference file and line, or the exact identity). A new test above about five seconds that is not a direct pw.x number is marked slow. The guard stays for any regime the fix does not cover, and its sibling refusals must still fire.
4. Run the check on D22, CPU only, by the recipe in ${RECIPE} (read it fully; every computation goes there, through the shared slot queue; nothing heavy on this workstation, which is shared). Ship your branch with a bundle as it says and use a D22 worktree /l/ladovj1/w-inc-${u.name} (a new -v2 etc. for a new version, never move a worktree under a running job).
5. Commit. Write notes/incompat/<ID>.md per item: the number measured and against what (file:line or identity, with the sample count where it is a median), what was missing (the term as found in the code, saying where the entry's stated term was wrong), the exact sentence of the user-guide page's Limits note (docs/guide/*.md, name the page) to delete or replace and the replacement, and the README feature-table row and cell to change. Do NOT edit PLAN.md, README.md, docs/guide/, notebooks or ${LIST}; the integrator does that.
Stop rule: per item, at most three D22 runs of the check or about 90 minutes of your own time. Past that, commit with the refusal in place and a sharpened message quoting what you measured, write the record, and mark the item partial or blocked. A partial never lifts the guard. A lifted refusal without a number is not acceptable. A T5 item lands only as its named identity, and its record says "no external reference by construction". A T4 item needs a pw.x/ph.x/Elk run on D22 first, its input and output committed under tests/data/qe/ (or the Elk equivalent).
${fix ? `\nTHIS IS A FIX ROUND. A verifier refuted part of this unit's earlier work, on the same branch incompat/${u.name}. Its findings:\n${fix}\nAddress them on the same branch: if \`git checkout incompat/${u.name}\` refuses because the branch is checked out in another worktree, find that worktree with \`git worktree list\` and work there instead. Re-run the check, commit, and update the records. If you cannot, put the refusal back for the refuted items and mark them partial.\n` : ''}
Return the structured result: unit, overall status, branch, final commit sha, D22 worktree, per item {id, status, number (the measured figure and its reference, or the discrepancy), check, test (file::name)}, d22_notes (anything learned about running on D22 that is not in docs/d22-notes.local.md, else empty), and a short summary.`
}

function verifyPrompt(u, r) {
  return `You are the adversarial verifier of one unit of the overnight incompatibility work in defuqumat. Unit ${u.name}, branch incompat/${u.name} at ${r.commit || '(its head)'}, D22 worktree ${r.d22_worktree || '/l/ladovj1/w-inc-' + u.name}. The implementer reported:
${JSON.stringify(r.items, null, 1)}
Summary: ${r.summary}

Your job is to try to show each claimed "done" item is wrong. You are in a worktree of your own; never edit the branch incompat/${u.name} and never commit. Do the revert of (a) in your own worktree (git checkout <branch> then git checkout 0feb4ee -- <modules>) or in a D22 worktree followed by git checkout -- . there. Computations go to D22 by ${RECIPE}; you may reuse the implementer's D22 worktree, but make a new one if the implementer's might still move. Re-running the implementer's own command only shows again that it passes, so instead:
(a) check the new test FAILS with the implementation reverted (the touched modules checked out from 0feb4ee under the new test), so it is not a tautology;
(b) read the asserted constant at the cited line of the reference file itself, not in the test;
(c) for an identity, change something the identity should be independent of (swap the arms, change the mesh or the cell) and see it still holds; and ask whether the alternative hypothesis would have read differently;
(d) run the refusal tests of the neighbouring regimes that the touched guard must still refuse;
(e) read the -rs skip list of your own log (a missing DEFUQUMAT_QE_ROOT reads as a pass);
(f) read the diff against CLAUDE.md's trap list (abs at a forced zero, np.asarray in a differentiated path, hardcoded dtype, a hand-rolled vmap over k, wedge symmetrisation, rule D4, a caller-built k-set without for_spin).
Partial or blocked items: confirm only that the guard still fires and the record states a measured discrepancy.
Return confirmed_ids (done items that survived), refuted_ids (with the reason in findings, concrete enough for a fix round), findings, and d22_notes.`
}

async function waitDeps(u) {
  const pre = [...new Set(u.ids.flatMap(id => DEPS[id] || []).map(id => unitOfId[id]))].filter(n => n !== u.name)
  if (!pre.length) return ''
  const outs = await Promise.all(pre.map(n => outcome[n]))
  const needed = u.ids.flatMap(id => DEPS[id] || [])
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
// Two integrations, so a result exists by morning without cutting the list: the first
// FIRST_BATCH units are integrated as soon as they all end, while the rest keep running.
const FIRST_BATCH = units.length
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
  return `You are the integrator of the overnight incompatibility work in defuqumat (user asleep; Fable's decisions apply). This is integration ${first ? 'ONE (the first ' + slim.length + ' units in rank order)' : 'TWO (the remaining units)'}. The outcome of each unit in this batch:
${JSON.stringify(slim)}
Only I1 to I48 were attempted in this session; I49 to I141 are left for another session (say so in the night file, and do not list them as failed). ${pending.length ? 'Units still running and NOT in this batch: ' + pending.join(', ') : ''}

Work in your own worktree only; do not touch the main checkout or master, and do not push.
0. ${first ? 'Create branch incompat-night from master (862235d) and check it out.' : 'Check out the existing branch incompat-night (if it is checked out in another worktree, find it with git worktree list and work there).'} ${first ? 'Write' : 'Update'} notes/INCOMPATIBILITIES-NIGHT.md FIRST, with one line per unit of this batch and its reported outcome, and commit it; update and commit it after every later step, so that if you die the branch and the record are readable.
1. Merge, one by one in rank order, every branch with confirmed items (git merge --no-ff incompat/<unit>), committing after each. Resolve conflicts by keeping both sides' terms (most are neighbouring if-branches in one guard); if a conflict is not plain, abort that merge and record it as unmerged rather than spend long on it. For a unit with confirmed and unconfirmed items, merge only if the unconfirmed items' guards still fire on it; otherwise take only its record files. From partial, blocked, refuted and skipped units take only the record files (git checkout incompat/<unit> -- notes/incompat/), when the branch exists.
2. Apply each landed item's record: the guide Limits sentence and README cell it names (one commit), move the landed entries of ${LIST} into its "Closed since" section with the number, and ${first ? 'write one PLAN.md §3 entry (next phase number after the last in PLAN.md)' : 'extend the PLAN.md §3 entry integration one wrote'} with a table: ID, number, check, branch. Fix the stale texts the sweeps found if a landed item touches them (.claude/rules/scope.md real-time bullet, tddft/spinkernel.py:51-67, notes/ELK-FEATURES.md §9, response/piezo.py:374 message).
3. Run the fast gate on D22 by ${RECIPE}, NOT through slot.sh: bundle incompat-night (range 0feb4ee..incompat-night), fetch it on D22 as ${first ? 'incompat-night' : '+incompat-night:incompat-night'}, a fresh worktree /l/ladovj1/w-inc-night${first ? '' : '2'}, and run tools/test-fast.sh there pinned to eight cores disjoint from the slot pool (taskset -c 0-3,12-15), with DEFUQUMAT_QE_ROOT=/l/ladovj1/qe-7.5/qe-7.5, /l/ladovj1/venv/bin first on PATH, -rs, in a systemd-run --user --scope with a memory cap, launched from a script with nohup and disown, polled at 60 s or more. Known not to be yours: tests/unit/test_calculator.py::test_the_subspace_size_changes_the_cost_and_not_the_answer fails at master; a core-count test in tests/unit/test_config.py fails under a one-core slot by construction. On a failure, attribute the failing file to the branch that touched it, git revert -m 1 that merge, record it as unmerged, re-gate; at most twice.
4. Finish notes/INCOMPATIBILITIES-NIGHT.md: per ID the outcome (landed with number / partial with the measured discrepancy / blocked and why / refuted and why / skipped / still running), the gate result with its counts, the follow-up deliverables that remain (guide prose, notebooks, timings), every "d22 notes" item from the reports (for the user to append to docs/d22-notes.local.md), and the D22 worktrees w-inc-* that are removable (remove none). Commit.
Return a short report: counts landed/partial/blocked/refuted/skipped, the gate result, the head of incompat-night, and anything the user must decide.`
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
