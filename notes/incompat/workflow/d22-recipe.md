# Running an incompatibility item on D22 (CPU only)

Every computation (pytest beyond one quick unit file, any SCF, any reference comparison) runs on
D22, never on this workstation. Read docs/d22-notes.local.md sections from 2026-10-14 onwards once.

Host: `ssh ladovj1@D22-0161.org.aalto.fi` (BatchMode works). Shared repo `/l/ladovj1/defumat`; the
base branch there is `incompat-base` = master 0feb4ee. Fetches into it go under
`flock /l/ladovj1/git.lock`.

1. Commit your work on your local branch (in your local worktree). Ship it:
   `git bundle create /tmp/.../<ID>.bundle 0feb4ee..HEAD` fails when empty; use
   `git bundle create <file> 0feb4ee..<your-branch-name>` (the range must end at a ref),
   `scp <file> ladovj1@D22-0161.org.aalto.fi:/l/ladovj1/bundles/incompat-<ID>.bundle`, then on D22:
   `cd /l/ladovj1/defumat && flock /l/ladovj1/git.lock git fetch -q /l/ladovj1/bundles/incompat-<ID>.bundle <your-branch>:incompat-<ID>`
   (add `-f`/`+` refspec on a re-ship: `+<your-branch>:incompat-<ID>`).
2. Worktree, one per item and never moved while a job runs from it:
   `git worktree add --detach /l/ladovj1/w-inc-<ID> incompat-<ID>`; for a later version make
   `/l/ladovj1/w-inc-<ID>-v2` etc. rather than moving a worktree under a running job.
3. Run every job through the shared CPU queue (one core, one thread, 6 GB cap, waits for a free slot):
   put the launch in a script file (not a long `&&` chain on the ssh line), e.g.
   `/l/ladovj1/review/incompat/<ID>/run1.sh` containing
   `export DEFUQUMAT_QE_ROOT=/l/ladovj1/qe-7.5/qe-7.5`
   `nohup bash /l/ladovj1/review/guide/slot.sh /l/ladovj1/w-inc-<ID> . /l/ladovj1/review/incompat/<ID>/<tag>.log -m pytest -q -rs --tb=line -m 'slow or not slow' tests/unit/<file>.py > /dev/null 2>&1 < /dev/null & disown`
   or a harness: `... slot.sh <wt> . <log> /l/ladovj1/review/incompat/<ID>/harness.py`
   (give a harness by ABSOLUTE path). The log ends with `# exit <status> after <s> s`.
   Poll the log at intervals of at least 60 s; a 6 GB scope kill leaves a log that just stops
   (check `systemctl --user show <scope> -p Result`).
4. Without `DEFUQUMAT_QE_ROOT` every test reading test-suite/ SKIPS and the summary reads like a pass:
   always run with `-rs` and read the skips.
5. References on D22: QE test-suite at /l/ladovj1/qe-7.5/qe-7.5; serial pw.x and ph.x in
   /l/ladovj1/review/timing/ (also /l/ladovj1/qe-7.5/bin/pw.x); Elk in /l/ladovj1/review/harmonics-elk/bin/.
   A new reference output is copied back and committed under tests/data/qe/ with its input.
6. Traps: `pkill -f` inside ssh kills the ssh itself (use pgrep + kill by pid); do not scp over a running
   script; a jit closing over a big array folds it in as a constant; tests/unit/test_calculator.py::
   test_the_subspace_size_changes_the_cost_and_not_the_answer fails at master already (not yours).
7. Do not delete anything on D22 that you did not create; do not touch other worktrees. Do not touch
   the GPU (JAX_PLATFORMS=cpu is set by slot.sh). Nothing goes to Triton.
8. Anything learned about running on D22 that is not in docs/d22-notes.local.md goes in your final
   report under "d22 notes", so it can be appended (do not edit that file yourself).
9. A unit that is a family (strain, finitefield, spiral, ultracell, wedge, magnon) uses its family
   name wherever this recipe says <ID>: branch incompat/<name>, bundle incompat-<name>, worktree w-inc-<name>,
   run directory review/incompat/<name>/. The local branch name is incompat/<name>; on D22 it is fetched
   as incompat-<name>.
