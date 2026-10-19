# Tier T3 of the incompatibility work, integration one

This file is the record of how the units of `notes/INCOMPATIBILITIES.md` in tier T3 (I86 to
I122) are brought onto the branch `incompat-t3`, which starts from master at `63804fc5`
(`6c5b3a36`, the merge of T2, plus the two commits of the T3 workflow files). The
implement/verify workflow stopped early at a usage limit, so this integration covers what had
finished. The rules are T2's: merge in item order with `--no-ff` and a commit after each,
record files only from a partial, refuted or unverified unit, one fast gate on D22 of the head,
a merge that fails the gate reverted. Slow tests are not run (the user's rule of 2026-10-16);
they are listed for the user. The outcome of each unit is taken from the integration input as
given (`t3-integ-input.json`, whose verifier findings are cut at about 1500 characters each); the
full verifier reports are in the D22 log directories each one names, `/l/ladovj1/review/incompat/<ID>-ver*/`.

## The units, as reported, and what this integration does

| unit | branch | commit | reported | integration |
|---|---|---|---|---|
| I86 | incompat/I86 | 90d92f55 | done, confirmed | merge |
| I87 | incompat/I87 | 9d858042 | done, confirmed (two record sentences to fix) | merge |
| I88 | incompat/I88 | 6967d36b | done, refuted on scope | record only |
| I89 | incompat/I89 | 3ad688ca | done, confirmed | merge |
| I90 | incompat/I90 | 0082ea99 | done, refuted (wedge regime) | record only |
| I91 | incompat/I91 | 3a36ecfb | partial, guard holds | record only |
| I92 | incompat/I92 | 3aae8e8e | done, confirmed | merge |
| I93 | incompat/I93 | f8891a9d | done, refuted (crossing filter) | record only |
| chi0 (I94 to I96) | incompat/chi0 | c2a83c97 | done, verifier did not finish | record only |
| I97 | incompat/I97 | c189b039 | done, confirmed | merge |
| I98 | incompat/I98 | 549c2b3a | done, confirmed (one record sentence false) | merge |
| I99 | incompat/I99 | 48f06b81 | done, refuted (not reachable from `Calculator`) | record only |
| I100 | incompat/I100 | 59f27d1f | done, confirmed | merge |
| I101 | incompat/I101 | fff605f3 | done, refuted (T=800 breaks the tolerance) | record only |
| I102 | incompat/I102 | 39f91dbb | done, confirmed | merge |
| I103 | incompat/I103 | ee7325ae | done, confirmed | merge |
| I104 | incompat/I104 | 24003058 | partial, refuted on its precision claims | record only |
| I105 | incompat/I105 | d1acce97 | partial, guard holds | record only |
| I106 | incompat/I106 | 65f9f183 | partial, confirmed as partial | record only |
| sourcefree (I107 to I109) | incompat/sourcefree | c5401217 | I108, I109 done, I107 partial; verifier did not run | record only |
| I110 | incompat/I110 | 8efa0ec4 | done, confirmed | merge |
| I114 | incompat/I114 | e80bc66f | partial; verifier did not run | record only |
| I111, rotate (I112, I113), I115, ultraseed (I116, I117), I118 to I122 | none | none | not started (session limit) | nothing |

## Progress

- Step 0: branch made from master `63804fc5`, this file written. Every branch tip matched the
  reported sha, and every branch sits on `6c5b3a36`.
