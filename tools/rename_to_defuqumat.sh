#!/usr/bin/env bash
# Rename the package defumat -> defuqumat in the working tree it is run from.
#
# Identifiers change (the import name, the CLI, every DEFUMAT_* variable, the
# cache directory); paths that exist on disk or on GitHub under the old name do
# not (the checkouts on Triton and D22, the scratch calculation directories),
# since renaming those is a move on each machine rather than an edit. Run it on
# a branch before merging it into a renamed master so both sides carry the same
# hunks.
#
#   tools/rename_to_defuqumat.sh            # tracked files plus *.local.md
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

self=tools/rename_to_defuqumat.sh

[ -d defumat ] && git mv defumat defuqumat
for f in slides/defumat.tex slides/defumat.pdf; do
    [ -e "$f" ] && git mv "$f" "${f/defumat/defuqumat}"
done

{
    git grep -I -l -i defumat || true
    find . -path ./.git -prune -o -name '*.local.md' -print | sed 's|^\./||'
} | sort -u | grep -v -x "$self" | while read -r f; do
    [ -f "$f" ] || continue
    grep -q -i defumat "$f" || continue
    perl -pi -e '
        s{(apps/|calculations/|/l/ladovj1/|w-k/|programs/)defumat}{$1\x01}g;
        s/defumat/defuqumat/g;
        s/DEFUMAT/DEFUQUMAT/g;
        s/\x01/defumat/g;
    ' "$f"
    # The notes name checkouts and calculation directories without their
    # prefix, as `defumat-jobs` or `defumat-gate`; those exist under the old name.
    case "$f" in
        *.md|*.tex) perl -pi -e 's/defuqumat-(55c|audit|bench|bessel|bigtests|card|chunk|docs|elk-b|factored|gate|gpu|hf|hier|hspin|hspin-card|jobs|ldos|ltf|ltf-old|magnetism|master|old|open|orient|owed|p121|p59|par|piezo|speed|spin|stall|thg|ulr-dual|vii1)\b/defumat-$1/g' "$f" ;;
    esac
done
