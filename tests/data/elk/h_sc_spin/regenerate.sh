#!/bin/bash
# Regenerate this fixture: ./regenerate.sh path/to/elk
# Seconds on one core. Keeps the committed set and deletes everything else Elk writes.
set -euo pipefail
cd "$(dirname "$0")"
ELK="${1:?usage: ./regenerate.sh path/to/elk}"
KEEP=(elk.in H.in scf.in regenerate.sh README.md GEOMETRY.OUT INFO.OUT STATE.OUT RHO3D.OUT MAG3D.OUT)
OMP_NUM_THREADS=1 "$ELK"
args=()
for f in "${KEEP[@]}"; do args+=(-not -name "$f"); done
find . -maxdepth 1 -type f "${args[@]}" -delete
ls -la "${KEEP[@]}"
