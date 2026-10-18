#!/usr/bin/env bash
# Finalize and evaluate the pointer-only baseline, then pull the results.
#
# Runs after main completes. The steps and their order come from RUNPOD.md.
#
#   ./scripts/baseline-evaluate.sh --host 81.27.69.177 --port 24431

set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
POD_ROOT=/workspace/decision
EVAL_ROOT=/workspace/decision-evaluation
TRANSFER_ROOT=/workspace/decision-transfer
RUNS=$POD_ROOT/runs/pointer-only
LEDGER=$POD_ROOT/runs/budget-pointer-only.json
MODEL_CACHE=$POD_ROOT/data/raw/processor-cache
PULL=$REPO/.audit/pointer-only

HOST=""; PORT=""; KEY="$HOME/.ssh/id_ed25519"; DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --key)  KEY="$2";  shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$HOST" ] && [ -n "$PORT" ] || { echo "usage: $0 --host HOST --port PORT [--dry-run]" >&2; exit 2; }

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -p "$PORT" -i "$KEY" "root@$HOST")
SCP=(scp -q -o BatchMode=yes -o ConnectTimeout=20 -P "$PORT" -i "$KEY")

run() {
  if [ "$DRY_RUN" -eq 1 ]; then printf '  remote$ %s\n' "$*"; else "${SSH[@]}" "$@"; fi
}

printf '== main must be finished ==\n'
run "test -f $RUNS/main/result.json && echo 'main result present' || { echo 'main result missing'; exit 1; }"

printf '== finalize, fitting temperature on calibration only ==\n'
run "cd $POD_ROOT && .venv/bin/python -m mdtrain finalize \
      --root $POD_ROOT \
      --checkpoint $RUNS/main/checkpoints/latest \
      --fit-temperature --device cuda \
      --budget-ledger $LEDGER \
      --output $POD_ROOT/runs/pointer-only-finalized"

printf '== transfer the locked archive ==\n'
if [ "$DRY_RUN" -eq 1 ]; then
  printf '  local$  scp dist/locked-evaluation-data.tar.gz dist/SHA256SUMS.locked -> root@%s:%s/dist/\n' "$HOST" "$TRANSFER_ROOT"
else
  "${SCP[@]}" "$REPO/dist/locked-evaluation-data.tar.gz" "$REPO/dist/SHA256SUMS.locked" \
    "root@$HOST:$TRANSFER_ROOT/dist/"
fi

printf '== verify and extract into a separate evaluation root ==\n'
# The code archive goes into the evaluation root as well. The locked contract checks
# verifier sources by path relative to that root, so a root holding only the locked
# data fails with "Missing or unconfined artifact: src/mdprep/common.py".
run "cd $TRANSFER_ROOT && sha256sum -c dist/SHA256SUMS.locked && \
     mkdir -p $EVAL_ROOT && \
     tar -xzf dist/training-code.tar.gz -C $EVAL_ROOT && \
     tar -xzf dist/locked-evaluation-data.tar.gz -C $EVAL_ROOT"

printf '== locked evaluation ==\n'
run "cd $EVAL_ROOT && HF_HUB_CACHE=$MODEL_CACHE \
     $POD_ROOT/.venv/bin/python -m mdtrain evaluate \
       --root $EVAL_ROOT \
       --artifact $POD_ROOT/runs/pointer-only-finalized \
       --splits test_id,test_ood --device cuda \
       --budget-ledger $LEDGER \
       --output $EVAL_ROOT/final-evaluation.json"

printf '== pull results ==\n'
if [ "$DRY_RUN" -eq 1 ]; then
  printf '  local$  scp root@%s:{final-evaluation.json, finalized model, freeze receipt} -> %s/\n' "$HOST" "$PULL"
else
  mkdir -p "$PULL"
  "${SCP[@]}" "root@$HOST:$EVAL_ROOT/final-evaluation.json" "$PULL/pointer-only-evaluation.json"
  "${SCP[@]}" -r "root@$HOST:$POD_ROOT/runs/pointer-only-finalized" "$PULL/pointer-only-finalized"
  "${SCP[@]}" "root@$HOST:$POD_ROOT/runs/pointer-only-finalized-freeze-receipt.json" "$PULL/" || \
    echo "  note: freeze receipt not found beside the artifact; check its actual name"
  "${SCP[@]}" "root@$HOST:$LEDGER" "$PULL/budget-pointer-only.json"
fi

cat <<EOF

== next ==
  The artifact is pulled to $PULL. Stop the pod once the copy is verified, because
  there is no network volume and the container disk does not survive a stop.
  Then compare against the trained run per slice:
    .venv/bin/python tools/chance_floor.py     (floors, unchanged)
  and read the pointer-only report against .audit/final-evaluation.json.
EOF
