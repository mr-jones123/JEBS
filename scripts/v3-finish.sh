#!/usr/bin/env bash
# Finalize and evaluate v3, run its external predictions, then checksum, pull, and verify
# every receipt. Everything stays on the network volume; this script never stops a pod or
# deletes the volume.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRANSFER_ROOT=/workspace/decision-transfer
V3_ROOT=/workspace/decision-v3
EVAL_ROOT=/workspace/decision-v3-evaluation
EXTERNAL_ROOT=/workspace/external
RESULTS_ROOT=/workspace/results
RUN_ROOT=$V3_ROOT/runs/v3-paired
LEDGER=$V3_ROOT/runs/budget-v3-paired.json
FINALIZED=$V3_ROOT/runs/v3-finalized
PULL=$REPO/.audit/v3-paid-run
EXTERNAL_SETS=(rule-probe hard-v1-test doclaynet rule-variants-v1)

HOST=""; PORT=""; KEY="$HOME/.ssh/id_ed25519"; DRY_RUN=0; FROM_EXTERNAL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --key) KEY="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --from-external) FROM_EXTERNAL=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$HOST" ] && [ -n "$PORT" ] || {
  echo "usage: $0 --host HOST --port PORT [--key PATH] [--from-external] [--dry-run]" >&2
  exit 2
}
[ -f "$KEY" ] || { echo "ssh key not found: $KEY" >&2; exit 2; }

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -p "$PORT" -i "$KEY" "root@$HOST")
SCP=(scp -q -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -P "$PORT" -i "$KEY")
run() {
  if [ "$DRY_RUN" -eq 1 ]; then printf '  remote$ %s\n' "$*"; else "${SSH[@]}" "$@"; fi
}
completed() {
  printf "python3 -c \"import json, sys; sys.exit(0 if json.load(open('%s')).get('status') == 'completed' else 1)\"" "$1"
}

if [ "$FROM_EXTERNAL" -eq 1 ]; then
  printf '== resume at external predictions: finalize and locked evaluation must already be complete ==\n'
  run "test -f $FINALIZED/artifact.json && test -f $EVAL_ROOT/final-evaluation.json && \
       $(completed $RESULTS_ROOT/v3-finalize-command.json) && $(completed $RESULTS_ROOT/v3-locked-evaluate-command.json) && \
       echo 'finalize and locked evaluation completed'"
else
  printf '== main must be complete ==\n'
  run "$(completed $RUN_ROOT/main/result.json) && echo 'main completed'"

  printf '== finalize v3 with scalar temperature ==\n'
  run "cd $V3_ROOT && HF_HUB_CACHE=$V3_ROOT/data/raw/processor-cache \
       .venv/bin/python -m mdtrain finalize --root $V3_ROOT \
         --checkpoint $RUN_ROOT/main/checkpoints/latest --fit-temperature --device cuda \
         --budget-ledger $LEDGER --output $FINALIZED \
       > $RESULTS_ROOT/v3-finalize-command.json"

  printf '== verify and extract the locked evaluation root only after finalization ==\n'
  run "cd $TRANSFER_ROOT/v3 && sha256sum -c dist/SHA256SUMS.locked && \
       rm -rf $EVAL_ROOT && mkdir -p $EVAL_ROOT && \
       tar -xzf dist/training-code.tar.gz -C $EVAL_ROOT && \
       tar -xzf dist/locked-evaluation-data.tar.gz -C $EVAL_ROOT"

  printf '== locked test_id and test_ood evaluation ==\n'
  run "cd $EVAL_ROOT && HF_HUB_CACHE=$V3_ROOT/data/raw/processor-cache \
       $V3_ROOT/.venv/bin/python -m mdtrain evaluate --root $EVAL_ROOT \
         --artifact $FINALIZED --splits test_id,test_ood --device cuda \
         --budget-ledger $LEDGER --output $EVAL_ROOT/final-evaluation.json \
       > $RESULTS_ROOT/v3-locked-evaluate-command.json"
fi

printf '== v3 external predictions ==\n'
for dataset in "${EXTERNAL_SETS[@]}"; do
  if [ "$dataset" = doclaynet ]; then media_root="$EXTERNAL_ROOT/doclaynet-candidate"; else media_root="$V3_ROOT"; fi
  run "cd $V3_ROOT && HF_HUB_CACHE=$V3_ROOT/data/raw/processor-cache \
       .venv/bin/python -m mdtrain predict --root $V3_ROOT --artifact $FINALIZED \
         --input $EXTERNAL_ROOT/requests/$dataset.requests.jsonl --media-root $media_root \
         --device cuda --budget-ledger $LEDGER \
         --output $RESULTS_ROOT/v3-$dataset.predictions.jsonl \
       > $RESULTS_ROOT/v3-$dataset.predict-receipt.json && $(completed $RESULTS_ROOT/v3-$dataset.predict-receipt.json)"
done

printf '== assemble the recovery bundle ==\n'
BUNDLE=v3-paid-results.tar.gz
run "cd /workspace && for path in decision-v3/runs/v3-paired/preflight.json decision-v3/runs/v3-paired/diagnostic/result.json \
       decision-v3/runs/v3-paired/pilot/result.json decision-v3/runs/v3-paired/main/result.json \
       decision-v3/runs/v3-paired/main/checkpoints/latest/state.json decision-v3/runs/budget-v3-paired.json \
       decision-v3/runs/pipeline-v3-paired.json decision-v3/runs/pipeline-v3-paired.log \
       decision-v3/runs/v3-finalized/checksums.json decision-v3/runs/v3-finalized-freeze-receipt.json \
       decision-v3-evaluation/final-evaluation.json results/v2-rule-variants-v1.predictions.jsonl; do \
       test -f \$path || { echo \"missing \$path\" >&2; exit 5; }; done && \
     tar -chzf $RESULTS_ROOT/$BUNDLE \
       decision-v3/runs/v3-paired/preflight.json \
       decision-v3/runs/v3-paired/diagnostic/result.json \
       decision-v3/runs/v3-paired/pilot/result.json \
       decision-v3/runs/v3-paired/main/result.json \
       decision-v3/runs/v3-paired/main/checkpoints/latest \
       decision-v3/runs/budget-v3-paired.json \
       decision-v3/runs/pipeline-v3-paired.json \
       decision-v3/runs/pipeline-v3-paired.log \
       decision-v3/runs/v3-finalized \
       decision-v3/runs/v3-finalized-freeze-receipt.json \
       decision-v3-evaluation/final-evaluation.json \
       results/budget-v2-predict-*.json \
       results/v2-rule-variants-v1.predictions.jsonl results/v2-rule-variants-v1.predict-receipt.json \
       results/v3-*.predictions.jsonl results/v3-*.json && \
     cd $RESULTS_ROOT && sha256sum $BUNDLE > $BUNDLE.sha256"

printf '== pull and verify ==\n'
if [ "$DRY_RUN" -eq 1 ]; then
  printf '  local$  scp root@%s:%s/{%s,%s.sha256} -> %s/ ; verify sha256 and seals\n' "$HOST" "$RESULTS_ROOT" "$BUNDLE" "$BUNDLE" "$PULL"
else
  mkdir -p "$PULL"
  "${SCP[@]}" "root@$HOST:$RESULTS_ROOT/$BUNDLE" "root@$HOST:$RESULTS_ROOT/$BUNDLE.sha256" "$PULL/"
  (cd "$PULL" && shasum -a 256 -c "$BUNDLE.sha256" && tar -xzf "$BUNDLE")
  "$REPO/.venv/bin/python" -c "from pathlib import Path; from mdtrain.storage import verify_seal; root=Path('$PULL'); print('checkpoint seal:', verify_seal(root/'decision-v3/runs/v3-paired/main/checkpoints/latest/inference')); print('finalized seal:', verify_seal(root/'decision-v3/runs/v3-finalized'))"
fi

cat <<EOF

== safe handoff ==
  Verified locally at $PULL/$BUNDLE. The network volume still holds everything.
  Stop the pod to end GPU billing; delete the volume only after you no longer need the copy on it.
EOF
