#!/usr/bin/env bash
# Finalize/evaluate v2, run external predictions, then checksum and pull every paid-run receipt.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRANSFER_ROOT=/workspace/decision-transfer
V2_ROOT=/workspace/decision-v2
EVAL_ROOT=/workspace/decision-v2-evaluation
EXTERNAL_ROOT=/workspace/external
RESULTS_ROOT=/workspace/results
RUN_ROOT=$V2_ROOT/runs/hardv1-paired
LEDGER=$V2_ROOT/runs/budget-hardv1-paired.json
FINALIZED=$V2_ROOT/runs/hardv1-v2-finalized
PULL=$REPO/.audit/hardv1-paid-run

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

# Keepalives: without them a stalled connection left a finished command's session open for two hours.
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -p "$PORT" -i "$KEY" "root@$HOST")
SCP=(scp -q -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -P "$PORT" -i "$KEY")
run() {
  if [ "$DRY_RUN" -eq 1 ]; then printf '  remote$ %s\n' "$*"; else "${SSH[@]}" "$@"; fi
}

if [ "$FROM_EXTERNAL" -eq 1 ]; then
printf '== resume at external predictions: finalize and locked evaluation must already be complete ==\n'
run "test -f $FINALIZED/artifact.json && test -f $EVAL_ROOT/final-evaluation.json && \
     python3 -c \"import json; [json.load(open(p))['status'] == 'completed' or exit(p) for p in ('$RESULTS_ROOT/v2-finalize-command.json', '$RESULTS_ROOT/v2-locked-evaluate-command.json')]\" && \
     echo 'finalize and locked evaluation completed'"
else
printf '== main must be complete ==\n'
run "cd $V2_ROOT && .venv/bin/python -c \"import json; from pathlib import Path; path=Path('$RUN_ROOT/main/result.json'); value=json.loads(path.read_text()); assert value.get('status') == 'completed', value\" && echo 'main completed'"

printf '== finalize v2 with scalar temperature ==\n'
run "cd $V2_ROOT && HF_HUB_CACHE=$V2_ROOT/data/raw/processor-cache \
     .venv/bin/python -m mdtrain finalize --root $V2_ROOT \
       --checkpoint $RUN_ROOT/main/checkpoints/latest --fit-temperature --device cuda \
       --budget-ledger $LEDGER --output $FINALIZED \
     > $RESULTS_ROOT/v2-finalize-command.json"

printf '== upload locked evaluation archive only after finalization ==\n'
if [ "$DRY_RUN" -eq 1 ]; then
  printf '  local$  scp dist/locked-evaluation-data.tar.gz dist/SHA256SUMS.locked -> root@%s:%s/v2/dist/\n' "$HOST" "$TRANSFER_ROOT"
else
  "${SCP[@]}" "$REPO/dist/locked-evaluation-data.tar.gz" "$REPO/dist/SHA256SUMS.locked" \
    "root@$HOST:$TRANSFER_ROOT/v2/dist/"
fi

printf '== verify and extract locked evaluation root ==\n'
run "cd $TRANSFER_ROOT/v2 && sha256sum -c dist/SHA256SUMS.locked && \
     rm -rf $EVAL_ROOT && mkdir -p $EVAL_ROOT && \
     tar -xzf dist/training-code.tar.gz -C $EVAL_ROOT && \
     tar -xzf dist/locked-evaluation-data.tar.gz -C $EVAL_ROOT"

printf '== locked test_id and test_ood evaluation ==\n'
run "cd $EVAL_ROOT && HF_HUB_CACHE=$V2_ROOT/data/raw/processor-cache \
     $V2_ROOT/.venv/bin/python -m mdtrain evaluate --root $EVAL_ROOT \
       --artifact $FINALIZED --splits test_id,test_ood --device cuda \
       --budget-ledger $LEDGER --output $EVAL_ROOT/final-evaluation.json \
     > $RESULTS_ROOT/v2-locked-evaluate-command.json"
fi

printf '== v2 external predictions ==\n'
for dataset in rule-probe hard-v1-test doclaynet; do
  if [ "$dataset" = doclaynet ]; then media_root="$EXTERNAL_ROOT/doclaynet-candidate"; else media_root="$V2_ROOT"; fi
  run "cd $V2_ROOT && HF_HUB_CACHE=$V2_ROOT/data/raw/processor-cache \
       .venv/bin/python -m mdtrain predict --root $V2_ROOT --artifact $FINALIZED \
         --input $EXTERNAL_ROOT/requests/$dataset.requests.jsonl --media-root $media_root \
         --device cuda --budget-ledger $LEDGER \
         --output $RESULTS_ROOT/v2-$dataset.predictions.jsonl \
       > $RESULTS_ROOT/v2-$dataset.predict-receipt.json"
done

printf '== assemble the complete recovery bundle before any pod stop ==\n'
run "test -f $RUN_ROOT/preflight.json && \
     test -f $RUN_ROOT/diagnostic/result.json && \
     test -f $RUN_ROOT/pilot/result.json && \
     test -f $RUN_ROOT/main/result.json && \
     test -f $RUN_ROOT/main/checkpoints/latest/state.json && \
     test -f $RUN_ROOT/main/checkpoints/latest/inference/checksums.json && \
     test -f $LEDGER && test -f $V2_ROOT/runs/pipeline-hardv1-paired.json && \
     test -f $V2_ROOT/runs/pipeline-hardv1-paired.log && \
     test -f $FINALIZED/checksums.json && \
     test -f $V2_ROOT/runs/hardv1-v2-finalized-freeze-receipt.json && \
     test -f $EVAL_ROOT/final-evaluation.json && \
     test -f $RESULTS_ROOT/budget-v1-predict.json"
run "cd /workspace && tar -chzf $RESULTS_ROOT/hardv1-paid-results.tar.gz \
       decision-v2/runs/hardv1-paired/preflight.json \
       decision-v2/runs/hardv1-paired/diagnostic/result.json \
       decision-v2/runs/hardv1-paired/pilot/result.json \
       decision-v2/runs/hardv1-paired/main/result.json \
       decision-v2/runs/hardv1-paired/main/checkpoints/latest \
       decision-v2/runs/budget-hardv1-paired.json \
       decision-v2/runs/pipeline-hardv1-paired.json \
       decision-v2/runs/pipeline-hardv1-paired.log \
       decision-v2/runs/hardv1-v2-finalized \
       decision-v2/runs/hardv1-v2-finalized-freeze-receipt.json \
       decision-v2-evaluation/final-evaluation.json \
       results/budget-v1-predict.json \
       results/v1-trained-rule-probe.predictions.jsonl \
       results/v1-trained-hard-v1-test.predictions.jsonl \
       results/v1-trained-doclaynet.predictions.jsonl \
       results/v1-pointer-only-rule-probe.predictions.jsonl \
       results/v1-pointer-only-hard-v1-test.predictions.jsonl \
       results/v1-pointer-only-doclaynet.predictions.jsonl \
       results/v2-rule-probe.predictions.jsonl \
       results/v2-hard-v1-test.predictions.jsonl \
       results/v2-doclaynet.predictions.jsonl \
       results/v1-trained-rule-probe.predict-receipt.json \
       results/v1-trained-hard-v1-test.predict-receipt.json \
       results/v1-trained-doclaynet.predict-receipt.json \
       results/v1-pointer-only-rule-probe.predict-receipt.json \
       results/v1-pointer-only-hard-v1-test.predict-receipt.json \
       results/v1-pointer-only-doclaynet.predict-receipt.json \
       results/v2-rule-probe.predict-receipt.json \
       results/v2-hard-v1-test.predict-receipt.json \
       results/v2-doclaynet.predict-receipt.json \
       results/v2-finalize-command.json \
       results/v2-locked-evaluate-command.json && \
     cd $RESULTS_ROOT && sha256sum hardv1-paid-results.tar.gz > hardv1-paid-results.tar.gz.sha256"

printf '== pull and verify before stop ==\n'
if [ "$DRY_RUN" -eq 1 ]; then
  printf '  local$  scp root@%s:%s/{hardv1-paid-results.tar.gz,hardv1-paid-results.tar.gz.sha256} -> %s/\n' "$HOST" "$RESULTS_ROOT" "$PULL"
  printf '  local$  (cd %s && shasum -a 256 -c hardv1-paid-results.tar.gz.sha256 && tar -xzf hardv1-paid-results.tar.gz)\n' "$PULL"
  printf '  local$  verify_seal extracted main checkpoint inference and finalized artifact\n'
else
  mkdir -p "$PULL"
  "${SCP[@]}" "root@$HOST:$RESULTS_ROOT/hardv1-paid-results.tar.gz" \
    "root@$HOST:$RESULTS_ROOT/hardv1-paid-results.tar.gz.sha256" "$PULL/"
  (cd "$PULL" && shasum -a 256 -c hardv1-paid-results.tar.gz.sha256 && \
    tar -xzf hardv1-paid-results.tar.gz)
  "$REPO/.venv/bin/python" -c "from pathlib import Path; from mdtrain.storage import verify_seal; root=Path('$PULL'); print('checkpoint seal:', verify_seal(root/'decision-v2/runs/hardv1-paired/main/checkpoints/latest/inference')); print('finalized seal:', verify_seal(root/'decision-v2/runs/hardv1-v2-finalized'))"
fi

cat <<EOF

== safe handoff ==
  The pulled recovery bundle checksum has been verified locally at:
    $PULL/hardv1-paid-results.tar.gz
  Inspect the extracted receipts, then stop the pod through RunPod. This script never stops it.
EOF
