#!/usr/bin/env bash
# Upload frozen v1 evaluators and v2 training inputs, score v1, then launch paired v2 training.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRANSFER_ROOT=/workspace/decision-transfer
V1_ROOT=/workspace/decision-v1
V2_ROOT=/workspace/decision-v2
EXTERNAL_ROOT=/workspace/external
RESULTS_ROOT=/workspace/results
MODEL_REVISION=ebb281ec70b05090aa6165b016eac8ec08e71b17
UV_VERSION=0.9.27

HOST=""; PORT=""; POD_ID=""; KEY="$HOME/.ssh/id_ed25519"; DRY_RUN=0; SKIP_V1=0
while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --pod-id) POD_ID="$2"; shift 2 ;;
    --key) KEY="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-v1) SKIP_V1=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$HOST" ] && [ -n "$PORT" ] && [ -n "$POD_ID" ] || {
  echo "usage: $0 --host HOST --port PORT --pod-id POD_ID [--key PATH] [--skip-v1] [--dry-run]" >&2
  exit 2
}
[ -f "$KEY" ] || { echo "ssh key not found: $KEY" >&2; exit 2; }

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -p "$PORT" -i "$KEY" "root@$HOST")
SCP=(scp -q -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -P "$PORT" -i "$KEY")
run() {
  if [ "$DRY_RUN" -eq 1 ]; then printf '  remote$ %s\n' "$*"; else "${SSH[@]}" "$@"; fi
}
# Uploads go through push_chunked.py: parts of 4 MiB with per-part retries, skipping
# parts and files the pod already holds by sha256. A plain scp restarts from zero
# on every dropped connection, which never completed over a slow, flaky uplink.
PUSH=(python3 "$REPO/scripts/push_chunked.py" --host "$HOST" --port "$PORT" --key "$KEY")
push_file() {
  local source=$1 destination=$2
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '  local$  push_chunked %s -> root@%s:%s\n' "$source" "$HOST" "$destination"
  else
    "${PUSH[@]}" "$source" "$destination"
  fi
}
push_tree() {
  local source=$1 destination=$2
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '  local$  push_chunked %s/ -> root@%s:%s/\n' "$source" "$HOST" "$destination"
  else
    "${PUSH[@]}" "$source" "$destination"
  fi
}

printf '== local preflight ==\n'
python3 - "$REPO" <<'PY'
import hashlib
import json
import sys
import tarfile
from pathlib import Path

root = Path(sys.argv[1])
required = (
    "dist-accepted-2026-10-16/training-code.tar.gz",
    ".audit/finalized/finalized-model/artifact.json",
    ".audit/pointer-only/pointer-only-finalized/artifact.json",
    "dist/training-code.tar.gz",
    "dist/training-data.tar.gz",
    "dist/bundles.json",
    "dist/SHA256SUMS.training",
    ".audit/rule-probe/rule-probe.requests.jsonl",
    ".audit/rule-probe/hard-v1-test.requests.jsonl",
    ".audit/rule-probe/doclaynet.requests.jsonl",
)
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise SystemExit("missing paid-run input(s):\n  " + "\n  ".join(missing))
if not (root / ".audit/doclaynet-candidate/media").is_dir():
    raise SystemExit("missing .audit/doclaynet-candidate/media")
index = json.loads((root / "dist/bundles.json").read_text())
for key in ("code", "training"):
    entry = index[key]
    path = root / entry["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise SystemExit(f"{entry['path']} differs from dist/bundles.json")
accepted = json.loads((root / "dist-accepted-2026-10-16/bundles.json").read_text())["code"]
accepted_path = root / "dist-accepted-2026-10-16" / Path(accepted["path"]).name
if hashlib.sha256(accepted_path.read_bytes()).hexdigest() != accepted["sha256"]:
    raise SystemExit("accepted v1 training-code.tar.gz differs from its frozen bundle index")
with tarfile.open(root / "dist/training-code.tar.gz", "r:gz") as archive:
    for name in ("train.py", "budget.py", "preflight.py", "__main__.py", "_worker.py"):
        if archive.extractfile(f"src/mdtrain/{name}").read() != (root / "src/mdtrain" / name).read_bytes():
            raise SystemExit(f"dist/training-code.tar.gz is stale: src/mdtrain/{name} differs; rebuild with mdprep.package")
print("  v1 artifacts, external requests/media, and v2 archives are present")
PY

printf '== provider allocation start ==\n'
if [ "$DRY_RUN" -eq 1 ]; then
  PAID_STARTED_AT='<provider lastStartedAt>'
  HOURLY_USD='<provider costPerHr>'
  printf '  local$  GET rest.runpod.io/v1/pods/%s with ~/.runpod-key; read lastStartedAt and costPerHr (key redacted)\n' "$POD_ID"
  printf '  PAID_STARTED_AT=%s HOURLY_USD=%s\n' "$PAID_STARTED_AT" "$HOURLY_USD"
else
  API_KEY_FILE="$HOME/.runpod-key"
  [ -f "$API_KEY_FILE" ] || { echo "RunPod key not found: $API_KEY_FILE" >&2; exit 2; }
  API_KEY=$(<"$API_KEY_FILE")
  POD_JSON=$(curl -fsS "https://rest.runpod.io/v1/pods/$POD_ID" -H "Authorization: Bearer $API_KEY")
  unset API_KEY
  # The provider reports e.g. "2026-10-17 17:32:04.95 +0000 UTC", which is not ISO 8601.
  # Normalize to RFC 3339 UTC; dropping the fraction moves the start earlier by under a
  # second, which only overcounts elapsed paid time.
  PAID_STARTED_AT=$(python3 -c 'import datetime as dt
import json
import re
import sys

value = json.load(sys.stdin).get("lastStartedAt")
if not isinstance(value, str) or not value:
    raise SystemExit("missing lastStartedAt")
match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:\.\d+)?\s*(?:Z|\+0000|\+00:00)?(?:\s*UTC)?", value.strip())
if not match:
    raise SystemExit(f"unrecognized lastStartedAt format: {value!r}")
parsed = dt.datetime.fromisoformat(f"{match[1]}T{match[2]}+00:00")
print(parsed.strftime("%Y-%m-%dT%H:%M:%SZ"))' <<<"$POD_JSON")
  # The ledger turns the fixed $8 budget into elapsed seconds at this pod's own price.
  HOURLY_USD=$(python3 -c 'import json
import math
import sys

value = json.load(sys.stdin).get("costPerHr")
if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
    raise SystemExit(f"invalid costPerHr: {value!r}")
print(value)' <<<"$POD_JSON")
  unset POD_JSON
  printf '  recorded provider lastStartedAt: %s, costPerHr: %s USD\n' "$PAID_STARTED_AT" "$HOURLY_USD"
fi

if [ "$DRY_RUN" -eq 0 ]; then
  printf '== reachability ==\n'
  "${SSH[@]}" 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; ffmpeg -version | head -1; ffprobe -version | head -1'
fi

printf '== create remote transfer roots ==\n'
run "mkdir -p $TRANSFER_ROOT/v1 $TRANSFER_ROOT/v2/dist $EXTERNAL_ROOT/requests $EXTERNAL_ROOT/doclaynet-candidate $RESULTS_ROOT"

printf '== upload frozen v1 code and artifacts ==\n'
push_file "$REPO/dist-accepted-2026-10-16/training-code.tar.gz" "$TRANSFER_ROOT/v1/training-code.tar.gz"
push_tree "$REPO/.audit/finalized/finalized-model" "$TRANSFER_ROOT/v1/finalized-model"
push_tree "$REPO/.audit/pointer-only/pointer-only-finalized" "$TRANSFER_ROOT/v1/pointer-only-finalized"

printf '== upload external requests and DocLayNet media ==\n'
for name in rule-probe hard-v1-test doclaynet; do
  push_file "$REPO/.audit/rule-probe/$name.requests.jsonl" "$EXTERNAL_ROOT/requests/$name.requests.jsonl"
done
push_tree "$REPO/.audit/doclaynet-candidate/media" "$EXTERNAL_ROOT/doclaynet-candidate/media"

printf '== upload v2 training archives ==\n'
for name in training-code.tar.gz training-data.tar.gz bundles.json SHA256SUMS.training; do
  push_file "$REPO/dist/$name" "$TRANSFER_ROOT/v2/dist/$name"
done

if [ "$SKIP_V1" -eq 1 ]; then
  printf '== skip v1 install and scoring (--skip-v1) ==\n'
else
printf '== install frozen v1 code ==\n'
run "rm -rf $V1_ROOT && mkdir -p $V1_ROOT && tar -xzf $TRANSFER_ROOT/v1/training-code.tar.gz -C $V1_ROOT && \
     cp -a $TRANSFER_ROOT/v1/finalized-model $V1_ROOT/finalized-model && \
     cp -a $TRANSFER_ROOT/v1/pointer-only-finalized $V1_ROOT/pointer-only-finalized && \
     cd $V1_ROOT && (command -v uv >/dev/null || python3 -m pip install uv==$UV_VERSION) && \
     uv sync --frozen --python 3.12 --extra training && \
     .venv/bin/hf download Qwen/Qwen3-VL-4B-Instruct --revision $MODEL_REVISION \
       --cache-dir $V1_ROOT/data/raw/processor-cache"

printf '== initialize the v1 prediction ledger at the provider start ==\n'
run "cd $V1_ROOT && .venv/bin/python -c \"from pathlib import Path; from mdtrain.budget import BudgetLedger; ledger=BudgetLedger(Path('$RESULTS_ROOT/budget-v1-predict.json'), paid_started_at='$PAID_STARTED_AT'); ledger.__enter__(); ledger.__exit__(None, None, None)\""

printf '== predict both v1 artifacts on all external sets ==\n'
for artifact in trained pointer-only; do
  if [ "$artifact" = trained ]; then artifact_path="$V1_ROOT/finalized-model"; else artifact_path="$V1_ROOT/pointer-only-finalized"; fi
  for dataset in rule-probe hard-v1-test doclaynet; do
    if [ "$dataset" = doclaynet ]; then media_root="$EXTERNAL_ROOT/doclaynet-candidate"; else media_root="$V1_ROOT"; fi
    run "cd $V1_ROOT && HF_HUB_CACHE=$V1_ROOT/data/raw/processor-cache \
      .venv/bin/python -m mdtrain predict --root $V1_ROOT --artifact $artifact_path \
        --input $EXTERNAL_ROOT/requests/$dataset.requests.jsonl --media-root $media_root \
        --device cuda --budget-ledger $RESULTS_ROOT/budget-v1-predict.json \
        --output $RESULTS_ROOT/v1-$artifact-$dataset.predictions.jsonl \
      > $RESULTS_ROOT/v1-$artifact-$dataset.predict-receipt.json"
  done
done
fi

printf '== verify, extract, and install v2 ==\n'
run "cd $TRANSFER_ROOT/v2 && sha256sum -c dist/SHA256SUMS.training && \
     rm -rf $V2_ROOT && mkdir -p $V2_ROOT && \
     tar -xzf dist/training-code.tar.gz -C $V2_ROOT && \
     tar -xzf dist/training-data.tar.gz -C $V2_ROOT && \
     cd $V2_ROOT && (command -v uv >/dev/null || python3 -m pip install uv==$UV_VERSION) && \
     uv sync --frozen --python 3.12 --extra training && \
     .venv/bin/hf download Qwen/Qwen3-VL-4B-Instruct --revision $MODEL_REVISION \
       --cache-dir $V2_ROOT/data/raw/processor-cache"

printf '== launch paired v2 pipeline detached ==\n'
run "cd $V2_ROOT && mkdir -p runs && \
     setsid env PAID_STARTED_AT='$PAID_STARTED_AT' \
       .venv/bin/python -m mdtrain pipeline --root $V2_ROOT --recipe mixed --schedule paired \
         --run-dir $V2_ROOT/runs/hardv1-paired \
         --budget-ledger $V2_ROOT/runs/budget-hardv1-paired.json \
         --paid-started-at '$PAID_STARTED_AT' --hourly-usd '$HOURLY_USD' \
         --output $V2_ROOT/runs/pipeline-hardv1-paired.json \
       > $V2_ROOT/runs/pipeline-hardv1-paired.log 2>&1 < /dev/null & \
     echo \"started pid \$! with provider allocation start $PAID_STARTED_AT\""

cat <<EOF

== next ==
  Monitor: ssh -p $PORT -i $KEY root@$HOST 'tail -f $V2_ROOT/runs/pipeline-hardv1-paired.log'
  After main completes, run scripts/hardv1-finish.sh with the same host, port, and key.
  Do not stop the pod before that script verifies the pulled results archive checksum.
EOF
