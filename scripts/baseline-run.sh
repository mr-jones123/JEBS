#!/usr/bin/env bash
# Drive the pointer-only baseline run over SSH.
#
# The pod is started by hand in the RunPod console, then this script does the rest:
# push the archives, install the frozen environment, download the pinned model, and
# start the baseline detached so a dropped connection cannot kill a paid run.
#
#   ./scripts/baseline-run.sh --host 81.27.69.179 --port 21041 --dry-run
#   ./scripts/baseline-run.sh --host 81.27.69.179 --port 21041 --key ~/.ssh/id_ed25519
#
# The pod-side steps mirror RUNPOD.md. Read it before changing anything here.

set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
POD_ROOT=/workspace/decision
TRANSFER_ROOT=/workspace/decision-transfer
MODEL_REVISION=ebb281ec70b05090aa6165b016eac8ec08e71b17
UV_VERSION=0.9.27

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

[ -n "$HOST" ] || { echo "usage: $0 --host HOST --port PORT [--key PATH] [--dry-run]" >&2; exit 2; }
[ -n "$PORT" ] || { echo "--port is required; the console shows it beside the SSH command" >&2; exit 2; }
[ -f "$KEY" ] || { echo "ssh key not found: $KEY" >&2; exit 2; }

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -p "$PORT" -i "$KEY" "root@$HOST")
SCP=(scp -q -o BatchMode=yes -o ConnectTimeout=15 -P "$PORT" -i "$KEY")

run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '  remote$ %s\n' "$*"
  else
    "${SSH[@]}" "$@"
  fi
}

printf '== local preflight ==\n'
python3 - "$REPO" <<'PY'
import hashlib, json, sys
from pathlib import Path

repo = Path(sys.argv[1])
index = json.loads((repo / "dist" / "bundles.json").read_text())
problems = []
for key in ("code", "training"):
    entry = index[key]
    path = repo / entry["path"]
    if not path.is_file():
        problems.append(f"{entry['path']} is missing")
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        problems.append(f"{entry['path']} does not match bundles.json")
    if key == "code":
        names = set(entry["members"])
        for expected in ("src/mdtrain/train.py", "src/mdprep/doclaynet.py"):
            if expected not in names:
                problems.append(f"{entry['path']} does not carry {expected}")
if problems:
    raise SystemExit("refusing to transfer:\n  " + "\n  ".join(problems))
print("  archives match bundles.json and the code archive carries the baseline flag")
PY

if [ "$DRY_RUN" -eq 0 ]; then
  printf '== reachability ==\n'
  "${SSH[@]}" 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; \
               ffmpeg -version | head -1; ffprobe -version | head -1'
fi

printf '== push ==\n'
for name in training-code.tar.gz training-data.tar.gz; do
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '  local$  scp dist/%s -> root@%s:%s/dist/\n' "$name" "$HOST" "$TRANSFER_ROOT"
  else
    "${SSH[@]}" "mkdir -p $TRANSFER_ROOT/dist"
    "${SCP[@]}" "$REPO/dist/$name" "root@$HOST:$TRANSFER_ROOT/dist/$name"
  fi
done
if [ "$DRY_RUN" -eq 1 ]; then
  printf '  local$  scp dist/bundles.json dist/SHA256SUMS.training -> root@%s:%s/dist/\n' "$HOST" "$TRANSFER_ROOT"
else
  "${SCP[@]}" "$REPO/dist/bundles.json" "$REPO/dist/SHA256SUMS.training" "root@$HOST:$TRANSFER_ROOT/dist/"
fi

printf '== verify and extract ==\n'
run "cd $TRANSFER_ROOT && sha256sum -c dist/SHA256SUMS.training && \
     mkdir -p $POD_ROOT && \
     tar -xzf dist/training-code.tar.gz -C $POD_ROOT && \
     tar -xzf dist/training-data.tar.gz -C $POD_ROOT"

printf '== install the frozen environment ==\n'
run "cd $POD_ROOT && \
     (command -v uv >/dev/null || python3 -m pip install uv==$UV_VERSION) && \
     uv sync --frozen --python 3.12 --extra training && \
     .venv/bin/hf download Qwen/Qwen3-VL-4B-Instruct \
       --revision $MODEL_REVISION --cache-dir $POD_ROOT/data/raw/processor-cache"

printf '== start the baseline detached ==\n'
# Its own run directory and its own ledger. It cannot reuse the accepted run's
# preflight receipt, because a receipt from one schedule is rejected by the other.
run "cd $POD_ROOT && mkdir -p runs && \
     PAID_STARTED_AT=\$(date -u +%Y-%m-%dT%H:%M:%SZ) && \
     echo \"allocation start recorded locally as \$PAID_STARTED_AT\" && \
     setsid nohup env PAID_STARTED_AT=\$PAID_STARTED_AT \
       .venv/bin/python -m mdtrain pipeline \
         --root $POD_ROOT --recipe mixed --schedule pointer-only \
         --run-dir $POD_ROOT/runs/pointer-only \
         --budget-ledger $POD_ROOT/runs/budget-pointer-only.json \
         --paid-started-at \$PAID_STARTED_AT \
         --output $POD_ROOT/runs/pipeline-pointer-only.json \
       > $POD_ROOT/runs/pipeline-pointer-only.log 2>&1 < /dev/null & \
     echo \"started pid \$!\""

cat <<EOF

== next ==
  follow the run:   ssh -p $PORT -i $KEY root@$HOST 'tail -f $POD_ROOT/runs/pipeline-pointer-only.log'
  stage results:    ssh -p $PORT -i $KEY root@$HOST 'cat $POD_ROOT/runs/pipeline-pointer-only.json'
  stop the pod:     do this yourself in the console, or with runpodctl if you have it.
                    The process deadline does not stop provider billing.

Record the real allocation start from the console into the ledger's paid-started-at if the
stamp above is later than the console shows. A network volume attached at creation survives
a stop; the container disk does not, and the previous run lost its artifact that way.
EOF
