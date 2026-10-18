#!/usr/bin/env bash
# Upload the frozen v2 scorer and v3 training inputs to the pod's network volume, score
# v2 on the held-out rule structures, then launch fresh paired v3 training detached.
# Everything lives under /workspace, which must be the network volume, so a stopped,
# replaced, or balance-stopped pod loses nothing already written.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRANSFER_ROOT=/workspace/decision-transfer
V2_ROOT=/workspace/decision-v2
V3_ROOT=/workspace/decision-v3
EXTERNAL_ROOT=/workspace/external
RESULTS_ROOT=/workspace/results
MODEL_REVISION=ebb281ec70b05090aa6165b016eac8ec08e71b17
UV_VERSION=0.9.27
V2_DIST="$REPO/dist-accepted-2026-10-18-v2"
V2_ARTIFACT="$REPO/.audit/hardv1-paid-run/decision-v2/runs/hardv1-v2-finalized"
EXTERNAL_SETS=(rule-probe hard-v1-test doclaynet rule-variants-v1)

HOST=""; PORT=""; POD_ID=""; KEY="$HOME/.ssh/id_ed25519"; DRY_RUN=0; REINSTALL_V3=0
while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --pod-id) POD_ID="$2"; shift 2 ;;
    --key) KEY="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --reinstall-v3) REINSTALL_V3=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$HOST" ] && [ -n "$PORT" ] && [ -n "$POD_ID" ] || {
  echo "usage: $0 --host HOST --port PORT --pod-id POD_ID [--key PATH] [--reinstall-v3] [--dry-run]" >&2
  exit 2
}
[ -f "$KEY" ] || { echo "ssh key not found: $KEY" >&2; exit 2; }

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -p "$PORT" -i "$KEY" "root@$HOST")
run() {
  if [ "$DRY_RUN" -eq 1 ]; then printf '  remote$ %s\n' "$*"; else "${SSH[@]}" "$@"; fi
}
PUSH=(python3 "$REPO/scripts/push_chunked.py" --host "$HOST" --port "$PORT" --key "$KEY")
push() {
  if [ "$DRY_RUN" -eq 1 ]; then printf '  local$  push_chunked %s -> root@%s:%s\n' "$1" "$HOST" "$2"; else "${PUSH[@]}" "$1" "$2"; fi
}
external_path() {
  case "$1" in
    rule-variants-v1) printf '%s/.audit/rule-structures-heldout/rule-variants-v1.requests.jsonl' "$REPO" ;;
    *) printf '%s/.audit/rule-probe/%s.requests.jsonl' "$REPO" "$1" ;;
  esac
}

printf '== local preflight ==\n'
V2_CODE_SHA=$(python3 - "$REPO" "$V2_DIST" "$V2_ARTIFACT" "${EXTERNAL_SETS[@]}" <<'PY'
import hashlib
import json
import sys
import tarfile
from pathlib import Path

root, v2_dist, v2_artifact = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
external = {name: (root / ".audit/rule-structures-heldout/rule-variants-v1.requests.jsonl" if name == "rule-variants-v1"
                   else root / f".audit/rule-probe/{name}.requests.jsonl") for name in sys.argv[4:]}
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
missing = [str(path) for path in (*external.values(), v2_artifact / "artifact.json",
                                  root / ".audit/doclaynet-candidate/media",
                                  *(root / "dist" / name for name in ("training-code.tar.gz", "training-data.tar.gz",
                                                                     "bundles.json", "SHA256SUMS.training",
                                                                     "locked-evaluation-data.tar.gz", "SHA256SUMS.locked")))
           if not path.exists()]
if missing:
    raise SystemExit("missing paid-run input(s):\n  " + "\n  ".join(missing))
for dist in (root / "dist", v2_dist):
    index = json.loads((dist / "bundles.json").read_text())
    for key in ("code", "training"):
        if sha(dist / Path(index[key]["path"]).name) != index[key]["sha256"]:
            raise SystemExit(f"{dist.name}/{Path(index[key]['path']).name} differs from its bundles.json")
with tarfile.open(root / "dist/training-code.tar.gz", "r:gz") as archive:
    for name in ("src/mdtrain/train.py", "src/mdtrain/budget.py", "src/mdtrain/preflight.py", "src/mdtrain/__main__.py",
                 "src/mdtrain/_worker.py", "src/mdprep/source_contracts.py", "src/mdprep/rule_variants.py"):
        if archive.extractfile(name).read() != (root / name).read_bytes():
            raise SystemExit(f"dist/training-code.tar.gz is stale: {name} differs; rebuild with mdprep.package")
locked = {"test_id": "4fe822e64f93d9db", "test_ood": "e0422fab965af43f", "policy": "1a29e04df2331830"}
for split, prefix in locked.items():
    if not sha(root / f"data/prepared/{split}.jsonl").startswith(prefix):
        raise SystemExit(f"data/prepared/{split}.jsonl changed; v3 would not be comparable to v1 and v2")
print(sha(v2_dist / "training-code.tar.gz"))
PY
)
printf '  inputs present, v3 archives fresh, locked splits unchanged, v2 code %s\n' "${V2_CODE_SHA:0:16}"

printf '== provider allocation start and price ==\n'
if [ "$DRY_RUN" -eq 1 ]; then
  PAID_STARTED_AT='<provider lastStartedAt>'; HOURLY_USD='<provider costPerHr>'
  printf '  local$  GET rest.runpod.io/v1/pods/%s; read lastStartedAt and costPerHr (key redacted)\n' "$POD_ID"
else
  API_KEY=$(<"$HOME/.runpod-key")
  POD_JSON=$(curl -fsS "https://rest.runpod.io/v1/pods/$POD_ID" -H "Authorization: Bearer $API_KEY")
  unset API_KEY
  # The provider reports e.g. "2026-10-17 17:32:04.95 +0000 UTC"; dropping the fraction only overcounts.
  read -r PAID_STARTED_AT HOURLY_USD < <(python3 -c 'import datetime as dt
import json
import math
import re
import sys

pod = json.load(sys.stdin)
value, price = pod.get("lastStartedAt"), pod.get("costPerHr")
match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:\.\d+)?\s*(?:Z|\+0000|\+00:00)?(?:\s*UTC)?", (value or "").strip())
if not match:
    raise SystemExit(f"unrecognized lastStartedAt: {value!r}")
if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
    raise SystemExit(f"invalid costPerHr: {price!r}")
started = dt.datetime.fromisoformat(f"{match[1]}T{match[2]}+00:00")
print(started.strftime("%Y-%m-%dT%H:%M:%SZ"), price)' <<<"$POD_JSON")
  unset POD_JSON
  printf '  lastStartedAt %s, costPerHr %s USD\n' "$PAID_STARTED_AT" "$HOURLY_USD"
fi

printf '== reachability and network volume ==\n'
run "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; \
     mountpoint -q /workspace || { echo '/workspace is not a separate mount: refusing to run without the network volume' >&2; exit 3; }; \
     df -h /workspace | tail -1"

printf '== uploads (skipped when the volume already holds them) ==\n'
run "mkdir -p $TRANSFER_ROOT/v2 $TRANSFER_ROOT/v3/dist $EXTERNAL_ROOT/requests $EXTERNAL_ROOT/doclaynet-candidate $RESULTS_ROOT"
push "$V2_DIST/training-code.tar.gz" "$TRANSFER_ROOT/v2/training-code.tar.gz"
push "$V2_ARTIFACT" "$TRANSFER_ROOT/v2/hardv1-v2-finalized"
for name in "${EXTERNAL_SETS[@]}"; do
  push "$(external_path "$name")" "$EXTERNAL_ROOT/requests/$name.requests.jsonl"
done
push "$REPO/.audit/doclaynet-candidate/media" "$EXTERNAL_ROOT/doclaynet-candidate/media"
for name in training-code.tar.gz training-data.tar.gz bundles.json SHA256SUMS.training \
            locked-evaluation-data.tar.gz SHA256SUMS.locked; do
  push "$REPO/dist/$name" "$TRANSFER_ROOT/v3/dist/$name"
done

printf '== install frozen v2 code and artifact (once per volume) ==\n'
run "if [ \"\$(cat $V2_ROOT/.installed 2>/dev/null)\" = '$V2_CODE_SHA' ]; then echo '  v2 already installed'; else \
     rm -rf $V2_ROOT && mkdir -p $V2_ROOT && tar -xzf $TRANSFER_ROOT/v2/training-code.tar.gz -C $V2_ROOT && \
     cp -a $TRANSFER_ROOT/v2/hardv1-v2-finalized $V2_ROOT/hardv1-v2-finalized && \
     cd $V2_ROOT && (command -v uv >/dev/null || python3 -m pip install uv==$UV_VERSION) && \
     uv sync --frozen --python 3.12 --extra training && \
     .venv/bin/hf download Qwen/Qwen3-VL-4B-Instruct --revision $MODEL_REVISION --cache-dir $V2_ROOT/data/raw/processor-cache && \
     echo '$V2_CODE_SHA' > $V2_ROOT/.installed; fi"

printf '== score v2 on the held-out rule structures ==\n'
# One ledger per pod: a replacement pod is a new allocation, and an open allocation cannot be extended.
V2_LEDGER=$RESULTS_ROOT/budget-v2-predict-$POD_ID.json
run "cd $V2_ROOT && .venv/bin/python -c \"from pathlib import Path; from mdtrain.budget import BudgetLedger; ledger=BudgetLedger(Path('$V2_LEDGER'), paid_started_at='$PAID_STARTED_AT', hourly_usd=float('$HOURLY_USD')); ledger.__enter__(); ledger.__exit__(None, None, None)\""
run "if [ -s $RESULTS_ROOT/v2-rule-variants-v1.predictions.jsonl ] && grep -q '\"status\": *\"completed\"' $RESULTS_ROOT/v2-rule-variants-v1.predict-receipt.json 2>/dev/null; then \
     echo '  v2 held-out predictions already present'; else \
     cd $V2_ROOT && HF_HUB_CACHE=$V2_ROOT/data/raw/processor-cache \
     .venv/bin/python -m mdtrain predict --root $V2_ROOT --artifact $V2_ROOT/hardv1-v2-finalized \
       --input $EXTERNAL_ROOT/requests/rule-variants-v1.requests.jsonl --media-root $V2_ROOT \
       --device cuda --budget-ledger $V2_LEDGER \
       --output $RESULTS_ROOT/v2-rule-variants-v1.predictions.jsonl \
     > $RESULTS_ROOT/v2-rule-variants-v1.predict-receipt.json; fi"

printf '== verify, extract, and install v3 ==\n'
# A v3 run directory on the volume holds checkpoints; never delete it implicitly.
run "if [ -d $V3_ROOT/runs ] && [ $REINSTALL_V3 -eq 0 ]; then \
     echo 'a v3 run already exists on the volume; resume it, or pass --reinstall-v3 to discard it' >&2; exit 4; fi; \
     cd $TRANSFER_ROOT/v3 && sha256sum -c dist/SHA256SUMS.training && \
     rm -rf $V3_ROOT && mkdir -p $V3_ROOT && \
     tar -xzf dist/training-code.tar.gz -C $V3_ROOT && tar -xzf dist/training-data.tar.gz -C $V3_ROOT && \
     cd $V3_ROOT && (command -v uv >/dev/null || python3 -m pip install uv==$UV_VERSION) && \
     uv sync --frozen --python 3.12 --extra training && \
     .venv/bin/hf download Qwen/Qwen3-VL-4B-Instruct --revision $MODEL_REVISION --cache-dir $V3_ROOT/data/raw/processor-cache"

printf '== launch fresh paired v3 pipeline detached ==\n'
# Background only the training command. A trailing '&' on 'cd && mkdir && setsid ...' backgrounds
# the whole list as a subshell that keeps the SSH channel open until training ends.
run "cd $V3_ROOT && mkdir -p runs && { \
     setsid env PAID_STARTED_AT='$PAID_STARTED_AT' \
       .venv/bin/python -m mdtrain pipeline --root $V3_ROOT --recipe mixed --schedule paired \
         --run-dir $V3_ROOT/runs/v3-paired \
         --budget-ledger $V3_ROOT/runs/budget-v3-paired.json \
         --paid-started-at '$PAID_STARTED_AT' --hourly-usd '$HOURLY_USD' \
         --output $V3_ROOT/runs/pipeline-v3-paired.json \
       > $V3_ROOT/runs/pipeline-v3-paired.log 2>&1 < /dev/null & } ; \
     echo \"started pid \$! with provider allocation start $PAID_STARTED_AT\""

cat <<EOF

== next ==
  Watch the pid above; when the pipeline exits, run scripts/v3-finish.sh with the same host, port, and key.
  Everything is on the network volume, so a stopped pod loses nothing; still pull before deleting the volume.
EOF
