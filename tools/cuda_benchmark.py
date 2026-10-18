#!/usr/bin/env python3
"""Time JEBS-4B on CUDA from an installed release and compare its answers with a reference run.

Run from the root of a downloaded experimentalmachines/JEBS-4B with its locked environment:

	.venv/bin/python cuda_benchmark.py --requests rule-probe.requests.jsonl --reference v3-rule-probe.predictions.jsonl
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from mdtrain.engine import DecisionModel  # noqa: E402
from mdtrain.evaluate import read_requests  # noqa: E402
from mdtrain.train import deterministic  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("cuda-benchmark.json"))
    args = parser.parse_args()

    deterministic()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = DecisionModel.load(ROOT / "model", device="cuda", root=ROOT, media_root=ROOT)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    after_load = torch.cuda.memory_allocated()

    requests = read_requests(args.requests)
    model.predict(requests[:5])
    rows, latencies = [], []
    for request in requests:
        torch.cuda.synchronize()
        tick = time.perf_counter()
        rows.extend(model.predict((request,)))
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - tick)

    reference = {json.loads(line)["id"]: json.loads(line) for line in args.reference.open()}
    mine = {row["id"]: row for row in rows}
    if mine.keys() != reference.keys():
        raise SystemExit("request ids differ from the reference run")
    differences = [abs(mine[i]["probabilities"][k] - reference[i]["probabilities"][k])
                   for i in mine for k in mine[i]["probabilities"]]
    ordered = sorted(latencies)
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "artifact_id": model.artifact_id,
        "questions": len(rows),
        "load_seconds": load_seconds,
        "latency_ms": {"median": 1000 * statistics.median(ordered),
                       "p95": 1000 * ordered[int(0.95 * (len(ordered) - 1))],
                       "max": 1000 * ordered[-1], "mean": 1000 * statistics.fmean(ordered)},
        "questions_per_second": len(rows) / sum(latencies),
        "gpu_memory_gib": {"after_load": after_load / 1024 ** 3,
                           "peak_allocated": torch.cuda.max_memory_allocated() / 1024 ** 3,
                           "peak_reserved": torch.cuda.max_memory_reserved() / 1024 ** 3},
        "same_answer_as_reference": sum(mine[i]["predicted_key"] == reference[i]["predicted_key"] for i in mine),
        "max_probability_difference": max(differences),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
