"""Qualify every current prepared input through the shared native compiler."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch

from mdprep.common import SPLITS, sha256_file, write_json, write_jsonl
from mdprep.native import MAX_POSITIONS, NativeCompiler, media_path
from mdprep.records import parse_record


def check(root: Path, max_positions: int = MAX_POSITIONS, limit: int | None = None) -> dict:
    if max_positions != MAX_POSITIONS:
        raise ValueError("The supported complete-input budget is fixed at 2048")
    if limit is not None and limit <= 0:
        raise ValueError("A diagnostic limit must be positive")
    torch.set_num_threads(2)
    compiler = NativeCompiler.from_pretrained(root)
    data = root / "data"
    checks, failures, prepared_hashes, media_hashes = [], [], {}, {}
    for split in SPLITS:
        path = data / "prepared" / f"{split}.jsonl"
        prepared_hashes[split] = sha256_file(path)
        with path.open() as stream:
            for index, line in enumerate(stream):
                if limit is not None and index >= limit:
                    break
                row = json.loads(line)
                try:
                    example = parse_record(row)
                    batch = compiler.compile((example.input,))
                    for relative in (*example.input.images, *(v.path for v in example.input.videos)):
                        if relative not in media_hashes:
                            media_hashes[relative] = sha256_file(media_path(data, relative))
                    checks.append({"id": example.input.id, "split": split, "modality": row["modality"],
                                   "origin": example.origin, "positions": int(batch.expanded_lengths[0]),
                                   "within_budget": True,
                                   "candidate_end_positions": batch.candidate_end_positions[0].tolist(),
                                   "decision_position": int(batch.decision_positions[0]),
                                   "media_fields": sorted(batch.native_inputs)})
                except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
                    failures.append({"id": row.get("id"), "split": split, "error": str(error)})
    qualified_paths = {f"prepared/{split}.jsonl": value for split, value in prepared_hashes.items()}
    for relative, expected in {**qualified_paths, **media_hashes}.items():
        if sha256_file(media_path(data, relative)) != expected:
            raise ValueError(f"Data changed during processor qualification: {relative}")
    successful = limit is None and not failures
    checks_name = "processor-checks.jsonl" if successful else "processor-diagnostic-checks.jsonl"
    checks_path = data / checks_name
    temporary = checks_path.with_suffix(".jsonl.partial")
    write_jsonl(temporary, checks)
    os.replace(temporary, checks_path)
    slices = {}
    for field in ("modality", "origin"):
        slices[field] = {}
        for value in sorted({row[field] for row in checks}):
            lengths = sorted(row["positions"] for row in checks if row[field] == value)
            slices[field][value] = {"questions": len(lengths), "max_positions": max(lengths),
                                    "p95_positions": lengths[min(len(lengths) - 1, int(len(lengths) * .95))]}
    summary = {**compiler.identity, "processor_id": compiler.processor_id,
               "status": "VERIFIED" if successful else "NOT_VERIFIED",
               "questions_checked": len(checks), "all_questions_checked": limit is None,
               "failures": len(failures), "failure_details": failures,
               "over_budget": sum("exceeds 2048" in row["error"] for row in failures),
               "scope": list(SPLITS), "slices": slices,
               "prepared_hashes": prepared_hashes, "media_hashes": media_hashes,
               "checks_path": checks_name, "checks_sha256": sha256_file(checks_path),
               "device": "cpu", "model_weights_loaded": False}
    destination = data / ("processor-contract.json" if successful else "processor-diagnostic.json")
    temporary = destination.with_suffix(".json.partial")
    write_json(temporary, summary)
    os.replace(temporary, destination)
    if failures:
        raise ValueError(f"{len(failures)} processor failures; see {destination}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--limit-per-split", type=int)
    args = parser.parse_args()
    result = check(args.root.resolve(), limit=args.limit_per_split)
    print(json.dumps({key: value for key, value in result.items() if key != "media_hashes"}, indent=2))


if __name__ == "__main__":
    main()
