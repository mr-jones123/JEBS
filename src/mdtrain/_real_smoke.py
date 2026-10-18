"""Inference-only real BF16 child. No implicit downloads or tiny-model substitution."""

import argparse
from pathlib import Path

import psutil

from mdprep.native import NativeCompiler
from mdtrain.engine import DecisionModel
from mdtrain.fixtures import read_requests
from mdtrain.storage import atomic_json
from mdtrain.train import deterministic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    deterministic()
    compiler = NativeCompiler.from_pretrained(args.root, media_root=args.directory)
    model = DecisionModel.create(args.root, compiler, device="cpu", kind="real")
    inputs = read_requests(args.directory / "requests.jsonl")
    predictions, shapes = [], []
    for item in inputs:
        batch = compiler.compile((item,))
        prediction = model.predict((item,))[0]
        predictions.append(prediction)
        shapes.append({"id": item.id, "positions": int(batch.expanded_lengths[0]),
                       "native_fields": sorted(batch.native_inputs)})
    model.save(args.directory / "real-pair", metadata={"scope": "bounded BF16 CPU inference only"})
    for prediction in predictions:
        prediction["artifact_id"] = model.artifact_id
    atomic_json(args.directory / "real-child.json", {
        "status": "VERIFIED", "model_kind": "real", "model_id": model.spec["model_id"],
        "revision": model.spec["model_revision"], "dtype": "bfloat16", "device": "cpu",
        "predictions": predictions, "inputs": shapes, "rss_bytes": psutil.Process().memory_info().rss,
        "backward_verified": False, "cuda_verified": False})


if __name__ == "__main__":
    main()
