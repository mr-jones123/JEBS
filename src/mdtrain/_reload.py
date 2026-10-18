"""Fresh-process paired artifact proof. Invoked by smoke and CUDA preflight."""

import argparse
from pathlib import Path

from mdtrain.engine import DecisionModel
from mdtrain.fixtures import read_requests
from mdtrain.storage import atomic_json
from mdtrain.train import deterministic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    deterministic()
    model = DecisionModel.load(args.artifact, root=args.root, device=args.device, media_root=args.media_root)
    predictions = [model.predict((item,))[0] for item in read_requests(args.input)]
    atomic_json(args.output, {"status": "VERIFIED", "model_kind": model.spec["kind"],
                              "device": args.device, "predictions": predictions})


if __name__ == "__main__":
    main()
