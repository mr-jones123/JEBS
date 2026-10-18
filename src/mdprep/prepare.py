"""Run CPU data builders, then assemble the complete local dataset."""

import argparse
import json
from pathlib import Path

from mdprep import hardv1, photos, rule_variants, text, videos, visuals
from mdprep.assemble import assemble
from mdprep.common import write_json

BUILDERS = {
    "text": text.prepare,
    "photos": photos.prepare,
    "visuals": visuals.prepare,
    "videos": videos.prepare,
    "hardv1": hardv1.prepare,
    "rule_variants": rule_variants.prepare,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--part", choices=("all", *BUILDERS, "assemble"), default="all")
    args = parser.parse_args()
    root = args.root.resolve()
    results = {}
    if args.part == "all":
        for name, builder in BUILDERS.items():
            print(f"Preparing {name}", flush=True)
            results[name] = builder(root)
            write_json(root / "data" / "sources" / f"{name}-build.json", results[name])
        results["assembly"] = assemble(root)
    elif args.part == "assemble":
        results["assembly"] = assemble(root)
    else:
        results[args.part] = BUILDERS[args.part](root)
        write_json(root / "data" / "sources" / f"{args.part}-build.json", results[args.part])
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
