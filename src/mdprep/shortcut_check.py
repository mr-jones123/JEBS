"""Measure fixed numeric-rank and rubric-gap shortcuts using train/dev only."""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from mdprep.common import sha256_file, write_json

_NUMERIC_FAMILIES=frozenset(f"image-{category}-lookup" for category in ("receipt", "bar_chart", "spatial"))
_RUBRIC=re.compile(r"(?:0 if below|Tier 0 below) (-?\d+).*?(?:2 if|Tier 2 from) (-?\d+).*?(?:3 if|Tier 3 at) (-?\d+)", re.S)


def _majority(counts):
    return min(counts, key=lambda key: (-counts[key], key))


def check(root: Path) -> dict:
    rows={}
    hashes={}
    for split in ("train", "dev"):
        path=root / "data/prepared" / f"{split}.jsonl"
        hashes[split]=sha256_file(path)
        with path.open() as stream:
            rows[split]=[json.loads(line) for line in stream]
    numeric={}
    structural_failures=[]
    for family in sorted(_NUMERIC_FAMILIES):
        ranks={}
        for split in rows:
            selected=[row for row in rows[split] if row["provenance"]["family"]==family]
            observed=[]
            for row in selected:
                criteria=row["question"]["criteria"]
                candidates=[float(value.removeprefix("$")) for value in criteria.values()]
                if len(set(candidates))!=len(candidates):
                    raise ValueError("Numeric control requires distinct visible candidates")
                correct=float(criteria[row["target"]["label"]].removeprefix("$"))
                observed.append((sorted(candidates).index(correct), len(candidates)))
            if not observed:
                raise ValueError(f"Missing numeric control slice {family}/{split}")
            ranks[split]=observed
        majority_rank=_majority(Counter(rank for rank, count in ranks["train"]))
        dev=ranks["dev"]
        numeric[family]={
            "train_questions": len(ranks["train"]), "dev_questions": len(dev),
            "train_correct_rank_counts": dict(sorted(Counter(rank for rank, count in ranks["train"]).items())),
            "dev_correct_rank_counts": dict(sorted(Counter(rank for rank, count in dev).items())),
            "candidate_counts": dict(sorted(Counter(count for rank, count in dev).items())),
            "second_smallest_accuracy": sum(rank==1 for rank, count in dev)/len(dev),
            "train_majority_rank": majority_rank,
            "train_majority_rank_dev_accuracy": sum(rank==majority_rank for rank, count in dev)/len(dev),
            "uniform_candidate_chance": sum(1/count for rank, count in dev)/len(dev),
        }
        if len({rank for rank, count in ranks["train"]})==1:
            structural_failures.append(family)
    training=defaultdict(Counter)
    prior=Counter()
    development=[]
    for split, selected in rows.items():
        for row in selected:
            if row["provenance"]["source"]!="synthetic-visuals-v1" or not row["provenance"]["family"].endswith(("-rubric", "-policy-rubric")):
                continue
            text=row["state"]+"\n"+row["question"]["instructions"]
            match=_RUBRIC.search(text)
            if not match:
                raise ValueError("Unparsed model-visible rubric in shortcut control")
            first, second, third=map(int, match.groups())
            feature=(second-first, third-second)
            label=row["target"]["label"]
            if split=="train":
                training[feature][label]+=1
                prior[label]+=1
            else:
                development.append((feature, label))
    if not prior or not development:
        raise ValueError("Rubric control requires train and development rows")
    prior_prediction=_majority(prior)
    correct=sum(_majority(training.get(feature, prior))==label for feature, label in development)
    rubric={
        "feature": "two adjacent gaps between the three displayed rubric thresholds",
        "train_questions": sum(prior.values()), "dev_questions": len(development),
        "training_gap_patterns": len(training),
        "known_gap_patterns_in_dev": sum(feature in training for feature, label in development),
        "correct": correct, "accuracy": correct/len(development),
        "train_majority_prior_dev_accuracy": sum(label==prior_prediction for feature, label in development)/len(development),
        "train_label_counts": dict(sorted(prior.items())),
        "dev_label_counts": dict(sorted(Counter(label for feature, label in development).items())),
    }
    return {"status": "MEASURED", "scope": "train/development generated visual questions only",
            "prepared_hashes": hashes, "numeric_rank": numeric, "rubric_gap": rubric,
            "structurally_fixed_numeric_rank_families": structural_failures,
            "media_or_audit_used_as_features": False, "final_test_used": False,
            "limitation": "These are narrow deterministic controls, not model quality or proof that all shortcuts are absent."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    result=check(args.root.resolve())
    write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__=="__main__":
    main()
