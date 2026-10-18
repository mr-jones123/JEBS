"""Private runtime worker. Operators use python -m mdtrain, not this module."""

import json
import signal
from pathlib import Path

from mdtrain.__main__ import parser
from mdtrain.budget import BudgetStopped
from mdtrain.deadline import request_stop
from mdtrain.evaluate import evaluate, finalize, predict
from mdtrain.preflight import preflight_cuda, smoke_real, smoke_tiny
from mdtrain.storage import atomic_json
from mdtrain.train import deterministic, resume, train


def pipeline(root: Path, *, recipe: str, run_dir: Path, budget_ledger: Path,
             paid_started_at: str, output: Path, schedule: str) -> dict:
    """Run all stages under the public CLI's single process guardian.

    The $8 process allowance at $0.89/hour excludes the $2 setup margin.
    Neither the ledger nor the guardian terminates a Pod or caps its billing.
    The operator must independently stop the Pod or install a provider stop.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    receipt = run_dir / "preflight.json"
    checked = preflight_cuda(root, recipe=recipe, budget_ledger=budget_ledger,
                             paid_started_at=paid_started_at, output=receipt, schedule=schedule)
    results = {"preflight": checked}
    if checked["status"] != "VERIFIED":
        return {"status": checked["status"], "stages": results}
    for stage in ("diagnostic", "pilot", "main"):
        result = train(root, recipe=recipe, stage=stage, run_dir=run_dir / stage,
                       preflight=receipt, budget_ledger=budget_ledger, schedule=schedule)
        results[stage] = result
        if result["status"] != "completed" or (stage != "main" and not result["accepted_learning_gate"]):
            return {"status": result["status"] if result["status"] != "completed" else "NOT_VERIFIED",
                    "stages": results, "reason": "Stage did not complete or pass its fixed-question learning gate"}
    return {"status": "completed", "stages": results, "timer_is_billing_guarantee": False}


def main() -> None:
    args = parser().parse_args()
    values = vars(args).copy()
    command = values.pop("command")
    # The guardian prices the ledger before the worker starts; every later open reuses that record.
    values.pop("hourly_usd", None)
    values["root"] = values["root"].resolve()
    if command in ("preflight", "train", "resume", "pipeline") or values.get("device") == "cuda":
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
    deterministic()
    if command == "smoke":
        function = smoke_tiny if values.pop("kind") == "tiny" else smoke_real
    elif command == "preflight":
        function = preflight_cuda
    elif command == "train":
        function = train
    elif command == "resume":
        function = resume
    elif command == "predict":
        function = predict
        values["input_path"] = values.pop("input")
    elif command == "evaluate":
        function = evaluate
        values["splits"] = tuple(values["splits"].split(","))
    elif command == "finalize":
        function = finalize
        values["fit_scalar_temperature"] = values.pop("fit_temperature")
    elif command == "pipeline":
        function = pipeline
    else:
        raise AssertionError("argparse admitted an unknown command")
    try:
        result = function(**values)
    except BudgetStopped as error:
        result = {"status": "budget_stopped", "reason": str(error), "timer_is_billing_guarantee": False}
    except KeyboardInterrupt:
        result = {"status": "interrupted", "checkpoint": None,
                  "reason": "Interrupted outside the admitted optimizer-boundary update loop"}
    if command == "pipeline":
        atomic_json(values["output"], result)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit({"completed": 0, "VERIFIED": 0, "NOT_VERIFIED": 2,
                      "budget_stopped": 3, "interrupted": 4}.get(result["status"], 1))


if __name__ == "__main__":
    main()
