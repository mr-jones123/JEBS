"""Local native runtime commands. Nothing creates, uploads to, or terminates a Pod."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from mdtrain.budget import BudgetLedger, BudgetStopped
from mdtrain.deadline import ledger_deadline, supervise
from mdtrain.data import DEFAULT_SCHEDULE, SCHEDULES
from mdtrain.storage import writer_lock
from mdtrain.engine import INFERENCE_DEVICES


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("smoke", "preflight", "train", "resume", "predict", "evaluate", "finalize", "pipeline"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, default=Path.cwd())
        if name == "smoke":
            command.add_argument("--kind", choices=("tiny", "real"), required=True)
            command.add_argument("--output", type=Path, required=True)
        if name in ("preflight", "train", "pipeline"):
            command.add_argument("--recipe", choices=("real", "mixed"), required=True)
        if name in ("smoke", "preflight", "train", "resume", "pipeline"):
            command.add_argument("--schedule", choices=tuple(SCHEDULES), default=DEFAULT_SCHEDULE)
        if name in ("preflight", "pipeline"):
            command.add_argument("--paid-started-at", required=True)
            command.add_argument("--hourly-usd", type=float, required=True)
            command.add_argument("--output", type=Path, required=True)
        if name in ("train", "pipeline"):
            command.add_argument("--run-dir", type=Path, required=True)
        if name == "train":
            command.add_argument("--stage", choices=("diagnostic", "pilot", "main"), required=True)
        if name in ("train", "resume"):
            command.add_argument("--preflight", type=Path, required=True)
        if name in ("resume", "finalize"):
            command.add_argument("--checkpoint", type=Path, required=True)
        if name in ("preflight", "train", "resume", "pipeline"):
            command.add_argument("--budget-ledger", type=Path, required=True)
        if name in ("predict", "evaluate", "finalize"):
            # Finalize fits calibration for a paid run, which is CUDA or CPU only.
            command.add_argument("--device", required=True,
                                 choices=INFERENCE_DEVICES if name != "finalize" else ("cpu", "cuda"))
            command.add_argument("--budget-ledger", type=Path)
            command.add_argument("--output", type=Path, required=True)
        if name in ("predict", "evaluate"):
            command.add_argument("--artifact", type=Path, required=True)
        if name == "predict":
            command.add_argument("--input", type=Path, required=True)
            command.add_argument("--media-root", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--splits", required=True)
        if name == "finalize":
            command.add_argument("--fit-temperature", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    command = [sys.executable, "-m", "mdtrain._worker", *sys.argv[1:]]
    paid = (args.command in ("preflight", "train", "resume", "pipeline")
            or (getattr(args, "device", None) == "cuda" and args.budget_ledger is not None))
    if not paid:
        raise SystemExit(subprocess.call(command))
    path = args.budget_ledger.resolve()
    try:
        with writer_lock(path.with_suffix(path.suffix + ".guardian.lock")):
            with BudgetLedger(path, paid_started_at=getattr(args, "paid_started_at", None),
                              hourly_usd=getattr(args, "hourly_usd", None)) as ledger:
                ledger.record.pop("process_deadline_elapsed", None)
                ledger.commit()
                ledger.admit(0)
                identity = ledger.record["id"]
                deadline = ledger_deadline(ledger.record, now_utc=time.time(), now_monotonic=time.monotonic())
            result = supervise(command, deadline=deadline, ledger_path=path, ledger_id=identity)
    except BudgetStopped as error:
        print(json.dumps({"status": "budget_stopped", "reason": str(error),
                          "timer_is_billing_guarantee": False}, allow_nan=False))
        raise SystemExit(3)
    if result["hard_stopped"]:
        print(json.dumps({"status": "budget_stopped", "deadline": result,
                          "reason": "Paid process group deadline reached. Recover only the last published checkpoint.",
                          "timer_is_billing_guarantee": False}, allow_nan=False))
        raise SystemExit(3)
    raise SystemExit(result["returncode"])


if __name__ == "__main__":
    main()
