"""Bounded real media and unlabeled requests for structural runtime exercises."""

import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from mdprep.records import Candidate, DecisionInput, TrainingExample, VideoRef, parse_request


def make_fixtures(directory: Path) -> tuple[TrainingExample, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (96, 96), "white")
    ImageDraw.Draw(image).rectangle((24, 24, 72, 72), fill="red")
    image.save(directory / "red.png")
    frames = []
    for x in (16, 32, 48, 64):
        frame = Image.new("RGB", (96, 96), "white")
        ImageDraw.Draw(frame).rectangle((x, 32, x + 16, 48), fill="blue")
        frames.append(frame.tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "96x96",
                    "-r", "2", "-i", "pipe:0", "-frames:v", "4", "-c:v", "libx264", "-crf", "0",
                    "-pix_fmt", "yuv420p", "-y", str(directory / "moving.mp4")],
                   input=b"".join(frames), check=True, capture_output=True)
    inputs = (
        DecisionInput("fixture-text", "The amount is 4. Approve amounts below 5.", "choice", "Choose the decision.",
                      (Candidate("private-approve", "Approve"), Candidate("private-deny", "Deny")), (), ()),
        DecisionInput("fixture-image", "", "noul", "Is the central rectangle red?",
                      (Candidate("false", "No"), Candidate("true", "Yes")), ("red.png",), ()),
        DecisionInput("fixture-mixed", "Count one point for a red central rectangle, otherwise zero.", "score",
                      "Choose the point count.", (Candidate("0", "Zero points"), Candidate("1", "One point")),
                      ("red.png",), ()),
        DecisionInput("fixture-video", "Observe the entire clip.", "choice", "Which way does the blue square move?",
                      (Candidate("private-right", "Right"), Candidate("private-left", "Left"),
                       Candidate("private-still", "It remains stationary")), (),
                      (VideoRef("moving.mp4", 2., 2., 4),)),
    )
    examples = tuple(TrainingExample(item, target, item.id, "generated_media", "structural_fixture",
                                     "runtime_smoke", "none") for item, target in zip(inputs, (0, 1, 1, 0)))
    write_requests(directory / "requests.jsonl", inputs)
    return examples


def write_requests(path: Path, inputs) -> None:
    with path.open("w") as stream:
        for item in inputs:
            stream.write(json.dumps(dataclasses.asdict(item)) + "\n")


def read_requests(path: Path) -> tuple[DecisionInput, ...]:
    with path.open() as stream:
        items = tuple(parse_request(json.loads(line)) for line in stream)
    if not items or len({item.id for item in items}) != len(items):
        raise ValueError("Prediction input must contain nonempty unique IDs")
    return items


def parameter_digest(model, *, frozen_only: bool = False) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if frozen_only and (name.startswith("pointer.") or ".lora_A." in name or ".lora_B." in name):
            continue
        digest.update(name.encode())
        value = parameter.detach().cpu().contiguous().view(torch.uint8).numpy()
        digest.update(memoryview(value))
    return digest.hexdigest()


def trainable_snapshot(model) -> dict:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
            if name.startswith("pointer.") or ".lora_A." in name or ".lora_B." in name}


def compare_states(left, right, *, exact: bool) -> None:
    if isinstance(left, torch.Tensor):
        if exact:
            if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or not torch.equal(left, right):
                raise AssertionError("Exact next-update tensor equality failed")
        else:
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
    elif isinstance(left, dict):
        if set(left) != set(right):
            raise AssertionError("Recovery state keys differ")
        for key in left:
            compare_states(left[key], right[key], exact=exact)
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError("Recovery state lengths differ")
        for a, b in zip(left, right):
            compare_states(a, b, exact=exact)
    elif left != right:
        raise AssertionError("Recovery state scalar differs")
