"""Render and probe the procedural clips independently of question generation."""

import json
import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from mdprep.common import sha256_file
from mdprep.video_specs import FPS, FRAME_COUNT, FRAME_SIZE, OOD_FAMILY, PALETTE

def _require_binary(preferred: Path, name: str) -> str:
    if preferred.is_file() and os.access(preferred, os.X_OK):
        return str(preferred)
    discovered = shutil.which(name)
    if discovered:
        return discovered
    raise RuntimeError(f"Required local executable not found: {name}")
def _draw_circle(draw: ImageDraw.ImageDraw, center: tuple[int, int], radius: int, color: str) -> None:
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="#25231F", width=3)


def _render_frame(render: dict, frame_index: int) -> Image.Image:
    image = Image.new("RGB", FRAME_SIZE, "#F4F1E8")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, FRAME_SIZE[0] - 9, FRAME_SIZE[1] - 9), outline="#C9C3B6", width=2)
    for decoration in render["decorations"]:
        x, y = decoration["center"]
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=decoration["color"])
    kind = render["kind"]
    if kind == "bounce":
        draw.line((24, 122, 232, 122), fill="#8D887E", width=3)
        draw.rectangle((112, 38, 144, 150), outline="#B8B1A4", width=2)
        _draw_circle(draw, (render["xs"][frame_index], 96), 16, PALETTE["amber"])
    elif kind == "color_sequence":
        draw.ellipse((88, 56, 168, 136), outline="#B8B1A4", width=3)
        _draw_circle(draw, (128, 96), 30, PALETTE[render["colors"][frame_index]])
    elif kind == "gate_order":
        draw.line((128, 18, 128, 174), fill="#5C8E91", width=4)
        _draw_circle(draw, (render["amber_xs"][frame_index], 70), 14, PALETTE["amber"])
        x = render["violet_xs"][frame_index]
        draw.rectangle((x - 14, 112, x + 14, 140), fill=PALETTE["violet"], outline="#25231F", width=3)
    elif kind == "collision":
        _draw_circle(draw, tuple(render["amber_points"][frame_index]), 16, PALETTE["amber"])
        _draw_circle(draw, tuple(render["violet_points"][frame_index]), 16, PALETTE["violet"])
    elif kind == OOD_FAMILY:
        draw.line((96, 16, 96, 176), fill=PALETTE["cyan"], width=4)
        draw.line((16, 96, 240, 96), fill=PALETTE["magenta"], width=4)
        x, y = render["points"][frame_index]
        draw.polygon(((x, y - 17), (x - 16, y + 14), (x + 16, y + 14)), fill=PALETTE["yellow"], outline="#25231F")
        draw.line(((x, y - 17), (x - 16, y + 14), (x + 16, y + 14), (x, y - 17)), fill="#25231F", width=3)
    else:
        raise ValueError(f"Unknown render kind: {kind}")
    return image


def _probe_video(path: Path, ffprobe: str) -> dict:
    command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,duration,nb_frames,nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected one video stream in {path}, found {len(streams)}")
    stream = streams[0]
    frame_rate = Fraction(stream["avg_frame_rate"])
    fps = float(frame_rate)
    raw_frame_count = stream.get("nb_read_frames") or stream.get("nb_frames")
    if not raw_frame_count or raw_frame_count == "N/A":
        raise RuntimeError(f"ffprobe did not report a frame count for {path}")
    frame_count = int(raw_frame_count)
    duration_s = float(stream["duration"])
    if frame_count != FRAME_COUNT or frame_rate != Fraction(FPS, 1):
        raise RuntimeError(
            f"Unexpected encoding metadata for {path}: fps={frame_rate}, frame_count={frame_count}"
        )
    expected_duration = Fraction(FRAME_COUNT, FPS)
    if abs(duration_s - float(expected_duration)) > 0.001:
        raise RuntimeError(f"Unexpected duration for {path}: {duration_s}")
    return {"fps": fps, "duration_s": duration_s, "frame_count": frame_count}


def _render_clip(job: dict, ffmpeg: str, ffprobe: str) -> dict:
    destination = job["absolute_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = destination.with_suffix(".building.mp4")
    if temporary_output.exists():
        temporary_output.unlink()
    try:
        with TemporaryDirectory(prefix=f"{destination.stem}-", dir=destination.parent) as temporary_directory:
            frame_directory = Path(temporary_directory)
            for frame_index in range(FRAME_COUNT):
                frame = _render_frame(job["scene"]["render"], frame_index)
                frame.save(frame_directory / f"frame_{frame_index:03d}.png", format="PNG", compress_level=9)
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(FPS),
                "-start_number",
                "0",
                "-i",
                str(frame_directory / "frame_%03d.png"),
                "-frames:v",
                str(FRAME_COUNT),
                "-an",
                "-c:v",
                "libx264rgb",
                "-preset",
                "veryslow",
                "-crf",
                "0",
                "-pix_fmt",
                "rgb24",
                "-r",
                str(FPS),
                "-threads",
                "1",
                "-map_metadata",
                "-1",
                "-fflags",
                "+bitexact",
                "-flags:v",
                "+bitexact",
                "-movflags",
                "+faststart",
                str(temporary_output),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
        temporary_output.replace(destination)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()
    metadata = _probe_video(destination, ffprobe)
    return {
        "relative_path": job["relative_path"],
        "metadata": metadata,
        "sha256": sha256_file(destination),
    }
