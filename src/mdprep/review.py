"""Render the stratified review pack; this does not mark it reviewed."""

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from mdprep.assemble import load_rows
from mdprep.native import read_video

REVIEW_TRANSLATION = str.maketrans({"\u2264":"<=", "\u2265":">=", "\u2013":"..",
                                  "\u2014":" - ", "\u201c":'"', "\u201d":'"',
                                  "\u2018":"'", "\u2019":"'"})


def render(root: Path) -> dict:
    data = root / "data"
    rows = load_rows(data / "review" / "review-200.jsonl")
    font = ImageFont.load_default(size=16)
    sheets = []
    for offset in range(0, len(rows), 8):
        chunk = rows[offset:offset + 8]
        sheet = Image.new("RGB", (1280, 2400), "white")
        for index, row in enumerate(chunk):
            x, y = index % 2 * 640, index // 2 * 600
            draw = ImageDraw.Draw(sheet)
            if row["images"]:
                with Image.open(data / row["images"][0]) as image:
                    thumbnail = ImageOps.contain(image.convert("RGB"), (620, 360))
                    sheet.paste(thumbnail, (x + 10, y + 10))
            elif row["videos"]:
                frames, _ = read_video(str(data / row["videos"][0]["path"]))
                for frame_index, frame in enumerate(frames[:4]):
                    thumbnail = ImageOps.contain(Image.fromarray(frame), (305, 170))
                    sheet.paste(thumbnail, (x + 10 + (frame_index % 2) * 310,
                                            y + 10 + (frame_index // 2) * 180))
            else:
                state_lines = textwrap.wrap(row["state"].translate(REVIEW_TRANSLATION), 65)[:16]
                draw.multiline_text((x + 10, y + 10), "\n".join(state_lines), fill="black", font=font, spacing=4)
            description = (
                f"{offset + index + 1}: {row['split']} / {row['modality']} / {row['question']['type']}\n"
                + row["question"]["instructions"] + "\n"
                + " | ".join(f"{k}: {v}" for k, v in row["question"]["criteria"].items())
                + f"\nTARGET: {row['target']['label']}\nID: {row['id']}"
            )
            if row["modality"] == "mixed":
                description = "TEXT: " + row["state"] + "\n" + description
            lines = []
            for line in description.splitlines():
                lines.extend(textwrap.wrap(line.translate(REVIEW_TRANSLATION), 65) or [""])
            draw.multiline_text((x + 10, y + 375), "\n".join(lines[:12]), fill="black", font=font, spacing=2)
            draw.rectangle((x, y, x + 639, y + 599), outline="#999999")
        destination = data / "review" / f"sheet-{offset // 8 + 1:02}.jpg"
        sheet.save(destination, quality=88)
        sheets.append(str(destination.relative_to(root)))
    return {"questions": len(rows), "sheets": sheets, "status": "prepared-for-independent-review"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(render(args.root.resolve()), indent=2))


if __name__ == "__main__":
    main()
