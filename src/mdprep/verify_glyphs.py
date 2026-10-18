"""Exact opaque-pixel recognition for the declared controlled renderer profile."""

import hashlib
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT_RELATIVE = Path("data/raw/visuals/atkinson-hyperlegible/AtkinsonHyperlegible-Regular.ttf")
FONT_SHA256 = "7fb917c89019896d0b52ee84b7cbb3304c18cb90b19a62f5e32712bd23e97669"
METHOD = "exact-opaque-glyph-dictionary-v1"


@dataclass(frozen=True)
class GlyphField:
    name: str
    xy: tuple[int, int]
    size: int
    color: tuple[int, int, int]
    candidates: tuple[str, ...]
    anchor: str | None = None
    box: tuple[int, int, int, int] | None = None


def resolve_font(image_path, font_path=None):
    if font_path is not None:
        return str(Path(font_path).resolve())
    for parent in Path(image_path).resolve().parents:
        candidate = parent / FONT_RELATIVE
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError(f"Controlled glyph recognition requires pinned font {FONT_RELATIVE}")


@lru_cache(maxsize=12)
def _font(path, size):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != FONT_SHA256:
        raise ValueError("Controlled recognition font hash mismatch")
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=512)
def _dictionary(font_path, xy, size, candidates, anchor, field_box):
    if not candidates or len(set(candidates)) != len(candidates) or any(not text for text in candidates):
        raise ValueError("Glyph dictionary must contain distinct nonempty complete strings")
    font = _font(font_path, size)
    canvas = Image.new("L", (450, 450))
    draw = ImageDraw.Draw(canvas)
    bounds = [draw.textbbox(xy, text, font=font, anchor=anchor) for text in candidates]
    box = (min(b[0] for b in bounds) - 1, min(b[1] for b in bounds) - 1,
           max(b[2] for b in bounds) + 1, max(b[3] for b in bounds) + 1)
    if field_box is not None:
        if any(b[0] < field_box[0] or b[1] < field_box[1] or
               b[2] > field_box[2] or b[3] > field_box[3] for b in bounds):
            raise ValueError("Glyph dictionary is clipped by the declared field")
        box = field_box
    if box[0] < 0 or box[1] < 0 or box[2] > 450 or box[3] > 450:
        raise ValueError("Complete glyph field lies outside the canvas")
    masks = {}
    for text in candidates:
        canvas.paste(0, (0, 0, 450, 450))
        draw.text(xy, text, font=font, fill=255, anchor=anchor)
        opaque = canvas.crop(box).point(lambda value: 255 if value == 255 else 0).tobytes()
        if not any(opaque):
            raise ValueError("Glyph dictionary contains an invisible string")
        masks.setdefault(opaque, []).append(text)
    return box, masks


def recognize(image, field, font_path, evidence=None):
    box, masks = _dictionary(font_path, field.xy, field.size, field.candidates, field.anchor, field.box)
    crop = image.crop(box).convert("RGB")
    entry = {"field": field.name, "box": list(box),
             "pixel_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
             "method": METHOD, "font_sha256": FONT_SHA256,
             "candidate_count": len(field.candidates), "status": "INCONCLUSIVE"}
    if evidence is not None:
        evidence.append(entry)
    pixels = crop.load()
    actual = bytes(255 if pixels[x,y] == field.color else 0
                   for y in range(crop.height) for x in range(crop.width))
    matches = masks.get(actual, ())
    if len(matches) != 1:
        raise ValueError(f"Unresolved exact glyph field {field.name!r}: {len(matches)} complete matches")
    text = matches[0]
    alpha = Image.new("L", (450,450))
    ImageDraw.Draw(alpha).text(field.xy, text, font=_font(font_path,field.size),
                              fill=255, anchor=field.anchor)
    alpha = alpha.crop(box)
    coverage = alpha.load()
    background = Image.new("RGB", crop.size)
    background_draw = ImageDraw.Draw(background)
    for y in range(crop.height):
        counts = Counter(pixels[x,y] for x in range(crop.width) if coverage[x,y] == 0)
        if not counts:
            raise ValueError("Glyph field has no observable row background")
        color = counts.most_common(1)[0][0]
        background_draw.line((0,y,crop.width-1,y), fill=color)
    reconstructed = Image.composite(Image.new("RGB",crop.size,field.color), background, alpha)
    if reconstructed.tobytes() != crop.tobytes():
        raise ValueError(f"Unexpected or damaged pixels in complete glyph field {field.name!r}")
    entry.update(text=text, status="EXTRACTED")
    return text
