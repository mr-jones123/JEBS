"""Deterministic synthetic visual-decision data builder."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from mdprep.common import (
    GENERATOR_VERSION,
    PART_QUOTAS,
    SPLITS,
    choice,
    fetch,
    noul,
    record,
    rng_for,
    score,
    sha256_file,
    stable_id,
    write_json,
    write_jsonl,
)

SOURCE = "synthetic-visuals-v1"
FONT_REVISION = "1cb311624b2ddf88e9e37873999d165a8cd28b46"
FONT_URL = (
    "https://raw.githubusercontent.com/googlefonts/atkinson-hyperlegible/"
    f"{FONT_REVISION}/fonts/ttf/AtkinsonHyperlegible-Regular.ttf"
)
FONT_LICENSE_URL = (
    "https://raw.githubusercontent.com/googlefonts/atkinson-hyperlegible/"
    f"{FONT_REVISION}/OFL.txt"
)
FONT_SHA256 = "7fb917c89019896d0b52ee84b7cbb3304c18cb90b19a62f5e32712bd23e97669"
FONT_LICENSE_SHA256 = "f32d22b3908fcad2c86a74000614ec22e6a7f66ea7e867e616026a27aebdc143"
WIDTH = 450
HEIGHT = 450
CATEGORIES = ("receipt", "bar_chart", "request_form", "dashboard", "spatial")
THEMES = (
    {"ink": "#17212b", "muted": "#65727e", "paper": "#fffdf8", "accent": "#19647e", "soft": "#dceff4"},
    {"ink": "#261f2d", "muted": "#746a7b", "paper": "#fffaf3", "accent": "#8a4f2d", "soft": "#f3dfcf"},
    {"ink": "#132a20", "muted": "#607269", "paper": "#f8fff9", "accent": "#24734d", "soft": "#dcefe3"},
)
OOD_LAYOUTS = {
    "receipt": "receipt-stacked-cards-ood",
    "bar_chart": "bar-horizontal-ood",
    "request_form": "form-field-cards-ood",
    "dashboard": "dashboard-sidebar-ood",
    "spatial": "spatial-coordinate-grid-ood",
}
ACTION_SETS = {
    "receipt": ("Decline", "Manual review", "Reimburse", "Priority reimburse"),
    "bar_chart": ("Observe", "Notify", "Investigate", "Escalate", "Emergency response"),
    "request_form": ("Return to requester", "Supervisor review", "Authorize"),
    "dashboard": ("Backlog", "Normal queue", "Fast track", "Incident bridge"),
    "spatial": ("Accept layout", "Inspect layout", "Revise layout", "Close area"),
}
MISSING_FIELDS = {
    "receipt": "a visible cost-center approval code",
    "bar_chart": "a visible audited-denominator badge",
    "request_form": "a visible supervisor-clearance code",
    "dashboard": "a visible customer-impact verification",
    "spatial": "a visible ceiling-height measurement",
}
METRIC_NAMES = {
    "receipt": "printed TOTAL in dollars",
    "bar_chart": "Nimbus count",
    "request_form": "requested units",
    "dashboard": "open-ticket count",
    "spatial": "red-triangle count",
}
DIRECTIONS = {
    "receipt": "maximum",
    "bar_chart": "minimum",
    "request_form": "maximum",
    "dashboard": "minimum",
    "spatial": "maximum",
}


def _font_files(root: Path) -> tuple[Path, Path]:
    font_dir = root / "data/raw/visuals/atkinson-hyperlegible"
    font_path = fetch(FONT_URL, font_dir / "AtkinsonHyperlegible-Regular.ttf")
    license_path = fetch(FONT_LICENSE_URL, font_dir / "OFL.txt")
    actual_font_hash = sha256_file(font_path)
    actual_license_hash = sha256_file(license_path)
    if actual_font_hash != FONT_SHA256:
        raise RuntimeError(f"Pinned font hash mismatch: {actual_font_hash}")
    if actual_license_hash != FONT_LICENSE_SHA256:
        raise RuntimeError(f"Pinned font license hash mismatch: {actual_license_hash}")
    return font_path, license_path


def _fonts(font_path: Path) -> dict[str, ImageFont.FreeTypeFont]:
    return {
        "xs": ImageFont.truetype(str(font_path), 14),
        "sm": ImageFont.truetype(str(font_path), 17),
        "body": ImageFont.truetype(str(font_path), 20),
        "strong": ImageFont.truetype(str(font_path), 23),
        "title": ImageFont.truetype(str(font_path), 30),
        "hero": ImageFont.truetype(str(font_path), 42),
    }


def _write(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: object,
           font: ImageFont.FreeTypeFont, fill: str, anchor: str | None = None) -> None:
    draw.text(xy, str(text), font=font, fill=fill, anchor=anchor)


def _metric_values(category: str, rng: Any) -> tuple[int, int]:
    if category == "receipt":
        low = rng.randint(38, 58)
        return low, low + rng.randint(22, 34)
    if category == "bar_chart":
        low = rng.randint(24, 39)
        return low, low + rng.randint(28, 39)
    if category == "request_form":
        low = rng.randint(24, 44)
        return low, low + rng.randint(28, 42)
    if category == "dashboard":
        low = rng.randint(3, 7)
        return low, low + rng.randint(5, 9)
    low = rng.randint(2, 4)
    return low, low + rng.randint(3, 4)


def _make_scene(split: str, index: int, attempt: int = 0) -> dict[str, Any]:
    category = CATEGORIES[index % len(CATEGORIES)]
    rng = rng_for(SOURCE, split, index, category, attempt)
    low, high = _metric_values(category, rng)
    is_ood = split == "test_ood"
    layout_family = OOD_LAYOUTS[category] if is_ood else f"{category}-standard-{index % 3}"
    scene: dict[str, Any] = {
        "category": category,
        "index": index,
        "theme": index % len(THEMES),
        "layout_family": layout_family,
        "ood": is_ood,
        "values": {"a": low, "b": high},
        "flag": bool(index % 2),
        "required_field_visible": bool((index // len(CATEGORIES)) % 2),
    }
    if category == "receipt":
        scene.update(
            merchant=("NORTH PINE", "COPPER & CO", "GREEN MARKET")[index % 3],
            receipt_number=f"R-{rng.randint(10000, 99999)}",
            date=f"2026-{(index % 9) + 1:02d}-{(index * 7 % 27) + 1:02d}",
            items=("Notebook", "Cable", "Tea", "Storage box"),
        )
    elif category == "bar_chart":
        comparator = high + 6 if scene["flag"] else max(5, high - 6)
        scene.update(
            title=("Resolved cases", "Weekly inspections", "Validated batches")[index % 3],
            labels=("Aster", "Cedar", "Nimbus", "Orchid"),
            fixed_values=(rng.randint(15, 70), comparator, None, rng.randint(18, 68)),
        )
    elif category == "request_form":
        scene.update(
            request_id=f"Q-{rng.randint(1000, 9999)}",
            department=("Facilities", "Research", "Operations")[index % 3],
            requester=("Morgan L.", "Avery K.", "Jordan P.")[index % 3],
        )
    elif category == "dashboard":
        scene.update(
            queue=("Atlas Support", "Beacon Ops", "Cedar Service")[index % 3],
            waiting=rng.randint(1, 6),
            resolved=rng.randint(18, 44),
            activities=rng.sample((
                "New request", "Assigned to agent", "Customer reply", "Awaiting confirmation",
                "Document received", "Callback scheduled", "Issue escalated", "Reply drafted",
                "Review completed", "Follow-up queued", "Team notified", "Status updated",
            ), 3),
        )
    else:
        relation = "left" if scene["flag"] else "right"
        candidate_positions = [
            (58 + column * 80, 144 + row * 68)
            for row in range(3)
            for column in range(5)
        ]
        rng.shuffle(candidate_positions)
        triangle_geometry = tuple(
            (x + rng.randint(-10, 10), y + rng.randint(-10, 10), rng.randint(13, 18))
            for x, y in candidate_positions[:8]
        )
        marker_x_positions = rng.sample((70, 126, 182, 238, 294, 350, 398), 2)
        left_x, right_x = sorted(marker_x_positions)
        blue_x, square_x = (
            (left_x, right_x) if relation == "left" else (right_x, left_x)
        )
        scene.update(
            map_name=("Packing zone", "Lab bay", "Loading area")[index % 3],
            relation=relation,
            triangle_geometry=triangle_geometry,
            blue_center=(blue_x, rng.randint(340, 365)),
            square_center=(square_x, rng.randint(340, 365)),
        )
    return scene


def _base_image(scene: dict[str, Any]) -> tuple[Image.Image, ImageDraw.ImageDraw, dict[str, str]]:
    theme = THEMES[scene["theme"]]
    image = Image.new("RGB", (WIDTH, HEIGHT), theme["paper"])
    return image, ImageDraw.Draw(image), theme


def _render_receipt(scene: dict[str, Any], value: int,
                    fonts: dict[str, ImageFont.FreeTypeFont]) -> Image.Image:
    image, draw, theme = _base_image(scene)
    if scene["ood"]:
        draw.rounded_rectangle((22, 20, 428, 430), radius=26, fill="#f4eee4", outline=theme["ink"], width=2)
        _write(draw, (48, 48), scene["merchant"], fonts["title"], theme["ink"])
        _write(draw, (48, 83), "PURCHASE SUMMARY", fonts["xs"], theme["accent"])
        _write(draw, (48, 110), scene["date"], fonts["sm"], theme["muted"])
        y = 148
        savings = 5 if scene["flag"] else 0
        fixed = (8, 11, 9)
        prices = (*fixed, value - sum(fixed) + savings)
        for name, price in zip(scene["items"], prices):
            draw.rounded_rectangle((42, y, 408, y + 43), radius=10, fill="#fffaf2")
            _write(draw, (58, y + 11), name, fonts["sm"], theme["ink"])
            _write(draw, (389, y + 11), f"${price:.2f}", fonts["sm"], theme["ink"], "ra")
            y += 50
        if scene["flag"]:
            _write(draw, (58, 354), "MEMBER SAVINGS", fonts["xs"], theme["accent"])
            _write(draw, (389, 354), "-$5.00", fonts["xs"], theme["accent"], "ra")
        _write(draw, (58, 397), "TOTAL", fonts["strong"], theme["ink"])
        _write(draw, (389, 397), f"${value:.2f}", fonts["strong"], theme["ink"], "ra")
        return image

    draw.rectangle((62, 18, 388, 432), fill="#fffefa", outline="#c9c4b8", width=2)
    _write(draw, (225, 46), scene["merchant"], fonts["title"], theme["ink"], "ma")
    _write(draw, (225, 80), "SALES RECEIPT", fonts["xs"], theme["muted"], "ma")
    _write(draw, (82, 109), scene["date"], fonts["sm"], theme["ink"])
    _write(draw, (367, 109), scene["receipt_number"], fonts["sm"], theme["ink"], "ra")
    draw.line((82, 137, 368, 137), fill="#8b8b83", width=1)
    savings = 5 if scene["flag"] else 0
    fixed = (8, 11, 9)
    prices = (*fixed, value - sum(fixed) + savings)
    y = 157
    for name, price in zip(scene["items"], prices):
        _write(draw, (84, y), name, fonts["sm"], theme["ink"])
        _write(draw, (366, y), f"${price:.2f}", fonts["sm"], theme["ink"], "ra")
        y += 37
    draw.line((82, 308, 368, 308), fill="#8b8b83", width=1)
    if scene["flag"]:
        _write(draw, (84, 326), "MEMBER SAVINGS", fonts["xs"], theme["accent"])
        _write(draw, (366, 326), "-$5.00", fonts["xs"], theme["accent"], "ra")
    _write(draw, (84, 366), "TOTAL", fonts["strong"], theme["ink"])
    _write(draw, (366, 366), f"${value:.2f}", fonts["strong"], theme["ink"], "ra")
    _write(draw, (225, 414), "Thank you", fonts["sm"], theme["muted"], "ma")
    return image


def _bar_values(scene: dict[str, Any], value: int) -> list[int]:
    return [value if item is None else item for item in scene["fixed_values"]]


def _render_bar_chart(scene: dict[str, Any], value: int,
                      fonts: dict[str, ImageFont.FreeTypeFont]) -> Image.Image:
    image, draw, theme = _base_image(scene)
    values = _bar_values(scene, value)
    labels = scene["labels"]
    _write(draw, (30, 28), scene["title"], fonts["title"], theme["ink"])
    _write(draw, (30, 65), "COUNT BY TEAM", fonts["xs"], theme["muted"])
    if scene["ood"]:
        x0, x1 = 112, 410
        y = 116
        for label, bar_value in zip(labels, values):
            _write(draw, (96, y + 13), label, fonts["sm"], theme["ink"], "ra")
            draw.rounded_rectangle((x0, y, x1, y + 34), radius=7, fill=theme["soft"])
            end = x0 + int((x1 - x0) * bar_value / 100)
            draw.rounded_rectangle((x0, y, end, y + 34), radius=7, fill=theme["accent"])
            _write(draw, (min(end + 8, 423), y + 8), bar_value, fonts["sm"], theme["ink"])
            y += 67
        _write(draw, (112, 404), "0", fonts["xs"], theme["muted"])
        _write(draw, (410, 404), "100", fonts["xs"], theme["muted"], "ra")
        return image

    chart = (52, 105, 415, 382)
    draw.line((chart[0], chart[3], chart[2], chart[3]), fill=theme["ink"], width=2)
    draw.line((chart[0], chart[1], chart[0], chart[3]), fill=theme["ink"], width=2)
    for tick in range(0, 101, 25):
        y = chart[3] - int((chart[3] - chart[1]) * tick / 100)
        draw.line((chart[0], y, chart[2], y), fill="#d6d9d7", width=1)
        _write(draw, (43, y), tick, fonts["xs"], theme["muted"], "rm")
    bar_width = 54
    for pos, (label, bar_value) in enumerate(zip(labels, values)):
        x = 75 + pos * 83
        top = chart[3] - int((chart[3] - chart[1]) * bar_value / 100)
        draw.rounded_rectangle((x, top, x + bar_width, chart[3]), radius=6, fill=theme["accent"])
        _write(draw, (x + bar_width // 2, top - 8), bar_value, fonts["sm"], theme["ink"], "mb")
        _write(draw, (x + bar_width // 2, 402), label, fonts["xs"], theme["ink"], "ma")
    return image


def _render_request_form(scene: dict[str, Any], value: int,
                         fonts: dict[str, ImageFont.FreeTypeFont]) -> Image.Image:
    image, draw, theme = _base_image(scene)
    _write(draw, (28, 27), "RESOURCE REQUEST", fonts["title"], theme["ink"])
    _write(draw, (28, 65), scene["request_id"], fonts["sm"], theme["accent"])
    fields = (
        ("REQUESTER", scene["requester"]),
        ("DEPARTMENT", scene["department"]),
        ("REQUESTED UNITS", str(value)),
        ("REQUESTER SIGNED", "Yes" if scene["flag"] else "No"),
    )
    if scene["ood"]:
        y = 102
        for label, field_value in fields:
            draw.rounded_rectangle((28, y, 422, y + 66), radius=12, fill=theme["soft"])
            _write(draw, (47, y + 10), label, fonts["xs"], theme["muted"])
            _write(draw, (47, y + 33), field_value, fonts["body"], theme["ink"])
            y += 76
        _write(draw, (28, 414), "Submitted information", fonts["xs"], theme["muted"])
        return image

    draw.rounded_rectangle((28, 96, 422, 390), radius=10, outline="#a8b0ac", width=2)
    y = 96
    for label, field_value in fields:
        draw.rectangle((28, y, 174, y + 73), fill=theme["soft"])
        _write(draw, (44, y + 26), label, fonts["xs"], theme["ink"])
        _write(draw, (198, y + 24), field_value, fonts["body"], theme["ink"])
        y += 73
        draw.line((28, y, 422, y), fill="#a8b0ac", width=1)
    _write(draw, (28, 416), "Internal request form", fonts["xs"], theme["muted"])
    return image


def _render_dashboard(scene: dict[str, Any], value: int,
                      fonts: dict[str, ImageFont.FreeTypeFont]) -> Image.Image:
    image, draw, theme = _base_image(scene)
    if scene["ood"]:
        draw.rectangle((0, 0, 116, HEIGHT), fill=theme["ink"])
        _write(draw, (18, 28), "PULSE", fonts["strong"], "#ffffff")
        for y, label in ((102, "Queue"), (148, "Agents"), (194, "Reports")):
            _write(draw, (18, y), label, fonts["sm"], "#dce6e1")
        _write(draw, (142, 31), scene["queue"], fonts["title"], theme["ink"])
        _write(draw, (142, 72), "LIVE QUEUE", fonts["xs"], theme["muted"])
        draw.rounded_rectangle((142, 111, 420, 228), radius=18, fill=theme["soft"])
        _write(draw, (163, 132), "OPEN TICKETS", fonts["xs"], theme["muted"])
        _write(draw, (163, 163), value, fonts["hero"], theme["ink"])
        draw.rounded_rectangle((142, 248, 274, 342), radius=14, outline=theme["accent"], width=2)
        draw.rounded_rectangle((288, 248, 420, 342), radius=14, outline=theme["accent"], width=2)
        _write(draw, (158, 264), "WAITING", fonts["xs"], theme["muted"])
        _write(draw, (158, 293), scene["waiting"], fonts["title"], theme["ink"])
        _write(draw, (304, 264), "RESOLVED", fonts["xs"], theme["muted"])
        _write(draw, (304, 293), scene["resolved"], fonts["title"], theme["ink"])
        if scene["flag"]:
            draw.rounded_rectangle((142, 370, 420, 420), radius=10, fill="#f5d7d0")
            _write(draw, (161, 385), "SERVICE NOTICE ACTIVE", fonts["sm"], "#7f2e22")
        return image

    draw.rectangle((0, 0, WIDTH, 72), fill=theme["ink"])
    _write(draw, (24, 22), "PULSE DESK", fonts["strong"], "#ffffff")
    _write(draw, (426, 25), scene["queue"], fonts["sm"], "#ffffff", "ra")
    cards = (("OPEN TICKETS", value), ("WAITING", scene["waiting"]), ("RESOLVED", scene["resolved"]))
    for idx, (label, number) in enumerate(cards):
        x = 24 + idx * 142
        draw.rounded_rectangle((x, 104, x + 122, 218), radius=14, fill=theme["soft"])
        _write(draw, (x + 14, 123), label, fonts["xs"], theme["muted"])
        _write(draw, (x + 14, 157), number, fonts["hero"], theme["ink"])
    _write(draw, (24, 259), "QUEUE ACTIVITY", fonts["xs"], theme["muted"])
    for idx, label in enumerate(scene["activities"]):
        y = 291 + idx * 43
        draw.rounded_rectangle((24, y, 426, y + 34), radius=7, fill="#ffffff", outline="#d5dcd8")
        _write(draw, (39, y + 8), label, fonts["sm"], theme["ink"])
    if scene["flag"]:
        draw.rounded_rectangle((260, 249, 426, 278), radius=7, fill="#f5d7d0")
        _write(draw, (275, 256), "SERVICE NOTICE", fonts["xs"], "#7f2e22")
    return image


def _triangle(draw: ImageDraw.ImageDraw, center: tuple[int, int], size: int, fill: str) -> None:
    x, y = center
    draw.polygon(((x, y - size), (x - size, y + size), (x + size, y + size)), fill=fill)


def _render_spatial(scene: dict[str, Any], value: int,
                    fonts: dict[str, ImageFont.FreeTypeFont]) -> Image.Image:
    image, draw, theme = _base_image(scene)
    _write(draw, (24, 25), scene["map_name"].upper(), fonts["title"], theme["ink"])
    _write(draw, (24, 63), "OBJECT MAP", fonts["xs"], theme["muted"])
    draw.rounded_rectangle((24, 96, 426, 420), radius=14, fill="#ffffff", outline="#b8c1bc", width=2)
    if scene["ood"]:
        for x in range(74, 426, 50):
            draw.line((x, 96, x, 420), fill="#e0e6e2", width=1)
        for y in range(120, 420, 50):
            draw.line((24, y, 426, y), fill="#e0e6e2", width=1)
        _write(draw, (38, 103), "GRID A", fonts["xs"], theme["muted"])
    for x, y, size in scene["triangle_geometry"][:value]:
        _triangle(draw, (x, y), size, "#c84d42")
    blue_x, blue_y = scene["blue_center"]
    square_x, square_y = scene["square_center"]
    draw.ellipse((blue_x - 20, blue_y - 20, blue_x + 20, blue_y + 20), fill="#3578b8")
    draw.rectangle((square_x - 18, square_y - 18, square_x + 18, square_y + 18), fill="#252a2c")
    _write(draw, (blue_x, blue_y + 34), "B", fonts["xs"], theme["muted"], "ma")
    _write(draw, (square_x, square_y + 34), "S", fonts["xs"], theme["muted"], "ma")
    return image


def _render(scene: dict[str, Any], variant: str,
            fonts: dict[str, ImageFont.FreeTypeFont], destination: Path) -> None:
    value = scene["values"][variant]
    if scene["category"] == "receipt":
        image = _render_receipt(scene, value, fonts)
    elif scene["category"] == "bar_chart":
        image = _render_bar_chart(scene, value, fonts)
    elif scene["category"] == "request_form":
        image = _render_request_form(scene, value, fonts)
    elif scene["category"] == "dashboard":
        image = _render_dashboard(scene, value, fonts)
    else:
        image = _render_spatial(scene, value, fonts)
    if scene["required_field_visible"]:
        evidence = {
            "receipt": "Cost-center approval code: CC-47",
            "bar_chart": "Audited denominator badge: 100",
            "request_form": "Supervisor clearance code: SV-19",
            "dashboard": "Customer-impact verification: VERIFIED",
            "spatial": "Ceiling height: 3 m",
        }[scene["category"]]
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 430, WIDTH, HEIGHT), fill="#ffffff")
        _write(draw, (12, 432), evidence, fonts["xs"], "#17212b")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=False, compress_level=6)


def _provenance(scene: dict[str, Any], variant: str) -> dict[str, Any]:
    return {
        "kind": "procedurally_generated",
        "license": "Project-created synthetic data; rendered font SIL-OFL-1.1",
        "license_url": FONT_LICENSE_URL,
        "attribution": "Local procedural scene generator; Atkinson Hyperlegible font by Braille Institute of America",
        "renderer": "Pillow",
        "scene_category": scene["category"],
        "scene_variant": variant,
        "layout_family": scene["layout_family"],
        "ood_basis": (
            "held-out layout family" if scene["ood"]
            else "in-distribution layout family"
        ),
        "source_metadata": "sources/visuals.json",
        "generated_data_terms": "Project-created deterministic synthetic scene and annotations; no external scene assets.",
        "generated_data_attribution": "multimodal-decision data preparation project",
        "font": "Atkinson Hyperlegible Regular",
        "font_revision": FONT_REVISION,
        "font_license": "SIL Open Font License 1.1",
        "font_license_url": FONT_LICENSE_URL,
        "font_attribution": "Copyright 2020 Braille Institute of America, Inc.",
    }


def _metric_display(category: str, value: int) -> str:
    return f"${value:.2f}" if category == "receipt" else str(value)


METRIC_SUPPORT = {
    "receipt": range(38, 93),
    "bar_chart": range(24, 79),
    "request_form": range(24, 87),
    "dashboard": range(3, 17),
    "spatial": range(2, 9),
}
LOOKUP_SUPPORT = {
    "receipt": range(38, 59),
    "bar_chart": range(24, 40),
    "spatial": range(2, 5),
}


def _numeric_candidates(category: str, value: int, rng: Any) -> dict[str, str]:
    support = LOOKUP_SUPPORT[category]
    if value not in support:
        raise ValueError("Numeric lookup value outside rendered lookup support")
    alternatives = [candidate for candidate in support if candidate != value]
    values = [value, *rng.sample(alternatives, min(3, len(alternatives)))]
    return {f"value_{candidate}": _metric_display(category, candidate) for candidate in sorted(values)}


def _score_bounds(category: str, rng: Any) -> tuple[int, int, int]:
    support = METRIC_SUPPORT[category]
    return tuple(sorted(rng.sample(range(support.start, support.stop + 1), 3)))


def _image_rows(split: str, group_id: str, scene: dict[str, Any],
                image_paths: dict[str, str]) -> list[dict[str, Any]]:
    category = scene["category"]
    low = scene["values"]["a"]
    high = scene["values"]["b"]
    common_state = "Use only the attached image."
    rows: list[dict[str, Any]] = []

    if category in LOOKUP_SUPPORT:
        criteria = _numeric_candidates(category, low, rng_for(group_id, "numeric-candidates"))
        prompt = {
            "receipt": "Which amount is printed next to TOTAL?",
            "bar_chart": "What count is printed for Nimbus?",
            "spatial": "How many red triangles are visible?",
        }[category]
        label = f"value_{low}"
    elif category == "request_form":
        criteria = {"facilities": "Facilities", "research": "Research", "operations": "Operations"}
        prompt = "Which department is shown on the request?"
        label = scene["department"].lower()
    elif category == "dashboard":
        criteria = {"atlas": "Atlas Support", "beacon": "Beacon Ops", "cedar": "Cedar Service", "other": "Delta Desk"}
        prompt = "Which queue name is displayed?"
        label = {"Atlas Support": "atlas", "Beacon Ops": "beacon", "Cedar Service": "cedar"}[scene["queue"]]
    else:
        raise ValueError("Unknown visual category")
    rows.append(record(
        split=split, source=SOURCE, group_id=group_id,
        family=f"image-{category}-lookup", modality="image", state=common_state,
        question=choice(prompt, criteria), label=label, images=[image_paths["a"]],
        provenance=_provenance(scene, "a"), variant="image-lookup-a",
        audit={"operation": "read_visible_field", "facts": {"variant": "a", "value": low},
               "field": METRIC_NAMES[category] if category not in ("request_form", "dashboard") else prompt,
               "result": criteria[label]},
    ))

    if category == "receipt":
        yes = scene["flag"]
        instructions = "Does the receipt visibly include a MEMBER SAVINGS line?"
        fact = {"member_savings_visible": yes}
    elif category == "bar_chart":
        values = _bar_values(scene, high)
        yes = values[2] > values[1]
        instructions = "Is the Nimbus bar larger than the Cedar bar?"
        fact = {"nimbus": values[2], "cedar": values[1]}
    elif category == "request_form":
        yes = scene["flag"]
        instructions = "Does REQUESTER SIGNED show Yes?"
        fact = {"requester_signed": "Yes" if yes else "No"}
    elif category == "dashboard":
        yes = scene["flag"]
        instructions = "Is a SERVICE NOTICE visible?"
        fact = {"service_notice_visible": yes}
    else:
        yes = scene["relation"] == "left"
        instructions = "Is the blue circle to the left of the black square?"
        fact = {"blue_circle_relation_to_black_square": scene["relation"]}
    rows.append(record(
        split=split, source=SOURCE, group_id=group_id,
        family=f"image-{category}-visible-relation", modality="image", state=common_state,
        question=noul(instructions), label=str(yes).lower(), images=[image_paths["b"]],
        provenance=_provenance(scene, "b"), variant="image-relation-b",
        audit={"operation": "evaluate_visible_predicate", "facts": fact,
               "rule": instructions, "result": yes},
    ))

    score_value = low if scene["index"] % 2 == 0 else high
    score_variant = "a" if scene["index"] % 2 == 0 else "b"
    t1, t2, t3 = _score_bounds(category, rng_for(group_id, "image-rubric"))
    score_target = sum(score_value >= threshold for threshold in (t1, t2, t3))
    score_instruction = (
        f"Score the visible {METRIC_NAMES[category]}: 0 if below {t1}; "
        f"1 if {t1}–{t2 - 1}; 2 if {t2}–{t3 - 1}; 3 if {t3} or above."
    )
    rows.append(record(
        split=split, source=SOURCE, group_id=group_id,
        family=f"image-{category}-rubric", modality="image", state=common_state,
        question=score(score_instruction, ["Tier 0", "Tier 1", "Tier 2", "Tier 3"]),
        label=str(score_target), images=[image_paths[score_variant]],
        provenance=_provenance(scene, score_variant), variant=f"image-score-{score_variant}",
        audit={"operation": "bucket", "facts": {"value": score_value},
               "rule": {"thresholds": [t1, t2, t3], "lower_inclusive": True},
               "result": score_target},
    ))
    return rows


def _threshold_policy(category: str, threshold: int) -> str:
    metric = METRIC_NAMES[category]
    if DIRECTIONS[category] == "maximum":
        return f"POLICY: Mark eligible only when the visible {metric} is at most {threshold}."
    return f"POLICY: Mark eligible only when the visible {metric} is at least {threshold}."


def _eligible(category: str, value: int, threshold: int) -> bool:
    if DIRECTIONS[category] == "maximum":
        return value <= threshold
    return value >= threshold


def _lookup_policy(category: str, rng: Any) -> tuple[str, dict[str, str], dict[str, Any]]:
    actions = ACTION_SETS[category]
    support = METRIC_SUPPORT[category]
    cutpoints = sorted(rng.sample(range(support.start, support.stop - 1), len(actions) - 1))
    ordered = rng.sample(actions, len(actions))
    criteria = {f"action_{position}": action for position, action in enumerate(actions)}
    bands: list[str] = []
    for position, action in enumerate(ordered):
        if position == 0:
            condition = f"value ≤ {cutpoints[0]}"
        elif position == len(ordered) - 1:
            condition = f"value > {cutpoints[-1]}"
        else:
            condition = f"{cutpoints[position - 1]} < value ≤ {cutpoints[position]}"
        bands.append(f"{condition}: {action}")
    state = f"ACTION TABLE for visible {METRIC_NAMES[category]} — " + "; ".join(bands) + "."
    return state, criteria, {
        "cutpoints": cutpoints,
        "band_actions": list(ordered),
    }


def _mixed_rows(split: str, group_id: str, scene: dict[str, Any],
                 image_paths: dict[str, str]) -> list[dict[str, Any]]:
    category = scene["category"]
    low = scene["values"]["a"]
    high = scene["values"]["b"]
    maximum = DIRECTIONS[category] == "maximum"
    strict = low - 1 if maximum else low + 1
    lenient = low + 1 if maximum else max(0, low - 1)
    midpoint = (low + high) // 2
    rows: list[dict[str, Any]] = []

    for policy_name, threshold in (("strict", strict), ("lenient", lenient)):
        result = _eligible(category, low, threshold)
        rows.append(record(
            split=split, source=SOURCE, group_id=group_id,
            family=f"mixed-{category}-policy-counterfactual", modality="mixed",
            state=_threshold_policy(category, threshold),
            question=noul("Under the displayed policy, is this case eligible?"),
            label=str(result).lower(), images=[image_paths["a"]],
            provenance=_provenance(scene, "a"), variant=f"mixed-policy-{policy_name}-a",
            audit={"operation": "threshold_compare", "facts": {"value": low},
                   "rule": {"direction": DIRECTIONS[category], "threshold": threshold},
                   "result": result, "pair": "same-image-changed-policy",
                   "changed_component": "policy_only"},
        ))

    for variant, value in (("a", low), ("b", high)):
        result = _eligible(category, value, midpoint)
        rows.append(record(
            split=split, source=SOURCE, group_id=group_id,
            family=f"mixed-{category}-evidence-counterfactual", modality="mixed",
            state=_threshold_policy(category, midpoint),
            question=noul("Under the displayed policy, is this case eligible?"),
            label=str(result).lower(), images=[image_paths[variant]],
            provenance=_provenance(scene, variant), variant=f"mixed-evidence-{variant}",
            audit={"operation": "threshold_compare", "facts": {"value": value},
                   "rule": {"direction": DIRECTIONS[category], "threshold": midpoint},
                   "result": result, "pair": "same-policy-changed-image-evidence",
                   "changed_component": "visual_evidence_only"},
        ))

    lookup_variant = "a" if (scene["index"] // len(CATEGORIES)) % 2 == 0 else "b"
    lookup_value = scene["values"][lookup_variant]
    lookup_state, lookup_criteria, lookup_audit = _lookup_policy(
        category, rng_for(group_id, "action-table")
    )
    matched_band = sum(lookup_value > cutpoint for cutpoint in lookup_audit["cutpoints"])
    target_action = lookup_audit["band_actions"][matched_band]
    lookup_label = next(key for key, action in lookup_criteria.items() if action == target_action)
    lookup_audit.update(matched_band=matched_band, chosen_action=target_action)
    rows.append(record(
        split=split, source=SOURCE, group_id=group_id,
        family=f"mixed-{category}-action-table", modality="mixed", state=lookup_state,
        question=choice("Which action does the displayed table assign to this image?", lookup_criteria),
        label=lookup_label, images=[image_paths[lookup_variant]],
        provenance=_provenance(scene, lookup_variant), variant=f"mixed-action-table-{lookup_variant}",
        audit={"operation": "interval_lookup", "facts": {"value": lookup_value},
               "rule": lookup_audit, "result": lookup_criteria[lookup_label]},
    ))

    t1, t2, t3 = _score_bounds(category, rng_for(group_id, "mixed-rubric"))
    score_target = sum(high >= threshold for threshold in (t1, t2, t3))
    rubric_state = (
        f"RUBRIC for visible {METRIC_NAMES[category]}: Tier 0 below {t1}; "
        f"Tier 1 from {t1} through {t2 - 1}; Tier 2 from {t2} through {t3 - 1}; "
        f"Tier 3 at {t3} or above."
    )
    rows.append(record(
        split=split, source=SOURCE, group_id=group_id,
        family=f"mixed-{category}-policy-rubric", modality="mixed", state=rubric_state,
        question=score("Apply the displayed rubric to the image.",
                       ["Tier 0", "Tier 1", "Tier 2", "Tier 3"]),
        label=str(score_target), images=[image_paths["b"]],
        provenance=_provenance(scene, "b"), variant="mixed-rubric-b",
        audit={"operation": "bucket", "facts": {"value": high},
               "rule": {"thresholds": [t1, t2, t3], "lower_inclusive": True},
               "result": score_target},
    ))

    missing = MISSING_FIELDS[category]
    limit_relation = "at most" if maximum else "at least"
    missing_state = (
        f"AUTHORIZATION RULE: Approve only if the visible {METRIC_NAMES[category]} is {limit_relation} "
        f"{midpoint} and the image also visibly supplies {missing}. Reject only when all required "
        "evidence is visible and the rule fails. Otherwise choose Insufficient evidence; never infer an absent field."
    )
    field_visible = scene["required_field_visible"]
    authorization = ("approve" if _eligible(category, low, midpoint) else "reject") if field_visible else "insufficient"
    rows.append(record(
        split=split, source=SOURCE, group_id=group_id,
        family=f"mixed-{category}-authorization-evidence", modality="mixed", state=missing_state,
        question=choice("What decision is justified by the image and authorization rule?", {
            "approve": "Approve", "reject": "Reject", "insufficient": "Insufficient evidence"
        }),
        label=authorization, images=[image_paths["a"]],
        provenance=_provenance(scene, "a"), variant="mixed-required-field-a",
        audit={"operation": "required_evidence_gate",
               "facts": {"metric_value": low, "required_field_visible": field_visible,
                         "required_field": missing},
               "rule": {"metric_limit": midpoint, "direction": DIRECTIONS[category], "requires_all_fields": True},
               "result": authorization},
    ))
    return rows


def prepare(root: Path) -> dict[str, Any]:
    root = root.resolve()
    font_path, license_path = _font_files(root)
    fonts = _fonts(font_path)
    media_dir = root / "data/media/visuals"
    media_dir.mkdir(parents=True, exist_ok=True)
    for stale_asset in media_dir.glob("*.png"):
        stale_asset.unlink()
    rows: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []
    group_counts: dict[str, int] = {}
    seen_facts: set[str] = set()
    seen_hashes: set[str] = set()

    for split in SPLITS:
        image_quota = PART_QUOTAS["visual_image"][split]
        mixed_quota = PART_QUOTAS["visual_mixed"][split]
        if image_quota % 3 or mixed_quota % 7 or image_quota // 3 != mixed_quota // 7:
            raise RuntimeError(f"Visual quotas cannot form complete scene bundles for {split}")
        group_count = image_quota // 3
        group_counts[split] = group_count
        for index in range(group_count):
            for attempt in range(10_000):
                scene = _make_scene(split, index, attempt)
                visible = {key: value for key, value in scene.items() if key not in {"index", "layout_family"}}
                signature = stable_id("visible-scene", visible)
                if signature in seen_facts:
                    continue
                group_id = stable_id(SOURCE, visible)
                image_paths = {}
                proposed = []
                for variant in ("a", "b"):
                    asset_id = stable_id("visual-asset", group_id, variant)
                    relative_path = f"media/visuals/{asset_id}.png"
                    destination = root / "data" / relative_path
                    _render(scene, variant, fonts, destination)
                    image_paths[variant] = relative_path
                    proposed.append({
                        "path": relative_path, "sha256": sha256_file(destination),
                        "group_id": group_id, "scene_variant": variant,
                        "scene_category": scene["category"], "layout_family": scene["layout_family"],
                        "scene_facts": json.loads(json.dumps(visible)),
                        "width": WIDTH, "height": HEIGHT,
                    })
                hashes = {asset["sha256"] for asset in proposed}
                if len(hashes) != 2 or hashes & seen_hashes:
                    for asset in proposed:
                        (root / "data" / asset["path"]).unlink()
                    continue
                seen_facts.add(signature)
                seen_hashes.update(hashes)
                assets.extend(proposed)
                break
            else:
                raise RuntimeError(f"Visible scene allocation exhausted: {split}/{index}")
            rows.extend(_image_rows(split, group_id, scene, image_paths))
            rows.extend(_mixed_rows(split, group_id, scene, image_paths))

    quota_counts = Counter((row["split"], row["modality"]) for row in rows)
    for split in SPLITS:
        expected_image = PART_QUOTAS["visual_image"][split]
        expected_mixed = PART_QUOTAS["visual_mixed"][split]
        if quota_counts[(split, "image")] != expected_image:
            raise RuntimeError(f"Image quota mismatch for {split}")
        if quota_counts[(split, "mixed")] != expected_mixed:
            raise RuntimeError(f"Mixed quota mismatch for {split}")

    primitive_counts = Counter(row["question"]["type"] for row in rows)
    modality_counts = Counter(row["modality"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    part_path = root / "data/parts/visuals.jsonl"
    row_count = write_jsonl(part_path, rows)
    source_metadata = {
        "source": SOURCE,
        "generator_version": GENERATOR_VERSION,
        "description": "Deterministic code-labelled synthetic decision scenes rendered with Pillow.",
        "redistribution": "Generated PNGs are project-created; the bundled font remains under SIL OFL 1.1.",
        "font": {
            "name": "Atkinson Hyperlegible Regular",
            "attribution": "Copyright 2020 Braille Institute of America, Inc.",
            "revision": FONT_REVISION,
            "source_url": FONT_URL,
            "sha256": sha256_file(font_path),
            "license": "SIL Open Font License 1.1",
            "license_url": FONT_LICENSE_URL,
            "license_path": str(license_path.relative_to(root / "data")),
            "license_sha256": sha256_file(license_path),
        },
        "rendering": {
            "library": "Pillow",
            "canvas": [WIDTH, HEIGHT],
            "color_mode": "RGB",
            "format": "PNG",
            "deterministic_seed": "mdprep.common.rng_for over source, split, group index, and category",
            "categories": list(CATEGORIES),
            "ood_layouts": OOD_LAYOUTS,
        },
        "asset_count": len(assets),
        "assets": assets,
    }
    write_json(root / "data/sources/visuals.json", source_metadata)
    return {
        "part": "visuals",
        "path": str(part_path),
        "rows": row_count,
        "assets": len(assets),
        "groups": sum(group_counts.values()),
        "groups_by_split": group_counts,
        "rows_by_split": dict(split_counts),
        "rows_by_modality": dict(modality_counts),
        "primitives": dict(primitive_counts),
        "source_metadata": str(root / "data/sources/visuals.json"),
        "known_limitations": [
            "Scenes are synthetic and use Latin-script English labels.",
            "OOD evaluates held-out generated layouts, not task-family or photographic domain shift.",
            "PNG byte identity assumes the same supported Pillow major version; scene facts and labels are version-stable.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Project root")
    args = parser.parse_args()
    print(prepare(args.root))


if __name__ == "__main__":
    main()
