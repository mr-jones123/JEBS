"""Pixel fixtures are drawn independently of the corpus generator and row labels."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from mdprep.common import choice, record
from mdprep.verify_glyphs import GlyphField, recognize
from mdprep.verify_visuals import extract_visual, visual_answer


FONT = (Path(__file__).resolve().parents[1] /
        "data/raw/visuals/atkinson-hyperlegible/AtkinsonHyperlegible-Regular.ttf")
THEMES = (
    ("#fffdf8", "#17212b", "#19647e", "#dceff4"),
    ("#fffaf3", "#261f2d", "#8a4f2d", "#f3dfcf"),
    ("#f8fff9", "#132a20", "#24734d", "#dcefe3"),
)
STRIPS = {
    "receipt": "Cost-center approval code: CC-47",
    "bar_chart": "Audited denominator badge: 100",
    "request_form": "Supervisor clearance code: SV-19",
    "dashboard": "Customer-impact verification: VERIFIED",
    "spatial": "Ceiling height: 3 m",
}


def write_text(draw, xy, text, size, color, anchor=None):
    draw.text(xy, text, font=ImageFont.truetype(str(FONT), size), fill=color, anchor=anchor)


def fixture(category, ood=False, theme=0, value=70, flag=True, strip=None,
            department="Research", queue="Beacon Ops", bar_text=None):
    paper, ink, accent, soft = THEMES[theme]
    image = Image.new("RGB", (450, 450), paper)
    draw = ImageDraw.Draw(image)
    if category == "receipt":
        if ood:
            draw.rounded_rectangle((22,20,428,430), 26, fill="#f4eee4", outline=ink, width=2)
        else:
            draw.rectangle((62,18,388,432), fill="#fffefa", outline="#c9c4b8", width=2)
        write_text(draw, (389,397) if ood else (366,366), f"${value:.2f}", 23, ink, "ra")
        if flag:
            write_text(draw, (58,354) if ood else (84,326), "MEMBER SAVINGS", 14, accent)
            write_text(draw, (389,354) if ood else (366,326), "-$5.00", 14, accent, "ra")
    elif category == "request_form":
        if ood:
            for y in (102,178,254,330):
                draw.rounded_rectangle((28,y,422,y+66), 12, fill=soft)
            write_text(draw, (47,264), "REQUESTED UNITS", 14, "#65727e")
        write_text(draw, (47,287) if ood else (198,266), str(value), 20, ink)
        write_text(draw, (47,211) if ood else (198,193), department, 20, ink)
        write_text(draw, (47,363) if ood else (198,339), "Yes" if flag else "No", 20, ink)
    elif category == "dashboard":
        if ood:
            draw.rectangle((0,0,116,450), fill=ink)
            write_text(draw, (142,31), queue, 30, ink)
            write_text(draw, (142,72), "LIVE QUEUE", 14, "#65727e")
            draw.rounded_rectangle((142,111,420,228), 18, fill=soft)
        else:
            draw.rectangle((0,0,450,72), fill=ink)
            write_text(draw, (426,25), queue, 17, "white", "ra")
            draw.rounded_rectangle((24,104,146,218), 14, fill=soft)
        write_text(draw, (163,163) if ood else (38,157), str(value), 42, ink)
        if flag:
            draw.rounded_rectangle((142,370,420,420) if ood else (260,249,426,278), 7, fill="#f5d7d0")
            write_text(draw, (161,385) if ood else (275,256),
                       "SERVICE NOTICE ACTIVE" if ood else "SERVICE NOTICE", 17 if ood else 14, "#7f2e22")
    elif category == "bar_chart":
        values = (24,64,value,42)
        if not ood:
            for tick in (0,25,50,75,100):
                y = 382-int(277*tick/100)
                draw.line((52,y,415,y), fill="#d6d9d7")
        for index, number in enumerate(values):
            text = str(bar_text if index == 2 and bar_text is not None else number)
            if ood:
                y = 116+67*index
                end = 112+int(298*number/100)
                draw.rounded_rectangle((112,y,410,y+34), 7, fill=soft)
                draw.rounded_rectangle((112,y,end,y+34), 7, fill=accent)
                write_text(draw, (end+8,y+8), text, 17, ink)
            else:
                x = 75+83*index
                top = 382-int(277*number/100)
                draw.rounded_rectangle((x,top,x+54,382), 6, fill=accent)
                write_text(draw, (x+27,top-8), text, 17, ink, "mb")
    elif category == "spatial":
        draw.polygon(((58,116),(40,152),(76,152)), fill="#c84d42")
        draw.polygon(((138,180),(120,216),(156,216)), fill="#c84d42")
        draw.ellipse((50,330,90,370), fill="#3578b8")
        draw.rectangle((200,330,236,366), fill="#252a2c")
    else:
        raise ValueError(category)
    if strip is not None:
        draw.rectangle((0,430,450,450), fill="white")
        write_text(draw, (12,432), strip, 14, "#17212b")
    return image


class GlyphDictionaryTests(unittest.TestCase):
    def test_all_numeric_values_and_money_use_the_full_independent_dictionary(self):
        for money in (False, True):
            strings = tuple(f"${n:.2f}" if money else str(n) for n in range(101))
            field = GlyphField("number", (220,100), 23, (23,33,43), strings, "ra", (90,90,230,140))
            for value in range(101):
                image = Image.new("RGB", (450,450), "white")
                literal = f"${value:.2f}" if money else str(value)
                write_text(ImageDraw.Draw(image), (220,100), literal, 23, "#17212b", "ra")
                with self.subTest(money=money, value=value):
                    self.assertEqual(recognize(image, field, str(FONT)), literal)

    def test_hidden_incomplete_unknown_extra_and_damaged_opaque_glyphs_fail(self):
        field = GlyphField("metric", (80,100), 23, (23,33,43), tuple(str(n) for n in range(101)),
                           box=(75,95,180,140))
        good = Image.new("RGB", (450,450), "white")
        write_text(ImageDraw.Draw(good), (80,100), "70", 23, "#17212b")
        self.assertEqual(recognize(good, field, str(FONT)), "70")
        for text in ("", "7?", "unknown", "70 extra"):
            changed = Image.new("RGB", (450,450), "white")
            write_text(ImageDraw.Draw(changed), (80,100), text, 23, "#17212b")
            with self.subTest(text=text), self.assertRaises(ValueError):
                recognize(changed, field, str(FONT))
        damaged = good.copy()
        point = next((x,y) for y in range(95,140) for x in range(75,180)
                     if damaged.getpixel((x,y)) == (23,33,43))
        damaged.putpixel(point, (255,255,255))
        with self.assertRaises(ValueError):
            recognize(damaged, field, str(FONT))
        incomplete = good.copy()
        ImageDraw.Draw(incomplete).rectangle((75,115,180,140), fill="white")
        with self.assertRaises(ValueError):
            recognize(incomplete, field, str(FONT))

    def test_pixel_ambiguous_dictionary_and_clipped_field_fail(self):
        image = Image.new("RGB", (450,450), "white")
        write_text(ImageDraw.Draw(image), (80,100), "A", 23, "#17212b")
        with self.assertRaisesRegex(ValueError, "2 complete matches"):
            recognize(image, GlyphField("ambiguous", (80,100), 23, (23,33,43), ("A", "A ")), str(FONT))
        with self.assertRaisesRegex(ValueError, "clipped"):
            recognize(image, GlyphField("clipped", (80,100), 23, (23,33,43), ("A",),
                                        box=(80,100,82,102)), str(FONT))


class ControlledPixelExtractionTests(unittest.TestCase):
    def extract(self, image, category, ood=False, evidence=None, font=FONT):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"actual.png"
            image.save(path)
            return extract_visual(path, category, ood, evidence, font_path=font)

    def test_all_categories_layouts_palettes_and_optional_presence(self):
        for category in STRIPS:
            for ood in (False,True):
                for theme in range(3):
                    for flag in (False,True):
                        value = 2 if category == "spatial" else 12 if category == "dashboard" else 70
                        image = fixture(category, ood, theme, value, flag, STRIPS[category] if flag else None)
                        with self.subTest(category=category, ood=ood, theme=theme, flag=flag):
                            visible = self.extract(image, category, ood)
                            self.assertEqual(visible["value"], value)
                            self.assertEqual(visible["required_field_visible"], flag)
                            self.assertEqual(visible["lookup"], "$70.00" if category == "receipt" else
                                             "Research" if category == "request_form" else
                                             "Beacon Ops" if category == "dashboard" else str(value))
                            self.assertEqual(visible["predicate"], True if category in {"bar_chart","spatial"} else flag)
                            if category == "bar_chart":
                                self.assertEqual(visible["bar_values"], [24,64,70,42])

    def test_complete_named_dictionaries_are_admitted(self):
        for ood in (False,True):
            for department in ("Facilities","Research","Operations"):
                visible = self.extract(fixture("request_form", ood, department=department), "request_form", ood)
                self.assertEqual(visible["lookup"], department)
            for queue in ("Atlas Support","Beacon Ops","Cedar Service"):
                visible = self.extract(fixture("dashboard", ood, value=12, queue=queue), "dashboard", ood)
                self.assertEqual(visible["lookup"], queue)

    def test_changed_numeric_pixels_change_the_observed_value_without_row_facts(self):
        for category in ("receipt","request_form","dashboard"):
            for ood in (False,True):
                for value in (0,16,38,70,92,100):
                    with self.subTest(category=category, ood=ood, value=value):
                        visible = self.extract(fixture(category, ood, value=value), category, ood)
                        self.assertEqual(visible["value"], value)

    def test_bar_text_disagreement_and_incomplete_geometry_fail(self):
        for ood in (False,True):
            with self.subTest(ood=ood), self.assertRaises(ValueError):
                self.extract(fixture("bar_chart", ood, bar_text=69), "bar_chart", ood)
            image = fixture("bar_chart", ood)
            point = (200,267) if ood else (268,300)
            image.putpixel(point, (255,253,248))
            with self.subTest(ood=ood), self.assertRaisesRegex(ValueError, "incomplete"):
                self.extract(image, "bar_chart", ood)

    def test_incomplete_unknown_and_extra_required_strips_fail_for_every_category(self):
        for category, strip in STRIPS.items():
            for ood in (False,True):
                for changed in (strip[:-1], strip+" ?", "Unknown evidence"):
                    with self.subTest(category=category, ood=ood, strip=changed), self.assertRaises(ValueError):
                        self.extract(fixture(category, ood, strip=changed), category, ood)

    def test_incomplete_optional_lines_and_named_fields_fail(self):
        for ood in (False,True):
            image = fixture("receipt", ood)
            ImageDraw.Draw(image).rectangle((280,352,400,375) if ood else (280,324,374,347), fill="white")
            with self.assertRaises(ValueError):
                self.extract(image, "receipt", ood)
            image = fixture("dashboard", ood, value=12)
            ImageDraw.Draw(image).rectangle((157,397,416,412) if ood else (271,265,424,276), fill="white")
            with self.assertRaises(ValueError):
                self.extract(image, "dashboard", ood)
            with self.assertRaises(ValueError):
                self.extract(fixture("request_form", ood, department="Research?"), "request_form", ood)
            with self.assertRaises(ValueError):
                self.extract(fixture("dashboard", ood, queue="Unknown Queue"), "dashboard", ood)

    def test_spatial_without_footer_never_requires_a_font(self):
        visible = self.extract(fixture("spatial"), "spatial", font="/nonexistent/font.ttf")
        self.assertEqual(visible, {"scene_category":"spatial", "required_field_visible":False,
                                   "value":2, "lookup":"2", "predicate":True})

    def test_pixel_changes_drive_numeric_lookup_and_required_evidence_decisions(self):
        lookup = record(
            split="train", source="synthetic-visuals-v1", group_id="literal-receipt",
            family="image-receipt-lookup", modality="image", state="Use only the attached image.",
            question=choice("Which amount is printed next to TOTAL?", {
                "value_38":"$38.00", "value_39":"$39.00", "value_40":"$40.00", "value_41":"$41.00"}),
            label="value_38", images=["media/receipt.png"],
            provenance={"scene_category":"receipt", "scene_variant":"a",
                        "layout_family":"receipt-standard-0", "ood_basis":"in-distribution layout family"})
        for value, expected in ((38,"value_38"), (39,"value_39")):
            visible = self.extract(fixture("receipt", value=value), "receipt")
            self.assertEqual(visual_answer(lookup, visible), expected)
        authorization = record(
            split="train", source="synthetic-visuals-v1", group_id="literal-authorization",
            family="mixed-receipt-authorization-evidence", modality="mixed",
            state="AUTHORIZATION RULE: Approve only if the visible printed TOTAL in dollars is at most 70 "
                  "and the image also visibly supplies a visible cost-center approval code. Reject only "
                  "when all required evidence is visible and the rule fails. Otherwise choose Insufficient "
                  "evidence; never infer an absent field.",
            question=choice("What decision is justified by the image and authorization rule?", {
                "approve":"Approve", "reject":"Reject", "insufficient":"Insufficient evidence"}),
            label="approve", images=["media/receipt.png"],
            provenance={"scene_category":"receipt", "scene_variant":"a",
                        "layout_family":"receipt-standard-0", "ood_basis":"in-distribution layout family"})
        for value, strip, expected in ((70,None,"insufficient"), (70,STRIPS["receipt"],"approve"),
                                       (71,STRIPS["receipt"],"reject")):
            visible = self.extract(fixture("receipt", value=value, strip=strip), "receipt")
            self.assertEqual(visual_answer(authorization, visible), expected)

    def test_evidence_records_actual_pixels_and_exact_recognition_method(self):
        image = fixture("receipt", value=70, flag=False)
        entries = []
        self.assertEqual(self.extract(image, "receipt", evidence=entries)["value"], 70)
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["field"], "total")
        self.assertEqual(entry["text"], "$70.00")
        self.assertEqual(entry["method"], "exact-opaque-glyph-dictionary-v1")
        self.assertEqual(entry["candidate_count"], 101)
        self.assertEqual(entry["status"], "EXTRACTED")
        self.assertEqual(entry["pixel_sha256"], hashlib.sha256(image.crop(entry["box"]).tobytes()).hexdigest())
