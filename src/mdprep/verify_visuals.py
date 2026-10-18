"""Controlled glyph dictionaries and colored geometry from actual generated PNGs."""

import re

from PIL import Image

from mdprep.verify_glyphs import GlyphField, recognize, resolve_font


def integer(text):
    matches=re.findall(r"(?<!\w)\d+(?:\.\d+)?",text)
    if len(matches)!=1: raise ValueError(f"Ambiguous numeric field: {text!r}")
    number=float(matches[0])
    if number!=int(number): raise ValueError(f"Nonintegral metric field: {text!r}")
    return int(number)


def components(image, predicate, box):
    x0,y0,x1,y1=box
    pixels=image.convert("RGB").load()
    points={(x,y) for y in range(y0,y1) for x in range(x0,x1) if predicate(pixels[x,y])}
    blobs=[]
    while points:
        start=points.pop()
        blob={start}
        queue=[start]
        while queue:
            x,y=queue.pop()
            for p in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if p in points:
                    points.remove(p)
                    blob.add(p)
                    queue.append(p)
        if len(blob)>100: blobs.append(blob)
    return blobs




def has_dark_ink(image, box):
    crop = image.crop(box).convert("RGB")
    pixels = crop.load()
    return any(max(pixels[x,y]) < 170 for y in range(crop.height) for x in range(crop.width))


NUMBERS = tuple(str(value) for value in range(101))
MONEY = tuple(f"${value:.2f}" for value in range(101))
DEPARTMENTS = ("Facilities", "Research", "Operations")
QUEUES = ("Atlas Support", "Beacon Ops", "Cedar Service")
REQUIRED_STRIPS = {
    "receipt": "Cost-center approval code: CC-47",
    "bar_chart": "Audited denominator badge: 100",
    "request_form": "Supervisor clearance code: SV-19",
    "dashboard": "Customer-impact verification: VERIFIED",
    "spatial": "Ceiling height: 3 m",
}
PALETTES = {
    (255, 253, 248): ((23, 33, 43), (25, 100, 126)),
    (255, 250, 243): ((38, 31, 45), (138, 79, 45)),
    (248, 255, 249): ((19, 42, 32), (36, 115, 77)),
}
FIELD_BOXES = {
    "total": ((278,360,378,400), (280,391,403,427)),
    "member_savings": ((82,324,275,347), (48,352,275,375)),
    "savings_amount": ((280,324,374,347), (280,352,400,375)),
    "requested_units": ((190,263,380,304), (40,283,330,319)),
    "department": ((190,185,410,230), (40,205,410,248)),
    "requester_signed": ((190,332,380,377), (40,359,330,396)),
    "open_tickets": ((35,158,143,207), (155,158,270,218)),
    "queue": ((240,18,435,58), (137,30,448,70)),
    "service_notice": ((271,254,424,276), (157,383,416,412)),
    "required_strip": ((0,430,450,450), (0,430,450,450)),
}


def extract_visual(path,category,ood,evidence=None,font_path=None):
    with Image.open(path) as original:
        image=original.convert("RGB")
    if image.size!=(450,450): raise ValueError("Unexpected visual canvas")
    if category not in REQUIRED_STRIPS:
        raise ValueError("Unknown visual category")
    resolved_font = None

    def field(name, xy, size, candidates, color, anchor=None, box=None):
        nonlocal resolved_font
        if resolved_font is None:
            resolved_font = resolve_font(path, font_path)
        if box is None:
            box = FIELD_BOXES[name][bool(ood)]
        return recognize(image, GlyphField(name, xy, size, color, candidates, anchor, box),
                         resolved_font, evidence)

    footer_left = 120 if category == "dashboard" and ood else 0
    required=has_dark_ink(image,(footer_left,434,450,450))
    if category == "dashboard" and ood:
        sidebar = image.getpixel((0,429))
        required = required or image.crop((0,430,116,450)).getextrema() != tuple((value,value) for value in sidebar)
    if required:
        field("required_strip", (12,432), 14, (REQUIRED_STRIPS[category],), (23,33,43))
    result={"required_field_visible":required, "scene_category":category}
    if category != "spatial":
        palette = PALETTES.get(image.getpixel((449,100)))
        if palette is None:
            raise ValueError("Unknown controlled visual palette")
        ink, accent = palette
    if category=="receipt":
        result["value"]=integer(field("total", (389,397) if ood else (366,366),
                                      23, MONEY, ink, "ra"))
        savings_box=(48,352,400,375) if ood else (82,324,374,347)
        savings=has_dark_ink(image,savings_box)
        if savings:
            field("member_savings", (58,354) if ood else (84,326),
                  14, ("MEMBER SAVINGS",), accent)
            field("savings_amount", (389,354) if ood else (366,326),
                  14, ("-$5.00",), accent, "ra")
        result["predicate"]=savings
        result["lookup"]=f"${result['value']:.2f}"
    elif category=="request_form":
        result["value"]=integer(field("requested_units", (47,287) if ood else (198,266),
                                      20, NUMBERS, ink))
        result["lookup"]=field("department", (47,211) if ood else (198,193),
                               20, DEPARTMENTS, ink)
        signed=field("requester_signed", (47,363) if ood else (198,339),
                     20, ("Yes","No"), ink)
        result["predicate"]=signed=="Yes"
    elif category=="dashboard":
        result["value"]=integer(field("open_tickets", (163,163) if ood else (38,157),
                                      42, NUMBERS, ink))
        result["lookup"]=field("queue", (142,31) if ood else (426,25),
                               30 if ood else 17, QUEUES, ink if ood else (255,255,255),
                               None if ood else "ra")
        notice_box=(157,383,416,412) if ood else (271,254,424,276)
        notice=has_dark_ink(image,notice_box)
        if notice:
            field("service_notice", (161,385) if ood else (275,256), 17 if ood else 14,
                  ("SERVICE NOTICE ACTIVE",) if ood else ("SERVICE NOTICE",), (127,46,34))
        result["predicate"]=notice
    elif category=="bar_chart":
        values=[]
        for position in range(4):
            if ood:
                y=116+67*position
                row=[px for px in range(112,411) if image.getpixel((px,y+17))==accent]
                if not row or row != list(range(112,max(row)+1)):
                    raise ValueError("Missing or incomplete horizontal bar")
                end=max(row)
                value=integer(field(f"bar_{position}", (min(end+8,423),y+8), 17, NUMBERS, ink,
                                    box=(end+4,y+2,min(450,end+65),y+34)))
                if end != 112+int(298*value/100): raise ValueError("Glyph/bar width disagreement")
            else:
                x=75+83*position
                ys=[y for y in range(105,383) if image.getpixel((x+27,y))==accent]
                if not ys or ys != list(range(min(ys),383)):
                    raise ValueError("Missing or incomplete vertical bar")
                top=min(ys)
                value=integer(field(f"bar_{position}", (x+27,top-8), 17, NUMBERS, ink, "mb",
                                    box=(x+2,max(0,top-38),x+53,top-3)))
                if top != 382-int(277*value/100): raise ValueError("Glyph/bar height disagreement")
            values.append(value)
        result.update(value=values[2],lookup=str(values[2]),predicate=values[2]>values[1],bar_values=values)
    elif category=="spatial":
        red=components(image,lambda c:c[0]>150 and 35<c[1]<130 and 25<c[2]<130 and c[0]>c[1]*1.5,(25,100,425,325))
        blue=components(image,lambda c:30<c[0]<100 and 70<c[1]<160 and c[2]>150,(25,315,425,390))
        black=components(image,lambda c:max(c)<65,(25,315,425,390))
        if len(blue)!=1 or len(black)!=1: raise ValueError("Unresolved spatial markers")
        for blob in red:
            xs,ys=zip(*blob)
            if not 20<=max(xs)-min(xs)<=40 or not 20<=max(ys)-min(ys)<=40: raise ValueError("Nontriangle red geometry")
            top=[x for x,y in blob if y==min(ys)]
            bottom=[x for x,y in blob if y==max(ys)]
            if len(top)>3 or len(bottom)<20: raise ValueError("Red component is not triangular")
        bx=sum(x for x,y in blue[0])/len(blue[0])
        sx=sum(x for x,y in black[0])/len(black[0])
        result.update(value=len(red),lookup=str(len(red)),predicate=bx<sx)
    return result


VISUAL_CONTRACTS = {
    "receipt": {
        "metric": "printed TOTAL in dollars",
        "lookup": "Which amount is printed next to TOTAL?",
        "relation": "Does the receipt visibly include a MEMBER SAVINGS line?",
        "actions": ("Decline", "Manual review", "Reimburse", "Priority reimburse"),
        "missing": "a visible cost-center approval code",
        "ood": "receipt-stacked-cards-ood",
    },
    "bar_chart": {
        "metric": "Nimbus count",
        "lookup": "What count is printed for Nimbus?",
        "relation": "Is the Nimbus bar larger than the Cedar bar?",
        "actions": ("Observe", "Notify", "Investigate", "Escalate", "Emergency response"),
        "missing": "a visible audited-denominator badge",
        "ood": "bar-horizontal-ood",
    },
    "request_form": {
        "metric": "requested units",
        "lookup": "Which department is shown on the request?",
        "relation": "Does REQUESTER SIGNED show Yes?",
        "actions": ("Return to requester", "Supervisor review", "Authorize"),
        "missing": "a visible supervisor-clearance code",
        "ood": "form-field-cards-ood",
    },
    "dashboard": {
        "metric": "open-ticket count",
        "lookup": "Which queue name is displayed?",
        "relation": "Is a SERVICE NOTICE visible?",
        "actions": ("Backlog", "Normal queue", "Fast track", "Incident bridge"),
        "missing": "a visible customer-impact verification",
        "ood": "dashboard-sidebar-ood",
    },
    "spatial": {
        "metric": "red-triangle count",
        "lookup": "How many red triangles are visible?",
        "relation": "Is the blue circle to the left of the black square?",
        "actions": ("Accept layout", "Inspect layout", "Revise layout", "Close area"),
        "missing": "a visible ceiling-height measurement",
        "ood": "spatial-coordinate-grid-ood",
    },
}


def _candidate_meanings(row, kind, expected=None):
    question = row["question"]
    criteria = question["criteria"]
    if question["type"] != kind or not isinstance(criteria, dict):
        raise ValueError("Unexpected visual question type")
    if kind != "choice":
        if criteria != expected:
            raise ValueError("Visual candidate meanings disagree")
        return criteria
    keys = row["audit"].get("original_candidate_keys")
    if (not isinstance(keys, list) or len(keys) != len(criteria)
            or not all(isinstance(key, str) for key in keys)
            or len(set(keys)) != len(keys)
            or list(criteria) != [f"option_{i}" for i in range(len(keys))]):
        raise ValueError("Unresolved visual candidate meanings")
    meanings = dict(zip(keys, criteria.values()))
    if len(set(criteria.values())) != len(criteria) or (expected is not None and meanings != expected):
        raise ValueError("Visual candidate meanings disagree")
    return meanings


def _rubric_thresholds(text, metric, mixed):
    number = r"(-?\d+)"
    if mixed:
        pattern = (
            rf"RUBRIC for visible {re.escape(metric)}: Tier 0 below {number}; "
            rf"Tier 1 from {number} through {number}; Tier 2 from {number} through {number}; "
            rf"Tier 3 at {number} or above\."
        )
    else:
        pattern = (
            rf"Score the visible {re.escape(metric)}: 0 if below {number}; "
            rf"1 if {number}–{number}; 2 if {number}–{number}; 3 if {number} or above\."
        )
    match = re.fullmatch(pattern, text)
    if not match:
        raise ValueError("Unresolved visible rubric")
    t1, start1, end1, t2, end2, t3 = map(int, match.groups())
    if not (t1 == start1 and t1 < t2 < t3 and end1 == t2 - 1 and end2 == t3 - 1):
        raise ValueError("Inconsistent visible rubric intervals")
    return t1, t2, t3


def _action_answer(state, metric, actions, value, meanings):
    prefix = f"ACTION TABLE for visible {metric} — "
    if not state.startswith(prefix) or not state.endswith("."):
        raise ValueError("Unresolved visible action table")
    bands = state[len(prefix):-1].split("; ")
    if len(bands) != len(actions):
        raise ValueError("Incomplete visible action table")
    cutpoints = []
    ordered = []
    for index, band in enumerate(bands):
        if index == 0:
            pattern = r"value ≤ (-?\d+): (.+)"
        elif index == len(bands) - 1:
            pattern = r"value > (-?\d+): (.+)"
        else:
            pattern = r"(-?\d+) < value ≤ (-?\d+): (.+)"
        match = re.fullmatch(pattern, band)
        if not match:
            raise ValueError("Unknown visible action operator")
        *numbers, action = match.groups()
        numbers = list(map(int, numbers))
        if index and numbers[0] != cutpoints[-1]:
            raise ValueError("Inconsistent visible action intervals")
        if index < len(bands) - 1:
            cutpoints.append(numbers[-1])
        ordered.append(action)
    if (set(ordered) != set(actions) or len(set(ordered)) != len(actions)
            or any(left >= right for left, right in zip(cutpoints, cutpoints[1:]))):
        raise ValueError("Inconsistent visible action table")
    action = ordered[sum(value > threshold for threshold in cutpoints)]
    return next(key for key, description in meanings.items() if description == action)


def visual_answer(row, visible):
    provenance = row["provenance"]
    category = provenance["scene_category"]
    if category not in VISUAL_CONTRACTS or visible.get("scene_category") != category:
        raise ValueError("Visual scene category disagrees")
    contract = VISUAL_CONTRACTS[category]
    family = provenance["family"]
    mode, separator, task = family.partition(f"-{category}-")
    allowed = {
        "image": {"lookup", "visible-relation", "rubric"},
        "mixed": {"policy-counterfactual", "evidence-counterfactual", "action-table",
                  "policy-rubric", "authorization-evidence"},
    }
    if not separator or mode not in allowed or task not in allowed[mode] or row["modality"] != mode:
        raise ValueError("Unknown visual decision family")
    ood = row["split"] == "test_ood"
    layout = provenance["layout_family"]
    layout_matches = layout == contract["ood"] if ood else bool(re.fullmatch(rf"{category}-standard-[0-2]", layout))
    if not layout_matches:
        raise ValueError("Visual layout declaration disagrees")
    if provenance["ood_basis"] != ("held-out layout family" if ood else "in-distribution layout family"):
        raise ValueError("Visual layout basis disagrees")
    variant = provenance["scene_variant"]
    variants = {"lookup": {"a"}, "visible-relation": {"b"}, "rubric": {"a", "b"},
                "policy-counterfactual": {"a"}, "evidence-counterfactual": {"a", "b"},
                "action-table": {"a", "b"}, "policy-rubric": {"b"}, "authorization-evidence": {"a"}}
    if variant not in variants[task]:
        raise ValueError("Unexpected visual scene variant")
    value = visible["value"]
    if type(value) is not int:
        raise ValueError("Unresolved visible metric")
    state = row["state"]
    instructions = row["question"]["instructions"]
    metric = contract["metric"]
    common_state = "Use only the attached image."
    if mode == "image" and state != common_state:
        raise ValueError("Unexpected visual state")
    if task == "lookup":
        if instructions != contract["lookup"]:
            raise ValueError("Unexpected visual lookup instruction")
        meanings = _candidate_meanings(row, "choice")
        if category in {"receipt", "bar_chart", "spatial"}:
            support = {"receipt": range(38, 59), "bar_chart": range(24, 40), "spatial": range(2, 5)}[category]
            if len(meanings) != min(4, len(support)):
                raise ValueError("Incomplete numeric candidate pool")
            lookup = f"${value:.2f}" if category == "receipt" else str(value)
            if visible["lookup"] != lookup:
                raise ValueError("Visible numeric lookup disagrees with metric")
            for key, description in meanings.items():
                match = re.fullmatch(r"value_(\d+)", key)
                if not match or int(match[1]) not in support:
                    raise ValueError("Unresolved numeric candidate meaning")
                candidate = int(match[1])
                expected = f"${candidate:.2f}" if category == "receipt" else str(candidate)
                if key != f"value_{candidate}" or description != expected:
                    raise ValueError("Numeric candidate meaning disagrees")
        else:
            expected = ({"facilities": "Facilities", "research": "Research", "operations": "Operations"}
                        if category == "request_form" else
                        {"atlas": "Atlas Support", "beacon": "Beacon Ops", "cedar": "Cedar Service", "other": "Delta Desk"})
            if meanings != expected:
                raise ValueError("Visual candidate meanings disagree")
        matches = [key for key, description in meanings.items() if description == visible["lookup"]]
        if len(matches) != 1:
            raise ValueError("Visible answer is missing or ambiguous among candidates")
        return matches[0]
    if task == "visible-relation":
        _candidate_meanings(row, "noul", {"false": "No", "true": "Yes"})
        if instructions != contract["relation"] or type(visible["predicate"]) is not bool:
            raise ValueError("Unexpected visual relation instruction or evidence")
        return str(visible["predicate"]).lower()
    if task in {"rubric", "policy-rubric"}:
        _candidate_meanings(row, "score", {str(index): f"Tier {index}" for index in range(4)})
        if mode == "mixed" and instructions != "Apply the displayed rubric to the image.":
            raise ValueError("Unexpected visual rubric instruction")
        thresholds = _rubric_thresholds(state if mode == "mixed" else instructions, metric, mode == "mixed")
        return str(sum(value >= threshold for threshold in thresholds))
    if task == "action-table":
        meanings = _candidate_meanings(row, "choice", {
            f"action_{index}": action for index, action in enumerate(contract["actions"])
        })
        if instructions != "Which action does the displayed table assign to this image?":
            raise ValueError("Unexpected visual action instruction")
        return _action_answer(state, metric, contract["actions"], value, meanings)
    if task == "authorization-evidence":
        meanings = _candidate_meanings(row, "choice", {
            "approve": "Approve", "reject": "Reject", "insufficient": "Insufficient evidence"
        })
        if instructions != "What decision is justified by the image and authorization rule?":
            raise ValueError("Unexpected visual authorization instruction")
        pattern = (
            rf"AUTHORIZATION RULE: Approve only if the visible {re.escape(metric)} is "
            rf"(at most|at least) (\d+) and the image also visibly supplies {re.escape(contract['missing'])}\. "
            r"Reject only when all required evidence is visible and the rule fails\. "
            r"Otherwise choose Insufficient evidence; never infer an absent field\."
        )
        if type(visible["required_field_visible"]) is not bool:
            raise ValueError("Unresolved required visual evidence")
    else:
        _candidate_meanings(row, "noul", {"false": "No", "true": "Yes"})
        if instructions != "Under the displayed policy, is this case eligible?":
            raise ValueError("Unexpected visual policy instruction")
        pattern = rf"POLICY: Mark eligible only when the visible {re.escape(metric)} is (at most|at least) (\d+)\."
    match = re.fullmatch(pattern, state)
    if not match:
        raise ValueError("Unresolved visible threshold policy")
    relation, limit = match.groups()
    eligible = value <= int(limit) if relation == "at most" else value >= int(limit)
    if task == "authorization-evidence":
        description = "Insufficient evidence" if not visible["required_field_visible"] else "Approve" if eligible else "Reject"
        return next(key for key, meaning in meanings.items() if meaning == description)
    return str(eligible).lower()
