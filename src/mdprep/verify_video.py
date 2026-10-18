"""Temporal answers derived from every decoded frame, without generator predicates."""

import math
import subprocess
from dataclasses import dataclass

from PIL import Image, ImageDraw

COLORS = {"amber": (228,154,49), "blue": (57,120,184), "cyan": (43,158,161), "green": (76,149,108), "magenta": (176,74,130), "red": (201,79,70), "violet": (118,86,165), "yellow": (224,184,61)}

VIDEO_STATE = (
    "Observe the entire four-frame clip in time order. Changes between consecutive "
    "sampled frames are evidence; do not answer from the final frame alone."
)


@dataclass(frozen=True)
class _TemporalTask:
    scene: str
    kind: str
    instructions: str
    candidates: tuple[tuple[str, str], ...]


_YES_NO = (("false", "No"), ("true", "Yes"))

# These admitted meanings are independent of generator templates and answer metadata.
_TASKS = {
    "bounce.motion_pattern": _TemporalTask(
        "bounce", "choice",
        "Which description matches the amber circle's complete horizontal motion?",
        (
            ("right_only", "It moved right throughout."),
            ("left_only", "It moved left throughout."),
            ("right_then_left", "It moved right, then reversed left."),
            ("left_then_right", "It moved left, then reversed right."),
            ("alternating_right_first", "It moved right, then left, then right."),
            ("alternating_left_first", "It moved left, then right, then left."),
        ),
    ),
    "bounce.any_reversal": _TemporalTask(
        "bounce", "noul",
        "Did the amber circle reverse horizontal direction before the clip ended?",
        _YES_NO,
    ),
    "bounce.reversal_count": _TemporalTask(
        "bounce", "score",
        "How many horizontal direction changes occurred across the three intervals?",
        (("0", "No direction changes"), ("1", "One direction change"), ("2", "Two direction changes")),
    ),
    "bounce.endpoint_intervals": _TemporalTask(
        "bounce", "choice",
        "Which directions describe the first interval and the final interval, in that order?",
        (
            ("right_right", "Right, then right"), ("right_left", "Right, then left"),
            ("left_right", "Left, then right"), ("left_left", "Left, then left"),
        ),
    ),
    "color.penultimate": _TemporalTask(
        "color_sequence", "choice",
        "Which color was visible immediately before the final sampled color?",
        (
            ("amber", "Amber"), ("blue", "Blue"), ("cyan", "Cyan"), ("green", "Green"),
            ("magenta", "Magenta"), ("red", "Red"), ("violet", "Violet"), ("yellow", "Yellow"),
        ),
    ),
    "color.return": _TemporalTask(
        "color_sequence", "noul",
        "After changing away from a color, did the circle ever return to that earlier color?",
        _YES_NO,
    ),
    "color.change_count": _TemporalTask(
        "color_sequence", "score",
        "How many color changes occurred between consecutive sampled frames?",
        (("0", "No changes"), ("1", "One change"), ("2", "Two changes"), ("3", "Three changes")),
    ),
    "color.final_first_seen": _TemporalTask(
        "color_sequence", "choice",
        "At which sampled frame did the final color first appear?",
        (
            ("frame_1", "First sampled frame"), ("frame_2", "Second sampled frame"),
            ("frame_3", "Third sampled frame"), ("frame_4", "Fourth sampled frame"),
        ),
    ),
    "gate.first_arrival": _TemporalTask(
        "gate_order", "choice",
        "Which object reached the center guide first?",
        (
            ("amber_first", "The amber circle"), ("violet_first", "The violet square"),
            ("same_time", "They reached it at the same sampled time"),
        ),
    ),
    "gate.amber_before_violet": _TemporalTask(
        "gate_order", "noul",
        "Did the amber circle reach the center guide before the violet square?",
        _YES_NO,
    ),
    "gate.arrival_gap": _TemporalTask(
        "gate_order", "score",
        "How many sampled-frame steps separated the two center-guide arrivals?",
        (("0", "Same sampled frame"), ("1", "One frame step"), ("2", "Two frame steps"), ("3", "Three frame steps")),
    ),
    "gate.second_frame": _TemporalTask(
        "gate_order", "choice",
        "At the second sampled frame, which object was exactly on the center guide?",
        (
            ("amber_only", "Only the amber circle"), ("violet_only", "Only the violet square"),
            ("both", "Both objects"), ("neither", "Neither object"),
        ),
    ),
    "collision.after_first_contact": _TemporalTask(
        "collision", "choice",
        "What happened after the circles first touched or overlapped?",
        (
            ("separated", "They later separated."),
            ("remained_in_contact", "They remained touching or overlapping through the end."),
            ("no_contact", "They never touched or overlapped."),
        ),
    ),
    "collision.before_final": _TemporalTask(
        "collision", "noul",
        "Did the two circles touch or overlap before the final sampled frame?",
        _YES_NO,
    ),
    "collision.contact_count": _TemporalTask(
        "collision", "score",
        "In how many sampled frames were the two circles touching or overlapping?",
        (("0", "Zero frames"), ("1", "One frame"), ("2", "Two frames"), ("3", "Three frames")),
    ),
    "collision.early_distance_trend": _TemporalTask(
        "collision", "choice",
        "Across the first two intervals, how did the distance between the circle centers change?",
        (
            ("closer_then_closer", "Closer, then closer again"),
            ("closer_then_farther", "Closer, then farther"),
            ("farther_then_closer", "Farther, then closer"),
            ("farther_then_farther", "Farther, then farther again"),
            ("includes_no_change", "At least one interval had no distance change"),
        ),
    ),
    "zigzag.guide_order": _TemporalTask(
        "zigzag_guides", "choice",
        "Which colored guide did the triangle intersect first?",
        (
            ("cyan_first", "The cyan vertical guide"), ("magenta_first", "The magenta horizontal guide"),
            ("same_time", "Both guides at the same sampled time"),
        ),
    ),
    "zigzag.vertical_reversal": _TemporalTask(
        "zigzag_guides", "noul",
        "Did the triangle reverse its vertical direction during the clip?",
        _YES_NO,
    ),
    "zigzag.vertical_reversal_count": _TemporalTask(
        "zigzag_guides", "score",
        "How many times did the triangle reverse vertical direction?",
        (("0", "No reversals"), ("1", "One reversal"), ("2", "Two reversals")),
    ),
    "zigzag.endpoint_vertical_motion": _TemporalTask(
        "zigzag_guides", "choice",
        "Which vertical directions describe the first interval and final interval, in that order?",
        (
            ("down_down", "Down, then down"), ("down_up", "Down, then up"),
            ("up_down", "Up, then down"), ("up_up", "Up, then up"),
        ),
    ),
}

_ID_SPLITS = frozenset(("train", "dev", "calibration", "policy", "test_id"))
_SCENE_FIELDS = {
    "bounce": ("focus_center",),
    "color_sequence": ("focus_color",),
    "gate_order": ("amber_circle_center", "violet_square_center"),
    "collision": ("amber_center", "violet_center"),
    "zigzag_guides": ("triangle_center",),
}


def video_answer(row, facts):
    """Return the original response key after binding the visible temporal decision.

    Unsupported meanings, contradictory metadata, and missing observed evidence raise
    ValueError. Targets and declared audit answers are never used.
    """
    provenance = row.get("provenance", {})
    family = provenance.get("family")
    task = _TASKS.get(family)
    if task is None:
        raise ValueError("Unsupported temporal task family")
    if row.get("audit", {}).get("scene_family") != task.scene:
        raise ValueError("Temporal scene/task family mismatch")
    held_out = task.scene == "zigzag_guides"
    if (
        (held_out and row.get("split") != "test_ood")
        or (not held_out and row.get("split") not in _ID_SPLITS)
        or provenance.get("ood_axis") != ("task_family" if held_out else "none")
    ):
        raise ValueError("Temporal task family does not match ID/OOD membership")
    if row.get("state") != VIDEO_STATE:
        raise ValueError("Unsupported temporal state instructions")
    question = row.get("question", {})
    if question.get("type") != task.kind or question.get("instructions") != task.instructions:
        raise ValueError("Unsupported temporal question meaning")
    criteria = question.get("criteria")
    if not isinstance(criteria, dict):
        raise ValueError("Missing temporal candidate meanings")
    if task.kind == "choice":
        original = row.get("audit", {}).get("original_candidate_keys")
        if (
            not isinstance(original, list)
            or len(original) != len(task.candidates)
            or not all(isinstance(key, str) for key in original)
            or len(set(original)) != len(original)
            or list(criteria) != [f"option_{i}" for i in range(len(original))]
            or dict(zip(original, criteria.values())) != dict(task.candidates)
        ):
            raise ValueError("Temporal choice candidate meanings do not match the task")
    elif tuple(criteria.items()) != task.candidates:
        raise ValueError("Temporal ordered candidate meanings do not match the task")
    if not isinstance(facts, (list, tuple)) or len(facts) != 4:
        raise ValueError("Temporal task requires four observed frame facts")
    for fact in facts:
        if not isinstance(fact, dict):
            raise ValueError("Missing temporal frame evidence")
        for field in _SCENE_FIELDS[task.scene]:
            value = fact.get(field)
            if field == "focus_color":
                if not isinstance(value, str) or value not in COLORS:
                    raise ValueError("Missing or unsupported observed color")
            elif (
                not isinstance(value, (list, tuple))
                or len(value) != 2
                or any(type(coordinate) not in (int, float) or not math.isfinite(coordinate) for coordinate in value)
            ):
                raise ValueError("Missing or invalid observed temporal geometry")
    if task.scene == "gate_order" and any(
        not any(fact[field][0] == 128 for fact in facts)
        for field in _SCENE_FIELDS[task.scene]
    ):
        raise ValueError("Missing observed center-guide arrival")
    if task.scene == "zigzag_guides" and any(
        not any(fact["triangle_center"][axis] == 96 for fact in facts)
        for axis in (0, 1)
    ):
        raise ValueError("Missing observed colored-guide intersection")
    answer = temporal_answer(family, facts)
    if answer not in dict(task.candidates):
        raise ValueError("Observed temporal answer is outside the admitted candidates")
    return answer


def color_points(image: Image.Image, color: str, region=None) -> list[tuple[int,int]]:
    rgb = COLORS[color]
    image = image.convert("RGB")
    x0,y0,x1,y1 = region or (0,0,image.width,image.height)
    return [(x,y) for y in range(y0,y1) for x in range(x0,x1) if sum((a-b)**2 for a,b in zip(image.getpixel((x,y)),rgb)) < 900]


def bounds(points):
    if len(points) < 12:
        raise ValueError("Unresolved colored geometry")
    xs,ys=zip(*points)
    return min(xs),min(ys),max(xs),max(ys)


def center(image, color, shape="circle", radius=16, occluder=None):
    observed=set(color_points(image,color))
    x0,y0,x1,y1=bounds(observed)
    guess_x,guess_y=round((x0+x1)/2),round((y0+y1)/2)
    if occluder and x1-x0<2*(radius-4):
        guess_x=x0+radius-3 if guess_x<occluder[0] else x1-radius+3
    local=Image.new("L",(49,49),0)
    draw=ImageDraw.Draw(local)
    if shape=="triangle":
        vertices=((24,7),(8,38),(40,38))
        draw.polygon(vertices,fill=255)
        draw.line((*vertices,vertices[0]),fill=0,width=3)
    elif shape=="square":
        draw.rectangle((24-radius,24-radius,24+radius,24+radius),fill=255,outline=0,width=3)
    elif shape=="circle":
        draw.ellipse((24-radius,24-radius,24+radius,24+radius),fill=255,outline=0,width=3)
    else:
        raise ValueError("Unsupported shape prior")
    template={(x-24,y-24) for y in range(49) for x in range(49) if local.getpixel((x,y))}
    fits=[]
    for cy in range(guess_y-4,guess_y+5):
        for cx in range(guess_x-4,guess_x+5):
            expected={(cx+x,cy+y) for x,y in template}
            if occluder:
                ox,oy=occluder
                expected={p for p in expected if (p[0]-ox)**2+(p[1]-oy)**2>16**2}
            fits.append((len(expected.symmetric_difference(observed)),cx,cy))
    fits.sort()
    if fits[0][0]==fits[1][0] or fits[0][0]>max(30,len(observed)//3):
        raise ValueError("Ambiguous or inconsistent decoded shape geometry")
    return fits[0][1],fits[0][2]


def extract_frames(path, scene_family):
    result=subprocess.run(["ffmpeg","-v","error","-i",str(path),"-f","rawvideo","-pix_fmt","rgb24","-fps_mode","passthrough","pipe:1"],check=True,capture_output=True)
    size=256*192*3
    if len(result.stdout)!=4*size:
        raise ValueError("Expected exactly four fully decoded frames")
    frames=[Image.frombytes("RGB",(256,192),result.stdout[i*size:(i+1)*size]) for i in range(4)]
    facts=[]
    for image in frames:
        if scene_family=="color_sequence":
            sample=image.getpixel((128,96))
            ranked=sorted((sum((a-b)**2 for a,b in zip(sample,rgb)),name) for name,rgb in COLORS.items())
            if ranked[0][0]>1600 or ranked[1][0]-ranked[0][0]<1000:
                raise ValueError("Unresolved video color")
            facts.append({"focus_color":ranked[0][1]})
        elif scene_family=="bounce":
            facts.append({"focus_center":center(image,"amber")})
        elif scene_family=="gate_order":
            facts.append({"amber_circle_center":center(image,"amber",radius=14),"violet_square_center":center(image,"violet",shape="square",radius=14)})
        elif scene_family=="zigzag_guides":
            facts.append({"triangle_center":center(image,"yellow",shape="triangle")})
        elif scene_family=="collision":
            violet=center(image,"violet")
            amber=center(image,"amber",occluder=violet)
            facts.append({"amber_center":amber,"violet_center":violet})
        else:
            raise ValueError("Unknown temporal scene family")
    return facts


def temporal_answer(family, facts):
    task=family.split(".",1)[1]
    if family.startswith("color."):
        colors=[f["focus_color"] for f in facts]
        if task=="penultimate": return colors[-2]
        if task=="change_count": return str(sum(a!=b for a,b in zip(colors,colors[1:])))
        if task=="final_first_seen": return f"frame_{colors.index(colors[-1])+1}"
        if task=="return": return str(any(colors[i]==colors[j] and any(c!=colors[i] for c in colors[i+1:j]) for i in range(4) for j in range(i+2,4))).lower()
    if family.startswith(("bounce.","zigzag.")):
        zig=family.startswith("zigzag.")
        points=[f["triangle_center" if zig else "focus_center"] for f in facts]
        values=[p[1 if zig else 0] for p in points]
        signs=[(b>a)-(b<a) for a,b in zip(values,values[1:])]
        if 0 in signs: raise ValueError("Unresolved motion interval")
        changes=sum(a!=b for a,b in zip(signs,signs[1:]))
        names=("down","up") if zig else ("right","left")
        first,last=(names[0] if s>0 else names[1] for s in (signs[0],signs[-1]))
        if task in {"any_reversal","vertical_reversal"}: return str(changes>0).lower()
        if task in {"reversal_count","vertical_reversal_count"}: return str(changes)
        if task in {"endpoint_intervals","endpoint_vertical_motion"}: return f"{first}_{last}"
        if task=="motion_pattern":
            return f"{first}_only" if changes==0 else f"{first}_then_{last}" if changes==1 else f"alternating_{first}_first"
        if task=="guide_order":
            a=next(i for i,p in enumerate(points) if p[0]==96)
            b=next(i for i,p in enumerate(points) if p[1]==96)
            return "cyan_first" if a<b else "magenta_first" if b<a else "same_time"
    if family.startswith("gate."):
        amber=[f["amber_circle_center"][0]==128 for f in facts]
        violet=[f["violet_square_center"][0]==128 for f in facts]
        a,b=amber.index(True),violet.index(True)
        if task=="first_arrival": return "amber_first" if a<b else "violet_first" if b<a else "same_time"
        if task=="amber_before_violet": return str(a<b).lower()
        if task=="arrival_gap": return str(abs(a-b))
        if task=="second_frame": return "both" if amber[1] and violet[1] else "amber_only" if amber[1] else "violet_only" if violet[1] else "neither"
    if family.startswith("collision."):
        distances=[math.dist(f["amber_center"],f["violet_center"]) for f in facts]
        contacts=[d<=32 for d in distances]
        if task=="before_final": return str(any(contacts[:-1])).lower()
        if task=="contact_count": return str(sum(contacts))
        if task=="after_first_contact": return "no_contact" if not any(contacts) else "separated" if not all(contacts[contacts.index(True):]) else "remained_in_contact"
        if task=="early_distance_trend":
            delta=[b-a for a,b in zip(distances,distances[1:])][:2]
            if any(0<abs(d)<1 for d in delta): raise ValueError("Subpixel distance trend is unobservable")
            return "includes_no_change" if 0 in delta else "_then_".join("farther" if d>0 else "closer" for d in delta)
    raise ValueError(f"Unsupported temporal predicate {family}")
