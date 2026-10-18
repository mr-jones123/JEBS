"""Qualify finite visible task grammars against source labels or parsed rules.

Unsupported or ambiguous wording fails closed. Photo answers reproduce upstream
human-verification labels, not independently established visible-world truth.
Generated text qualification covers these controlled rules, not arbitrary prose.
"""

import csv
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal

from mdprep.common import sha256_file, stable_id
from mdprep.text_catalog import ASSETS, SOURCE_REVISION

_ID_SPLITS = frozenset({"train", "dev", "calibration", "policy", "test_id"})
_BANK_INSTRUCTIONS = frozenset({
    "Choose the banking support intent that best matches the customer message.",
    "Which single support category most precisely describes this request?",
    "Route this message to the most appropriate banking intent.",
    "Select the intent an annotator should assign to this customer query.",
    "Identify the narrowest matching support reason for the message.",
    "Which candidate intent is the best destination for this request?",
    "Classify the customer's issue using one of the offered banking intents.",
    "Pick the support intent whose meaning best fits the quoted message.",
})
_PHOTO_HELD_OUT_CLASSES = frozenset(name.casefold() for name in (
    "Aircraft", "Airplane", "Ambulance", "Auto rickshaw", "Barge", "Bicycle",
    "Boat", "Bus", "Canoe", "Car", "Cart", "Catamaran", "Helicopter", "Jet ski",
    "Land vehicle", "Limousine", "Minibus", "Motorcycle", "Personal water craft",
    "Pickup truck", "Police car", "Raft", "Rickshaw", "Sailboat", "Segway",
    "Ship", "Snowmobile", "Submarine", "Taxi", "Train", "Tram", "Truck",
    "Unicycle", "Van", "Vehicle", "Watercraft", "Wheelchair",
))
_BOOL_CRITERIA = {"false": "No", "true": "Yes"}
_TEXT_INSTRUCTIONS = {
    "choice": frozenset({
        "Apply the stated policy to the evidence and choose the required action.",
        "Using only the listed evidence, select the policy outcome.",
        "Choose the outcome produced by the rule; do not assume missing facts.",
        "Evaluate the request against every stated constraint and select one action.",
    }),
    "noul": frozenset({
        "Does the request satisfy every stated authorization condition?",
        "Using only the evidence shown, should this request be authorized?",
        "Is the action permitted under the complete rule below?",
        "Answer whether all mandatory policy constraints are met.",
    }),
    "score": frozenset({
        "Assign the ordinal response severity defined by the scoring rule.",
        "What severity level follows from the listed evidence and point rule?",
        "Score this case using the stated additive rubric and cap.",
        "Choose the required response tier after applying the severity rubric.",
    }),
}
_MONEY = r"(?:0|[1-9]\d*|[1-9]\d{0,2}(?:,\d{3})+)(?:\.\d{1,2})?"
_INTEGER = r"(?:0|[1-9]\d*)"


@dataclass(frozen=True)
class _TextTask:
    kind: str
    held_out: bool
    pattern: re.Pattern
    criteria: dict[str, str]


_TEXT_TASKS = {
    "id_transfer_release_constraints": _TextTask(
        "choice", False, re.compile(
            r"Transfer release policy:\n"
            r"Identity status, recipient age, and amount are required\. Decline if identity failed "
            rf"or amount exceeds USD (?P<decline>{_MONEY})\. Allow when identity is verified, the recipient is at "
            rf"least (?P<minimum_age>{_INTEGER}) days old, and amount is at most USD (?P<allow>{_MONEY})\. "
            r"Send any other complete request for manual review\. Missing required evidence is insufficient\."
            r"\n\nEvidence:\n- Identity check: (?P<identity>verified|failed|not provided)"
            rf"\n- Recipient age: (?P<age>{_INTEGER} days|not provided)"
            rf"\n- Transfer amount: (?P<amount>USD {_MONEY}|not provided)"
        ),
        {"allow": "Allow automatically", "review": "Send for manual review",
         "decline": "Decline the request", "insufficient": "Insufficient evidence to decide"},
    ),
    "ood_dispute_deadline_and_priority": _TextTask(
        "choice", True, re.compile(
            r"Held-out dispute rule:\n"
            r"A dispute needs statement age, duplicate-charge status, receipt status, and amount\. "
            rf"Deny when the statement is older than (?P<deadline>{_INTEGER}) days\. Expedite an otherwise eligible "
            r"dispute when duplicate evidence is present or the amount is at least USD "
            rf"(?P<priority>{_MONEY})\. Otherwise file it in the standard queue\. "
            r"Missing required evidence is insufficient\."
            rf"\n\nEvidence:\n- Statement age: (?P<age>{_INTEGER} days|not provided)"
            r"\n- Duplicate-charge evidence: (?P<duplicate>present|absent|not provided)"
            r"\n- Receipt evidence: (?P<receipt>present|absent|not provided)"
            rf"\n- Disputed amount: (?P<amount>USD {_MONEY}|not provided)"
        ),
        {"file": "File the dispute in the standard queue", "expedite": "File and expedite the dispute",
         "deny": "Do not file because the policy conditions fail",
         "insufficient": "Insufficient evidence to decide"},
    ),
    "id_card_purchase_all_constraints": _TextTask(
        "noul", False, re.compile(
            r"Card authorization rule:\n"
            r"A card purchase is authorized only when the card is active, the purchase country "
            r"is supported, and the purchase amount does not exceed the remaining daily limit\."
            r"\n\nRequest evidence:\nCard active: (?P<active>yes|no)"
            r"\nPurchase country supported: (?P<supported>yes|no)"
            rf"\nPurchase amount: USD (?P<amount>{_MONEY})"
            rf"\nRemaining daily limit: USD (?P<limit>{_MONEY})"
        ), _BOOL_CRITERIA,
    ),
    "ood_cash_access_time_place_balance": _TextTask(
        "noul", True, re.compile(
            r"Held-out cash-access rule:\n"
            r"Cash access is authorized only at a bank ATM, between (?P<start>\d{2}):00 and (?P<end>\d{2}):00, "
            rf"and when available balance is at least the requested amount plus USD (?P<buffer>{_MONEY})\."
            r"\n\nRequest evidence:\nVenue: (?P<venue>bank_atm|retail_kiosk|merchant_counter)"
            rf"\nHour \(24-hour clock\): (?P<hour>{_INTEGER})"
            rf"\nAvailable balance: USD (?P<balance>{_MONEY})"
            rf"\nRequested cash: USD (?P<request>{_MONEY})"
        ), _BOOL_CRITERIA,
    ),
    "id_incident_additive_severity": _TextTask(
        "score", False, re.compile(
            r"Incident severity rubric:\n"
            rf"Add (?P<unauthorized_points>{_INTEGER}) points for an unauthorized transaction; "
            rf"add (?P<other_points>{_INTEGER}) each for a missing card, "
            rf"amount at least USD (?P<threshold>{_MONEY}), and repeated occurrence\. "
            rf"Cap the total at severity (?P<cap>{_INTEGER})\."
            r"\n\nObserved evidence:\nUnauthorized transaction: (?P<unauthorized>yes|no)"
            r"\nCard missing: (?P<missing>yes|no)"
            rf"\nAmount: USD (?P<amount>{_MONEY})"
            r"\nRepeated occurrence: (?P<repeated>yes|no)"
        ),
        {"0": "Routine — no immediate intervention",
         "1": "Elevated — monitor and contact the customer",
         "2": "High — restrict the affected channel and investigate",
         "3": "Critical — restrict all access and escalate immediately"},
    ),
    "ood_merchant_network_exposure": _TextTask(
        "score", True, re.compile(
            r"Held-out merchant exposure rubric:\n"
            rf"Add (?P<low_points>{_INTEGER}) points for 1–(?P<low>{_INTEGER}) affected accounts, "
            rf"(?P<middle_points>{_INTEGER}) for (?P<middle_start>{_INTEGER})–(?P<high>{_INTEGER}), "
            rf"or (?P<high_points>{_INTEGER}) for more than (?P<high_start>{_INTEGER}); "
            rf"add (?P<cross_points>{_INTEGER}) for cross-border spread and "
            rf"(?P<bypass_points>{_INTEGER}) for a control bypass\. Cap at level (?P<cap>{_INTEGER})\."
            rf"\n\nObserved evidence:\nAffected accounts: (?P<accounts>{_INTEGER})"
            r"\nCross-border spread: (?P<cross>yes|no)"
            r"\nControl bypass observed: (?P<bypass>yes|no)"
        ),
        {"0": "Limited exposure", "1": "Material exposure",
         "2": "Severe exposure", "3": "Systemic exposure"},
    ),
}


def _choice_candidates(row):
    originals = row["audit"]["original_candidate_keys"]
    descriptions = list(row["question"]["criteria"].values())
    if (len(originals) != len(descriptions) or not originals
            or any(not isinstance(key, str) or not key for key in originals)
            or len(set(originals)) != len(originals)
            or len(set(descriptions)) != len(descriptions)):
        raise ValueError("Ambiguous choice candidate binding")
    return list(zip(originals, descriptions))


def _admit_question(row, kind, instructions, criteria):
    question = row["question"]
    if question["type"] != kind or question["instructions"] not in instructions:
        raise ValueError("Unsupported visible question semantics")
    visible = dict(_choice_candidates(row)) if kind == "choice" else question["criteria"]
    if visible != criteria:
        raise ValueError("Visible candidate meanings do not match the admitted task")


def _decimal(value):
    return Decimal(value.removeprefix("USD ").replace(",", ""))

def answer_key(row):
    return row["audit"]["original_target_key"] if row["question"]["type"]=="choice" else row["target"]["label"]


def source_tables(root):
    banking=root/"data/raw/banking77"/SOURCE_REVISION
    for name,spec in ASSETS.items():
        payload=(banking/name).read_bytes()
        actual=hashlib.sha1(f"blob {len(payload)}\0".encode()+payload).hexdigest()
        if actual!=spec["git_blob_sha1"]: raise ValueError(f"Pinned BANKING77 source changed: {name}")
    with (banking/"train.csv").open(newline="") as stream:
        bank={number:(r["text"].strip(),r["category"].strip()) for number,r in enumerate(csv.DictReader(stream),2)}
    photo=root/"data/raw/open_images_v7"
    receipt=json.loads((photo/"acquisition.json").read_text())
    for name,digest in receipt["local_files"].items():
        if sha256_file(photo/name)!=digest: raise ValueError(f"Photo raw source changed: {name}")
    labels=defaultdict(dict)
    with (photo/"eligible_train_human_labels.csv").open(newline="") as stream:
        for item in csv.DictReader(stream):
            if item["Source"]!="verification" or item["Confidence"] not in {"0","1","0.0","1.0"}: continue
            image,label=item["ImageID"].lower(),item["LabelName"]
            confidence=int(float(item["Confidence"]))
            if label in labels[image] and labels[image][label]!=confidence: raise ValueError("Conflicting raw photo label")
            labels[image][label]=confidence
    with (photo/"oidv7-class-descriptions.csv").open(newline="") as stream:
        classes={r[0]:r[1] for r in csv.reader(stream) if len(r)==2}
    with (photo/"eligible_train_image_metadata.csv").open(newline="") as stream:
        metadata={r["ImageID"].lower():r for r in csv.DictReader(stream)}
    return bank,labels,classes,metadata,receipt


def reconcile(row,tables):
    bank,labels,classes,metadata,receipt=tables
    provenance,audit=row["provenance"],row["audit"]
    target=answer_key(row)
    if provenance["source"]=="banking77":
        utterance,label=bank[provenance["upstream_row_number"]]
        if row["state"]!="Customer message:\n"+utterance or target!=label:
            raise ValueError("BANKING77 raw utterance/label mismatch")
        if audit["ground_truth"]["intent_key"]!=label: raise ValueError("BANKING77 audit mismatch")
        if audit["source_evidence"]["utterance_sha256"]!=hashlib.sha256(utterance.encode()).hexdigest(): raise ValueError("BANKING77 utterance hash mismatch")
        if row["question"]["type"] != "choice" or row["question"]["instructions"] not in _BANK_INSTRUCTIONS:
            raise ValueError("Unsupported BANKING77 visible question semantics")
        candidates = _choice_candidates(row)
        if len(candidates) != 5 or label not in dict(candidates):
            raise ValueError("BANKING77 candidate evidence mismatch")
        for key, description in candidates:
            words = key.rstrip("?").replace("_", " ").lower()
            if not words or description != words[0].upper() + words[1:]:
                raise ValueError("BANKING77 candidate description mismatch")
        normalized=" ".join(re.findall(r"[\w]+",unicodedata.normalize("NFKC",utterance).casefold(),flags=re.UNICODE))
        if row["group_id"]!=stable_id("banking77-normalized",normalized): raise ValueError("BANKING77 normalized source group mismatch")
        return label
    if provenance["source"] != "open_images_v7_train_human_verified":
        raise ValueError("Unsupported reconciled source")
    if row["state"] != "Use only the visible photograph and the named human-verification decision.":
        raise ValueError("Unsupported photo visible state semantics")
    image=provenance["image_id"]
    raw=metadata[image]
    if row["group_id"]!=stable_id(provenance["source"],image): raise ValueError("Photo group mismatch")
    for field,rawfield in (("author_profile_url","AuthorProfileURL"),("original_url","OriginalURL"),("original_landing_url","OriginalLandingURL"),("original_license_url_exact","License")):
        if provenance[field]!=raw[rawfield]: raise ValueError(f"Photo provenance mismatch: {field}")
    if provenance["creator_group"]!=stable_id("open-images-creator",raw["AuthorProfileURL"]) or provenance["original_asset_group"]!=stable_id("open-images-original-md5",raw["OriginalMD5"]):
        raise ValueError("Photo creator/original source grouping mismatch")
    if audit["annotation_source_sha256"]!=receipt["local_files"]["eligible_train_human_labels.csv"]: raise ValueError("Photo annotation hash mismatch")
    if row["split"] not in _ID_SPLITS | {"test_ood"}:
        raise ValueError("Unknown photo split")
    raw_labels = labels[image]
    if any(label_id not in classes for label_id in raw_labels):
        raise ValueError("Unresolved raw photo class")
    has_held_out_positive = any(
        confidence == 1 and classes[label_id].casefold() in _PHOTO_HELD_OUT_CLASSES
        for label_id, confidence in raw_labels.items()
    )
    if has_held_out_positive != (row["split"] == "test_ood"):
        raise ValueError("Photo positive-class holdout violation")
    facts=audit["facts"]
    if (not facts or len({f["label_id"] for f in facts}) != len(facts)
            or len({f["class_name"] for f in facts}) != len(facts)):
        raise ValueError("Missing or ambiguous source label evidence")
    for fact in facts:
        if fact["class_name"]!=classes[fact["label_id"]] or fact["confidence"]!=labels[image].get(fact["label_id"]) or fact["source"]!="verification": raise ValueError("Photo raw annotation mismatch")
    question=row["question"]
    if question["type"] == "noul":
        if len(facts) != 1:
            raise ValueError("Photo boolean evidence binding mismatch")
        _admit_question(
            row, "noul",
            {f'Is “{facts[0]["class_name"]}” human-verified as present in this photograph?'},
            _BOOL_CRITERIA,
        )
        expected = str(bool(facts[0]["confidence"])).lower()
    elif question["type"] == "score":
        if len(facts) != 3:
            raise ValueError("Photo count evidence binding mismatch")
        names = "; ".join(f'“{f["class_name"]}”' for f in facts)
        _admit_question(
            row, "score",
            {f"Among these three classes — {names} — how many are human-verified as present?"},
            {"0": "None", "1": "One", "2": "Two", "3": "All three"},
        )
        expected = str(sum(f["confidence"] for f in facts))
    elif question["type"] == "choice":
        meanings = {
            "Which class is human-verified as present in this photograph?": 1,
            "Which class is human-verified as absent from this photograph?": 0,
        }
        if question["instructions"] not in meanings or len(facts) != 4:
            raise ValueError("Unsupported photo choice semantics")
        confidence = meanings[question["instructions"]]
        eligible = [f["class_name"] for f in facts if f["confidence"] == confidence]
        candidates = _choice_candidates(row)
        if len(eligible) != 1 or {name for _, name in candidates} != {f["class_name"] for f in facts}:
            raise ValueError("Photo choice evidence binding mismatch")
        expected = next(key for key, name in candidates if name == eligible[0])
    else:
        raise ValueError("Unsupported photo question type")
    if target!=expected: raise ValueError("Photo target does not reproduce raw labels")
    return expected


def text_answer(row):
    """Compute a controlled rule's answer from its complete visible semantics.

    The registry is deliberately independent of generator templates and audit
    answers. Full-state matching prevents unparsed overrides or duplicate fields.
    """
    family = row["provenance"]["family"]
    task = _TEXT_TASKS.get(family)
    if task is None:
        raise ValueError("Unknown text policy family")
    split = row["split"]
    if split not in _ID_SPLITS | {"test_ood"} or task.held_out != (split == "test_ood"):
        raise ValueError("Text task-family holdout violation")
    _admit_question(row, task.kind, _TEXT_INSTRUCTIONS[task.kind], task.criteria)
    match = task.pattern.fullmatch(row["state"])
    if match is None:
        raise ValueError("Unsupported or ambiguous visible text policy/evidence")
    values = match.groupdict()
    if family == "id_transfer_release_constraints":
        decline, allow = _decimal(values["decline"]), _decimal(values["allow"])
        if allow > decline:
            raise ValueError("Contradictory transfer decision bounds")
        if "not provided" in (values["identity"], values["age"], values["amount"]):
            return "insufficient"
        age, amount = int(values["age"].removesuffix(" days")), _decimal(values["amount"])
        if values["identity"] == "failed" or amount > decline:
            return "decline"
        return "allow" if age >= int(values["minimum_age"]) and amount <= allow else "review"
    if family == "ood_dispute_deadline_and_priority":
        if "not provided" in (values["age"], values["amount"], values["duplicate"], values["receipt"]):
            return "insufficient"
        age, amount = int(values["age"].removesuffix(" days")), _decimal(values["amount"])
        if age > int(values["deadline"]):
            return "deny"
        return "expedite" if values["duplicate"] == "present" or amount >= _decimal(values["priority"]) else "file"
    if family == "id_card_purchase_all_constraints":
        return str(values["active"] == "yes" and values["supported"] == "yes"
                   and _decimal(values["amount"]) <= _decimal(values["limit"])).lower()
    if family == "ood_cash_access_time_place_balance":
        start, end, hour = int(values["start"]), int(values["end"]), int(values["hour"])
        if not 0 <= start <= end <= 23 or not 0 <= hour <= 23:
            raise ValueError("Invalid or ambiguous cash-access hours")
        return str(values["venue"] == "bank_atm" and start <= hour <= end
                   and _decimal(values["balance"]) >= _decimal(values["request"]) + _decimal(values["buffer"])).lower()
    cap = int(values["cap"])
    if cap > 3:
        raise ValueError("Rubric cap exceeds the visible response scale")
    if family == "id_incident_additive_severity":
        points = int(values["unauthorized_points"]) * (values["unauthorized"] == "yes")
        points += int(values["other_points"]) * (
            (values["missing"] == "yes") + (values["repeated"] == "yes")
            + (_decimal(values["amount"]) >= _decimal(values["threshold"]))
        )
        return str(min(cap, points))
    low, high = int(values["low"]), int(values["high"])
    if (low < 1 or high <= low or int(values["middle_start"]) != low + 1
            or int(values["high_start"]) != high or int(values["accounts"]) < 1):
        raise ValueError("Gapped, overlapping, or invalid exposure rubric")
    accounts = int(values["accounts"])
    points = int(values["low_points"] if accounts <= low else
                 values["middle_points"] if accounts <= high else values["high_points"])
    points += int(values["cross_points"]) * (values["cross"] == "yes")
    points += int(values["bypass_points"]) * (values["bypass"] == "yes")
    return str(min(cap, points))
