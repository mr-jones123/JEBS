"""Prepare deterministic text-only decision data from BANKING77 and local rules."""

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from mdprep.common import (
    PART_QUOTAS,
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
from mdprep.text_catalog import (
    ASSETS,
    CHOICE_INSTRUCTIONS,
    CITATION,
    DISPUTE_CRITERIA,
    EXPOSURE_LEVELS,
    HELD_OUT_INTENTS,
    LICENSE_NAME,
    LICENSE_URL,
    NOUL_INSTRUCTIONS,
    POLICY_CRITERIA,
    PROJECT_DATA_PROVENANCE,
    RAW_BASE_URL,
    SCORE_INSTRUCTIONS,
    SEVERITY_LEVELS,
    SOURCE_API_URL,
    SOURCE_COUNTS,
    SOURCE_QUESTION_TEMPLATES,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
)




@dataclass(frozen=True)
class SourceExample:
    text: str
    category: str
    row_number: int
    normalized_text: str
    cluster_id: str
    cluster_size: int


def _git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload).hexdigest()


def _acquire_assets(root: Path) -> tuple[dict[str, Path], list[dict]]:
    raw_directory = root / "data" / "raw" / "banking77" / SOURCE_REVISION
    paths: dict[str, Path] = {}
    metadata: list[dict] = []
    for name, specification in ASSETS.items():
        upstream_path = specification["upstream_path"]
        url = f"{RAW_BASE_URL}/{upstream_path}"
        path = fetch(url, raw_directory / name)
        actual_blob = _git_blob_sha1(path)
        expected_blob = specification["git_blob_sha1"]
        if actual_blob != expected_blob:
            raise RuntimeError(
                f"BANKING77 asset integrity failure for {name}: "
                f"expected git blob {expected_blob}, got {actual_blob}"
            )
        paths[name] = path
        metadata.append(
            {
                "name": name,
                "path": path.relative_to(root).as_posix(),
                "upstream_path": upstream_path,
                "url": url,
                "revision": SOURCE_REVISION,
                "git_blob_sha1": actual_blob,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "role": specification["role"],
            }
        )
    return paths, metadata


def _read_categories(path: Path) -> list[str]:
    value = json.loads(path.read_text())
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError("BANKING77 categories.json is not a non-empty string list")
    if len(value) != len(set(value)):
        raise ValueError("BANKING77 categories.json contains duplicate labels")
    missing = set(HELD_OUT_INTENTS) - set(value)
    if missing:
        raise ValueError(f"Held-out BANKING77 intents are absent: {sorted(missing)}")
    return value

def _validate_license(path: Path) -> None:
    text = path.read_text(encoding="utf-8").casefold()
    markers = (
        "attribution 4.0 international",
        "creative commons attribution 4.0 international public license",
    )
    if not all(marker in text for marker in markers):
        raise ValueError("Pinned BANKING77 license is not the expected CC BY 4.0 text")


def _read_csv(path: Path, categories: set[str]) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["text", "category"]:
            raise ValueError(f"Unexpected BANKING77 CSV columns in {path}: {reader.fieldnames}")
        for row_number, row in enumerate(reader, start=2):
            text = row["text"].strip()
            category = row["category"].strip()
            if not text:
                raise ValueError(f"Empty BANKING77 utterance at {path}:{row_number}")
            if category not in categories:
                raise ValueError(
                    f"Unknown BANKING77 category {category!r} at {path}:{row_number}"
                )
            rows.append((row_number, text, category))
    return rows


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[\w]+", normalized, flags=re.UNICODE))


def _cluster_training_rows(
    rows: list[tuple[int, str, str]],
) -> tuple[list[SourceExample], dict]:
    clusters: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for row in rows:
        normalized = _normalize_text(row[1])
        if not normalized:
            raise ValueError(f"BANKING77 utterance normalizes to empty at row {row[0]}")
        clusters[normalized].append(row)

    examples: list[SourceExample] = []
    conflicting_clusters = 0
    conflicting_rows = 0
    duplicate_clusters = 0
    for normalized, members in clusters.items():
        labels = {member[2] for member in members}
        if len(members) > 1:
            duplicate_clusters += 1
        if len(labels) != 1:
            conflicting_clusters += 1
            conflicting_rows += len(members)
            continue
        row_number, text, category = min(members, key=lambda member: member[0])
        examples.append(
            SourceExample(
                text=text,
                category=category,
                row_number=row_number,
                normalized_text=normalized,
                cluster_id=stable_id("banking77-normalized", normalized),
                cluster_size=len(members),
            )
        )

    examples.sort(key=lambda example: (example.category, example.cluster_id))
    audit = {
        "normalization": "Unicode NFKC, casefold, retain Unicode word characters, collapse whitespace",
        "raw_rows": len(rows),
        "normalized_clusters": len(clusters),
        "duplicate_clusters": duplicate_clusters,
        "conflicting_label_clusters_excluded": conflicting_clusters,
        "conflicting_label_rows_excluded": conflicting_rows,
        "eligible_single_label_clusters": len(examples),
    }
    return examples, audit


def _balanced_take(
    pools: dict[str, deque[SourceExample]], count: int, split: str,
) -> list[SourceExample]:
    labels = sorted(label for label, pool in pools.items() if pool)
    if not labels:
        raise RuntimeError(f"No source examples are available for {split}")
    selected: list[SourceExample] = []
    offset = int(stable_id("banking77-label-offset", split), 16) % len(labels)
    cursor = offset
    while len(selected) < count:
        made_progress = False
        for step in range(len(labels)):
            label = labels[(cursor + step) % len(labels)]
            if pools[label]:
                selected.append(pools[label].popleft())
                made_progress = True
                if len(selected) == count:
                    break
        if not made_progress:
            available = sum(len(pool) for pool in pools.values())
            raise RuntimeError(
                f"BANKING77 has insufficient eligible clusters for {split}; "
                f"needed {count}, selected {len(selected)}, remaining {available}"
            )
        cursor = (cursor + 1) % len(labels)
    return selected


def _select_source_examples(
    examples: list[SourceExample],
) -> dict[str, list[SourceExample]]:
    held_out = set(HELD_OUT_INTENTS)
    id_pools: dict[str, deque[SourceExample]] = {}
    ood_pools: dict[str, deque[SourceExample]] = {}
    grouped: dict[str, list[SourceExample]] = defaultdict(list)
    for example in examples:
        grouped[example.category].append(example)

    for category, members in grouped.items():
        rng_for("banking77-source-order", category).shuffle(members)
        destination = ood_pools if category in held_out else id_pools
        destination[category] = deque(members)

    selections: dict[str, list[SourceExample]] = {}
    for split in ("train", "dev", "calibration", "policy", "test_id"):
        selections[split] = _balanced_take(id_pools, SOURCE_COUNTS[split], split)
    selections["test_ood"] = _balanced_take(
        ood_pools, SOURCE_COUNTS["test_ood"], "test_ood"
    )
    return selections


def _humanize_intent(label: str) -> str:
    words = label.rstrip("?").replace("_", " ").lower()
    return words[0].upper() + words[1:]

def _select_instruction(
    instructions: tuple[str, ...], family: str, split: str, index: int,
) -> str:
    wording_rng = rng_for("synthetic-question-wording", family, split, index)
    return instructions[wording_rng.randrange(len(instructions))]


def _intent_blocks(examples: list[SourceExample], split: str) -> list[tuple[SourceExample, dict[str, str]]]:
    """Partition the selected queues without replacement into balanced five-intent sets."""
    if len(examples) % 5:
        raise ValueError("BANKING77 allocation must be divisible by five")
    grouped: dict[str, list[SourceExample]] = defaultdict(list)
    for example in examples:
        if (example.category in HELD_OUT_INTENTS) != (split == "test_ood"):
            raise ValueError("BANKING77 intent crossed its holdout boundary")
        grouped[example.category].append(example)
    block_count = len(examples) // 5
    if not block_count or max(map(len, grouped.values())) > block_count:
        raise ValueError("BANKING77 queues cannot form balanced five-intent blocks")
    for attempt in range(128):
        pools = {label: deque(sorted(values, key=lambda e: stable_id("intent-queue", e.cluster_id)))
                 for label, values in grouped.items()}
        output = []
        diversity: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        for block in range(block_count):
            remaining_blocks = block_count - block
            if max(map(len, pools.values())) > remaining_blocks:
                raise ValueError("BANKING77 queues cannot form balanced five-intent blocks")
            labels = sorted((label for label, pool in pools.items() if pool),
                            key=lambda label: (-len(pools[label]), stable_id("intent-block", split, attempt, block, label)))[:5]
            if len(labels) != 5:
                raise ValueError("Fewer than five BANKING77 intents remain")
            signature = tuple(sorted(labels))
            criteria = {label: _humanize_intent(label) for label in signature}
            for label in labels:
                diversity[label].add(signature)
                output.append((pools[label].popleft(), criteria))
        if all(len(values) < 2 or len(diversity[label]) >= 2 for label, values in grouped.items()):
            return output
    raise ValueError("BANKING77 distractor diversity failed after 128 deterministic allocations")


def _source_row(
    example: SourceExample, split: str, criteria: dict[str, str], variant: int,
) -> dict:
    question_text = SOURCE_QUESTION_TEMPLATES[
        int(stable_id(example.cluster_id, "question-wording"), 16)
        % len(SOURCE_QUESTION_TEMPLATES)
    ]
    provenance = {
        "dataset": "BANKING77",
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "revision_api_url": SOURCE_API_URL,
        "upstream_path": "banking_data/train.csv",
        "upstream_split": "train",
        "upstream_row_number": example.row_number,
        "license": LICENSE_NAME,
        "license_url": LICENSE_URL,
        "attribution": "PolyAI-LDN task-specific-datasets; BANKING77 authors",
        "citation": CITATION,
    }
    audit = {
        "task": "human_annotated_intent_classification",
        "ground_truth": {
            "intent_key": example.category,
            "basis": "public human label in BANKING77 train.csv",
        },
        "source_evidence": {
            "upstream_row_number": example.row_number,
            "utterance_sha256": hashlib.sha256(example.text.encode()).hexdigest(),
            "normalized_text": example.normalized_text,
            "normalized_cluster_id": example.cluster_id,
            "cluster_size": example.cluster_size,
        },
        "candidate_intent_keys": list(criteria),
        "split_assignment": {
            "unit": "normalized utterance cluster",
            "held_out_intent": example.category in HELD_OUT_INTENTS,
            "held_out_intent_set": list(HELD_OUT_INTENTS),
        },
    }
    return record(
        split=split,
        source="banking77",
        group_id=example.cluster_id,
        family="banking77_intent_routing",
        modality="text",
        state=f"Customer message:\n{example.text}",
        question=choice(question_text, criteria),
        label=example.category,
        provenance=provenance,
        audit=audit,
        variant=variant,
    )


def _policy_choice_row(split: str, index: int, held_out: bool) -> dict:
    target_index = index % 4
    case_rng = rng_for("text-policy-choice-facts", split, index, held_out)
    if held_out:
        labels = tuple(DISPUTE_CRITERIA)
        target = labels[target_index]
        if target == "file":
            days = case_rng.randint(1, 30)
            duplicate = False
            receipt = bool(case_rng.randrange(2))
            amount = case_rng.randint(20, 499)
        elif target == "expedite":
            days = case_rng.randint(1, 30)
            duplicate = (index // 4) % 2 == 0
            receipt = bool(case_rng.randrange(2))
            amount = case_rng.randint(500, 1800) if not duplicate else case_rng.randint(20, 1800)
        elif target == "deny":
            days = case_rng.randint(31, 90)
            duplicate = bool(case_rng.randrange(2))
            receipt = bool(case_rng.randrange(2))
            amount = case_rng.randint(20, 1800)
        else:
            days = case_rng.randint(1, 30)
            duplicate = bool(case_rng.randrange(2))
            receipt = bool(case_rng.randrange(2))
            amount = case_rng.randint(20, 1800)
            missing_field = (index // 4) % 4
            if missing_field == 0:
                days = None
            elif missing_field == 1:
                duplicate = None
            elif missing_field == 2:
                receipt = None
            else:
                amount = None
        facts = {
            "days_since_statement": days,
            "merchant_charge_is_duplicate": duplicate,
            "receipt_available": receipt,
            "amount_usd": amount,
        }
        rendered = [
            f"Statement age: {days} days" if days is not None else "Statement age: not provided",
            (
                f"Duplicate-charge evidence: {'present' if duplicate else 'absent'}"
                if duplicate is not None else "Duplicate-charge evidence: not provided"
            ),
            (
                f"Receipt evidence: {'present' if receipt else 'absent'}"
                if receipt is not None else "Receipt evidence: not provided"
            ),
            f"Disputed amount: USD {amount}" if amount is not None else "Disputed amount: not provided",
        ]
        policy = (
            "A dispute needs statement age, duplicate-charge status, receipt status, and amount. "
            "Deny when the statement is older than 30 days. Expedite an otherwise eligible "
            "dispute when duplicate evidence is present or the amount is at least USD 500. "
            "Otherwise file it in the standard queue. Missing required evidence is insufficient."
        )
        family = "ood_dispute_deadline_and_priority"
        criteria = DISPUTE_CRITERIA
        calculation = {
            "required_fields_present": None not in (days, duplicate, receipt, amount),
            "deadline_days": 30,
            "expedite_if_duplicate_or_amount_at_least": 500,
        }
        state = "Held-out dispute rule:\n" + policy + "\n\nEvidence:\n- " + "\n- ".join(rendered)
    else:
        labels = tuple(POLICY_CRITERIA)
        target = labels[target_index]
        if target == "allow":
            identity = True
            recipient_age = case_rng.randint(2, 45)
            amount = case_rng.randint(20, 1000)
        elif target == "review":
            identity = True
            if (index // 4) % 2 == 0:
                recipient_age = case_rng.randint(0, 1)
                amount = case_rng.randint(20, 1000)
            else:
                recipient_age = case_rng.randint(2, 45)
                amount = case_rng.randint(1001, 2000)
        elif target == "decline":
            identity = (index // 4) % 2 != 0
            recipient_age = case_rng.randint(0, 45)
            amount = case_rng.randint(2001, 5000) if identity else case_rng.randint(20, 2000)
        else:
            identity = True
            recipient_age = case_rng.randint(0, 45)
            amount = case_rng.randint(20, 2000)
            missing_field = (index // 4) % 3
            if missing_field == 0:
                identity = None
            elif missing_field == 1:
                recipient_age = None
            else:
                amount = None
        facts = {
            "identity_verified": identity,
            "recipient_age_days": recipient_age,
            "amount_usd": amount,
        }
        identity_text = (
            "verified" if identity is True else "failed" if identity is False else "not provided"
        )
        amount_text = f"USD {amount}" if amount is not None else "not provided"
        recipient_age_text = (
            f"{recipient_age} days" if recipient_age is not None else "not provided"
        )
        policy = (
            "Identity status, recipient age, and amount are required. Decline if identity failed "
            "or amount exceeds USD 2,000. Allow when identity is verified, the recipient is at "
            "least 2 days old, and amount is at most USD 1,000. Send any other complete request "
            "for manual review. Missing required evidence is insufficient."
        )
        family = "id_transfer_release_constraints"
        criteria = POLICY_CRITERIA
        calculation = {
            "required_fields_present": None not in (identity, recipient_age, amount),
            "allow_amount_ceiling_usd": 1000,
            "decline_amount_above_usd": 2000,
            "minimum_recipient_age_days": 2,
        }
        state = (
            "Transfer release policy:\n" + policy + "\n\nEvidence:\n"
            f"- Identity check: {identity_text}\n"
            f"- Recipient age: {recipient_age_text}\n"
            f"- Transfer amount: {amount_text}"
        )
    instruction = _select_instruction(CHOICE_INSTRUCTIONS, family, split, index)
    group_id = stable_id("text-policy-choice", family, policy, json.dumps(facts, sort_keys=True))
    return record(
        split=split,
        source="mdprep_text_rules",
        group_id=group_id,
        family=family,
        modality="text",
        state=state,
        question=choice(instruction, criteria),
        label=target,
        provenance={
            "dataset": "deterministic code-labelled banking policies",
            "generator": "mdprep.text",
            **PROJECT_DATA_PROVENANCE,
            "upstream_split": None,
            "held_out_rule_family": held_out,
        },
        audit={
            "task": "policy_constraint_decision",
            "facts": facts,
            "rule_family": family,
            "rule": policy,
            "calculation": calculation,
            "expected_label": target,
            "evidence_binding": "Only explicitly listed evidence is admissible; null means missing.",
            "held_out_rule_family": held_out,
            "template_family": "heldout_dispute_memo" if held_out else "id_transfer_evidence_list",
        },
        variant=index,
    )


def _policy_noul_row(split: str, index: int, held_out: bool) -> dict:
    expected = index % 2 == 0
    case_rng = rng_for("text-policy-noul-facts", split, index, held_out)
    if held_out:
        request = case_rng.randint(20, 500)
        facts = {
            "venue": "bank_atm",
            "hour": case_rng.randint(6, 22),
            "available_balance_usd": request + 25 + case_rng.randint(0, 800),
            "request_usd": request,
        }
        if not expected:
            failure = (index // 2) % 3
            if failure == 0:
                facts["venue"] = ("retail_kiosk", "merchant_counter")[case_rng.randrange(2)]
            elif failure == 1:
                facts["hour"] = case_rng.choice((0, 1, 2, 3, 4, 5, 23))
            else:
                facts["available_balance_usd"] = request + case_rng.randint(0, 24)
        policy = (
            "Cash access is authorized only at a bank ATM, between 06:00 and 22:00, "
            "and when available balance is at least the requested amount plus USD 25."
        )
        state = (
            "Held-out cash-access rule:\n" + policy + "\n\nRequest evidence:\n"
            f"Venue: {facts['venue']}\nHour (24-hour clock): {facts['hour']}\n"
            f"Available balance: USD {facts['available_balance_usd']}\n"
            f"Requested cash: USD {facts['request_usd']}"
        )
        family = "ood_cash_access_time_place_balance"
        checks = {
            "bank_atm": facts["venue"] == "bank_atm",
            "within_hours": 6 <= facts["hour"] <= 22,
            "balance_buffer_met": facts["available_balance_usd"] >= facts["request_usd"] + 25,
        }
    else:
        remaining = case_rng.randint(40, 1500)
        facts = {
            "card_active": True,
            "country_supported": True,
            "amount_usd": case_rng.randint(5, remaining),
            "daily_remaining_usd": remaining,
        }
        if not expected:
            failure = (index // 2) % 3
            if failure == 0:
                facts["card_active"] = False
            elif failure == 1:
                facts["country_supported"] = False
            else:
                facts["amount_usd"] = remaining + case_rng.randint(1, 500)
        policy = (
            "A card purchase is authorized only when the card is active, the purchase country "
            "is supported, and the purchase amount does not exceed the remaining daily limit."
        )
        state = (
            "Card authorization rule:\n" + policy + "\n\nRequest evidence:\n"
            f"Card active: {'yes' if facts['card_active'] else 'no'}\n"
            f"Purchase country supported: {'yes' if facts['country_supported'] else 'no'}\n"
            f"Purchase amount: USD {facts['amount_usd']}\n"
            f"Remaining daily limit: USD {facts['daily_remaining_usd']}"
        )
        family = "id_card_purchase_all_constraints"
        checks = {
            "card_active": facts["card_active"],
            "country_supported": facts["country_supported"],
            "within_remaining_limit": facts["amount_usd"] <= facts["daily_remaining_usd"],
        }
    computed = all(checks.values())
    if computed != expected:
        raise AssertionError(f"Internal rule case mismatch for {family} index {index}")
    group_id = stable_id("text-policy-noul", family, policy, json.dumps(facts, sort_keys=True))
    return record(
        split=split,
        source="mdprep_text_rules",
        group_id=group_id,
        family=family,
        modality="text",
        state=state,
        question=noul(_select_instruction(NOUL_INSTRUCTIONS, family, split, index)),
        label=str(expected).lower(),
        provenance={
            "dataset": "deterministic code-labelled banking policies",
            "generator": "mdprep.text",
            **PROJECT_DATA_PROVENANCE,
            "upstream_split": None,
            "held_out_rule_family": held_out,
        },
        audit={
            "task": "all_constraints_authorization",
            "facts": facts,
            "rule_family": family,
            "rule": policy,
            "constraint_checks": checks,
            "calculation": "logical AND of all constraint checks",
            "expected_label": str(expected).lower(),
            "held_out_rule_family": held_out,
            "template_family": "heldout_cash_access_form" if held_out else "id_card_authorization_form",
        },
        variant=index,
    )


def _policy_score_row(split: str, index: int, held_out: bool) -> dict:
    target_level = index % 4
    case_rng = rng_for("text-policy-score-facts", split, index, held_out)
    if held_out:
        low_ceiling = case_rng.randint(2, 20)
        high_ceiling = case_rng.randint(21, 100)
        if target_level == 0:
            facts = {
                "affected_accounts": case_rng.randint(1, low_ceiling),
                "cross_border": False,
                "control_bypass": False,
            }
        elif target_level == 1:
            facts = {
                "affected_accounts": case_rng.randint(low_ceiling + 1, high_ceiling),
                "cross_border": False,
                "control_bypass": False,
            }
        elif target_level == 2:
            facts = {
                "affected_accounts": case_rng.randint(high_ceiling + 1, high_ceiling + 100),
                "cross_border": False,
                "control_bypass": False,
            }
        else:
            facts = {
                "affected_accounts": case_rng.randint(high_ceiling + 1, high_ceiling + 200),
                "cross_border": (index // 4) % 2 == 0,
                "control_bypass": (index // 4) % 2 != 0,
            }
        points = (
            (0 if facts["affected_accounts"] <= low_ceiling else 1 if facts["affected_accounts"] <= high_ceiling else 2)
            + int(facts["cross_border"])
            + int(facts["control_bypass"])
        )
        computed_level = min(3, points)
        levels = EXPOSURE_LEVELS
        family = "ood_merchant_network_exposure"
        rule = (
            f"Add 0 points for 1–{low_ceiling} affected accounts, 1 for {low_ceiling + 1}–{high_ceiling}, "
            f"or 2 for more than {high_ceiling}; add 1 for cross-border spread and 1 for a control bypass. Cap at level 3."
        )
        state = (
            "Held-out merchant exposure rubric:\n" + rule + "\n\nObserved evidence:\n"
            f"Affected accounts: {facts['affected_accounts']}\n"
            f"Cross-border spread: {'yes' if facts['cross_border'] else 'no'}\n"
            f"Control bypass observed: {'yes' if facts['control_bypass'] else 'no'}"
        )
    else:
        amount_cents = case_rng.randint(100, 49_999)
        facts = {
            "unauthorized": False,
            "card_missing": False,
            "amount_usd": amount_cents / 100,
            "repeated": False,
        }
        if target_level == 1:
            one_point_factor = (index // 4) % 3
            if one_point_factor == 0:
                facts["card_missing"] = True
            elif one_point_factor == 1:
                amount_cents = case_rng.randint(50_000, 500_000)
                facts["amount_usd"] = amount_cents / 100
            else:
                facts["repeated"] = True
        elif target_level == 2:
            facts["unauthorized"] = True
        elif target_level == 3:
            facts["unauthorized"] = True
            one_point_factor = (index // 4) % 3
            if one_point_factor == 0:
                facts["card_missing"] = True
            elif one_point_factor == 1:
                amount_cents = case_rng.randint(50_000, 500_000)
                facts["amount_usd"] = amount_cents / 100
            else:
                facts["repeated"] = True
        points = (
            2 * int(facts["unauthorized"])
            + int(facts["card_missing"])
            + int(facts["amount_usd"] >= 500)
            + int(facts["repeated"])
        )
        computed_level = min(3, points)
        levels = SEVERITY_LEVELS
        family = "id_incident_additive_severity"
        rule = (
            "Add 2 points for an unauthorized transaction; add 1 each for a missing card, "
            "amount at least USD 500, and repeated occurrence. Cap the total at severity 3."
        )
        state = (
            "Incident severity rubric:\n" + rule + "\n\nObserved evidence:\n"
            f"Unauthorized transaction: {'yes' if facts['unauthorized'] else 'no'}\n"
            f"Card missing: {'yes' if facts['card_missing'] else 'no'}\n"
            f"Amount: USD {facts['amount_usd']:.2f}\n"
            f"Repeated occurrence: {'yes' if facts['repeated'] else 'no'}"
        )
    if computed_level != target_level:
        raise AssertionError(f"Internal severity case mismatch for {family} index {index}")
    group_id = stable_id("text-policy-score", family, rule, json.dumps(facts, sort_keys=True))
    return record(
        split=split,
        source="mdprep_text_rules",
        group_id=group_id,
        family=family,
        modality="text",
        state=state,
        question=score(
            _select_instruction(SCORE_INSTRUCTIONS, family, split, index), levels
        ),
        label=str(target_level),
        provenance={
            "dataset": "deterministic code-labelled banking policies",
            "generator": "mdprep.text",
            **PROJECT_DATA_PROVENANCE,
            "upstream_split": None,
            "held_out_rule_family": held_out,
        },
        audit={
            "task": "ordinal_severity_scoring",
            "facts": facts,
            "rule_family": family,
            "rule": rule,
            "rubric_parameters": {"zero_point_ceiling": low_ceiling, "one_point_ceiling": high_ceiling} if held_out else {},
            "raw_points": points,
            "cap": 3,
            "expected_level": target_level,
            "expected_label": str(target_level),
            "held_out_rule_family": held_out,
            "template_family": "heldout_exposure_brief" if held_out else "id_incident_observation",
        },
        variant=index,
    )


def _synthetic_rows(split: str, count: int, held_out: bool, seen_evidence: set[str]) -> list[dict]:
    builders = (_policy_choice_row, _policy_noul_row, _policy_score_row)
    rows: list[dict] = []
    counters = [0, 0, 0]
    for index in range(count):
        builder_index = index % len(builders)
        for _ in range(10_000):
            row = builders[builder_index](split, counters[builder_index], held_out)
            counters[builder_index] += 1
            if row["group_id"] not in seen_evidence:
                seen_evidence.add(row["group_id"])
                rows.append(row)
                break
        else:
            raise RuntimeError(f"Synthetic evidence pool exhausted: {split}/{builder_index}")
    return rows


def _count_primitives(rows: list[dict]) -> dict[str, int]:
    counts = Counter(row["question"]["type"] for row in rows)
    return {primitive: counts.get(primitive, 0) for primitive in ("choice", "noul", "score")}


def prepare(root: Path) -> dict:
    """Acquire BANKING77 and write the exact text-part quota deterministically."""
    root = root.expanduser().resolve()
    asset_paths, asset_metadata = _acquire_assets(root)
    categories = _read_categories(asset_paths["categories.json"])
    category_set = set(categories)
    train_rows = _read_csv(asset_paths["train.csv"], category_set)
    test_rows = _read_csv(asset_paths["test.csv"], category_set)
    _validate_license(asset_paths["LICENSE"])
    clustered_examples, cluster_audit = _cluster_training_rows(train_rows)
    source_examples = _select_source_examples(clustered_examples)
    selected_by_intent = {
        split: dict(sorted(Counter(example.category for example in examples).items()))
        for split, examples in source_examples.items()
    }
    eligible_clusters_by_intent = dict(
        sorted(Counter(example.category for example in clustered_examples).items())
    )

    rows: list[dict] = []
    per_split_primitives: dict[str, dict[str, int]] = {}
    synthetic_counts: dict[str, int] = {}
    seen_evidence: set[str] = set()
    for split, quota in PART_QUOTAS["text"].items():
        source_count = SOURCE_COUNTS[split]
        synthetic_count = quota - source_count
        if synthetic_count < 0:
            raise ValueError(f"Source allocation exceeds text quota for {split}")
        split_rows = [
            _source_row(example, split, criteria, variant)
            for variant, (example, criteria) in enumerate(_intent_blocks(source_examples[split], split))
        ]
        split_rows.extend(
            _synthetic_rows(split, synthetic_count, held_out=split == "test_ood", seen_evidence=seen_evidence)
        )
        rng_for("text-row-order", split).shuffle(split_rows)
        if len(split_rows) != quota:
            raise AssertionError(
                f"Text split {split} has {len(split_rows)} rows; expected {quota}"
            )
        if any(row["modality"] != "text" or row["images"] or row["videos"] for row in split_rows):
            raise AssertionError(f"Non-text media leaked into text split {split}")
        rows.extend(split_rows)
        per_split_primitives[split] = _count_primitives(split_rows)
        synthetic_counts[split] = synthetic_count

    source_metadata = {
        "source": "BANKING77",
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "revision_api_url": SOURCE_API_URL,
        "revision_discovery": "GitHub commits API; all asset URLs are pinned to the resolved commit",
        "license": LICENSE_NAME,
        "license_url": LICENSE_URL,
        "attribution": "PolyAI-LDN task-specific-datasets and the BANKING77 authors",
        "citation": CITATION,
        "assets": asset_metadata,
        "asset_counts": {
            "train_rows": len(train_rows),
            "test_rows_evaluation_only": len(test_rows),
            "categories": len(categories),
            "license_files": 1,
        },
        "usage": {
            "train_csv": "Only source of human-labelled generated rows",
            "test_csv": "Downloaded and counted, but never selected or transformed into rows",
            "held_out_intents": list(HELD_OUT_INTENTS),
            "held_out_policy": "All training utterances with these intents are excluded from ID splits",
            "source_rows_selected": {split: len(values) for split, values in source_examples.items()},
            "distinct_real_train_utterances_in_train": len(source_examples["train"]),
            "source_rows_selected_by_intent": selected_by_intent,
            "eligible_clusters_by_intent": eligible_clusters_by_intent,
        },
        "duplicate_cluster_audit": cluster_audit,
    }
    write_json(root / "data" / "sources" / "banking77.json", source_metadata)
    output_path = root / "data" / "parts" / "text.jsonl"
    written = write_jsonl(output_path, rows)
    if written != sum(PART_QUOTAS["text"].values()):
        raise AssertionError(f"Wrote {written} text rows, expected 9000")

    return {
        "part": "text",
        "path": output_path.relative_to(root).as_posix(),
        "rows": written,
        "splits": dict(PART_QUOTAS["text"]),
        "source_examples": {split: len(values) for split, values in source_examples.items()},
        "synthetic_examples": synthetic_counts,
        "primitive_counts": _count_primitives(rows),
        "primitive_counts_by_split": per_split_primitives,
        "source_revision": SOURCE_REVISION,
        "source_assets": len(asset_metadata),
        "source_asset_counts": source_metadata["asset_counts"],
        "distinct_real_train_utterances_in_train": len(source_examples["train"]),
        "held_out_intents": list(HELD_OUT_INTENTS),
        "held_out_generated_families": [
            "ood_dispute_deadline_and_priority",
            "ood_cash_access_time_place_balance",
            "ood_merchant_network_exposure",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Project root")
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
