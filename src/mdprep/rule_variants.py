"""Generate counterfactual rule variants and a held-out structure evaluation set."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Iterable

from mdprep.common import (
    GENERATOR_VERSION,
    PART_QUOTAS,
    choice,
    noul,
    record,
    rng_for,
    score,
    sha256_file,
    stable_id,
    write_json,
    write_jsonl,
)
from mdprep.native import MAX_POSITIONS, NativeCompiler
from mdprep.records import parse_request, validate_record
from mdprep.text import _policy_choice_row, _policy_noul_row, _policy_score_row
from mdprep.text_catalog import POLICY_CRITERIA, SEVERITY_LEVELS

SOURCE = "rule-variants-v1"
PART = "rule_variants"
TEMPLATE_START_INDEX = 2_000_000
PROBE_START_INDEX = 1_000_000
PART_A_COUNTS = {"train": 2_400, "dev": 150, "calibration": 150}
PART_B_COUNTS = {"train": 3_000, "dev": 200, "calibration": 200}
PAIR_SAME_LABEL_INTERVAL = 10
RENDERINGS = (0, 1, 2, 3)
HELD_OUT_RENDERING = 4
WORD = re.compile(r"[a-z0-9]+")
FORBIDDEN_WORDING = (
    "=>",
    "entry reads",
    "is recorded for",
    "must be",
    "inclusive alternative rule",
    "may be",
    "eligible cases have",
    "the rule accepts",
    "values of",
    "approval covers",
    "is the selected",
    "the case lists",
    "identity is failed",
)
INCLUSIVE_AT_MOST = (
    "at most", "no more than", "or less", "does not exceed", "not above",
    "no higher than", "at or below",
)
INCLUSIVE_AT_LEAST = (
    "at least", "no less than", "or more", "minimum of", "not below",
    "at or above",
)
BOOL_WORDING = {
    "receipt_present": ("a receipt is present", "no receipt is present"),
    "item_opened": ("the item has been opened", "the item was not opened"),
    "report_attached": ("the report has been attached", "no report has been attached"),
    "inspection_complete": ("the inspection is complete", "the inspection is incomplete"),
    "address_verified": ("the address has been verified", "the address is not verified"),
    "fragile": ("the parcel is marked fragile", "the parcel is not fragile"),
    "tracking_enabled": ("parcel tracking is currently enabled", "parcel tracking is not enabled"),
    "owner_verified": ("the owner has been verified", "the owner is not verified"),
    "admin_approved": ("an administrator approved the change", "no administrator approved the change"),
    "training_complete": ("training is complete", "training is incomplete"),
    "device_managed": ("the device has managed status", "the device has unmanaged status"),
    "escort_assigned": ("an escort is assigned", "no escort is assigned"),
    "deposit_recorded": ("a deposit is recorded", "no deposit is recorded"),
    "safety_check": ("the safety check is complete", "the safety check is incomplete"),
    "budget_attached": ("a budget is attached", "no budget is attached"),
    "reviewer_assigned": ("a reviewer is assigned", "no reviewer is assigned"),
    "referral_present": ("a referral is present", "no referral is present"),
    "documents_ready": ("all required documents are ready", "required documents are not ready"),
}
ACTION_PROSE = {
    "allow": "approve the request",
    "review": "send the request for review",
    "decline": "decline the request",
    "insufficient": "defer the decision until more evidence is available",
}


TEMPLATE_SPECS = (
    ("transfer", "id_transfer_release_constraints", _policy_choice_row),
    ("card_purchase", "id_card_purchase_all_constraints", _policy_noul_row),
    ("incident", "id_incident_additive_severity", _policy_score_row),
)

DOMAIN_FIELDS = {
    "refunds": (
        ("receipt_present", "bool", ()),
        ("item_opened", "bool", ()),
        ("order_value", "number", ()),
        ("return_channel", "category", ("courier", "store", "locker")),
        ("customer_tier", "category", ("basic", "plus", "premium")),
    ),
    "insurance_claims": (
        ("report_attached", "bool", ()),
        ("damage_estimate", "number", ()),
        ("policy_age_days", "number", ()),
        ("incident_kind", "category", ("weather", "collision", "theft")),
        ("inspection_complete", "bool", ()),
    ),
    "shipping": (
        ("address_verified", "bool", ()),
        ("parcel_weight", "number", ()),
        ("fragile", "bool", ()),
        ("destination_zone", "category", ("local", "regional", "remote")),
        ("tracking_enabled", "bool", ()),
    ),
    "account_limits": (
        ("owner_verified", "bool", ()),
        ("change_count", "number", ()),
        ("account_age_days", "number", ()),
        ("plan", "category", ("starter", "team", "enterprise")),
        ("admin_approved", "bool", ()),
    ),
    "access_control": (
        ("training_complete", "bool", ()),
        ("clearance_level", "number", ()),
        ("device_managed", "bool", ()),
        ("area", "category", ("lobby", "lab", "archive")),
        ("escort_assigned", "bool", ()),
    ),
    "equipment_rental": (
        ("deposit_recorded", "bool", ()),
        ("renter_age", "number", ()),
        ("equipment_class", "category", ("standard", "powered", "specialist")),
        ("safety_check", "bool", ()),
        ("rental_days", "number", ()),
    ),
    "grant_applications": (
        ("budget_attached", "bool", ()),
        ("requested_units", "number", ()),
        ("organization_kind", "category", ("school", "charity", "cooperative")),
        ("reviewer_assigned", "bool", ()),
        ("project_months", "number", ()),
    ),
    "service_appointments": (
        ("referral_present", "bool", ()),
        ("party_size", "number", ()),
        ("service_kind", "category", ("consultation", "repair", "installation")),
        ("documents_ready", "bool", ()),
        ("lead_days", "number", ()),
    ),
}

NUMERIC_FIELD_SPECS = {
    "order_value": {"minimum": 5, "maximum": 5_000, "unit": "USD"},
    "damage_estimate": {"minimum": 100, "maximum": 50_000, "unit": "USD"},
    "policy_age_days": {"minimum": 1, "maximum": 365, "unit": "days"},
    "parcel_weight": {"minimum": 1, "maximum": 150, "unit": "kilograms"},
    "change_count": {"minimum": 0, "maximum": 30, "unit": "changes"},
    "account_age_days": {"minimum": 1, "maximum": 365, "unit": "days"},
    "clearance_level": {"minimum": 1, "maximum": 5, "unit": "level"},
    "renter_age": {"minimum": 18, "maximum": 85, "unit": "years"},
    "rental_days": {"minimum": 1, "maximum": 365, "unit": "days"},
    "requested_units": {"minimum": 1, "maximum": 500, "unit": "units"},
    "project_months": {"minimum": 1, "maximum": 60, "unit": "months"},
    "party_size": {"minimum": 1, "maximum": 20, "unit": "people"},
    "lead_days": {"minimum": 1, "maximum": 365, "unit": "days"},
}
NUMERIC_SUBJECTS = {
    "order_value": (
        "the order total", "the purchase being returned", "the refund request",
        "the returned order", "the original order",
    ),
    "damage_estimate": (
        "the estimated damage", "the claim's damage estimate", "the assessed loss",
        "the repair estimate", "the reported damage",
    ),
    "policy_age_days": (
        "the policy", "the insurance policy", "the coverage", "the claimant's policy",
        "the active policy",
    ),
    "parcel_weight": (
        "the parcel", "the shipment", "the package", "the consignment", "the item being shipped",
    ),
    "change_count": (
        "the number of requested changes", "the change request count", "the account change total",
        "the number of account changes", "the pending change count",
    ),
    "account_age_days": (
        "the account", "the customer account", "the workspace account", "the active account",
        "the requesting account",
    ),
    "clearance_level": (
        "the worker's clearance", "the visitor's clearance", "the access clearance",
        "the assigned clearance", "the requester's clearance",
    ),
    "renter_age": (
        "the renter", "the customer", "the named renter", "the equipment renter",
        "the person renting the equipment",
    ),
    "rental_days": (
        "the rental period", "the equipment booking", "the requested hire", "the reservation",
        "the equipment rental",
    ),
    "requested_units": (
        "the grant request", "the application", "the requested allocation",
        "the proposed award", "the funding request",
    ),
    "project_months": (
        "the project", "the proposed project", "the funded work", "the project schedule",
        "the planned program",
    ),
    "party_size": (
        "the party", "the appointment group", "the visiting party", "the customer group",
        "the booked group",
    ),
    "lead_days": (
        "the appointment lead time", "the advance notice", "the booking notice",
        "the scheduling lead time", "the notice before service",
    ),
}
NUMERIC_METRIC_SUBJECTS = {
    "order_value": "the order total",
    "damage_estimate": "the damage estimate",
    "policy_age_days": "the policy age",
    "parcel_weight": "the parcel weight",
    "change_count": "the change count",
    "account_age_days": "the account age",
    "clearance_level": "the clearance level",
    "renter_age": "the renter's age",
    "rental_days": "the rental duration",
    "requested_units": "the requested allocation",
    "project_months": "the project duration",
    "party_size": "the party size",
    "lead_days": "the appointment lead time",
}
CATEGORY_PHRASES = {
    "return_channel": (
        "the return uses the {value} channel",
        "the {value} channel handles the return",
        "the customer chose a {value} return",
        "the item is returning through {value}",
        "the return route is {value}",
    ),
    "customer_tier": (
        "the customer is on the {value} tier",
        "the account has {value} customer status",
        "the customer's membership tier is {value}",
        "the customer belongs to the {value} tier",
        "the order comes from a {value}-tier customer",
    ),
    "incident_kind": (
        "the incident is classified as {value}",
        "the claim concerns {value}",
        "the reported cause is {value}",
        "the loss category is {value}",
        "the incident type is {value}",
    ),
    "destination_zone": (
        "the destination is in the {value} zone",
        "the parcel is bound for a {value} destination",
        "the delivery zone is {value}",
        "the shipment's destination class is {value}",
        "the address falls within the {value} zone",
    ),
    "plan": (
        "the account uses the {value} plan",
        "the subscription is on the {value} plan",
        "the workspace has a {value} subscription",
        "the customer account is on {value}",
        "the current plan is {value}",
    ),
    "area": (
        "access is requested for the {value}",
        "the destination area is the {value}",
        "the request concerns the {value}",
        "the controlled area is the {value}",
        "the visitor is entering the {value}",
    ),
    "equipment_class": (
        "the equipment is in the {value} class",
        "the renter chose {value} equipment",
        "the requested item is classified as {value}",
        "the booking covers {value} equipment",
        "the rental category is {value}",
    ),
    "organization_kind": (
        "the applicant is a {value}",
        "the applying organization is a {value}",
        "the proposal comes from a {value}",
        "the organization is registered as a {value}",
        "the grant applicant is a {value}",
    ),
    "service_kind": (
        "the appointment is for a {value}",
        "the requested service is {value}",
        "the booking concerns a {value}",
        "the customer scheduled a {value}",
        "the service category is {value}",
    ),
}
COMBINATORS = ("all_of", "any_of", "unless_exception", "first_match", "additive_points")
HELD_OUT_COMBINATIONS = (
    ("insurance_claims", "first_match"),
    ("shipping", "unless_exception"),
    ("refunds", "additive_points"),
    ("access_control", "any_of"),
    ("equipment_rental", "all_of"),
)
CHOICE_OUTCOMES = {
    "allow": "Approve the request",
    "review": "Send the request for review",
    "decline": "Reject the request",
    "insufficient": "Defer until more evidence is available",
}
SCORE_LEVELS = ["No escalation", "Low escalation", "Medium escalation", "High escalation"]


def _semantic_label(row: dict) -> str:
    if row["question"]["type"] == "choice":
        return row["audit"]["original_target_key"]
    return row["target"]["label"]


def _serialized_label(row: dict, semantic: str) -> str:
    if row["question"]["type"] != "choice":
        return semantic
    return f"option_{row['audit']['original_candidate_keys'].index(semantic)}"


def _provenance(part: str) -> dict:
    return {
        "dataset": "deterministic counterfactual rule grammar",
        "generator": "mdprep.rule_variants",
        "license": "Project-generated synthetic data",
        "upstream_split": None,
        "held_out_rule_family": False,
        "generation_part": part,
    }


def transfer_label(facts: dict, parameters: dict) -> str:
    if None in (facts["identity_verified"], facts["recipient_age_days"], facts["amount_usd"]):
        return parameters["missing_label"]
    declined_identity = facts["identity_verified"] is parameters["declined_identity_status"]
    if declined_identity or facts["amount_usd"] > parameters["decline_above_usd"]:
        return "decline"
    if (facts["identity_verified"] is True
            and facts["recipient_age_days"] >= parameters["minimum_age_days"]
            and facts["amount_usd"] <= parameters["allow_ceiling_usd"]):
        return "allow"
    return "review"


def card_label(facts: dict, parameters: dict) -> str:
    checks = []
    if "card_active" in parameters["required_constraints"]:
        checks.append(facts["card_active"] is parameters["required_card_active"])
    if "country_supported" in parameters["required_constraints"]:
        checks.append(facts["country_supported"] is parameters["required_country_supported"])
    if "amount_limit" in parameters["required_constraints"]:
        checks.append(facts["amount_usd"] <= facts["daily_remaining_usd"] + parameters["limit_offset_usd"])
    return str(all(checks)).lower()


def incident_label(facts: dict, parameters: dict) -> str:
    points = (
        parameters["unauthorized_weight"] * int(facts["unauthorized"])
        + parameters["missing_weight"] * int(facts["card_missing"])
        + parameters["amount_weight"] * int(facts["amount_usd"] >= parameters["amount_threshold_usd"])
        + parameters["repeated_weight"] * int(facts["repeated"])
    )
    return str(min(parameters["cap"], points))


def _transfer_parameters(facts: dict, original: str, variant: int, same: bool) -> tuple[dict, dict]:
    rng = rng_for(SOURCE, "transfer-parameters", facts, variant)
    amount = facts["amount_usd"]
    age = facts["recipient_age_days"]
    first = {
        "allow_ceiling_usd": 800 + rng.randrange(0, 8) * 100,
        "decline_above_usd": 1_800 + rng.randrange(0, 8) * 200,
        "minimum_age_days": 1 + rng.randrange(0, 4),
        "declined_identity_status": False,
        "missing_label": "insufficient",
    }
    if original == "allow":
        first.update(allow_ceiling_usd=amount + 20 + rng.randrange(180),
                     decline_above_usd=amount + 300 + rng.randrange(900),
                     minimum_age_days=max(0, age - rng.randrange(3)))
    elif original == "review" and None not in (facts["identity_verified"], age, amount):
        first.update(decline_above_usd=amount + 200 + rng.randrange(600))
        if age < 2:
            first.update(minimum_age_days=age + 1 + rng.randrange(3),
                         allow_ceiling_usd=amount + 100)
        else:
            first.update(minimum_age_days=max(0, age - 1),
                         allow_ceiling_usd=max(0, amount - 1 - rng.randrange(100)))
    elif original == "decline":
        if facts["identity_verified"] is False:
            first.update(declined_identity_status=False,
                         decline_above_usd=amount + 500)
        else:
            first.update(declined_identity_status=False,
                         decline_above_usd=max(0, amount - 1 - rng.randrange(100)))
    elif original == "insufficient":
        first["missing_label"] = "insufficient"
    if transfer_label(facts, first) != original:
        raise AssertionError("transfer parameter construction failed")

    second = dict(first)
    if same:
        if original in {"insufficient", "review"} and None in facts.values():
            second["decline_above_usd"] += 137
        elif original == "allow":
            second["allow_ceiling_usd"] += 137
        elif original == "decline":
            second["allow_ceiling_usd"] += 137
        else:
            second["decline_above_usd"] += 137
    elif original == "allow":
        second["decline_above_usd"] = max(0, amount - 1)
    elif original == "review":
        if None in facts.values():
            second["missing_label"] = "insufficient" if original == "review" else "review"
        else:
            second.update(declined_identity_status=not facts["identity_verified"],
                          decline_above_usd=amount + 500,
                          minimum_age_days=0,
                          allow_ceiling_usd=amount + 50)
    elif original == "decline":
        second.update(declined_identity_status=not facts["identity_verified"],
                      decline_above_usd=amount + 500,
                      minimum_age_days=(age or 0) + 1,
                      allow_ceiling_usd=max(0, amount - 1))
    else:
        second["missing_label"] = "review"
    if (transfer_label(facts, second) == original) is not same:
        raise AssertionError("transfer pair relation failed")
    return first, second


def _card_parameters(facts: dict, original: str, variant: int, same: bool) -> tuple[dict, dict]:
    constraints = ["card_active", "country_supported", "amount_limit"]
    passes = {
        "card_active": facts["card_active"],
        "country_supported": facts["country_supported"],
        "amount_limit": facts["amount_usd"] <= facts["daily_remaining_usd"],
    }
    failing = [name for name in constraints if not passes[name]]
    rng = rng_for(SOURCE, "card-parameters", facts, variant)
    if original == "true":
        passing = [name for name in constraints if passes[name]]
        required = [passing[variant % len(passing)]]
    else:
        required = [failing[0]]
    first = {
        "required_constraints": required,
        "required_card_active": True,
        "required_country_supported": True,
        "limit_offset_usd": 0,
        "comparison_wording": ("at_most", "no_more_than", "or_less", "not_above")[variant % 4],
    }
    if card_label(facts, first) != original:
        raise AssertionError("card parameter construction failed")
    second = {**first, "required_constraints": list(first["required_constraints"]),
              "comparison_wording": ("no_more_than", "or_less", "not_above", "at_most")[variant % 4]}
    if same:
        second["limit_offset_usd"] = 17 if "amount_limit" not in required else (
            1 if passes["amount_limit"] else -1)
    elif original == "true":
        second["required_constraints"] = ["card_active"]
        second["required_card_active"] = not facts["card_active"]
    else:
        passing = [name for name in constraints if passes[name]]
        selected = passing[0]
        second["required_constraints"] = [selected]
    if (card_label(facts, second) == original) is not same:
        raise AssertionError("card pair relation failed")
    return first, second


def _incident_parameters(facts: dict, original: str, variant: int, same: bool) -> tuple[dict, dict]:
    rng = rng_for(SOURCE, "incident-parameters", facts, variant)
    first = {
        "unauthorized_weight": 2,
        "missing_weight": 1,
        "amount_weight": 1,
        "repeated_weight": 1,
        "amount_threshold_usd": 500,
        "cap": 2 if original in {"0", "1", "2"} and variant % 3 == 0 else 3,
    }
    active_one = [name for name, active in (
        ("missing_weight", facts["card_missing"]),
        ("amount_weight", facts["amount_usd"] >= 500),
        ("repeated_weight", facts["repeated"]),
    ) if active]
    if incident_label(facts, first) != original:
        raise AssertionError("incident parameter construction failed")
    second = dict(first)
    if same:
        inactive = [name for name, active in (
            ("missing_weight", facts["card_missing"]),
            ("repeated_weight", facts["repeated"]),
        ) if not active]
        if inactive:
            second[inactive[rng.randrange(len(inactive))]] = 2
        else:
            second["amount_threshold_usd"] = 501 if facts["amount_usd"] < 500 else 499
    elif original == "0":
        second.update(amount_threshold_usd=max(0, facts["amount_usd"] - 1), amount_weight=1)
    elif original == "1":
        second[active_one[0]] = 2
    elif original == "2":
        second.update(unauthorized_weight=3, cap=3)
    else:
        second["cap"] = 0
    if (incident_label(facts, second) == original) is not same:
        raise AssertionError("incident pair relation failed")
    return first, second


def _render_transfer(parameters: dict, style: int) -> str:
    failed_identity = parameters["declined_identity_status"] is False
    identity_clause = (
        "the identity check failed" if failed_identity
        else "the identity check passed"
    )
    missing = "insufficient evidence" if parameters["missing_label"] == "insufficient" else "manual review"
    decline_comparison = "more than" if style % 2 == 0 else "greater than"
    age_comparison = (
        f"has been registered for at least {parameters['minimum_age_days']} days"
        if style % 2 == 0
        else f"has been registered for {parameters['minimum_age_days']} days or more"
    )
    amount_comparison = "does not exceed" if style % 3 else "is no higher than"
    clauses = [
        f"Decline a transfer when {identity_clause} or the transfer amount is "
        f"{decline_comparison} USD {parameters['decline_above_usd']:,}.",
        f"Release requires a verified identity, a recipient relationship that {age_comparison}, "
        f"and a transfer amount that {amount_comparison} USD {parameters['allow_ceiling_usd']:,}.",
        f"Route all remaining complete transfers to review; missing fields mean {missing}.",
    ]
    if style % 2:
        clauses = [clauses[1], clauses[0], clauses[2]]
    return " ".join(clauses)


def _render_card(parameters: dict, style: int) -> str:
    descriptions = []
    if "card_active" in parameters["required_constraints"]:
        descriptions.append(
            "the payment card has active status"
            if parameters["required_card_active"]
            else "the payment card has inactive status"
        )
    if "country_supported" in parameters["required_constraints"]:
        descriptions.append(
            "the transaction country does appear on the supported list"
            if parameters["required_country_supported"]
            else "the transaction country does not appear on the supported list"
        )
    if "amount_limit" in parameters["required_constraints"]:
        offset = parameters["limit_offset_usd"]
        if offset > 0:
            limit = f"the remaining allowance plus {offset} USD"
        elif offset < 0:
            limit = f"the remaining allowance minus {abs(offset)} USD"
        else:
            limit = "the remaining allowance"
        descriptions.append({
            "at_most": f"the purchase amount is at most {limit}",
            "no_more_than": f"the purchase amount is no more than {limit}",
            "or_less": f"the purchase amount is {limit} or less",
            "not_above": f"the purchase amount is not above {limit}",
        }[parameters["comparison_wording"]])
    wording_index = ("at_most", "no_more_than", "or_less", "not_above").index(
        parameters["comparison_wording"]
    )
    lead = (
        "Approve the purchase only when ",
        "Approve the purchase when ",
        "Approval requires that ",
        "Return yes only if ",
    )[(style + wording_index) % 4]
    return lead + "; and ".join(descriptions) + "."


def _render_incident(parameters: dict, style: int) -> str:
    factors = [
        f"unauthorized activity contributes {parameters['unauthorized_weight']}",
        f"a missing card contributes {parameters['missing_weight']}",
        f"an amount of at least USD {parameters['amount_threshold_usd']:g} contributes {parameters['amount_weight']}",
        f"a repeat contributes {parameters['repeated_weight']}",
    ]
    if style % 2:
        factors.reverse()
    return ("Compute severity points: " + "; ".join(factors)
            + f". Report no more than level {parameters['cap']}.")



def _template_state(kind: str, rule: str, facts: dict) -> str:
    if kind == "transfer":
        recipient_age = (
            "not provided" if facts["recipient_age_days"] is None
            else f"{facts['recipient_age_days']} days"
        )
        amount = (
            "not provided" if facts["amount_usd"] is None
            else f"USD {facts['amount_usd']:,}"
        )
        return (
            f"Transfer instructions:\n{rule}\n\nCase data:\n"
            f"Identity verified: {facts['identity_verified']}\n"
            f"Recipient relationship age: {recipient_age}\n"
            f"Transfer amount: {amount}"
        )
    if kind == "card_purchase":
        return (
            f"Purchase instructions:\n{rule}\n\nCase data:\n"
            f"Card active: {facts['card_active']}\nCountry supported: {facts['country_supported']}\n"
            f"Purchase amount: USD {facts['amount_usd']:,}\n"
            f"Remaining daily allowance: USD {facts['daily_remaining_usd']:,}"
        )
    return (
        f"Incident instructions:\n{rule}\n\nCase data:\n"
        f"Unauthorized: {facts['unauthorized']}\nCard missing: {facts['card_missing']}\n"
        f"Transaction amount: USD {facts['amount_usd']:,.2f}\nRepeated: {facts['repeated']}"
    )


def _template_row(split: str, kind: str, family: str, facts: dict, parameters: dict,
                  pair_id: str, pair_variant: str, variant: int, style: int, same: bool) -> dict:
    if kind == "transfer":
        semantic = transfer_label(facts, parameters)
        rule = _render_transfer(parameters, style)
        question = choice("Select the outcome required by the transfer instructions.", POLICY_CRITERIA)
        computation = {"function": "transfer_label", "result": semantic}
        condition_count = 5
    elif kind == "card_purchase":
        semantic = card_label(facts, parameters)
        rule = _render_card(parameters, style)
        question = noul("Does the purchase satisfy every required instruction?")
        computation = {"function": "logical_all_of_required_constraints", "result": semantic}
        condition_count = len(parameters["required_constraints"])
    else:
        semantic = incident_label(facts, parameters)
        rule = _render_incident(parameters, style)
        question = score("Choose the severity level produced by the instructions.", SEVERITY_LEVELS)
        computation = {"function": "weighted_sum_then_cap", "result": semantic}
        condition_count = 4
    group_id = stable_id(SOURCE, "template-evidence", pair_id, facts)
    return record(
        split=split,
        source=SOURCE,
        group_id=group_id,
        family=family,
        modality="text",
        state=_template_state(kind, rule, facts),
        question=question,
        label=semantic,
        provenance=_provenance("template_variants"),
        audit={
            "task": "counterfactual_template_rule",
            "facts": facts,
            "rule": rule,
            "rule_parameters": parameters,
            "computation": computation,
            "expected_label": semantic,
            "pair_id": pair_id,
            "pair_variant": pair_variant,
            "label_relation": "same" if same else "changed",
            "part": "A",
            "domain": kind,
            "combinator": f"template_{kind}",
            "condition_count": condition_count,
            "rendering": style,
            "structure_family": family,
            "source_index_start": TEMPLATE_START_INDEX,
        },
        variant=(variant, pair_variant, style),
    )


def build_template_rows(split: str, count: int) -> list[dict]:
    if count % (2 * len(TEMPLATE_SPECS)):
        raise ValueError("template row count must form complete pairs evenly across templates")
    pairs_per_template = count // (2 * len(TEMPLATE_SPECS))
    rows = []
    for template_offset, (kind, family, builder) in enumerate(TEMPLATE_SPECS):
        seen_facts = set()
        for base in range(pairs_per_template):
            transfer_schedule = (
                "insufficient", "insufficient", "insufficient", "insufficient",
                "allow", "allow", "allow", "allow", "review", "decline",
            )
            desired_original = transfer_schedule[base % len(transfer_schedule)] if kind == "transfer" else None
            source_index = TEMPLATE_START_INDEX + template_offset * 100_000 + base
            original_row = builder(split, source_index, False)
            facts = original_row["audit"]["facts"]
            fact_signature = json.dumps(facts, sort_keys=True, separators=(",", ":"))
            while (original_row["provenance"]["family"] != family
                   or fact_signature in seen_facts
                   or (desired_original is not None and _semantic_label(original_row) != desired_original)):
                source_index += 1
                original_row = builder(split, source_index, False)
                facts = original_row["audit"]["facts"]
                fact_signature = json.dumps(facts, sort_keys=True, separators=(",", ":"))
            seen_facts.add(fact_signature)
            original = _semantic_label(original_row)
            same = base % PAIR_SAME_LABEL_INTERVAL == 0
            if kind == "transfer":
                parameters = _transfer_parameters(facts, original, base, same)
            elif kind == "card_purchase":
                parameters = _card_parameters(facts, original, base, same)
            else:
                parameters = _incident_parameters(facts, original, base, same)
            pair_id = stable_id(SOURCE, "template-pair", split, kind, base, facts)
            label_count = 2 if kind == "card_purchase" else 4
            style = (base // label_count) % len(RENDERINGS)
            for index, value in enumerate(parameters):
                rows.append(_template_row(
                    split, kind, family, facts, value, pair_id,
                    "original" if index == 0 else ("edit_keeps_label" if same else "edit_changes_label"),
                    base * 2 + index, style, same,
                ))
    return rows


def _facts_for(domain: str, split: str, base: int) -> dict:
    rng = rng_for(SOURCE, "structure-facts", domain, split, base)
    facts = {}
    for name, kind, values in DOMAIN_FIELDS[domain]:
        if kind == "bool":
            facts[name] = bool(rng.randrange(2))
        elif kind == "number":
            spec = NUMERIC_FIELD_SPECS[name]
            facts[name] = rng.randint(spec["minimum"], spec["maximum"])
        else:
            facts[name] = values[rng.randrange(len(values))]
    return facts


def _condition(field: tuple[str, str, tuple], facts: dict, desired: bool, salt: int) -> dict:
    name, kind, values = field
    actual = facts[name]
    if kind == "bool":
        return {"field": name, "operator": "is", "value": actual if desired else not actual}
    if kind == "category":
        if desired:
            expected = actual
        else:
            expected = next(value for value in values if value != actual)
        return {"field": name, "operator": "equals", "value": expected}

    spec = NUMERIC_FIELD_SPECS[name]
    rng = rng_for(SOURCE, "numeric-condition", name, actual, desired, salt)
    boundary_bucket = rng.randrange(20)
    near_boundary = boundary_bucket < 7
    if desired and boundary_bucket == 0:
        return {"field": name, "operator": "at_least" if rng.randrange(2) else "at_most",
                "value": actual}

    operator = "at_least" if rng.randrange(2) else "at_most"
    positive_distance = desired == (operator == "at_least")
    room = actual - spec["minimum"] if positive_distance else spec["maximum"] - actual
    minimum_distance = 1 if near_boundary else 2
    if room < minimum_distance:
        operator = "at_most" if operator == "at_least" else "at_least"
        positive_distance = desired == (operator == "at_least")
        room = actual - spec["minimum"] if positive_distance else spec["maximum"] - actual
    if room < minimum_distance:
        raise AssertionError(f"numeric range for {name} cannot support requested boundary distance")
    distance = 1 if near_boundary else rng.randint(2, room)
    signed_distance = distance if positive_distance else -distance
    threshold = actual - signed_distance
    return {"field": name, "operator": operator, "value": threshold}



def evaluate_condition(condition: dict, facts: dict) -> bool:
    actual = facts[condition["field"]]
    operator = condition["operator"]
    if operator == "is" or operator == "equals":
        return actual == condition["value"]
    if operator == "at_least":
        return actual >= condition["value"]
    if operator == "at_most":
        return actual <= condition["value"]
    raise ValueError(f"unknown condition operator: {operator}")


def evaluate_structure(parameters: dict, facts: dict) -> str:
    combinator = parameters["combinator"]
    conditions = parameters["conditions"]
    results = [evaluate_condition(condition, facts) for condition in conditions]
    if combinator == "all_of":
        return str(all(results)).lower()
    if combinator == "any_of":
        return str(any(results)).lower()
    if combinator == "unless_exception":
        return str(all(results[:-1]) and not results[-1]).lower()
    if combinator == "first_match":
        for result, action in zip(results, parameters["actions"]):
            if result:
                return action
        return parameters["default_action"]
    if combinator == "additive_points":
        points = sum(weight for result, weight in zip(results, parameters["weights"]) if result)
        return str(min(parameters["cap"], points))
    raise ValueError(f"unknown combinator: {combinator}")


def _truth_pattern(combinator: str, target: str, count: int) -> tuple[list[bool], dict]:
    if combinator == "all_of":
        truth = target == "true"
        return ([True] * count if truth else [False] + [True] * (count - 1)), {}
    if combinator == "any_of":
        truth = target == "true"
        return ([True] + [False] * (count - 1) if truth else [False] * count), {}
    if combinator == "unless_exception":
        truth = target == "true"
        return ([True] * (count - 1) + [False] if truth else [True] * (count - 1) + [True]), {}
    if combinator == "first_match":
        outcomes = list(CHOICE_OUTCOMES)
        default = outcomes[-1]
        if target == default:
            actions = [outcomes[index % (len(outcomes) - 1)] for index in range(count)]
            return [False] * count, {"actions": actions, "default_action": default}
        position = min(outcomes.index(target), count - 1)
        results = [False] * count
        results[position] = True
        actions = [outcomes[(index + 1) % (len(outcomes) - 1)] for index in range(count)]
        actions[position] = target
        return results, {"actions": actions, "default_action": default}
    level = int(target)
    results = [False] * count
    weights = [1] * count
    if level == 1:
        results[0] = True
    elif level == 2:
        results[0] = True
        weights[0] = 2
    elif level == 3:
        results[0] = True
        weights[0] = 2
        if count > 1:
            results[1] = True
        else:
            weights[0] = 3
    return results, {"weights": weights, "cap": 3}


def _targets_for(combinator: str) -> tuple[str, ...]:
    if combinator in {"all_of", "any_of", "unless_exception"}:
        return ("false", "true")
    if combinator == "first_match":
        return tuple(CHOICE_OUTCOMES)
    return ("0", "1", "2", "3")


def _structure_parameters(domain: str, combinator: str, facts: dict, count: int,
                          target: str, salt: int) -> dict:
    pattern, extra = _truth_pattern(combinator, target, count)
    fields = DOMAIN_FIELDS[domain]
    conditions = [
        _condition(fields[(salt + index) % len(fields)], facts, desired, salt * 11 + index)
        for index, desired in enumerate(pattern)
    ]
    parameters = {"domain": domain, "combinator": combinator, "conditions": conditions, **extra}
    actual = evaluate_structure(parameters, facts)
    if actual != target:
        raise AssertionError(f"structure construction produced {actual}, expected {target}")
    return parameters


def _field_words(field: str) -> str:
    return field.replace("_", " ")


def _format_numeric(field: str, value: int) -> str:
    unit = NUMERIC_FIELD_SPECS[field]["unit"]
    if unit == "USD":
        return f"USD {value:,}"
    if unit == "level":
        return f"level {value}"
    return f"{value:,} {unit}"


def _numeric_comparison(operator: str, value: str, style: int) -> str:
    if operator == "at_most":
        return (
            f"at most {value}",
            f"no more than {value}",
            f"{value} or less",
        )[style]
    return (
        f"at least {value}",
        f"no less than {value}",
        f"{value} or more",
    )[style]


def _render_numeric_condition(field: str, operator: str, value: int, style: int) -> str:
    formatted = _format_numeric(field, value)
    if style >= 3:
        metric = NUMERIC_METRIC_SUBJECTS[field]
        if operator == "at_most":
            verb = "does not exceed" if style == 3 else "is not above"
        else:
            verb = "has a minimum of" if style == 3 else "is not below"
        return f"{metric} {verb} {formatted}"

    subject = NUMERIC_SUBJECTS[field][style]
    comparison = _numeric_comparison(operator, formatted, style)
    if field in {"policy_age_days", "account_age_days"}:
        return f"{subject} has been active for {comparison}"
    if field == "renter_age":
        return f"{subject} has an age of {comparison}"
    if field == "party_size":
        return f"{subject} includes {comparison}"
    if field in {"rental_days", "project_months"}:
        return f"{subject} lasts {comparison}"
    if field == "lead_days":
        return f"{subject} allows {comparison}"
    if field == "parcel_weight":
        return f"{subject} weighs {comparison}"
    if field in {"order_value", "damage_estimate"}:
        return f"{subject} totals {comparison}"
    if field == "requested_units":
        return f"{subject} requests {comparison}"
    if field in {"change_count", "clearance_level"}:
        return f"{subject} is {comparison}"
    raise ValueError(f"no numeric phrase template for field: {field}")


def _render_condition(condition: dict, style: int) -> str:
    operator = condition["operator"]
    if operator == "is":
        positive, negative = BOOL_WORDING[condition["field"]]
        return positive if condition["value"] else negative
    if operator == "equals":
        return CATEGORY_PHRASES[condition["field"]][style].format(value=condition["value"])
    return _render_numeric_condition(
        condition["field"], operator, condition["value"], style,
    )


def render_structure(parameters: dict, style: int) -> str:
    conditions = [_render_condition(condition, style) for condition in parameters["conditions"]]
    combinator = parameters["combinator"]
    if combinator == "all_of":
        lead = (
            "Approve the request only when all of the following conditions hold",
            "Every listed requirement is necessary for approval",
            "Approval requires every listed condition",
            "The request qualifies only if each of these conditions holds",
            "All of the following conditions are required for approval",
        )[style]
        return lead + ": " + "; ".join(conditions) + "."
    if combinator == "any_of":
        lead = (
            "Approve the request when at least one of these conditions holds",
            "Any one of these conditions is sufficient for approval",
            "Approval is permitted if one or more of these conditions holds",
            "The request qualifies when any listed condition holds",
            "A single matching condition permits approval",
        )[style]
        return lead + ": " + "; ".join(conditions) + "."
    if combinator == "unless_exception":
        requirements = conditions[:-1]
        exception = conditions[-1]
        leads = (
            "Approve only when every requirement is met and the exception does not apply",
            "The requirements permit approval unless the exception applies",
            "Require the complete checklist, but reject the request when the exception applies",
            "Apply every prerequisite, then deny approval if the exception applies",
            "The exception overrides an otherwise complete checklist",
        )
        return leads[style] + ". Requirements: " + "; ".join(requirements) + f". Exception: {exception}."
    if combinator == "first_match":
        entries = [
            f"{'if' if index == 0 else 'otherwise, if'} {condition}, {ACTION_PROSE[action]}"
            for index, (condition, action) in enumerate(zip(conditions, parameters["actions"]))
        ]
        leads = (
            "Check these conditions in order and apply the first one that fits",
            "Review the conditions in order and stop as soon as one applies",
            "Earlier conditions take priority, so use the first matching instruction",
            "Apply the first matching instruction in this ordered list",
            "Work through this ordered policy and follow the first applicable instruction",
        )
        default = ACTION_PROSE[parameters["default_action"]]
        return leads[style] + ": " + "; ".join(entries) + f"; if none fit, {default}."
    entries = [
        f"award {weight} {'point' if weight == 1 else 'points'} when {condition}"
        for condition, weight in zip(conditions, parameters["weights"])
    ]
    leads = (
        "Add the points for every condition that holds",
        "Total the points from all applicable conditions",
        "Calculate the score from each condition that applies",
        "Accumulate points for every satisfied condition",
        "Each applicable condition contributes the stated number of points",
    )
    return leads[style] + ": " + "; ".join(entries) + f". Cap the final level at {parameters['cap']}."


def _facts_text(domain: str, facts: dict) -> str:
    lines = []
    for field, value in facts.items():
        if field in NUMERIC_FIELD_SPECS:
            rendered = _format_numeric(field, value)
        elif isinstance(value, bool):
            rendered = str(value).lower()
        else:
            rendered = str(value).replace("_", " ")
        lines.append(f"{_field_words(field).title()}: {rendered}")
    return f"Domain: {_field_words(domain)}\n" + "\n".join(lines)


def _structure_question(combinator: str) -> dict:
    if combinator == "first_match":
        return choice("Which outcome does the ordered rule select?", CHOICE_OUTCOMES)
    if combinator == "additive_points":
        return score("Which level does the point rule produce?", SCORE_LEVELS)
    return noul("Does this case receive approval under the rule?")


def _structure_row(split: str, domain: str, combinator: str, facts: dict, parameters: dict,
                   pair_id: str, pair_variant: str, variant: int, style: int, same: bool,
                   external: bool = False) -> dict:
    semantic = evaluate_structure(parameters, facts)
    rule = render_structure(parameters, style)
    family = f"random_{domain}_{combinator}"
    results = [evaluate_condition(condition, facts) for condition in parameters["conditions"]]
    row = record(
        split=split,
        source=SOURCE,
        group_id=stable_id(SOURCE, "structure-evidence", pair_id, facts),
        family=family,
        modality="text",
        state=(
            f"Decision rule:\n{rule}\n\nCase facts:\n"
            f"Case reference: {pair_id[:12]}\n{_facts_text(domain, facts)}"
        ),
        question=_structure_question(combinator),
        label=semantic,
        provenance={**_provenance("heldout_structure_eval" if external else "random_structures"),
                    "held_out_rule_family": external},
        audit={
            "task": "random_rule_structure",
            "facts": facts,
            "rule": rule,
            "rule_parameters": parameters,
            "computation": {"condition_results": results, "result": semantic},
            "expected_label": semantic,
            "pair_id": pair_id,
            "pair_variant": pair_variant,
            "label_relation": "same" if same else "changed",
            "part": "external" if external else "B",
            "domain": domain,
            "combinator": combinator,
            "condition_count": len(parameters["conditions"]),
            "rendering": style,
            "structure_family": f"{domain}:{combinator}:{len(parameters['conditions'])}",
        },
        variant=(variant, pair_variant, style),
    )
    validate_record(row)
    return row


def _training_combinations() -> list[tuple[str, str]]:
    held_out = set(HELD_OUT_COMBINATIONS)
    return [(domain, combinator) for domain in DOMAIN_FIELDS for combinator in COMBINATORS
            if (domain, combinator) not in held_out]


def build_structure_rows(split: str, count: int) -> list[dict]:
    if count % 2:
        raise ValueError("structure rows must form complete pairs")
    pairs = count // 2
    combinations = _training_combinations()
    rows = []
    for base in range(pairs):
        combination_index = base % len(combinations)
        domain, combinator = combinations[combination_index]
        facts = _facts_for(domain, split, base)
        targets = _targets_for(combinator)
        block = base // len(combinations)
        original_target = targets[(block // len(RENDERINGS) + combination_index) % len(targets)]
        same = base % PAIR_SAME_LABEL_INTERVAL == 0
        changed_target = original_target if same else targets[(targets.index(original_target) + 1) % len(targets)]
        count_conditions = 2 + (block // (len(targets) * len(RENDERINGS))) % 4
        first = _structure_parameters(domain, combinator, facts, count_conditions, original_target, base * 2)
        second = _structure_parameters(domain, combinator, facts, count_conditions, changed_target, base * 2 + 1)
        pair_id = stable_id(SOURCE, "structure-pair", split, domain, combinator, base, facts)
        style = block % len(RENDERINGS)
        rows.append(_structure_row(split, domain, combinator, facts, first, pair_id,
                                   "original", base * 2, style, same))
        rows.append(_structure_row(split, domain, combinator, facts, second, pair_id,
                                   "edit_keeps_label" if same else "edit_changes_label",
                                   base * 2 + 1, style, same))
    return rows


def build_heldout_rows(pairs_per_combination: int = 60) -> list[dict]:
    rows = []
    for combo_index, (domain, combinator) in enumerate(HELD_OUT_COMBINATIONS):
        targets = _targets_for(combinator)
        for base in range(pairs_per_combination):
            facts = _facts_for(domain, "heldout", combo_index * 10_000 + base)
            original = targets[base % len(targets)]
            changed = targets[(targets.index(original) + 1) % len(targets)]
            condition_count = 2 + (base // len(targets)) % 4
            first = _structure_parameters(domain, combinator, facts, condition_count, original, base * 2)
            second = _structure_parameters(domain, combinator, facts, condition_count, changed, base * 2 + 1)
            pair_id = stable_id(SOURCE, "heldout-pair", domain, combinator, base, facts)
            rows.append(_structure_row("test_ood", domain, combinator, facts, first, pair_id,
                                       "original", base * 2, HELD_OUT_RENDERING, False, True))
            rows.append(_structure_row("test_ood", domain, combinator, facts, second, pair_id,
                                       "edit_changes_label", base * 2 + 1,
                                       HELD_OUT_RENDERING, False, True))
    return rows


def _request(row: dict) -> dict:
    question = row["question"]
    if question["type"] == "choice":
        keys = row["audit"]["original_candidate_keys"]
    else:
        keys = list(question["criteria"])
    request = {
        "id": row["id"],
        "state": row["state"],
        "kind": question["type"],
        "instructions": question["instructions"],
        "candidates": [
            {"key": key, "description": description}
            for key, description in zip(keys, question["criteria"].values())
        ],
        "images": [],
        "videos": [],
    }
    parse_request(request)
    return request


def _key(row: dict) -> dict:
    label = _semantic_label(row)
    request = _request(row)
    return {
        "id": row["id"],
        "dataset": "rule_structures_heldout",
        "variant": row["audit"]["pair_variant"],
        "base_id": row["audit"]["pair_id"],
        "domain": row["audit"]["domain"],
        "combinator": row["audit"]["combinator"],
        "condition_count": row["audit"]["condition_count"],
        "rendering": row["audit"]["rendering"],
        "label": label,
        "original_label": label if row["audit"]["pair_variant"] == "original" else None,
        "candidate_keys": [candidate["key"] for candidate in request["candidates"]],
    }


def _five_grams(text: str) -> set[tuple[str, ...]]:
    words = WORD.findall(text.casefold())
    return {tuple(words[index:index + 5]) for index in range(max(0, len(words) - 4))}


def leakage_overlaps(generated_rules: Iterable[str], reference_rules: Iterable[tuple[str, str]]) -> list[dict]:
    reference_index: dict[tuple[str, ...], set[str]] = collections.defaultdict(set)
    for reference_id, rule in reference_rules:
        for gram in _five_grams(rule):
            reference_index[gram].add(reference_id)
    overlaps = []
    reference_grams = set(reference_index)
    for generated_index, rule in enumerate(generated_rules):
        for gram in sorted(_five_grams(rule) & reference_grams):
            overlaps.append({
                "generated_index": generated_index,
                "reference_ids": sorted(reference_index[gram]),
                "five_gram": " ".join(gram),
            })
    return overlaps


def _existing_rows(root: Path) -> list[dict]:
    rows = []
    for split in ("train", "dev", "calibration", "policy", "test_id", "test_ood"):
        path = root / "data" / "prepared" / f"{split}.jsonl"
        if path.is_file():
            with path.open(encoding="utf-8") as stream:
                rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def _probe_states(root: Path) -> set[str]:
    path = root / ".audit" / "rule-probe" / "rule-probe.requests.jsonl"
    if not path.is_file():
        raise FileNotFoundError("rule-probe request file is required for collision exclusion")
    with path.open(encoding="utf-8") as stream:
        return {json.loads(line)["state"] for line in stream if line.strip()}


def _reference_rules(rows: Iterable[dict]) -> list[tuple[str, str]]:
    references = []
    for row in rows:
        if row["split"] not in {"test_id", "test_ood"}:
            continue
        rule = row.get("audit", {}).get("rule")
        references.append((row["id"], rule if isinstance(rule, str) and rule.strip() else row["state"]))
    return references


def recompute_label(row: dict) -> str:
    audit = row["audit"]
    facts = audit["facts"]
    if audit["part"] == "A":
        if audit["domain"] == "transfer":
            semantic = transfer_label(facts, audit["rule_parameters"])
        elif audit["domain"] == "card_purchase":
            semantic = card_label(facts, audit["rule_parameters"])
        else:
            semantic = incident_label(facts, audit["rule_parameters"])
    else:
        semantic = evaluate_structure(audit["rule_parameters"], facts)
    return _serialized_label(row, semantic)


def _label_counts(rows: Iterable[dict], field: str) -> dict:
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in rows:
        value = row["question"]["type"] if field == "question_type" else row["audit"][field]
        counts[str(value)][_semantic_label(row)] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(counts.items())}


def _surface_key(row: dict) -> tuple:
    audit = row["audit"]
    word_bucket = len(WORD.findall(audit["rule"])) // 10
    return (word_bucket, audit["condition_count"], audit["combinator"], audit["domain"],
            audit["rendering"], row["question"]["type"])


def _surface_baseline(train: list[dict], dev: list[dict]) -> dict:
    feature_labels: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    type_labels: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in train:
        label = _semantic_label(row)
        feature_labels[_surface_key(row)][label] += 1
        type_labels[row["question"]["type"]][label] += 1
    correct = majority_correct = 0
    per_type = {}
    for question_type in sorted(type_labels):
        selected = [row for row in dev if row["question"]["type"] == question_type]
        fallback = max(sorted(type_labels[question_type].items()), key=lambda item: item[1])[0]
        type_correct = type_majority_correct = 0
        for row in selected:
            counts = feature_labels.get(_surface_key(row))
            prediction = max(sorted(counts.items()), key=lambda item: item[1])[0] if counts else fallback
            actual = _semantic_label(row)
            type_correct += prediction == actual
            type_majority_correct += fallback == actual
        correct += type_correct
        majority_correct += type_majority_correct
        per_type[question_type] = {
            "questions": len(selected),
            "surface_accuracy": type_correct / len(selected),
            "majority_accuracy": type_majority_correct / len(selected),
            "delta": type_correct / len(selected) - type_majority_correct / len(selected),
        }
    return {
        "features": ["rule_length_bucket", "condition_count", "combinator", "domain", "rendering", "question_type"],
        "questions": len(dev),
        "accuracy": correct / len(dev),
        "per_type_majority_accuracy": majority_correct / len(dev),
        "delta": correct / len(dev) - majority_correct / len(dev),
        "per_type": per_type,
        "pass": all(value["delta"] <= 0.05 + 1e-12 for value in per_type.values()),
    }


def _evidence_only_pair_bound(rows: list[dict]) -> dict:
    pairs: dict[str, list[str]] = collections.defaultdict(list)
    for row in rows:
        pairs[row["audit"]["pair_id"]].append(_semantic_label(row))
    if any(len(labels) != 2 for labels in pairs.values()):
        raise AssertionError("every evidence group must contain exactly one pair")
    correct = sum(max(collections.Counter(labels).values()) for labels in pairs.values())
    accuracy = correct / (2 * len(pairs))
    return {
        "method": "best constant prediction for each evidence-identical pair after removing the rule",
        "pairs": len(pairs),
        "questions": 2 * len(pairs),
        "upper_bound_accuracy": accuracy,
        "chance_reference": 0.5,
        "pass": accuracy <= 0.56,
    }


def _shortcut_receipt(rows: list[dict]) -> dict:
    train = [row for row in rows if row["split"] == "train"]
    dev = [row for row in rows if row["split"] == "dev"]
    baseline = _surface_baseline(train, dev)
    evidence = _evidence_only_pair_bound(rows)
    receipt = {
        "status": "VERIFIED" if baseline["pass"] and evidence["pass"] else "NOT_VERIFIED",
        "label_balance": {
            "question_type": _label_counts(rows, "question_type"),
            "combinator": _label_counts(rows, "combinator"),
            "domain": _label_counts(rows, "domain"),
            "condition_count": _label_counts(rows, "condition_count"),
        },
        "surface_feature_baseline_dev": baseline,
        "evidence_only_pair_baseline": evidence,
    }
    if receipt["status"] != "VERIFIED":
        raise AssertionError(f"shortcut checks failed: {receipt}")
    return receipt


def wording_lint(rows: list[dict]) -> dict:
    forbidden_counts = {phrase: 0 for phrase in FORBIDDEN_WORDING}
    numeric_without_inclusivity = []
    boundary_counts = collections.Counter()
    numeric_checked = 0
    for row in rows:
        audit = row["audit"]
        rule = audit["rule"].casefold()
        for phrase in FORBIDDEN_WORDING:
            forbidden_counts[phrase] += rule.count(phrase)
        if audit["part"] in {"B", "external"}:
            for condition in audit["rule_parameters"]["conditions"]:
                if condition["operator"] not in {"at_most", "at_least"}:
                    continue
                numeric_checked += 1
                rendered = _render_condition(condition, audit["rendering"]).casefold()
                inclusive = INCLUSIVE_AT_MOST if condition["operator"] == "at_most" else INCLUSIVE_AT_LEAST
                if rendered not in rule or not any(phrase in rendered for phrase in inclusive):
                    numeric_without_inclusivity.append({
                        "id": row["id"],
                        "field": condition["field"],
                        "operator": condition["operator"],
                    })
                distance = audit["facts"][condition["field"]] - condition["value"]
                if distance == 0:
                    boundary_counts["at_threshold"] += 1
                elif distance == -1:
                    boundary_counts["one_below"] += 1
                elif distance == 1:
                    boundary_counts["one_above"] += 1
                else:
                    boundary_counts["farther"] += 1
            continue
        domain = audit["domain"]
        if domain == "transfer":
            numeric_checked += 3
            expected = (
                ("more than", "greater than"),
                INCLUSIVE_AT_LEAST,
                INCLUSIVE_AT_MOST,
            )
            for phrases in expected:
                if not any(phrase in rule for phrase in phrases):
                    numeric_without_inclusivity.append({"id": row["id"], "domain": domain})
        elif domain == "card_purchase" and "amount_limit" in audit["rule_parameters"]["required_constraints"]:
            numeric_checked += 1
            if not any(phrase in rule for phrase in INCLUSIVE_AT_MOST):
                numeric_without_inclusivity.append({"id": row["id"], "domain": domain})
        elif domain == "incident":
            numeric_checked += 1
            if "at least" not in rule:
                numeric_without_inclusivity.append({"id": row["id"], "domain": domain})
    boundary = {
        key: boundary_counts[key]
        for key in ("at_threshold", "one_below", "one_above", "farther")
    }
    boundary_total = sum(boundary.values())
    near_boundary = (
        boundary["at_threshold"] + boundary["one_below"] + boundary["one_above"]
    )
    boundary["near_boundary_total"] = near_boundary
    boundary["near_boundary_share"] = near_boundary / boundary_total if boundary_total else 0.0
    boundary["farther_share"] = boundary["farther"] / boundary_total if boundary_total else 0.0
    boundary_pass = (
        all(boundary[key] > 0 for key in ("at_threshold", "one_below", "one_above", "farther"))
        and 0.30 <= boundary["near_boundary_share"] <= 0.40
    )
    passed = (
        not any(forbidden_counts.values())
        and not numeric_without_inclusivity
        and boundary_pass
    )
    return {
        "status": "VERIFIED" if passed else "NOT_VERIFIED",
        "rules_checked": len(rows),
        "forbidden_phrase_counts": forbidden_counts,
        "numeric_conditions_checked": numeric_checked,
        "numeric_comparisons_without_inclusivity": len(numeric_without_inclusivity),
        "numeric_comparison_issues": numeric_without_inclusivity,
        "numeric_boundary_facts": boundary,
        "numeric_boundary_target": {
            "near_boundary_definition": "fact equals threshold or is exactly one unit either side",
            "near_boundary_share_minimum": 0.30,
            "near_boundary_share_maximum": 0.40,
            "farther_definition": "absolute fact-to-threshold distance is greater than one unit",
        },
        "numeric_boundary_coverage_pass": boundary_pass,
    }


def _generation_receipt(rows: list[dict]) -> dict:
    counts = lambda values: dict(sorted(collections.Counter(values).items()))
    rendering_sets: dict[str, set[int]] = collections.defaultdict(set)
    for row in rows:
        if row["split"] == "train" and row["audit"]["part"] == "B":
            rendering_sets[row["audit"]["structure_family"]].add(row["audit"]["rendering"])
    return {
        "status": "VERIFIED",
        "source": SOURCE,
        "questions": len(rows),
        "pairs": len(rows) // 2,
        "splits": counts(row["split"] for row in rows),
        "parts": counts(row["audit"]["part"] for row in rows),
        "combinators": counts(row["audit"]["combinator"] for row in rows),
        "domains": counts(row["audit"]["domain"] for row in rows),
        "question_types": counts(row["question"]["type"] for row in rows),
        "condition_counts": counts(str(row["audit"]["condition_count"]) for row in rows),
        "renderings_per_structure": {
            key: sorted(values) for key, values in sorted(rendering_sets.items())
        },
        "pair_relations": counts(row["audit"]["label_relation"] for row in rows[::2]),
        "template_start_index": TEMPLATE_START_INDEX,
        "probe_start_index": PROBE_START_INDEX,
        "probe_bases_per_template": 120,
        "numeric_field_specs": {
            field: dict(spec) for field, spec in sorted(NUMERIC_FIELD_SPECS.items())
        },
    }


def _external_manifest(root: Path, rows: list[dict], output: Path) -> dict:
    requests = [_request(row) for row in rows]
    keys = [_key(row) for row in rows]
    originals = {key["base_id"]: key["label"] for key in keys if key["variant"] == "original"}
    for key in keys:
        key["original_label"] = originals[key["base_id"]]
        if key["variant"] == "edit_changes_label" and key["label"] == key["original_label"]:
            raise AssertionError("held-out label-changing sibling did not change")
    request_path = output / f"{SOURCE}.requests.jsonl"
    key_path = output / f"{SOURCE}.key.jsonl"
    write_jsonl(request_path, requests)
    write_jsonl(key_path, keys)
    return {
        "status": "VERIFIED",
        "dataset": "rule_structures_heldout",
        "pairs": len(rows) // 2,
        "questions": len(rows),
        "reserved_combinator_domain_combinations": [list(value) for value in HELD_OUT_COMBINATIONS],
        "reserved_rendering": HELD_OUT_RENDERING,
        "canonical_renderings": list(RENDERINGS),
        "requests": request_path.relative_to(root).as_posix(),
        "key": key_path.relative_to(root).as_posix(),
        "request_sha256": sha256_file(request_path),
        "key_sha256": sha256_file(key_path),
        "scorer": "tools/score_predictions.py",
    }


def build_rows() -> list[dict]:
    rows = []
    for split in ("train", "dev", "calibration"):
        rows.extend(build_template_rows(split, PART_A_COUNTS[split]))
        rows.extend(build_structure_rows(split, PART_B_COUNTS[split]))
    if collections.Counter(row["split"] for row in rows) != collections.Counter(PART_QUOTAS[PART]):
        raise AssertionError("rule-variant split counts differ from PART_QUOTAS")
    for row in rows:
        validate_record(row)
        if recompute_label(row) != row["target"]["label"]:
            raise AssertionError(f"label recomputation failed: {row['id']}")
    if len({row["id"] for row in rows}) != len(rows):
        raise AssertionError("rule-variant record IDs are not unique")
    rendering_sets: dict[str, set[int]] = collections.defaultdict(set)
    for row in rows:
        if row["split"] == "train" and row["audit"]["part"] == "B":
            rendering_sets[row["audit"]["structure_family"]].add(row["audit"]["rendering"])
    if any(styles != set(RENDERINGS) for styles in rendering_sets.values()):
        raise AssertionError("every admitted random structure family must use all four renderings")
    return rows


def prepare(root: Path) -> dict:
    root = root.resolve()
    audit = root / ".audit" / SOURCE
    audit.mkdir(parents=True, exist_ok=True)
    existing = [
        row for row in _existing_rows(root)
        if row.get("provenance", {}).get("source") != SOURCE
    ]
    existing_states = {row["state"] for row in existing}
    probe_states = _probe_states(root)
    rows = build_rows()
    generated_states = {row["state"] for row in rows}
    if len(generated_states) != len(rows):
        raise AssertionError("generated states are not unique")
    collisions = generated_states & (existing_states | probe_states)
    if collisions:
        raise AssertionError(f"generated state collision count: {len(collisions)}")

    heldout_rows = build_heldout_rows()
    heldout_states = {row["state"] for row in heldout_rows}
    if heldout_states & (existing_states | probe_states | generated_states):
        raise AssertionError("held-out structure state collided with canonical or probe data")
    training_combinations = {(row["audit"]["domain"], row["audit"]["combinator"]) for row in rows
                             if row["audit"]["part"] == "B"}
    if training_combinations & set(HELD_OUT_COMBINATIONS):
        raise AssertionError("held-out combinator/domain combination leaked into canonical data")
    if {row["audit"]["rendering"] for row in rows} & {HELD_OUT_RENDERING}:
        raise AssertionError("held-out rendering leaked into canonical data")

    all_generated = [*rows, *heldout_rows]
    generated_rules = [row["audit"]["rule"] for row in all_generated]
    references = _reference_rules(existing)
    overlaps = leakage_overlaps(generated_rules, references)
    forbidden_phrases = (
        "cash access",
        "bank atm",
        "available balance",
        "requested cash",
        "dispute deadline",
        "duplicate charge",
        "statement age",
        "merchant exposure",
        "cross border spread",
        "control bypass",
    )
    concept_hits = [
        {"generated_index": index, "phrase": phrase}
        for index, rule in enumerate(generated_rules)
        for phrase in forbidden_phrases
        if phrase in " ".join(WORD.findall(rule.casefold()))
    ]
    leakage = {
        "status": "VERIFIED" if not overlaps and not concept_hits else "NOT_VERIFIED",
        "n": 5,
        "generated_rules": len(generated_rules),
        "reference_rules": len(references),
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
        "forbidden_concept_hit_count": len(concept_hits),
        "forbidden_concept_hits": concept_hits,
        "reference_splits": ["test_id", "test_ood"],
        "forbidden_heldout_concepts": [
            "cash access / ATM time-place-balance",
            "dispute deadline and priority",
            "merchant network exposure",
        ],
    }
    write_json(audit / "leakage-check.json", leakage)
    if overlaps or concept_hits:
        raise AssertionError(
            f"leakage guard failed: five_grams={len(overlaps)} concepts={len(concept_hits)}"
        )

    part_path = root / "data" / "parts" / f"{PART}.jsonl"
    write_jsonl(part_path, rows)
    source_path = root / "data" / "sources" / f"{SOURCE}.json"
    write_json(source_path, {
        "source": SOURCE,
        "revision": GENERATOR_VERSION,
        "repository": "local deterministic generator",
        "upstream_split": None,
        "license": "Project-generated synthetic data",
        "license_url": None,
        "processing": "Counterfactual pairs with code-computed labels; no model or human judgments.",
        "known_limits": ["Synthetic grammar coverage is finite.", "External held-out rows are not canonical training data."],
    })
    generation = _generation_receipt(rows)
    wording = wording_lint(all_generated)
    if wording["status"] != "VERIFIED":
        raise AssertionError(f"wording checks failed: {wording}")
    generation["wording_lint"] = wording
    generation.update({
        "part_path": part_path.relative_to(root).as_posix(),
        "part_sha256": sha256_file(part_path),
        "source_manifest": source_path.relative_to(root).as_posix(),
        "source_manifest_sha256": sha256_file(source_path),
        "state_collisions_with_existing": 0,
        "state_collisions_with_probe": 0,
    })
    write_json(audit / "generation.json", generation)
    shortcut = _shortcut_receipt(rows)
    shortcut.update({"part_sha256": sha256_file(part_path), "dev_questions": PART_QUOTAS[PART]["dev"]})
    write_json(audit / "shortcut-check.json", shortcut)

    external_output = root / ".audit" / "rule-structures-heldout"
    external_output.mkdir(parents=True, exist_ok=True)
    manifest = _external_manifest(root, heldout_rows, external_output)
    write_json(external_output / "manifest.json", manifest)
    return {
        "status": "VERIFIED",
        "part": PART,
        "source": SOURCE,
        "rows": len(rows),
        "splits": dict(PART_QUOTAS[PART]),
        "generation_receipt": (audit / "generation.json").relative_to(root).as_posix(),
        "leakage_receipt": (audit / "leakage-check.json").relative_to(root).as_posix(),
        "shortcut_receipt": (audit / "shortcut-check.json").relative_to(root).as_posix(),
        "external_manifest": (external_output / "manifest.json").relative_to(root).as_posix(),
    }


def compile_external_eval(root: Path) -> dict:
    root = root.resolve()
    manifest_path = root / ".audit" / "rule-structures-heldout" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requests_path = root / manifest["requests"]
    compiler = NativeCompiler.from_pretrained(root, media_root=root / "data")
    lengths = []
    with requests_path.open(encoding="utf-8") as stream:
        for line in stream:
            request = parse_request(json.loads(line))
            batch = compiler.compile((request,))
            lengths.append(int(batch.expanded_lengths[0]))
    if not lengths or max(lengths) > MAX_POSITIONS:
        raise AssertionError("held-out structure evaluation exceeds the native position budget")
    compilation = {
        "status": "VERIFIED",
        "questions": len(lengths),
        "max_positions": max(lengths),
        "position_limit": MAX_POSITIONS,
        "processor_id": compiler.processor_id,
    }
    manifest["compilation"] = compilation
    write_json(manifest_path, manifest)
    return compilation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--compile-eval", action="store_true")
    arguments = parser.parse_args()
    result = compile_external_eval(arguments.root) if arguments.compile_eval else prepare(arguments.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
