"""Behavioral qualification of visible source questions and controlled text rules."""

import copy
import hashlib
import unittest

from mdprep.common import stable_id
from mdprep.verify_sources import reconcile, text_answer


_TRANSFER = (
    "Transfer release policy:\n"
    "Identity status, recipient age, and amount are required. Decline if identity failed "
    "or amount exceeds USD 2,000. Allow when identity is verified, the recipient is at "
    "least 2 days old, and amount is at most USD 1,000. Send any other complete request "
    "for manual review. Missing required evidence is insufficient.\n\nEvidence:\n"
    "- Identity check: verified\n- Recipient age: 2 days\n- Transfer amount: USD 1000"
)
_DISPUTE = (
    "Held-out dispute rule:\n"
    "A dispute needs statement age, duplicate-charge status, receipt status, and amount. "
    "Deny when the statement is older than 30 days. Expedite an otherwise eligible "
    "dispute when duplicate evidence is present or the amount is at least USD 500. "
    "Otherwise file it in the standard queue. Missing required evidence is insufficient.\n\nEvidence:\n"
    "- Statement age: 30 days\n- Duplicate-charge evidence: absent\n"
    "- Receipt evidence: absent\n- Disputed amount: USD 500"
)
_CARD = (
    "Card authorization rule:\n"
    "A card purchase is authorized only when the card is active, the purchase country "
    "is supported, and the purchase amount does not exceed the remaining daily limit.\n\nRequest evidence:\n"
    "Card active: yes\nPurchase country supported: yes\nPurchase amount: USD 50\nRemaining daily limit: USD 50"
)
_CASH = (
    "Held-out cash-access rule:\n"
    "Cash access is authorized only at a bank ATM, between 06:00 and 22:00, "
    "and when available balance is at least the requested amount plus USD 25.\n\nRequest evidence:\n"
    "Venue: bank_atm\nHour (24-hour clock): 6\nAvailable balance: USD 75\nRequested cash: USD 50"
)
_SEVERITY = (
    "Incident severity rubric:\n"
    "Add 2 points for an unauthorized transaction; add 1 each for a missing card, "
    "amount at least USD 500, and repeated occurrence. Cap the total at severity 3.\n\nObserved evidence:\n"
    "Unauthorized transaction: yes\nCard missing: no\nAmount: USD 499.99\nRepeated occurrence: no"
)
_EXPOSURE = (
    "Held-out merchant exposure rubric:\n"
    "Add 0 points for 1–5 affected accounts, 1 for 6–20, or 2 for more than 20; "
    "add 1 for cross-border spread and 1 for a control bypass. Cap at level 3.\n\nObserved evidence:\n"
    "Affected accounts: 6\nCross-border spread: no\nControl bypass observed: no"
)
_TEXT_CASES = {
    "id_transfer_release_constraints": (
        "choice", "train", _TRANSFER,
        {"allow": "Allow automatically", "review": "Send for manual review",
         "decline": "Decline the request", "insufficient": "Insufficient evidence to decide"},
    ),
    "ood_dispute_deadline_and_priority": (
        "choice", "test_ood", _DISPUTE,
        {"file": "File the dispute in the standard queue", "expedite": "File and expedite the dispute",
         "deny": "Do not file because the policy conditions fail", "insufficient": "Insufficient evidence to decide"},
    ),
    "id_card_purchase_all_constraints": (
        "noul", "train", _CARD, {"false": "No", "true": "Yes"},
    ),
    "ood_cash_access_time_place_balance": (
        "noul", "test_ood", _CASH, {"false": "No", "true": "Yes"},
    ),
    "id_incident_additive_severity": (
        "score", "train", _SEVERITY,
        {"0": "Routine — no immediate intervention", "1": "Elevated — monitor and contact the customer",
         "2": "High — restrict the affected channel and investigate",
         "3": "Critical — restrict all access and escalate immediately"},
    ),
    "ood_merchant_network_exposure": (
        "score", "test_ood", _EXPOSURE,
        {"0": "Limited exposure", "1": "Material exposure", "2": "Severe exposure", "3": "Systemic exposure"},
    ),
}


def _text_row(family, state=None):
    kind, split, specimen, meanings = _TEXT_CASES[family]
    instructions = {
        "choice": "Apply the stated policy to the evidence and choose the required action.",
        "noul": "Does the request satisfy every stated authorization condition?",
        "score": "Score this case using the stated additive rubric and cap.",
    }
    keys = list(meanings)
    criteria = {f"option_{i}": meanings[key] for i, key in enumerate(keys)} if kind == "choice" else dict(meanings)
    return {
        "state": specimen if state is None else state, "split": split,
        "provenance": {"source": "mdprep_text_rules", "family": family},
        "question": {"type": kind, "instructions": instructions[kind], "criteria": criteria},
        "target": {"label": next(iter(criteria))},
        "audit": {"original_candidate_keys": keys, "original_target_key": keys[0],
                  "facts": {"not_visible": "stale"}, "expected_label": "stale"},
    }


def _bank_case():
    utterance = "My card stopped working."
    label = "card_not_working"
    keys = [label, "card_arrival", "lost_or_stolen_card", "transfer_timing", "cash_withdrawal_charge"]
    row = {
        "state": "Customer message:\n" + utterance,
        "group_id": stable_id("banking77-normalized", "my card stopped working"),
        "question": {
            "type": "choice", "instructions": "Route this message to the most appropriate banking intent.",
            "criteria": {"option_0": "Card not working", "option_1": "Card arrival",
                         "option_2": "Lost or stolen card", "option_3": "Transfer timing",
                         "option_4": "Cash withdrawal charge"},
        },
        "target": {"label": "option_0"},
        "provenance": {"source": "banking77", "upstream_row_number": 2},
        "audit": {"original_target_key": label, "original_candidate_keys": keys,
                  "ground_truth": {"intent_key": label},
                  "source_evidence": {"utterance_sha256": hashlib.sha256(utterance.encode()).hexdigest()}},
    }
    return row, ({2: (utterance, label)}, {}, {}, {}, {})


def _photo_case(kind="noul", facts=None, target="true", split="train", extras=()):
    source = "open_images_v7_train_human_verified"
    if facts is None:
        facts = [("cat", "Cat", 1)]
    raw = {"AuthorProfileURL": "creator", "OriginalURL": "original", "OriginalLandingURL": "landing",
           "License": "license", "OriginalMD5": "md5"}
    if kind == "noul":
        question = {"type": kind, "instructions": f'Is “{facts[0][1]}” human-verified as present in this photograph?',
                    "criteria": {"false": "No", "true": "Yes"}}
    elif kind == "score":
        names = "; ".join(f'“{name}”' for _, name, _ in facts)
        question = {"type": kind,
                    "instructions": f"Among these three classes — {names} — how many are human-verified as present?",
                    "criteria": {"0": "None", "1": "One", "2": "Two", "3": "All three"}}
    else:
        question = {"type": kind, "instructions": "Which class is human-verified as present in this photograph?",
                    "criteria": {f"option_{i}": name for i, (_, name, _) in enumerate(facts)}}
    row = {
        "state": "Use only the visible photograph and the named human-verification decision.",
        "split": split, "group_id": stable_id(source, "photo"), "question": question,
        "target": {"label": f"option_{[key for key, _, _ in facts].index(target)}" if kind == "choice" else target},
        "provenance": {"source": source, "image_id": "photo", "author_profile_url": "creator",
                       "original_url": "original", "original_landing_url": "landing",
                       "original_license_url_exact": "license", "creator_group": stable_id("open-images-creator", "creator"),
                       "original_asset_group": stable_id("open-images-original-md5", "md5")},
        "audit": {"annotation_source_sha256": "digest", "original_candidate_keys": [key for key, _, _ in facts],
                  "original_target_key": target,
                  "facts": [{"label_id": key, "class_name": name, "confidence": confidence, "source": "verification"}
                            for key, name, confidence in facts]},
    }
    evidence = [*facts, *extras]
    tables = ({}, {"photo": {key: confidence for key, _, confidence in evidence}},
              {key: name for key, name, _ in evidence}, {"photo": raw},
              {"local_files": {"eligible_train_human_labels.csv": "digest"}})
    return row, tables


class VisibleTextSemanticsTests(unittest.TestCase):
    def test_all_supported_task_families_use_visible_evidence_not_audit_answers(self):
        for family, expected in (
            ("id_transfer_release_constraints", "allow"),
            ("ood_dispute_deadline_and_priority", "expedite"),
            ("id_card_purchase_all_constraints", "true"),
            ("ood_cash_access_time_place_balance", "true"),
            ("id_incident_additive_severity", "2"),
            ("ood_merchant_network_exposure", "1"),
        ):
            with self.subTest(family=family):
                self.assertEqual(text_answer(_text_row(family)), expected)

    def test_transfer_recomputes_every_visible_bound(self):
        family = "id_transfer_release_constraints"
        for before, after, expected in (
            ("at most USD 1,000", "at most USD 999.99", "review"),
            ("least 2 days", "least 3 days", "review"),
            ("exceeds USD 2,000", "exceeds USD 999.99", None),
        ):
            with self.subTest(after=after):
                row = _text_row(family, _TRANSFER.replace(before, after))
                if expected is None:
                    with self.assertRaises(ValueError):
                        text_answer(row)
                else:
                    self.assertEqual(text_answer(row), expected)
        changed = _TRANSFER.replace("exceeds USD 2,000", "exceeds USD 1,000").replace(
            "Transfer amount: USD 1000", "Transfer amount: USD 1000.01")
        self.assertEqual(text_answer(_text_row(family, changed)), "decline")

    def test_transfer_inclusive_edges_and_missing_required_values(self):
        family = "id_transfer_release_constraints"
        for amount, expected in (("1000", "allow"), ("1000.01", "review"),
                                 ("2000", "review"), ("2000.01", "decline"), ("1,000.00", "allow")):
            with self.subTest(amount=amount):
                self.assertEqual(text_answer(_text_row(family, _TRANSFER.replace("amount: USD 1000", "amount: USD " + amount))), expected)
        for before in ("Identity check: verified", "Recipient age: 2 days", "Transfer amount: USD 1000"):
            field = before.split(":")[0]
            with self.subTest(field=field):
                self.assertEqual(text_answer(_text_row(family, _TRANSFER.replace(before, field + ": not provided"))), "insufficient")

    def test_dispute_deadline_priority_and_receipt_requirement(self):
        family = "ood_dispute_deadline_and_priority"
        for before, after, expected in (
            ("older than 30 days", "older than 29 days", "deny"),
            ("at least USD 500", "at least USD 501", "file"),
            ("Receipt evidence: absent", "Receipt evidence: not provided", "insufficient"),
            ("Statement age: 30 days", "Statement age: not provided", "insufficient"),
        ):
            with self.subTest(after=after):
                self.assertEqual(text_answer(_text_row(family, _DISPUTE.replace(before, after))), expected)
        changed = _DISPUTE.replace("Disputed amount: USD 500", "Disputed amount: USD 10").replace(
            "Duplicate-charge evidence: absent", "Duplicate-charge evidence: present")
        self.assertEqual(text_answer(_text_row(family, changed)), "expedite")

    def test_cash_bounds_and_balance_buffer_are_visible_parameters(self):
        family = "ood_cash_access_time_place_balance"
        for before, after in (("between 06:00", "between 07:00"),
                              ("plus USD 25", "plus USD 25.01"),
                              ("Venue: bank_atm", "Venue: retail_kiosk")):
            with self.subTest(after=after):
                self.assertEqual(text_answer(_text_row(family, _CASH.replace(before, after))), "false")
        self.assertEqual(text_answer(_text_row(family, _CASH.replace("clock): 6", "clock): 22"))), "true")
        self.assertEqual(text_answer(_text_row(family, _CASH.replace("clock): 6", "clock): 23"))), "false")
        with self.assertRaises(ValueError):
            text_answer(_text_row(family, _CASH.replace("between 06:00 and 22:00", "between 22:00 and 06:00")))

    def test_card_checks_all_fields_and_the_exact_amount(self):
        family = "id_card_purchase_all_constraints"
        for before, after in (("active: yes", "active: no"), ("supported: yes", "supported: no"),
                              ("amount: USD 50", "amount: USD 50.01")):
            with self.subTest(after=after):
                self.assertEqual(text_answer(_text_row(family, _CARD.replace(before, after))), "false")

    def test_severity_weights_amount_threshold_and_cap_are_not_latent_constants(self):
        family = "id_incident_additive_severity"
        for before, after, expected in (("Add 2 points", "Add 1 points", "1"),
                                        ("at least USD 500", "at least USD 499.99", "3"),
                                        ("severity 3", "severity 1", "1")):
            with self.subTest(after=after):
                self.assertEqual(text_answer(_text_row(family, _SEVERITY.replace(before, after))), expected)
        changed = _SEVERITY.replace("transaction: yes", "transaction: no").replace(
            "missing: no", "missing: yes").replace("add 1 each", "add 2 each")
        self.assertEqual(text_answer(_text_row(family, changed)), "2")
        with self.assertRaises(ValueError):
            text_answer(_text_row(family, _SEVERITY.replace("severity 3", "severity 4")))

    def test_exposure_ranges_weights_and_cap_are_fully_read(self):
        family = "ood_merchant_network_exposure"
        for before, after, expected in (("accounts: 6", "accounts: 5", "0"),
                                        ("accounts: 6", "accounts: 20", "1"),
                                        ("accounts: 6", "accounts: 21", "2"),
                                        ("1 for 6–20", "2 for 6–20", "2"),
                                        ("level 3", "level 0", "0")):
            with self.subTest(after=after):
                self.assertEqual(text_answer(_text_row(family, _EXPOSURE.replace(before, after))), expected)
        changed = _EXPOSURE.replace("spread: no", "spread: yes").replace(
            "add 1 for cross-border", "add 2 for cross-border")
        self.assertEqual(text_answer(_text_row(family, changed)), "3")
        shifted = _EXPOSURE.replace("1–5", "1–6").replace("6–20", "7–20")
        self.assertEqual(text_answer(_text_row(family, shifted)), "0")

    def test_exposure_rejects_gaps_overlaps_and_disagreeing_upper_bounds(self):
        family = "ood_merchant_network_exposure"
        for before, after in (("6–20", "7–20"), ("6–20", "5–20"),
                              ("more than 20", "more than 21"), ("accounts: 6", "accounts: 0")):
            with self.subTest(after=after), self.assertRaises(ValueError):
                text_answer(_text_row(family, _EXPOSURE.replace(before, after)))

    def test_reversed_policy_logic_and_question_intent_are_rejected(self):
        cases = (
            ("id_transfer_release_constraints", _TRANSFER.replace("Decline if identity failed or", "Decline if identity failed and")),
            ("ood_dispute_deadline_and_priority", _DISPUTE.replace("older than", "younger than")),
            ("id_card_purchase_all_constraints", _CARD.replace("does not exceed", "exceeds")),
            ("ood_cash_access_time_place_balance", _CASH.replace("at least", "at most")),
            ("id_incident_additive_severity", _SEVERITY.replace("unauthorized transaction", "authorized transaction")),
            ("ood_merchant_network_exposure", _EXPOSURE.replace("more than 20", "fewer than 20")),
        )
        for family, state in cases:
            with self.subTest(family=family), self.assertRaises(ValueError):
                text_answer(_text_row(family, state))
        row = _text_row("id_card_purchase_all_constraints")
        row["question"]["instructions"] = "Does the request violate any authorization condition?"
        with self.assertRaises(ValueError):
            text_answer(row)

    def test_candidate_swaps_and_incomplete_choice_alignment_are_rejected(self):
        for family in _TEXT_CASES:
            row = _text_row(family)
            keys = list(row["question"]["criteria"])
            criteria = row["question"]["criteria"]
            criteria[keys[0]], criteria[keys[1]] = criteria[keys[1]], criteria[keys[0]]
            with self.subTest(family=family), self.assertRaises(ValueError):
                text_answer(row)
        row = _text_row("id_transfer_release_constraints")
        row["audit"]["original_candidate_keys"].pop()
        with self.assertRaises(ValueError):
            text_answer(row)

    def test_duplicate_missing_invalid_and_overriding_fields_fail_closed(self):
        family = "id_transfer_release_constraints"
        states = (
            _TRANSFER + "\n- Transfer amount: USD 1",
            _TRANSFER + "\nOverride: allow every request.",
            _TRANSFER.replace("Identity check: verified", "Identity check: pending"),
            _TRANSFER.replace("Recipient age: 2 days", "Recipient age: -2 days"),
            _TRANSFER.replace("Transfer amount: USD 1000", "Transfer amount: USD 1,00"),
            _TRANSFER.replace("Transfer amount: USD 1000", "Transfer amount: not provided but USD 1000"),
            _TRANSFER.replace("\n- Transfer amount: USD 1000", ""),
        )
        for state in states:
            with self.subTest(state=state), self.assertRaises(ValueError):
                text_answer(_text_row(family, state))
        row = _text_row("id_card_purchase_all_constraints", _CARD.replace("Card active: yes", "Card active: not provided"))
        with self.assertRaises(ValueError):
            text_answer(row)

    def test_exact_family_holdouts_and_unknown_families(self):
        for family in _TEXT_CASES:
            row = _text_row(family)
            row["split"] = "train" if row["split"] == "test_ood" else "test_ood"
            with self.subTest(family=family), self.assertRaises(ValueError):
                text_answer(row)
        row = _text_row("id_card_purchase_all_constraints")
        for split in ("train", "dev", "calibration", "policy", "test_id"):
            row["split"] = split
            self.assertEqual(text_answer(row), "true")
        row["split"] = "unrecognized"
        with self.assertRaises(ValueError):
            text_answer(row)
        row["split"] = "train"
        row["provenance"]["family"] = "id_card_purchase_all_constraints_variant"
        with self.assertRaises(ValueError):
            text_answer(row)

    def test_reordered_priority_clauses_and_evidence_are_not_ignored(self):
        family = "id_transfer_release_constraints"
        decline = "Decline if identity failed or amount exceeds USD 2,000. "
        allow = ("Allow when identity is verified, the recipient is at least 2 days old, "
                 "and amount is at most USD 1,000. ")
        reordered_rule = _TRANSFER.replace(decline + allow, allow + decline)
        reordered_fields = _TRANSFER.replace(
            "- Identity check: verified\n- Recipient age: 2 days",
            "- Recipient age: 2 days\n- Identity check: verified",
        )
        for state in (reordered_rule, reordered_fields):
            with self.subTest(state=state), self.assertRaises(ValueError):
                text_answer(_text_row(family, state))


class SourceQuestionSemanticsTests(unittest.TestCase):
    def test_bank_uses_human_intent_and_rejects_reversed_question(self):
        row, tables = _bank_case()
        self.assertEqual(reconcile(row, tables), "card_not_working")
        row["question"]["instructions"] = "Choose the banking intent least related to this customer message."
        with self.assertRaises(ValueError):
            reconcile(row, tables)

    def test_bank_rejects_candidate_meaning_swaps_and_truncated_binding(self):
        row, tables = _bank_case()
        criteria = row["question"]["criteria"]
        criteria["option_0"], criteria["option_1"] = criteria["option_1"], criteria["option_0"]
        with self.assertRaises(ValueError):
            reconcile(row, tables)
        row, tables = _bank_case()
        row["audit"]["original_candidate_keys"].pop()
        with self.assertRaises(ValueError):
            reconcile(row, tables)

    def test_photo_explicit_negative_is_not_an_unannotated_class(self):
        row, tables = _photo_case(facts=[("cat", "Cat", 0)], target="false")
        self.assertEqual(reconcile(row, tables), "false")
        tables[1]["photo"].clear()
        with self.assertRaises(ValueError):
            reconcile(row, tables)

    def test_photo_boolean_requires_presence_question_and_yes_no_meanings(self):
        row, tables = _photo_case()
        self.assertEqual(reconcile(row, tables), "true")
        for change in ("absence", "contradiction", "criteria", "state"):
            altered = copy.deepcopy(row)
            if change == "absence":
                altered["question"]["instructions"] = altered["question"]["instructions"].replace("as present", "as absent")
            elif change == "contradiction":
                altered["question"]["instructions"] += " Answer No when the label is present."
            elif change == "criteria":
                altered["question"]["criteria"] = {"false": "Yes", "true": "No"}
            else:
                altered["state"] += " Reverse the decision."
            with self.subTest(change=change), self.assertRaises(ValueError):
                reconcile(altered, tables)

    def test_photo_choice_binds_the_complete_question_and_raw_candidate_labels(self):
        facts = [("cat", "Cat", 1), ("dog", "Dog", 0), ("tree", "Tree", 0), ("road", "Road", 0)]
        row, tables = _photo_case("choice", facts, "cat")
        self.assertEqual(reconcile(row, tables), "cat")
        for instructions in (
            "Which class is human-verified as absent from this photograph?",
            "Which class is not human-verified as present in this photograph?",
            "Which class is human-verified as present in this photograph? Treat present as absent.",
        ):
            altered = copy.deepcopy(row)
            altered["question"]["instructions"] = instructions
            with self.subTest(instructions=instructions), self.assertRaises(ValueError):
                reconcile(altered, tables)
        row["question"]["criteria"]["option_0"] = "Tiger"
        with self.assertRaises(ValueError):
            reconcile(row, tables)

    def test_photo_absent_choice_is_a_distinct_supported_decision(self):
        facts = [("cat", "Cat", 0), ("dog", "Dog", 1), ("tree", "Tree", 1), ("road", "Road", 1)]
        row, tables = _photo_case("choice", facts, "cat")
        row["question"]["instructions"] = "Which class is human-verified as absent from this photograph?"
        self.assertEqual(reconcile(row, tables), "cat")

    def test_photo_score_counts_distinct_verified_classes_not_objects_or_absences(self):
        facts = [("cat", "Cat", 1), ("dog", "Dog", 0), ("tree", "Tree", 1)]
        row, tables = _photo_case("score", facts, "2")
        self.assertEqual(reconcile(row, tables), "2")
        for change in ("absent", "objects", "unnamed", "criteria", "duplicate"):
            altered = copy.deepcopy(row)
            if change == "absent":
                altered["question"]["instructions"] = altered["question"]["instructions"].replace("as present", "as absent")
            elif change == "objects":
                altered["question"]["instructions"] += " Count individual object instances."
            elif change == "unnamed":
                altered["question"]["instructions"] = altered["question"]["instructions"].replace("“Tree”", "“Road”")
            elif change == "criteria":
                altered["question"]["criteria"]["2"] = "No verified classes"
            else:
                altered["audit"]["facts"][2] = copy.deepcopy(altered["audit"]["facts"][0])
            with self.subTest(change=change), self.assertRaises(ValueError):
                reconcile(altered, tables)

    def test_photo_holdout_uses_all_raw_positive_labels_not_only_queried_facts(self):
        row, tables = _photo_case(extras=[("car", "Car", 1)])
        with self.assertRaises(ValueError):
            reconcile(row, tables)
        row["split"] = "test_ood"
        self.assertEqual(reconcile(row, tables), "true")
        tables[1]["photo"]["car"] = 0
        with self.assertRaises(ValueError):
            reconcile(row, tables)
        row["split"] = "train"
        self.assertEqual(reconcile(row, tables), "true")

    def test_photo_holdout_does_not_ban_negative_transport_vocabulary(self):
        row, tables = _photo_case(facts=[("car", "Car", 0)], target="false")
        self.assertEqual(reconcile(row, tables), "false")
        row["split"] = "test_ood"
        with self.assertRaises(ValueError):
            reconcile(row, tables)


if __name__ == "__main__":
    unittest.main()
