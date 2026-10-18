"""Source provenance, intent catalog, and wording for text-data preparation."""

SOURCE_REPOSITORY = "https://github.com/PolyAI-LDN/task-specific-datasets"
SOURCE_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE_API_URL = (
    "https://api.github.com/repos/PolyAI-LDN/task-specific-datasets/commits/"
    f"{SOURCE_REVISION}"
)
RAW_BASE_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    f"{SOURCE_REVISION}"
)
LICENSE_NAME = "CC-BY-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
CITATION = (
    "Casanueva et al. (2020), Efficient Intent Detection with Dual Sentence "
    "Encoders, ACL Workshop on NLP for Conversational AI."
)

PROJECT_DATA_PROVENANCE = {
    "license": "Project-created procedural data; no external asset license applies",
    "license_url": None,
    "attribution": "Generated locally by multimodal-decision mdprep.text",
    "project_created_data_terms": (
        "Project-created deterministic text and labels; no third-party content is incorporated."
    ),
    "font_terms": "Not applicable: these are text-only records with no font assets.",
}

ASSETS = {
    "train.csv": {
        "upstream_path": "banking_data/train.csv",
        "git_blob_sha1": "98e2543cf482d0dca7bfb175ebe35d98efad95be",
        "role": "source examples",
    },
    "test.csv": {
        "upstream_path": "banking_data/test.csv",
        "git_blob_sha1": "799687a8367359432985b8b13d85a2baf73f92dd",
        "role": "evaluation-only; excluded from every generated row",
    },
    "categories.json": {
        "upstream_path": "banking_data/categories.json",
        "git_blob_sha1": "cdd2a5c77a4079a455f8fb7e751d1ecee0e2a5a4",
        "role": "intent label catalog",
    },
    "LICENSE": {
        "upstream_path": "LICENSE",
        "git_blob_sha1": "2f244ac814036ecd9ba9f69782e89ce6b1dca9eb",
        "role": "license text",
    },
}

# These intents form a semantic OOD slice. No utterance carrying one of these
# labels is eligible for an ID split. The raw upstream test split remains wholly
# excluded; test_ood draws only from upstream training examples.
HELD_OUT_INTENTS = (
    "apple_pay_or_google_pay",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cash_withdrawal_not_recognised",
    "disposable_card_limits",
    "verify_source_of_funds",
)

SOURCE_COUNTS = {
    "train": 3600,
    "dev": 360,
    "calibration": 360,
    "policy": 200,
    "test_id": 540,
    "test_ood": 210,
}

# Candidate retrieval stays within a banking subdomain before using lexical
# similarity as a fallback. This makes wrong options plausible rather than
# arbitrary while retaining the immutable upstream intent keys as label IDs.
INTENT_GROUPS = (
    (
        "card_arrival", "card_delivery_estimate", "card_linking",
        "card_not_working", "lost_or_stolen_card", "compromised_card",
        "get_physical_card", "order_physical_card", "getting_spare_card",
        "activate_my_card", "card_about_to_expire", "card_swallowed",
        "change_pin", "pin_blocked", "contactless_not_working",
        "card_acceptance", "visa_or_mastercard",
    ),
    (
        "card_payment_wrong_exchange_rate", "card_payment_fee_charged",
        "card_payment_not_recognised", "pending_card_payment",
        "declined_card_payment", "reverted_card_payment?",
        "transaction_charged_twice", "request_refund", "Refund_not_showing_up",
        "extra_charge_on_statement",
    ),
    (
        "pending_cash_withdrawal", "wrong_amount_of_cash_received",
        "declined_cash_withdrawal", "cash_withdrawal_not_recognised",
        "wrong_exchange_rate_for_cash_withdrawal", "cash_withdrawal_charge",
        "atm_support",
    ),
    (
        "cancel_transfer", "transfer_not_received_by_recipient",
        "declined_transfer", "pending_transfer", "transfer_timing",
        "beneficiary_not_allowed", "transfer_fee_charged", "failed_transfer",
        "receiving_money", "transfer_into_account",
        "balance_not_updated_after_bank_transfer",
    ),
    (
        "automatic_top_up", "top_up_by_bank_transfer_charge", "pending_top_up",
        "top_up_limits", "supported_cards_and_currencies", "top_up_reverted",
        "topping_up_by_card", "verify_top_up", "top_up_by_cash_or_cheque",
        "top_up_failed", "top_up_by_card_charge",
    ),
    (
        "exchange_rate", "fiat_currency_support", "exchange_via_app",
        "exchange_charge", "country_support",
    ),
    (
        "getting_virtual_card", "disposable_card_limits",
        "virtual_card_not_working", "get_disposable_virtual_card",
        "apple_pay_or_google_pay",
    ),
    (
        "age_limit", "edit_personal_details", "why_verify_identity",
        "unable_to_verify_identity", "verify_source_of_funds",
        "verify_my_identity", "passcode_forgotten", "lost_or_stolen_phone",
        "terminate_account",
    ),
    (
        "balance_not_updated_after_cheque_or_cash_deposit",
        "direct_debit_payment_not_recognised",
    ),
)

SOURCE_QUESTION_TEMPLATES = (
    "Choose the banking support intent that best matches the customer message.",
    "Which single support category most precisely describes this request?",
    "Route this message to the most appropriate banking intent.",
    "Select the intent an annotator should assign to this customer query.",
    "Identify the narrowest matching support reason for the message.",
    "Which candidate intent is the best destination for this request?",
    "Classify the customer's issue using one of the offered banking intents.",
    "Pick the support intent whose meaning best fits the quoted message.",
)

CHOICE_INSTRUCTIONS = (
    "Apply the stated policy to the evidence and choose the required action.",
    "Using only the listed evidence, select the policy outcome.",
    "Choose the outcome produced by the rule; do not assume missing facts.",
    "Evaluate the request against every stated constraint and select one action.",
)
NOUL_INSTRUCTIONS = (
    "Does the request satisfy every stated authorization condition?",
    "Using only the evidence shown, should this request be authorized?",
    "Is the action permitted under the complete rule below?",
    "Answer whether all mandatory policy constraints are met.",
)
SCORE_INSTRUCTIONS = (
    "Assign the ordinal response severity defined by the scoring rule.",
    "What severity level follows from the listed evidence and point rule?",
    "Score this case using the stated additive rubric and cap.",
    "Choose the required response tier after applying the severity rubric.",
)

POLICY_CRITERIA = {
    "allow": "Allow automatically",
    "review": "Send for manual review",
    "decline": "Decline the request",
    "insufficient": "Insufficient evidence to decide",
}
DISPUTE_CRITERIA = {
    "file": "File the dispute in the standard queue",
    "expedite": "File and expedite the dispute",
    "deny": "Do not file because the policy conditions fail",
    "insufficient": "Insufficient evidence to decide",
}
SEVERITY_LEVELS = [
    "Routine — no immediate intervention",
    "Elevated — monitor and contact the customer",
    "High — restrict the affected channel and investigate",
    "Critical — restrict all access and escalate immediately",
]
EXPOSURE_LEVELS = [
    "Limited exposure",
    "Material exposure",
    "Severe exposure",
    "Systemic exposure",
]
