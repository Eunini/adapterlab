import hashlib
import json
import re
from collections import Counter, defaultdict

from .schemas import DatasetInput


def canonical(text):
    return " ".join(text.split()).casefold()


def prepare_dataset(request: DatasetInput):
    unique, group_labels, records, representatives = {}, {}, [], {}
    duplicates = 0
    for row in request.records:
        key, label = canonical(row.text), row.label.strip()
        if len(key) < 3 or not label:
            raise ValueError("Empty text or label after whitespace normalization")
        group = row.group.strip() or hashlib.sha256(key.encode()).hexdigest()
        if key in unique:
            if unique[key] != label:
                raise ValueError("Identical text has conflicting labels")
            existing = representatives[key]
            if existing["group"] != group:
                raise ValueError("Identical text has conflicting group assignments")
            existing["text"] = min(existing["text"], " ".join(row.text.split()))
            duplicates += 1
            continue
        unique[key] = label
        if group in group_labels and group_labels[group] != label:
            raise ValueError("Each group ID must belong to exactly one label")
        group_labels[group] = label
        records.append({"text": " ".join(row.text.split()), "label": label, "group": group})
        representatives[key] = records[-1]
    labels = sorted({row["label"] for row in records})
    if not 2 <= len(labels) <= 20:
        raise ValueError("Datasets must contain between 2 and 20 labels")
    by_label = defaultdict(set)
    for row in records:
        by_label[row["label"]].add(row["group"])
    assignment = {}
    for label, groups in by_label.items():
        if len(groups) < 5:
            raise ValueError(f"Label {label!r} needs at least five distinct groups")
        ordered = sorted(
            groups, key=lambda group: hashlib.sha256(f"42:{group}".encode()).hexdigest()
        )
        test_count = max(1, len(ordered) * 15 // 100)
        val_count = max(1, len(ordered) * 15 // 100)
        for index, group in enumerate(ordered):
            assignment[group] = (
                "test"
                if index < test_count
                else "validation"
                if index < test_count + val_count
                else "train"
            )
    records.sort(key=lambda row: (row["label"], canonical(row["text"])))
    for row in records:
        row["split"] = assignment[row["group"]]
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    fingerprint = hashlib.sha256(encoded).hexdigest()
    pii = sum(
        bool(re.search(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b|\b(?:\d[ -]?){10,}\b", row["text"]))
        for row in records
    )
    report = {
        "examples": len(records),
        "labels": labels,
        "label_counts": dict(Counter(row["label"] for row in records)),
        "splits": dict(Counter(row["split"] for row in records)),
        "groups": len(assignment),
        "duplicates_removed": duplicates,
        "possible_pii_examples": pii,
        "split_method": "label-stratified groups, deterministic SHA-256 order",
        "fingerprint": fingerprint,
    }
    return records, report


TEMPLATES = {
    "billing": [
        "Invoice {ref} has a charge I do not recognize.",
        "Why was I charged twice for subscription {ref}?",
        "The total on receipt {ref} is higher than expected.",
        "I need a breakdown of the fees on bill {ref}.",
        "My payment for invoice {ref} is not reflected in the balance.",
        "There is an unexpected transaction on statement {ref}.",
        "Please explain the extra amount on invoice {ref}.",
        "The amount deducted for purchase {ref} looks incorrect.",
    ],
    "login": [
        "I cannot sign in to account {ref} after resetting my password.",
        "My login for profile {ref} keeps being rejected.",
        "The verification code for account {ref} never arrives.",
        "Signing into user {ref} returns an authentication error.",
        "I am locked out of profile {ref}.",
        "The password reset link for account {ref} has expired.",
        "Two-factor authentication is stopping access to profile {ref}.",
        "The sign-in page says the credentials for account {ref} are invalid.",
    ],
    "delivery": [
        "Order {ref} has not arrived despite the delivery estimate.",
        "Can you check the shipping status of parcel {ref}?",
        "The tracking information for shipment {ref} has not updated.",
        "Package {ref} was marked delivered but is missing.",
        "My delivery for order {ref} went to the wrong address.",
        "When will the courier bring parcel {ref}?",
        "Shipment {ref} seems stuck at the distribution center.",
        "I need an updated arrival date for order {ref}.",
    ],
    "cancellation": [
        "Please cancel subscription {ref} before the next renewal.",
        "I want to stop the recurring plan {ref}.",
        "How do I end my membership {ref}?",
        "Turn off automatic renewal for subscription {ref}.",
        "I no longer need service plan {ref}; please close it.",
        "Please terminate my monthly membership {ref}.",
        "I would like to unsubscribe from package {ref}.",
        "Do not renew the service associated with plan {ref}.",
    ],
    "account": [
        "I need to update the contact details on profile {ref}.",
        "Please change the display name for account {ref}.",
        "Where can I edit the address linked to user {ref}?",
        "My phone number has changed for profile {ref}.",
        "I want to replace the contact email on account {ref}.",
        "The company details on profile {ref} need updating.",
        "Help me correct the personal information on user {ref}.",
        "I need to modify my communication preferences for account {ref}.",
    ],
    "technical": [
        "The application crashes when opening project {ref}.",
        "Uploading document {ref} causes the app to freeze.",
        "The dashboard for workspace {ref} shows a blank screen.",
        "I get a server error while loading resource {ref}.",
        "The export for report {ref} fails halfway through.",
        "Buttons in workspace {ref} are unresponsive.",
        "The page for item {ref} never finishes loading.",
        "The app stops responding while processing file {ref}.",
    ],
}


def synthetic_examples(namespace="demo", variations=5):
    result = []
    endings = [
        "This started today.",
        "Please check when you can.",
        "I have already tried again.",
        "Could someone investigate?",
        "The problem is still happening.",
    ]
    for label, templates in TEMPLATES.items():
        for family, template in enumerate(templates):
            for index in range(variations):
                ref = f"{namespace}-{family + 1:02d}{index + 1:02d}"
                result.append(
                    {
                        "text": template.format(ref=ref) + " " + endings[index % len(endings)],
                        "label": label,
                        "group": f"{label}-family-{family}",
                    }
                )
    return result


def demo_dataset():
    return DatasetInput(
        name="Support intents",
        description="Synthetic support requests. Template families stay in one split.",
        license="CC0-1.0 (synthetic examples)",
        synthetic=True,
        records=synthetic_examples(),
    )
