import pytest
from adapterlab.data import canonical, demo_dataset, prepare_dataset
from adapterlab.schemas import Example


def test_grouped_splits_are_disjoint_and_every_label_is_represented():
    records, report = prepare_dataset(demo_dataset())
    assert report["examples"] == 240
    assert report["splits"] == {"train": 180, "validation": 30, "test": 30}
    sets = {
        split: {row["group"] for row in records if row["split"] == split}
        for split in ["train", "validation", "test"]
    }
    assert not sets["train"] & sets["validation"]
    assert not sets["test"] & (sets["train"] | sets["validation"])
    for split in sets:
        assert {row["label"] for row in records if row["split"] == split} == set(report["labels"])


def test_input_order_does_not_change_dataset_fingerprint():
    first = demo_dataset()
    second = first.model_copy(update={"records": list(reversed(first.records))})
    assert prepare_dataset(first) == prepare_dataset(second)


def test_duplicate_text_is_removed_and_conflicting_labels_are_rejected():
    request = demo_dataset()
    request.records.append(request.records[0])
    assert prepare_dataset(request)[1]["duplicates_removed"] == 1
    request.records.append(request.records[0].model_copy(update={"label": "login"}))
    with pytest.raises(ValueError, match="conflicting labels"):
        prepare_dataset(request)


def test_groups_cannot_mix_labels_and_need_enough_support():
    request = demo_dataset()
    request.records[-1] = request.records[-1].model_copy(update={"group": request.records[0].group})
    with pytest.raises(ValueError, match="exactly one label"):
        prepare_dataset(request)
    request = demo_dataset()
    for row in request.records:
        row.group = row.label
    with pytest.raises(ValueError, match="five distinct groups"):
        prepare_dataset(request)


def test_pii_warning_does_not_echo_private_text():
    request = demo_dataset()
    request.records[0] = Example(
        text="Contact member@example.test about the invoice",
        label="billing",
        group="private-billing",
    )
    _, report = prepare_dataset(request)
    assert report["possible_pii_examples"] == 1
    assert "member@example.test" not in str(report)
    assert canonical(" Hello\n WORLD ") == "hello world"


def test_case_variant_duplicates_have_order_independent_fingerprints():
    request = demo_dataset()
    request.records.append(
        request.records[0].model_copy(update={"text": request.records[0].text.upper()})
    )
    reverse = request.model_copy(update={"records": list(reversed(request.records))})
    assert prepare_dataset(request) == prepare_dataset(reverse)
    request.records[-1] = request.records[-1].model_copy(update={"group": "another-group"})
    with pytest.raises(ValueError, match="conflicting group assignments"):
        prepare_dataset(request)
