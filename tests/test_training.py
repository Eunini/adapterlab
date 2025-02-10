import hashlib
import json
import zipfile

import pytest
from adapterlab.data import demo_dataset, prepare_dataset
from adapterlab.schemas import TrainConfig
from adapterlab.training import classification_metrics, load_artifact, predict, train


def test_macro_f1_keeps_missing_predictions_in_the_denominator():
    result = classification_metrics([0, 0, 1, 1], [0, 0, 0, 0], ["first", "second"])
    assert result["accuracy"] == 0.5
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["confusion_matrix"] == [[2, 0], [2, 0]]


@pytest.mark.parametrize("strategy", ["lora", "head-only", "full"])
def test_real_optimizer_export_and_reload_work_offline(training_settings, tmp_path, strategy):
    request = demo_dataset()
    request.records = [row for row in request.records if row.label in ["billing", "login"]][::5]
    records, report = prepare_dataset(request)
    output = tmp_path / strategy
    events = []
    config = TrainConfig(strategy=strategy, epochs=1, batch_size=4, max_length=32)
    result = train(
        records, report, config, training_settings, output, lambda **values: events.append(values)
    )
    assert result["steps"] > 0
    assert result["test"]["samples"] == report["splits"]["test"]
    assert result["trainable_parameters"] <= result["total_parameters"]
    if strategy != "full":
        assert result["trainable_parameters"] < result["total_parameters"]
    assert any("train_loss" in event.get("values", {}) for event in events)
    first = predict(output, training_settings, "please help with my invoice")
    second = predict(
        output,
        training_settings,
        "please help with my invoice",
        loaded=load_artifact(output, training_settings),
    )
    assert first["probabilities"] == second["probabilities"]
    assert sum(row["probability"] for row in first["probabilities"]) == pytest.approx(1)
    assert (output / "model.zip").is_file()
    with zipfile.ZipFile(output / "model.zip") as bundle:
        hashes = json.loads(bundle.read("checksums.json"))
        assert all(
            hashlib.sha256(bundle.read(file)).hexdigest() == sha for file, sha in hashes.items()
        )
