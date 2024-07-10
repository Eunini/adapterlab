"""CPU fine-tuning and evaluation. Model downloads are pinned and safetensors-only."""

import hashlib
import json
import math
import os
import platform
import random
import time
import zipfile
from datetime import UTC, datetime
from importlib.resources import files as package_files
from pathlib import Path

from .schemas import TrainConfig


def libraries():
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(2)
    return (
        torch,
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )


def classification_metrics(actual, predicted, labels):
    matrix = [[0] * len(labels) for _ in labels]
    for truth, guess in zip(actual, predicted):
        matrix[truth][guess] += 1
    per_class = []
    for index, label in enumerate(labels):
        tp = matrix[index][index]
        fp = sum(row[index] for row in matrix) - tp
        fn = sum(matrix[index]) - tp
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        per_class.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": sum(matrix[index]),
            }
        )
    return {
        "accuracy": sum(t == p for t, p in zip(actual, predicted)) / len(actual) if actual else 0,
        "macro_f1": sum(row["f1"] for row in per_class) / len(labels),
        "samples": len(actual),
        "labels": labels,
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


def evaluate(model, encoded, labels, batch_size=16):
    import torch

    actual, predicted, losses = [], [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(encoded["labels"]), batch_size):
            batch = {key: value[start : start + batch_size] for key, value in encoded.items()}
            output = model(**batch)
            actual.extend(batch["labels"].tolist())
            predicted.extend(output.logits.argmax(dim=-1).tolist())
            losses.append((output.loss.item(), len(batch["labels"])))
    result = classification_metrics(actual, predicted, labels)
    result["loss"] = sum(loss * count for loss, count in losses) / sum(count for _, count in losses)
    return result


def base_model(settings, labels):
    *_, AutoModel, _tokenizer = libraries()
    model, info = AutoModel.from_pretrained(
        settings.model_path,
        revision=settings.model_revision,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={label: i for i, label in enumerate(labels)},
        use_safetensors=True,
        trust_remote_code=False,
        token=False,
        attn_implementation="eager",
        output_loading_info=True,
    )
    if any(not key.startswith("classifier.") for key in info["missing_keys"]):
        raise ValueError("Base checkpoint is missing encoder or pooling weights")
    return model


def load_artifact(directory, settings, metadata=None):
    _, _, PeftModel, _, _, AutoModel, AutoTokenizer = libraries()
    directory = Path(directory)
    metadata = metadata or json.loads((directory / "run.json").read_text())
    checkpoint = directory / "checkpoint"
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False
    )
    if metadata["config"]["strategy"] == "lora":
        model = PeftModel.from_pretrained(
            base_model(settings, metadata["labels"]), checkpoint, local_files_only=True
        )
    else:
        model = AutoModel.from_pretrained(checkpoint, local_files_only=True, use_safetensors=True)
    model.eval()
    return model, tokenizer, metadata


def train(records, report, config: TrainConfig, settings, output_dir, callback):
    started = time.monotonic()
    torch, LoraConfig, _, TaskType, get_peft_model, _, AutoTokenizer = libraries()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    labels = report["labels"]
    label_to_id = {label: index for index, label in enumerate(labels)}
    callback(status="preparing")
    tokenizer = AutoTokenizer.from_pretrained(
        settings.model_path,
        revision=settings.model_revision,
        trust_remote_code=False,
        token=False,
        do_lower_case=True,
    )
    model = base_model(settings, labels)
    encoded = {}
    for split in ["train", "validation", "test"]:
        rows = [row for row in records if row["split"] == split]
        batch = tokenizer(
            [row["text"] for row in rows],
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor([label_to_id[row["label"]] for row in rows])
        encoded[split] = batch
    baseline = evaluate(model, encoded["test"], labels)
    if config.strategy == "lora":
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=config.rank,
                lora_alpha=config.rank * 2,
                lora_dropout=0.05,
                target_modules=["query", "value"],
                modules_to_save=["classifier"],
                bias="none",
            ),
        )
    elif config.strategy == "head-only":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("classifier.")
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=0.01,
    )
    size = len(encoded["train"]["labels"])
    steps_per_epoch = math.ceil(size / config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    callback(
        status="training",
        step=0,
        total_steps=total_steps,
        values={
            "baseline_accuracy": baseline["accuracy"],
            "baseline_macro_f1": baseline["macro_f1"],
            "trainable_parameters": trainable,
            "total_parameters": total,
        },
    )
    output_dir = Path(output_dir)
    checkpoint = output_dir / "checkpoint"
    checkpoint.mkdir(parents=True, exist_ok=True)
    history, step, best_score, best_loss, best_epoch = [], 0, -1, float("inf"), 0
    generator = torch.Generator().manual_seed(config.seed)
    for epoch in range(config.epochs):
        order = torch.randperm(size, generator=generator)
        model.train()
        for offset in range(0, size, config.batch_size):
            indices = order[offset : offset + config.batch_size]
            batch = {key: value[indices] for key, value in encoded["train"].items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError("Training loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1)
            optimizer.step()
            step += 1
            point = {
                "step": step,
                "epoch": round(epoch + min(offset + config.batch_size, size) / size, 4),
                "train_loss": float(loss.item()),
                "learning_rate": config.learning_rate,
            }
            history.append(point)
            callback(
                step=step,
                epoch=point["epoch"],
                values={"train_loss": point["train_loss"], "learning_rate": config.learning_rate},
            )
        validation = evaluate(model, encoded["validation"], labels)
        callback(
            step=step,
            epoch=epoch + 1,
            values={
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "validation_macro_f1": validation["macro_f1"],
            },
        )
        if validation["macro_f1"] > best_score or (
            validation["macro_f1"] == best_score and validation["loss"] < best_loss
        ):
            best_score, best_loss, best_epoch = (
                validation["macro_f1"],
                validation["loss"],
                epoch + 1,
            )
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
    callback(status="evaluating")
    metadata = {
        "config": config.model_dump(),
        "labels": labels,
        "base_model": settings.model_path,
        "base_revision": settings.model_revision,
        "dataset_fingerprint": report["fingerprint"],
        "trained_at": datetime.now(UTC).isoformat(),
        "synthetic_data": bool(report.get("synthetic", False)),
    }
    model, tokenizer, _ = load_artifact(output_dir, settings, metadata)
    test = evaluate(model, encoded["test"], labels)
    result = {
        "baseline": baseline,
        "baseline_description": "Pretrained encoder with an untrained classification head",
        "test": test,
        "best_epoch": best_epoch,
        "validation_macro_f1": best_score,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_pct": round(100 * trainable / total, 4),
        "steps": step,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "base_model": settings.model_path,
        "base_revision": settings.model_revision,
        "dataset_fingerprint": report["fingerprint"],
        "torch_version": torch.__version__,
        "hardware": {"device": "cpu", "threads": 2, "machine": platform.machine()},
    }
    for filename, value in [
        ("run.json", metadata),
        ("evaluation.json", result),
        ("dataset_manifest.json", report),
        ("training_history.json", history),
    ]:
        (output_dir / filename).write_text(json.dumps(value, indent=2) + "\n")
    (output_dir / "MODEL_CARD.md").write_text(
        f"# AdapterLab model export\n\nBase model: `{settings.model_path}` at `{settings.model_revision}`.\n\n"
        f"Recipe: {config.strategy}. Labels: {', '.join(labels)}.\n\n"
        "Selected using validation macro F1; the test split is not used for checkpoint selection.\n\n"
        f"Dataset fingerprint: `{report['fingerprint']}`. Synthetic dataset: {metadata['synthetic_data']}.\n\n"
        "Results apply to the supplied dataset. They do not establish production accuracy.\n"
    )
    (output_dir / "BASE_MODEL_LICENSE.txt").write_text(
        package_files("adapterlab").joinpath("resources/APACHE-2.0.txt").read_text()
    )
    files = [file for file in output_dir.rglob("*") if file.is_file() and file.name != "model.zip"]
    hashes = {
        str(file.relative_to(output_dir)): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in files
    }
    (output_dir / "checksums.json").write_text(json.dumps(hashes, indent=2) + "\n")
    with zipfile.ZipFile(output_dir / "model.zip", "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for file in [*files, output_dir / "checksums.json"]:
            bundle.write(file, file.relative_to(output_dir))
    return result


def predict(directory, settings, text, loaded=None):
    import torch

    model, tokenizer, metadata = loaded or load_artifact(directory, settings)
    batch = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=metadata["config"]["max_length"]
    )
    started = time.monotonic()
    with torch.inference_mode():
        probabilities = model(**batch).logits.softmax(-1)[0].tolist()
    ranked = sorted(
        [
            {"label": label, "probability": probability}
            for label, probability in zip(metadata["labels"], probabilities)
        ],
        key=lambda row: row["probability"],
        reverse=True,
    )
    return {
        "prediction": ranked[0]["label"],
        "probabilities": ranked,
        "latency_ms": round((time.monotonic() - started) * 1000, 2),
    }
