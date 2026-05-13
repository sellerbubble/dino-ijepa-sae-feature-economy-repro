"""Paper-scale torch probe trainers for public full reruns."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from feature_economy.artifacts import (
    current_git_commit,
    validate_probe_summary,
    validate_run_manifest,
)
from feature_economy.data import ManifestDataset

from .linear_probe import (
    _classification_readout_features,
    _labels_to_class_indices,
    _labels_to_int,
    _load_targets,
)
from .metrics import depth_metrics, topk_accuracy


def train_native_paper_scale_probe(
    *,
    train_features_npz: str | Path,
    train_manifest_path: str | Path,
    val_features_npz: str | Path,
    val_manifest_path: str | Path,
    task_type: str,
    task_id: str,
    model_id: str,
    output_dir: str | Path,
    expected_train_split: str | None = "train",
    expected_val_split: str | None = "val",
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    train_targets_npz: str | Path | None = None,
    val_targets_npz: str | Path | None = None,
    target_key: str = "targets",
    decoder_hidden_channels: int = 256,
) -> Path:
    """Train a torch paper-scale native probe on saved global features."""

    if task_type == "dense_depth":
        return _train_paper_scale_depth_probe(
            train_array_npz=train_features_npz,
            train_manifest_path=train_manifest_path,
            val_array_npz=val_features_npz,
            val_manifest_path=val_manifest_path,
            array_key="features",
            task_id=task_id,
            model_id=model_id,
            sae_id=None,
            output_dir=output_dir,
            expected_train_split=expected_train_split,
            expected_val_split=expected_val_split,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            train_targets_npz=train_targets_npz,
            val_targets_npz=val_targets_npz,
            target_key=target_key,
            decoder_hidden_channels=decoder_hidden_channels,
            record_type="native_probe_summary",
        )
    return _train_paper_scale_classification_probe(
        train_array_npz=train_features_npz,
        train_manifest_path=train_manifest_path,
        val_array_npz=val_features_npz,
        val_manifest_path=val_manifest_path,
        array_key="features",
        task_type=task_type,
        task_id=task_id,
        model_id=model_id,
        sae_id=None,
        output_dir=output_dir,
        expected_train_split=expected_train_split,
        expected_val_split=expected_val_split,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        record_type="native_probe_summary",
    )


def train_sae_paper_scale_probe(
    *,
    train_codes_npz: str | Path,
    train_manifest_path: str | Path,
    val_codes_npz: str | Path,
    val_manifest_path: str | Path,
    task_type: str,
    task_id: str,
    model_id: str,
    sae_id: str,
    output_dir: str | Path,
    expected_train_split: str | None = "train",
    expected_val_split: str | None = "val",
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    train_targets_npz: str | Path | None = None,
    val_targets_npz: str | Path | None = None,
    target_key: str = "targets",
    decoder_hidden_channels: int = 256,
) -> Path:
    """Train a torch paper-scale SAE-code probe on saved global codes."""

    if task_type == "dense_depth":
        return _train_paper_scale_depth_probe(
            train_array_npz=train_codes_npz,
            train_manifest_path=train_manifest_path,
            val_array_npz=val_codes_npz,
            val_manifest_path=val_manifest_path,
            array_key="codes",
            task_id=task_id,
            model_id=model_id,
            sae_id=sae_id,
            output_dir=output_dir,
            expected_train_split=expected_train_split,
            expected_val_split=expected_val_split,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            train_targets_npz=train_targets_npz,
            val_targets_npz=val_targets_npz,
            target_key=target_key,
            decoder_hidden_channels=decoder_hidden_channels,
            record_type="sae_probe_summary",
        )
    return _train_paper_scale_classification_probe(
        train_array_npz=train_codes_npz,
        train_manifest_path=train_manifest_path,
        val_array_npz=val_codes_npz,
        val_manifest_path=val_manifest_path,
        array_key="codes",
        task_type=task_type,
        task_id=task_id,
        model_id=model_id,
        sae_id=sae_id,
        output_dir=output_dir,
        expected_train_split=expected_train_split,
        expected_val_split=expected_val_split,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        record_type="sae_probe_summary",
    )


def _train_paper_scale_classification_probe(
    *,
    train_array_npz: str | Path,
    train_manifest_path: str | Path,
    val_array_npz: str | Path,
    val_manifest_path: str | Path,
    array_key: str,
    task_type: str,
    task_id: str,
    model_id: str,
    sae_id: str | None,
    output_dir: str | Path,
    expected_train_split: str | None,
    expected_val_split: str | None,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    record_type: str,
) -> Path:
    if task_type not in {"classification", "count_classification"}:
        raise ValueError("paper-scale public trainer currently supports classification/counting")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    torch = _require_torch()
    torch.manual_seed(seed)

    train_features, train_labels = _load_readout_arrays(
        array_npz=train_array_npz,
        array_key=array_key,
        manifest_path=train_manifest_path,
        task_type=task_type,
        expected_split=expected_train_split,
    )
    val_features, val_labels = _load_readout_arrays(
        array_npz=val_array_npz,
        array_key=array_key,
        manifest_path=val_manifest_path,
        task_type=task_type,
        expected_split=expected_val_split,
    )
    classes = np.unique(np.concatenate([train_labels, val_labels]))
    if classes.size < 2:
        raise ValueError("paper-scale probe requires at least two classes")
    train_targets = _labels_to_class_indices(train_labels, classes)
    val_targets = _labels_to_class_indices(val_labels, classes)

    model = torch.nn.Linear(train_features.shape[1], classes.size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(train_features, dtype=torch.float32),
        torch.as_tensor(train_targets, dtype=torch.long),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_tensor = torch.as_tensor(val_features, dtype=torch.float32, device=device)
    val_target_tensor = torch.as_tensor(val_targets, dtype=torch.long, device=device)

    history: list[dict[str, float | int]] = []
    best_state: dict[str, object] | None = None
    best_primary = -float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * int(batch_labels.shape[0])
            train_count += int(batch_labels.shape[0])

        model.eval()
        with torch.no_grad():
            val_logits_tensor = model(val_tensor)
            val_loss = float(criterion(val_logits_tensor, val_target_tensor).item())
        val_logits = val_logits_tensor.detach().cpu().numpy()
        metrics = _classification_metrics(task_type, val_logits, val_labels, classes)
        primary = metrics["top1"] if task_type == "classification" else metrics["accuracy"]
        epoch_record = {
            "epoch": int(epoch + 1),
            "train_loss": float(train_loss / max(train_count, 1)),
            "val_loss": val_loss,
            **metrics,
        }
        history.append(epoch_record)
        if primary > best_primary:
            best_primary = primary
            best_state = {
                "epoch": int(epoch + 1),
                "model": copy.deepcopy(model.state_dict()),
                "metrics": dict(metrics),
                "val_loss": val_loss,
            }

    if best_state is None:
        raise RuntimeError("paper-scale probe did not produce a best checkpoint")
    model.load_state_dict(best_state["model"])
    model.eval()
    with torch.no_grad():
        val_logits = model(val_tensor).detach().cpu().numpy()
    metrics = _classification_metrics(task_type, val_logits, val_labels, classes)
    weights = _linear_weights_with_bias(model).astype("float32")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "probe.pt"
    torch.save(
        {
            "model_state_dict": best_state["model"],
            "classes": classes,
            "history": history,
            "best_state": {
                "epoch": best_state["epoch"],
                "metrics": best_state["metrics"],
                "val_loss": best_state["val_loss"],
            },
            "args": {
                "task_type": task_type,
                "task_id": task_id,
                "model_id": model_id,
                "sae_id": sae_id,
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "backend": "paper_scale_torch",
            },
        },
        checkpoint_path,
    )
    probe_arrays = {
        "logits": val_logits.astype("float32"),
        "weights": weights,
        "classes": classes,
        "labels": val_labels.astype("int64"),
    }
    probe_outputs_path = output_dir / "probe_outputs.npz"
    compatibility_logits_path = output_dir / "probe_logits.npz"
    np.savez_compressed(probe_outputs_path, **probe_arrays)
    np.savez_compressed(compatibility_logits_path, **probe_arrays)

    summary = {
        "record_type": record_type,
        "task_id": task_id,
        "model_id": model_id,
        "seed": seed,
        "metrics": metrics,
        "checkpoint": str(checkpoint_path),
        "fixture": False,
        "backend": "paper_scale_torch",
        "probe_backend": "paper_scale_torch",
        "best_epoch": int(best_state["epoch"]),
        "selection": {
            "checkpoint_rule": (
                "best_validation_top1"
                if task_type == "classification"
                else "best_validation_accuracy"
            )
        },
        "history": history,
        "probe_outputs": str(probe_outputs_path),
        "compatibility_probe_logits": str(compatibility_logits_path),
    }
    if sae_id is not None:
        summary["sae_id"] = sae_id
    validate_probe_summary(summary)
    summary_path = output_dir / f"{record_type}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    inputs = {
        f"train_{array_key}_npz": str(train_array_npz),
        f"val_{array_key}_npz": str(val_array_npz),
        "train_manifest": str(train_manifest_path),
        "val_manifest": str(val_manifest_path),
        "task_type": task_type,
        "task_id": task_id,
        "model_id": model_id,
        "backend": "paper_scale_torch",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
    }
    if sae_id is not None:
        inputs["sae_id"] = sae_id
    manifest = {
        "run_id": "_".join(
            part for part in ["paper_scale_probe", task_id, model_id, sae_id or "native"] if part
        ),
        "command": "feature-economy probe-sae" if sae_id is not None else "feature-economy probe-native",
        "git_commit": current_git_commit(),
        "config_files": [],
        "inputs": inputs,
        "outputs": {
            "summary": str(summary_path),
            "checkpoint": str(checkpoint_path),
            "probe_outputs": str(probe_outputs_path),
            "probe_logits": str(compatibility_logits_path),
        },
        "fixture": False,
    }
    validate_run_manifest(manifest)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def _train_paper_scale_depth_probe(
    *,
    train_array_npz: str | Path,
    train_manifest_path: str | Path,
    val_array_npz: str | Path,
    val_manifest_path: str | Path,
    array_key: str,
    task_id: str,
    model_id: str,
    sae_id: str | None,
    output_dir: str | Path,
    expected_train_split: str | None,
    expected_val_split: str | None,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    train_targets_npz: str | Path | None,
    val_targets_npz: str | Path | None,
    target_key: str,
    decoder_hidden_channels: int,
    record_type: str,
) -> Path:
    if train_targets_npz is None or val_targets_npz is None:
        raise ValueError("paper-scale dense-depth probe requires train and val targets")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if decoder_hidden_channels <= 0:
        raise ValueError("decoder_hidden_channels must be positive")

    torch = _require_torch()
    torch.manual_seed(seed)

    train_features, train_targets = _load_dense_arrays(
        array_npz=train_array_npz,
        array_key=array_key,
        manifest_path=train_manifest_path,
        expected_split=expected_train_split,
        targets_npz=train_targets_npz,
        target_key=target_key,
    )
    val_features, val_targets = _load_dense_arrays(
        array_npz=val_array_npz,
        array_key=array_key,
        manifest_path=val_manifest_path,
        expected_split=expected_val_split,
        targets_npz=val_targets_npz,
        target_key=target_key,
    )
    decoder = _build_depth_decoder(
        input_channels=train_features.shape[1],
        hidden_channels=decoder_hidden_channels,
        torch=torch,
    ).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=weight_decay)
    train_dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(train_features, dtype=torch.float32),
        torch.as_tensor(train_targets, dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_tensor = torch.as_tensor(val_features, dtype=torch.float32, device=device)
    val_target_tensor = torch.as_tensor(val_targets, dtype=torch.float32, device=device)

    history: list[dict[str, float | int]] = []
    best_state: dict[str, object] | None = None
    best_rmse = float("inf")
    for epoch in range(epochs):
        decoder.train()
        train_loss = 0.0
        train_count = 0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            prediction = decoder(batch_features).squeeze(1)
            valid = batch_targets > 0
            if not bool(valid.any()):
                continue
            loss = torch.nn.functional.mse_loss(prediction[valid], batch_targets[valid])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * int(batch_targets.shape[0])
            train_count += int(batch_targets.shape[0])

        decoder.eval()
        with torch.no_grad():
            val_prediction = decoder(val_tensor).squeeze(1).detach().cpu().numpy()
        metrics = depth_metrics(val_prediction, val_targets)
        epoch_record = {
            "epoch": int(epoch + 1),
            "train_loss": float(train_loss / max(train_count, 1)),
            **metrics,
        }
        history.append(epoch_record)
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_state = {
                "epoch": int(epoch + 1),
                "model": copy.deepcopy(decoder.state_dict()),
                "metrics": dict(metrics),
            }

    if best_state is None:
        raise RuntimeError("paper-scale depth probe did not produce a best checkpoint")
    decoder.load_state_dict(best_state["model"])
    decoder.eval()
    with torch.no_grad():
        val_prediction = decoder(val_tensor).squeeze(1).detach().cpu().numpy()
    metrics = depth_metrics(val_prediction, val_targets)
    weights = _depth_input_weight_proxy(decoder).astype("float32")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "probe.pt"
    torch.save(
        {
            "model_state_dict": best_state["model"],
            "history": history,
            "best_state": {
                "epoch": best_state["epoch"],
                "metrics": best_state["metrics"],
            },
            "args": {
                "task_type": "dense_depth",
                "task_id": task_id,
                "model_id": model_id,
                "sae_id": sae_id,
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "decoder_hidden_channels": decoder_hidden_channels,
                "backend": "paper_scale_torch",
            },
        },
        checkpoint_path,
    )
    probe_arrays = {
        "prediction": val_prediction.astype("float32"),
        "weights": weights,
        "targets": val_targets.astype("float32"),
        "input_feature_weight_proxy": weights[:-1],
    }
    probe_outputs_path = output_dir / "probe_outputs.npz"
    compatibility_logits_path = output_dir / "probe_logits.npz"
    np.savez_compressed(probe_outputs_path, **probe_arrays)
    np.savez_compressed(compatibility_logits_path, **probe_arrays)
    summary = {
        "record_type": record_type,
        "task_id": task_id,
        "model_id": model_id,
        "seed": seed,
        "metrics": metrics,
        "checkpoint": str(checkpoint_path),
        "fixture": False,
        "backend": "paper_scale_torch",
        "probe_backend": "paper_scale_torch",
        "best_epoch": int(best_state["epoch"]),
        "selection": {"checkpoint_rule": "best_validation_rmse"},
        "history": history,
        "readout": {
            "type": "light_depth_decoder",
            "decoder_hidden_channels": int(decoder_hidden_channels),
        },
        "probe_outputs": str(probe_outputs_path),
        "compatibility_probe_logits": str(compatibility_logits_path),
        "compatibility_note": (
            "probe_logits.npz stores a first-layer input-channel weight proxy "
            "for public ranking compatibility; probe_outputs.npz is the "
            "authoritative decoder prediction artifact."
        ),
    }
    if sae_id is not None:
        summary["sae_id"] = sae_id
    validate_probe_summary(summary)
    summary_path = output_dir / f"{record_type}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    inputs = {
        f"train_{array_key}_npz": str(train_array_npz),
        f"val_{array_key}_npz": str(val_array_npz),
        "train_targets_npz": str(train_targets_npz),
        "val_targets_npz": str(val_targets_npz),
        "train_manifest": str(train_manifest_path),
        "val_manifest": str(val_manifest_path),
        "task_type": "dense_depth",
        "task_id": task_id,
        "model_id": model_id,
        "backend": "paper_scale_torch",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "decoder_hidden_channels": decoder_hidden_channels,
    }
    if sae_id is not None:
        inputs["sae_id"] = sae_id
    manifest = {
        "run_id": "_".join(
            part for part in ["paper_scale_depth_probe", task_id, model_id, sae_id or "native"] if part
        ),
        "command": "feature-economy probe-sae" if sae_id is not None else "feature-economy probe-native",
        "git_commit": current_git_commit(),
        "config_files": [],
        "inputs": inputs,
        "outputs": {
            "summary": str(summary_path),
            "checkpoint": str(checkpoint_path),
            "probe_outputs": str(probe_outputs_path),
            "probe_logits": str(compatibility_logits_path),
        },
        "fixture": False,
    }
    validate_run_manifest(manifest)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def _load_readout_arrays(
    *,
    array_npz: str | Path,
    array_key: str,
    manifest_path: str | Path,
    task_type: str,
    expected_split: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = ManifestDataset(
        manifest_path,
        task_type=task_type,
        expected_split=expected_split,
    )
    arrays = np.load(array_npz)
    if array_key not in arrays:
        raise ValueError(f"{array_npz} does not contain array {array_key!r}")
    features = np.asarray(arrays[array_key], dtype=np.float32)
    if features.shape[0] != len(dataset):
        raise ValueError(
            f"{array_key} rows ({features.shape[0]}) must match manifest rows ({len(dataset)})"
        )
    labels = _labels_to_int(np.asarray([record.label for record in dataset.records]))
    return _classification_readout_features(features).astype("float32"), labels


def _load_dense_arrays(
    *,
    array_npz: str | Path,
    array_key: str,
    manifest_path: str | Path,
    expected_split: str | None,
    targets_npz: str | Path,
    target_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = ManifestDataset(
        manifest_path,
        task_type="dense_depth",
        expected_split=expected_split,
    )
    arrays = np.load(array_npz)
    if array_key not in arrays:
        raise ValueError(f"{array_npz} does not contain array {array_key!r}")
    features = np.asarray(arrays[array_key], dtype=np.float32)
    targets = np.asarray(_load_targets(targets_npz, target_key), dtype=np.float32)
    if features.shape[0] != len(dataset):
        raise ValueError(
            f"{array_key} rows ({features.shape[0]}) must match manifest rows ({len(dataset)})"
        )
    if targets.shape[0] != len(dataset):
        raise ValueError(
            f"target rows ({targets.shape[0]}) must match manifest rows ({len(dataset)})"
        )
    if features.shape[:-1] != targets.shape:
        raise ValueError(
            "dense-depth features must have shape target_shape + [feature_dim]; "
            f"got features {features.shape}, targets {targets.shape}"
        )
    return np.moveaxis(features, -1, 1).astype("float32"), targets


def _classification_metrics(
    task_type: str,
    logits: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    if task_type == "classification":
        topk = (1, min(5, classes.size))
        return topk_accuracy(logits, _labels_to_class_indices(labels, classes), topk=topk)
    predictions = classes[np.argmax(logits, axis=1)]
    return {"accuracy": float(np.mean(predictions == labels))}


def _linear_weights_with_bias(model) -> np.ndarray:
    weight = model.weight.detach().cpu().numpy().T
    bias = model.bias.detach().cpu().numpy()[None, :]
    return np.concatenate([weight, bias], axis=0)


def _build_depth_decoder(*, input_channels: int, hidden_channels: int, torch):
    return torch.nn.Sequential(
        torch.nn.Conv2d(input_channels, hidden_channels, kernel_size=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(hidden_channels, 1, kernel_size=1),
        torch.nn.Softplus(),
    )


def _depth_input_weight_proxy(decoder) -> np.ndarray:
    first_conv = decoder[0]
    weight = first_conv.weight.detach().cpu().numpy()
    channel_score = np.mean(np.abs(weight), axis=(0, 2, 3))
    bias = np.asarray([0.0], dtype=np.float32)
    return np.concatenate([channel_score, bias], axis=0)


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on user environment.
        raise RuntimeError("paper-scale probe backend requires PyTorch") from exc
    return torch
