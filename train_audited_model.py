"""Train and compare baseline/augmented EfficientNetB0 on audited photos only."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "reports" / "image_audit_manifest.csv"
MODEL_DIR = ROOT / "models"
EVAL_DIR = ROOT / "reports" / "evaluation_v2"
DATASET_STATS = ROOT / "reports" / "dataset_statistics.json"
CLASS_NAMES = ["leafs", "saplings", "trees"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
SEED = 42


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


def load_records() -> pd.DataFrame:
    if not MANIFEST.exists():
        raise FileNotFoundError("Run audit_real_dataset.py before training.")
    frame = pd.read_csv(MANIFEST, keep_default_na=False)
    frame = frame[
        (frame["source"] == "real_dataset")
        & frame["training_eligible"].astype(str).str.lower().eq("true")
        & frame["verified_class"].isin(CLASS_NAMES)
    ].copy()
    frame["absolute_path"] = frame["relative_path"].map(lambda p: str(ROOT / "to review real dataset" / p))
    missing = [path for path in frame["absolute_path"] if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing audited images: {missing[:3]}")
    frame["label_id"] = frame["verified_class"].map({name: i for i, name in enumerate(CLASS_NAMES)})
    return frame.sort_values("relative_path").reset_index(drop=True)


def make_grouped_split(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Choose nested stratified group folds using class counts, never model scores."""
    y = frame["label_id"].to_numpy(dtype=np.int32)
    groups = frame["group_id"].astype(str).to_numpy()
    all_ids = np.arange(len(frame))
    class_count = len(CLASS_NAMES)
    best = None
    best_score = float("inf")

    for outer_seed in (SEED,):
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=outer_seed)
        outer_folds = list(outer.split(all_ids, y, groups))
        for fold_index, (remainder_ids, test_ids) in enumerate(outer_folds):
            inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=outer_seed + fold_index)
            remainder_y = y[remainder_ids]
            remainder_groups = groups[remainder_ids]
            for train_rel, val_rel in inner.split(remainder_ids, remainder_y, remainder_groups):
                train_ids = remainder_ids[train_rel]
                val_ids = remainder_ids[val_rel]
                split_ids = {"train": train_ids, "validation": val_ids, "test": test_ids}
                proportions = {"train": 0.68, "validation": 0.16, "test": 0.16}
                score = 0.0
                for split_name, ids in split_ids.items():
                    observed = np.bincount(y[ids], minlength=class_count)
                    expected = np.bincount(y, minlength=class_count) * proportions[split_name]
                    score += float(np.sum(((observed - expected) ** 2) / np.maximum(expected, 1.0)))
                    if np.any(observed == 0):
                        score += 1000.0
                global_ratios = [len(split_ids[key]) / len(frame) for key in proportions]
                target_ratios = list(proportions.values())
                score += float(sum((actual - target) ** 2 for actual, target in zip(global_ratios, target_ratios))) * 100
                split_groups = {key: set(groups[value]) for key, value in split_ids.items()}
                if (
                    split_groups["train"] & split_groups["validation"]
                    or split_groups["train"] & split_groups["test"]
                    or split_groups["validation"] & split_groups["test"]
                ):
                    raise RuntimeError("Grouped split leaked a source group.")
                if score < best_score:
                    best_score = score
                    best = split_ids

    if best is None:
        raise RuntimeError("Could not construct a grouped train/validation/test split.")
    for split_name, ids in best.items():
        frame.loc[ids, "split"] = split_name
    return best


def image_dataset(frame: pd.DataFrame, shuffle: bool = False) -> tf.data.Dataset:
    paths = frame["absolute_path"].astype(str).to_numpy()
    labels = frame["label_id"].to_numpy(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")
        return tf.cast(image, tf.float32), label

    # The audited dataset is small enough to cache decoded/resized images in
    # memory. Augmentation remains in the model, so cached inputs are still
    # transformed differently on each training pass.
    dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE).cache()
    if shuffle:
        dataset = dataset.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def make_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=SEED),
            tf.keras.layers.RandomRotation(0.08, fill_mode="reflect", seed=SEED + 1),
            tf.keras.layers.RandomZoom(0.10, fill_mode="reflect", seed=SEED + 2),
            tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", seed=SEED + 3),
            tf.keras.layers.RandomBrightness(0.08, value_range=(0.0, 255.0), seed=SEED + 4),
            tf.keras.layers.RandomContrast(0.08, seed=SEED + 5),
        ],
        name="train_augmentation",
    )


def make_model(augment: bool) -> tuple[tf.keras.Model, tf.keras.Model | None]:
    inputs = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3), name="rgb_0_255")
    augmentation = make_augmentation() if augment else None
    x = augmentation(inputs) if augmentation is not None else inputs
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="head_batch_norm")(x)
    x = tf.keras.layers.Dropout(0.35, seed=SEED, name="head_dropout")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(0.25, seed=SEED + 1, name="head_dropout_2")(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="category_probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name="AuditedTreeImageCategory")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, backbone


def save_augmentation_preview(frame: pd.DataFrame, augmentation: tf.keras.layers.Layer) -> None:
    selected = []
    for class_name in CLASS_NAMES:
        examples = frame[frame["verified_class"] == class_name].head(2)
        selected.extend(examples["absolute_path"].tolist())
    if not selected:
        return
    batch = []
    for path in selected:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            batch.append(np.asarray(image, dtype=np.float32))
    augmented = augmentation(tf.convert_to_tensor(np.stack(batch)), training=True).numpy()
    cols = len(selected)
    fig, axes = plt.subplots(2, cols, figsize=(3.0 * cols, 6.0), squeeze=False)
    for index, (path, original, transformed) in enumerate(zip(selected, batch, augmented)):
        axes[0, index].imshow(np.clip(original / 255.0, 0, 1))
        axes[0, index].set_title(Path(path).name[:18], fontsize=8)
        axes[1, index].imshow(np.clip(transformed / 255.0, 0, 1))
        axes[1, index].set_title("training augmentation", fontsize=8)
        for axis in (axes[0, index], axes[1, index]):
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "augmentation_examples.png", dpi=150)
    plt.close(fig)


def class_weights(train_frame: pd.DataFrame) -> dict[int, float]:
    labels = train_frame["label_id"].to_numpy(dtype=np.int32)
    present = np.unique(labels)
    values = compute_class_weight(class_weight="balanced", classes=present, y=labels)
    weights = dict(zip(present.tolist(), values.tolist()))
    return {int(key): float(value) for key, value in weights.items()}


def fit_candidate(
    name: str,
    augment: bool,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    weights: dict[int, float],
) -> tuple[tf.keras.Model, pd.DataFrame]:
    # Start both candidates from identical classifier-head and shuffle seeds;
    # their intended difference is the training-only augmentation layer.
    set_seed()
    candidate_dir = MODEL_DIR / "audited_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = candidate_dir / f"{name}.h5"
    train_ds = image_dataset(train_frame, shuffle=True)
    validation_ds = image_dataset(validation_frame)
    model, backbone = make_model(augment)
    history_rows = []

    def callbacks():
        return [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ]

    # With only 72 training images, keep the ImageNet feature extractor frozen
    # to limit overfitting and make both candidates practical to train on CPU.
    # Validation early stopping selects the best epoch; save a complete model
    # once at the end rather than repeatedly writing large checkpoints.
    print(f"\n=== {name}: frozen ImageNet feature extractor ===", flush=True)
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        shuffle=False,
        epochs=10,
        class_weight=weights,
        callbacks=callbacks(),
        verbose=2,
    )
    history_rows.extend(
        {
            "candidate": name,
            "stage": "frozen_backbone",
            "epoch": i + 1,
            **{key: values[i] for key, values in history.history.items()},
        }
        for i in range(len(history.epoch))
    )
    # Keras HDF5 cannot serialize the Random* augmentation Sequential reliably.
    # Keep a reusable no-augmentation baseline; if the augmented model wins,
    # export it through an inference graph without training-time augmentation.
    if not augment:
        model.save(checkpoint, include_optimizer=False)
    return model, pd.DataFrame(history_rows)


def inference_model_from_candidate(model: tf.keras.Model) -> tf.keras.Model:
    """Copy candidate weights into the matching architecture without augmentation."""
    inference_model, _ = make_model(augment=False)
    for target_layer in inference_model.layers:
        source_layer = model.get_layer(target_layer.name)
        target_layer.set_weights(source_layer.get_weights())
    return inference_model


def evaluate_model(model: tf.keras.Model, frame: pd.DataFrame) -> dict:
    dataset = image_dataset(frame)
    y_true = frame["label_id"].to_numpy(dtype=np.int32)
    probabilities = model.predict(dataset, verbose=0)
    predictions = probabilities.argmax(axis=1)
    report = classification_report(
        y_true,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, predictions, labels=list(range(len(CLASS_NAMES))))
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "support": {name: int(report[name]["support"]) for name in CLASS_NAMES},
        "y_true": y_true,
        "predictions": predictions,
        "probabilities": probabilities,
    }


def plot_confusion(matrix: list[list[int]], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Greens", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_history(history: pd.DataFrame) -> None:
    for candidate in history["candidate"].unique():
        current = history[history["candidate"] == candidate]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(current["epoch"], current["accuracy"], label="train")
        axes[0].plot(current["epoch"], current["val_accuracy"], label="validation")
        axes[0].set_title(f"{candidate}: accuracy")
        axes[1].plot(current["epoch"], current["loss"], label="train")
        axes[1].plot(current["epoch"], current["val_loss"], label="validation")
        axes[1].set_title(f"{candidate}: loss")
        for axis in axes:
            axis.set_xlabel("Epoch")
            axis.legend()
            axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(EVAL_DIR / f"{candidate}_training_curves.png", dpi=180)
        plt.close(fig)


def main(preview_only: bool = False, reuse_baseline: bool = False) -> None:
    set_seed()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_records()
    split_ids = make_grouped_split(frame)
    split_frames = {name: frame.iloc[ids].copy().reset_index(drop=True) for name, ids in split_ids.items()}
    for name, values in split_frames.items():
        values["split"] = name
    combined = pd.concat(split_frames.values(), ignore_index=True)

    split_counts = {
        name: {
            "total": len(values),
            "class_counts": dict(Counter(values["verified_class"])),
            "group_count": int(values["group_id"].nunique()),
        }
        for name, values in split_frames.items()
    }
    group_sets = {name: set(values["group_id"]) for name, values in split_frames.items()}
    if (group_sets["train"] & group_sets["validation"] or group_sets["train"] & group_sets["test"] or group_sets["validation"] & group_sets["test"]):
        raise RuntimeError("At least one source group crosses a split.")

    combined[["relative_path", "verified_class", "group_id", "split", "sha256"]].to_csv(
        EVAL_DIR / "split_manifest.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "seed": SEED,
        "dataset_manifest": "reports/image_audit_manifest.csv",
        "training_source": "to review real dataset",
        "excluded_source_folder": "datasets",
        "split_strategy": "Nested 5-fold/4-fold StratifiedGroupKFold; split chosen by class-count balance only; same filename-index groups held together.",
        "grouping_limitations": "Filename number/variant groups approximate capture subjects; original tree/session identifiers were not recorded.",
        "image_size": list(IMAGE_SIZE),
        "preprocessing": "RGB decode, bilinear resize to 224x224, float32 [0,255]; EfficientNetB0 built-in preprocessing. Uploaded images are EXIF-oriented before resize.",
        "resize_method": "bilinear",
        "class_names": CLASS_NAMES,
        "split_counts": split_counts,
        "validated_class_counts": dict(Counter(frame["verified_class"])),
        "augmentation": {
            "mode": "On the fly in the model; training split only",
            "transforms": [
                "horizontal flip, p=0.5",
                "rotation factor 0.08",
                "zoom factor 0.10",
                "translation factor 0.05 each axis",
                "brightness factor 0.08",
                "contrast factor 0.08",
            ],
            "vertical_flip": False,
            "blur_or_noise": False,
            "generated_files": 0,
        },
        "model": "EfficientNetB0 ImageNet weights with a frozen feature extractor; 128-unit dense classification head; dropout 0.35/0.25; sparse categorical cross-entropy; balanced class weights; validation early stopping (maximum 10 epochs). Augmentation is excluded from the exported inference graph.",
        "device": "CPU",
        "initialization": "Cached ImageNet EfficientNetB0 no-top weights, independent of the old project model.",
    }
    (EVAL_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Publish the known grouped split counts before fitting so the dashboard
    # never has to display placeholder split totals during a long CPU run.
    stats = json.loads(DATASET_STATS.read_text(encoding="utf-8"))
    stats.update(
        validated_images=len(frame),
        class_counts=dict(Counter(frame["verified_class"])),
        train_count=split_counts["train"]["total"],
        validation_count=split_counts["validation"]["total"],
        test_count=split_counts["test"]["total"],
        split_class_counts={key: value["class_counts"] for key, value in split_counts.items()},
    )
    DATASET_STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    aug = make_augmentation()
    save_augmentation_preview(split_frames["train"], aug)
    if preview_only:
        print(json.dumps({"split_counts": split_counts, "preview": str(EVAL_DIR / "augmentation_examples.png")}, indent=2))
        return

    weights = class_weights(split_frames["train"])
    print("TensorFlow", tf.__version__, "GPU devices", tf.config.list_physical_devices("GPU"), flush=True)
    print("Split counts", json.dumps(split_counts), flush=True)
    print("Class weights", weights, flush=True)

    baseline_checkpoint = MODEL_DIR / "audited_candidates" / "baseline.h5"
    if reuse_baseline and baseline_checkpoint.exists():
        print(f"Reusing completed baseline checkpoint: {baseline_checkpoint}", flush=True)
        baseline_model = tf.keras.models.load_model(baseline_checkpoint, compile=False)
        baseline_history = pd.DataFrame()
    else:
        baseline_model, baseline_history = fit_candidate(
            "baseline", False, split_frames["train"], split_frames["validation"], weights
        )
    augmented_model, augmented_history = fit_candidate(
        "augmented", True, split_frames["train"], split_frames["validation"], weights
    )
    history = pd.concat([baseline_history, augmented_history], ignore_index=True)
    history.to_csv(EVAL_DIR / "training_history.csv", index=False)
    plot_history(history)

    # Validation selects the final candidate. The test set is not used to select it.
    val_results = {
        "baseline": evaluate_model(baseline_model, split_frames["validation"]),
        "augmented": evaluate_model(augmented_model, split_frames["validation"]),
    }
    selected_name = max(val_results, key=lambda key: (val_results[key]["macro_f1"], val_results[key]["accuracy"]))
    selected_model = baseline_model if selected_name == "baseline" else inference_model_from_candidate(augmented_model)

    # One final comparative pass over the same untouched grouped holdout.
    test_results = {
        "baseline": evaluate_model(baseline_model, split_frames["test"]),
        "augmented": evaluate_model(augmented_model, split_frames["test"]),
    }
    final_result = test_results[selected_name]
    final_model_path = MODEL_DIR / "TreeCategory_EfficientNetB0.h5"
    selected_model.save(final_model_path, include_optimizer=False)
    model_metadata = {
        "model_path": final_model_path.name,
        "class_names": CLASS_NAMES,
        "input_size": list(IMAGE_SIZE),
        "color_mode": "RGB",
        "resize_method": "bilinear",
        "input_range": [0, 255],
        "preprocessing": "EfficientNetB0 application preprocessing is embedded in the model.",
        "uncertainty_threshold": 0.45,
        "confidence_interpretation": "Uncalibrated softmax score used only for image-category ranking and a fixed abstention threshold.",
        "selected_candidate": selected_name,
        "selected_by": "validation macro F1; ties broken by validation accuracy",
        "task": "coarse tree-image framing/content category, not species or disease diagnosis",
    }
    (MODEL_DIR / "TreeCategory_EfficientNetB0.metadata.json").write_text(
        json.dumps(model_metadata, indent=2), encoding="utf-8"
    )

    safe_metrics = {}
    for candidate in ("baseline", "augmented"):
        safe_metrics[candidate] = {
            split_name: {
                "accuracy": float(results["accuracy"]),
                "macro_f1": float(results["macro_f1"]),
                "weighted_f1": float(results["weighted_f1"]),
                "support": results["support"],
                "classification_report": results["classification_report"],
                "confusion_matrix": results["confusion_matrix"],
            }
            for split_name, results in (("validation", val_results[candidate]), ("test", test_results[candidate]))
        }
        plot_confusion(
            test_results[candidate]["confusion_matrix"],
            EVAL_DIR / f"{candidate}_confusion_matrix_test.png",
            f"{candidate.title()} model: grouped test",
        )
    (EVAL_DIR / "model_comparison.json").write_text(
        json.dumps({"selected_candidate": selected_name, "validation_and_test": safe_metrics}, indent=2),
        encoding="utf-8",
    )
    (EVAL_DIR / "classification_report_test.json").write_text(
        json.dumps(final_result["classification_report"], indent=2), encoding="utf-8"
    )
    plot_confusion(final_result["confusion_matrix"], EVAL_DIR / "confusion_matrix.png", "Selected model: grouped test")
    error_rows = []
    for eval_index, (true_id, pred_id, row) in enumerate(
        zip(final_result["y_true"], final_result["predictions"], split_frames["test"].to_dict("records"))
    ):
        if int(true_id) != int(pred_id):
            error_rows.append({
                "relative_path": row["relative_path"],
                "true_label": CLASS_NAMES[int(true_id)],
                "predicted_label": CLASS_NAMES[int(pred_id)],
                "group_id": row["group_id"],
                "confidence": float(final_result["probabilities"][eval_index, int(pred_id)]),
            })
    pd.DataFrame(error_rows, columns=["relative_path", "true_label", "predicted_label", "group_id", "confidence"]).to_csv(
        EVAL_DIR / "misclassified_test_images.csv", index=False, encoding="utf-8-sig"
    )

    stats = json.loads(DATASET_STATS.read_text(encoding="utf-8"))
    stats.update(
        validated_images=len(frame),
        class_counts=dict(Counter(frame["verified_class"])),
        train_count=split_counts["train"]["total"],
        validation_count=split_counts["validation"]["total"],
        test_count=split_counts["test"]["total"],
        split_class_counts={key: value["class_counts"] for key, value in split_counts.items()},
        selected_candidate=selected_name,
        test_accuracy=final_result["accuracy"],
        macro_f1=final_result["macro_f1"],
        weighted_f1=final_result["weighted_f1"],
        augmentation_test_accuracy=test_results["augmented"]["accuracy"],
        baseline_test_accuracy=test_results["baseline"]["accuracy"],
    )
    DATASET_STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "reports" / "training_metrics.json").write_text(
        json.dumps(
            {
                "selected_candidate": selected_name,
                "selection_metric": "validation macro F1, then validation accuracy",
                "training_source": "to review real dataset",
                "split_strategy": metadata["split_strategy"],
                "validation": safe_metrics[selected_name]["validation"],
                "test": safe_metrics[selected_name]["test"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    shutil.copy2(EVAL_DIR / "confusion_matrix.png", ROOT / "reports" / "confusion_matrix.png")
    print(json.dumps({"selected_candidate": selected_name, "test": safe_metrics[selected_name]["test"], "model": str(final_model_path)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-only", action="store_true", help="Create grouped split metadata and augmentation samples without training.")
    parser.add_argument("--reuse-baseline", action="store_true", help="Reuse a completed baseline.h5 checkpoint before fitting the augmented candidate.")
    args = parser.parse_args()
    main(preview_only=args.preview_only, reuse_baseline=args.reuse_baseline)
