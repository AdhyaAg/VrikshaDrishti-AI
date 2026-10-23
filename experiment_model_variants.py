"""Compare compact EfficientNetB0 classifier heads on the audited grouped split.

This is an exploratory run: it reads the published train/validation/test split,
uses augmentation on training images only, selects by validation macro F1, and
evaluates the selected candidate on the held-out test split once.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import train_audited_model as pipeline


ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "reports" / "evaluation_v2" / "split_manifest.csv"
ACTIVE_MODEL_PATH = ROOT / "models" / "TreeCategory_EfficientNetB0.h5"
CANDIDATE_DIR = ROOT / "models" / "audited_candidates"
OUTPUT_PATH = ROOT / "reports" / "evaluation_v2" / "head_variant_comparison.json"


def load_split_frames() -> dict[str, pd.DataFrame]:
    manifest = pd.read_csv(SPLIT_PATH, keep_default_na=False)
    manifest["absolute_path"] = manifest["relative_path"].map(
        lambda relative: str(ROOT / "to review real dataset" / relative)
    )
    manifest["label_id"] = manifest["verified_class"].map(
        {name: index for index, name in enumerate(pipeline.CLASS_NAMES)}
    )
    if manifest["label_id"].isna().any():
        raise ValueError("The split manifest contains an unknown class.")
    if not all(Path(path).is_file() for path in manifest["absolute_path"]):
        raise FileNotFoundError("At least one image in the published split is missing.")
    return {
        name: manifest[manifest["split"] == name].reset_index(drop=True)
        for name in ("train", "validation", "test")
    }


def make_variant(variant: str, source_backbone_weights: list[np.ndarray]) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(*pipeline.IMAGE_SIZE, 3), name="rgb_0_255")
    augmented = pipeline.make_augmentation()(inputs)
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(*pipeline.IMAGE_SIZE, 3),
    )
    backbone.set_weights(source_backbone_weights)
    backbone.trainable = False
    x = backbone(augmented, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="head_batch_norm")(x)

    if variant == "linear_augmented":
        x = tf.keras.layers.Dropout(0.4, seed=pipeline.SEED, name="head_dropout")(x)
        outputs = tf.keras.layers.Dense(
            len(pipeline.CLASS_NAMES),
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(1e-3),
            name="category_probabilities",
        )(x)
    elif variant == "compact_augmented":
        x = tf.keras.layers.Dropout(0.35, seed=pipeline.SEED, name="head_dropout")(x)
        x = tf.keras.layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-3),
            name="head_dense",
        )(x)
        x = tf.keras.layers.Dropout(0.35, seed=pipeline.SEED + 1, name="head_dropout_2")(x)
        outputs = tf.keras.layers.Dense(
            len(pipeline.CLASS_NAMES), activation="softmax", name="category_probabilities"
        )(x)
    elif variant == "balanced_augmented":
        x = tf.keras.layers.Dropout(0.35, seed=pipeline.SEED, name="head_dropout")(x)
        x = tf.keras.layers.Dense(128, activation="relu", name="head_dense")(x)
        x = tf.keras.layers.Dropout(0.25, seed=pipeline.SEED + 1, name="head_dropout_2")(x)
        outputs = tf.keras.layers.Dense(
            len(pipeline.CLASS_NAMES), activation="softmax", name="category_probabilities"
        )(x)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    model = tf.keras.Model(inputs, outputs, name=f"AuditedTreeImageCategory_{variant}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def to_inference_model(model: tf.keras.Model, variant: str) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(*pipeline.IMAGE_SIZE, 3), name="rgb_0_255")
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(*pipeline.IMAGE_SIZE, 3),
    )
    backbone.trainable = False
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="head_batch_norm")(x)

    if variant == "linear_augmented":
        x = tf.keras.layers.Dropout(0.4, seed=pipeline.SEED, name="head_dropout")(x)
        outputs = tf.keras.layers.Dense(
            len(pipeline.CLASS_NAMES),
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(1e-3),
            name="category_probabilities",
        )(x)
    elif variant == "compact_augmented":
        x = tf.keras.layers.Dropout(0.35, seed=pipeline.SEED, name="head_dropout")(x)
        x = tf.keras.layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-3),
            name="head_dense",
        )(x)
        x = tf.keras.layers.Dropout(0.35, seed=pipeline.SEED + 1, name="head_dropout_2")(x)
        outputs = tf.keras.layers.Dense(
            len(pipeline.CLASS_NAMES), activation="softmax", name="category_probabilities"
        )(x)
    else:
        x = tf.keras.layers.Dropout(0.35, seed=pipeline.SEED, name="head_dropout")(x)
        x = tf.keras.layers.Dense(128, activation="relu", name="head_dense")(x)
        x = tf.keras.layers.Dropout(0.25, seed=pipeline.SEED + 1, name="head_dropout_2")(x)
        outputs = tf.keras.layers.Dense(
            len(pipeline.CLASS_NAMES), activation="softmax", name="category_probabilities"
        )(x)

    target = tf.keras.Model(inputs, outputs, name=f"AuditedTreeImageCategory_{variant}_inference")
    target.set_weights(model.get_weights())
    return target


def balanced_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    target_count = int(frame["verified_class"].value_counts().max())
    chunks = []
    for class_name in pipeline.CLASS_NAMES:
        examples = frame[frame["verified_class"] == class_name]
        if examples.empty:
            raise ValueError(f"Training split has no images for class {class_name}.")
        repeats = int(np.ceil(target_count / len(examples)))
        chunks.append(pd.concat([examples] * repeats, ignore_index=True).iloc[:target_count])
    return pd.concat(chunks, ignore_index=True).sample(frac=1, random_state=pipeline.SEED).reset_index(drop=True)


def main(variants: list[str]) -> None:
    tf.keras.utils.set_random_seed(pipeline.SEED)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    splits = load_split_frames()

    active = tf.keras.models.load_model(ACTIVE_MODEL_PATH, compile=False)
    active_validation = pipeline.evaluate_model(active, splits["validation"])
    baseline_validation = {
        key: float(active_validation[key]) for key in ("accuracy", "macro_f1", "weighted_f1")
    }
    backbone_weights = active.get_layer("efficientnetb0").get_weights()
    weights = pipeline.class_weights(splits["train"])
    validation_ds = pipeline.image_dataset(splits["validation"])
    validation_results: dict[str, dict] = {}
    best_model: tf.keras.Model | None = None
    best_variant: str | None = None
    best_variant_metrics: tuple[float, float] = (-1.0, -1.0)

    balanced_frame = balanced_training_frame(splits["train"])
    for variant in variants:
        tf.keras.utils.set_random_seed(pipeline.SEED)
        model = make_variant(variant, backbone_weights)
        train_frame = balanced_frame if variant == "balanced_augmented" else splits["train"]
        train_ds = pipeline.image_dataset(train_frame, shuffle=True)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
            ),
        ]
        print(f"\n=== {variant} ===", flush=True)
        history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=20,
            class_weight=None if variant == "balanced_augmented" else weights,
            callbacks=callbacks,
            verbose=2,
        )
        result = pipeline.evaluate_model(model, splits["validation"])
        validation_results[variant] = {
            "accuracy": float(result["accuracy"]),
            "macro_f1": float(result["macro_f1"]),
            "weighted_f1": float(result["weighted_f1"]),
            "epochs_run": len(history.epoch),
            "support": result["support"],
        }
        print(json.dumps({"variant": variant, "validation": validation_results[variant]}, indent=2), flush=True)
        candidate_metrics = (
            validation_results[variant]["macro_f1"], validation_results[variant]["accuracy"]
        )
        if candidate_metrics > best_variant_metrics:
            best_variant = variant
            best_variant_metrics = candidate_metrics
            best_model = model
        else:
            del model

    baseline_metrics = (baseline_validation["macro_f1"], baseline_validation["accuracy"])
    selected_name = best_variant if best_variant_metrics > baseline_metrics else "baseline"
    test_result = None
    if selected_name != "baseline":
        assert best_model is not None and best_variant is not None
        # Test is evaluated only after validation has selected a candidate.
        test = pipeline.evaluate_model(best_model, splits["test"])
        test_result = {
            key: float(test[key]) for key in ("accuracy", "macro_f1", "weighted_f1")
        }
        test_result["support"] = test["support"]
        inference_model = to_inference_model(best_model, best_variant)
        inference_path = CANDIDATE_DIR / f"{best_variant}_inference.h5"
        inference_model.save(inference_path, include_optimizer=False)
        validation_results[best_variant]["inference_model_path"] = str(inference_path.relative_to(ROOT))

    payload = {
        "selection_rule": "Validation macro F1, ties broken by validation accuracy; baseline retained unless a candidate beats it.",
        "split_source": "reports/evaluation_v2/split_manifest.csv",
        "training_images": len(splits["train"]),
        "effective_training_samples_per_epoch": {
            "original": len(splits["train"]),
            "balanced_augmented": len(balanced_frame),
        },
        "validation_images": len(splits["validation"]),
        "test_images": len(splits["test"]),
        "augmentation": "Existing randomized on-the-fly augmentation; training split only.",
        "baseline_validation": baseline_validation,
        "candidates_validation": validation_results,
        "selected_by_validation": selected_name,
        "selected_candidate_test": test_result,
        "promoted_to_active_model": False,
    }
    if OUTPUT_PATH.exists():
        previous = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        prior_variants = previous.get("candidates_validation", {})
        payload["candidates_validation"] = {**prior_variants, **validation_results}
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("linear_augmented", "compact_augmented", "balanced_augmented"),
        action="append",
        help="Run only selected candidate variants; may be passed more than once.",
    )
    args = parser.parse_args()
    main(args.variant or ["linear_augmented", "compact_augmented", "balanced_augmented"])
