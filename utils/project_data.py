"""Read the saved audit and evaluation artifacts used by the existing UI."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATS_PATH = ROOT / "reports" / "dataset_statistics.json"
RUN_METADATA_PATH = ROOT / "reports" / "evaluation_v2" / "run_metadata.json"
COMPARISON_PATH = ROOT / "reports" / "evaluation_v2" / "model_comparison.json"
METADATA_PATH = ROOT / "models" / "TreeCategory_EfficientNetB0.metadata.json"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_project_statistics() -> dict:
    stats = _read_json(STATS_PATH) or {
        "source_original_images": 0,
        "source_unique_images": 0,
        "validated_images": 0,
        "class_counts": {},
        "train_count": 0,
        "validation_count": 0,
        "test_count": 0,
        "pending_review": 0,
        "excluded": 0,
        "species_verified_count": 0,
        "health_verified_count": 0,
    }
    # The split is fixed and published before model fitting. Read it from the
    # run metadata if a long training run has not yet written final metrics.
    run = _read_json(RUN_METADATA_PATH)
    if run and run.get("split_counts"):
        split_counts = run["split_counts"]
        field_names = {"train": "train_count", "validation": "validation_count", "test": "test_count"}
        for split, field in field_names.items():
            if not stats.get(field):
                stats[field] = split_counts.get(split, {}).get("total", 0)
        stats.setdefault("split_class_counts", {
            split: values.get("class_counts", {}) for split, values in split_counts.items()
        })
    return stats


def load_model_comparison() -> dict | None:
    """Return saved baseline-versus-augmentation metrics for the UI."""
    payload = _read_json(COMPARISON_PATH)
    if payload and payload.get("validation_and_test"):
        return payload
    return None


def load_model_summary() -> dict | None:
    payload = load_model_comparison()
    if not payload:
        return None
    candidate = payload.get("selected_candidate")
    selected = payload.get("validation_and_test", {}).get(candidate, {})
    test = selected.get("test")
    if not test:
        return None
    return {
        "candidate": candidate,
        "test": test,
        "validation": selected.get("validation", {}),
    }


def load_active_model_metadata() -> dict:
    metadata = _read_json(METADATA_PATH)
    if metadata:
        return metadata
    return {
        "class_names": ["leafs", "saplings", "trees", "unclear"],
        "task": "legacy image-category model",
    }


def percentage(part: int | float, total: int | float) -> str:
    if not total:
        return "0%"
    return f"{100 * part / total:.0f}%"


def display_class_name(name: str) -> str:
    return {
        "leafs": "Leaf-focused",
        "saplings": "Sapling-focused",
        "trees": "Tree-focused",
        "unclear": "Uncertain",
    }.get(name, name.replace("_", " ").capitalize())
