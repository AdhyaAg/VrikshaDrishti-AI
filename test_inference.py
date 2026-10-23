"""Smoke-check the Streamlit inference path on the untouched audited test split."""

import csv
import math
from pathlib import Path

from utils.predictor import predict_tree_health

ROOT = Path(__file__).resolve().parent
SPLIT_MANIFEST = ROOT / "reports" / "evaluation_v2" / "split_manifest.csv"
IMAGE_ROOT = ROOT / "to review real dataset"


def main():
    if not SPLIT_MANIFEST.exists():
        raise FileNotFoundError("Run train_audited_model.py to create the grouped test manifest first.")
    with SPLIT_MANIFEST.open("r", newline="", encoding="utf-8-sig") as handle:
        test_rows = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    if not test_rows:
        raise RuntimeError("The audited grouped test split is empty.")

    correct = 0
    for row in test_rows:
        image_path = IMAGE_ROOT / row["relative_path"]
        result = predict_tree_health(str(image_path))
        if result["health_score"] is not None:
            raise AssertionError("The category model must not return a health score.")
        if not result["top5"] or not all(math.isfinite(float(score)) for _, score in result["top5"]):
            raise AssertionError(f"Invalid category probabilities for {row['relative_path']}.")
        predicted = result["top5"][0][0]
        correct += predicted == row["verified_class"]
        print(f"{row['relative_path']}: expected={row['verified_class']} predicted={predicted} top_class_score={result['confidence']:.1f}% category_status={result['status']}")

    print(f"Inference smoke check: {len(test_rows)} primary Real Dataset test images; argmax agreement={correct}/{len(test_rows)} ({correct / len(test_rows):.1%}).")
    print("Species and health are not inferred; low-confidence category results can abstain as unclear.")


if __name__ == "__main__":
    main()
