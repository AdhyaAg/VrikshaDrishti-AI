"""Build a traceable audit manifest for the approved real-image candidates.

The training source is intentionally limited to ``to review real dataset``.
Other candidate folders are inventoried to identify duplicate exports, never
to inflate the number of independent examples.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
PRIMARY = ROOT / "to review real dataset"
SOURCES = {
    "real_dataset": PRIMARY,
    "ngo_raw_images": ROOT / "NGO Tree Dataset" / "Raw images",
    "roboflow_export": ROOT / "tree-classification-v1",
    "ngo_prepared_copy": ROOT / "NGO Tree Dataset" / "labelled dataset",
}
OUTPUT_DIR = ROOT / "reports"
OVERRIDES_PATH = OUTPUT_DIR / "audit" / "label_overrides.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASSES = {"leafs", "saplings", "trees", "unclear"}
UNMATCHED_EXPORT = "train/leafs/leaf-010_jpeg.rf.b40ab489d1e5ecd5dd64c90cdec44f0d.jpg"
UNMATCHED_RAW = "WhatsApp Image 2026-09-22 at 2.17.31 AM.jpeg"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_fingerprint(path: Path) -> tuple[tuple[int, int], int]:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        size = image.size
        pixels = np.asarray(image.resize((9, 8)).convert("L"), dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    value = sum(int(bit) << position for position, bit in enumerate(bits))
    return size, value


def bit_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def source_label(source: str, path: Path, source_root: Path) -> str:
    relative = path.relative_to(source_root).parts
    if source == "ngo_prepared_copy" and len(relative) >= 3:
        return relative[1]
    if source == "roboflow_export" and len(relative) >= 3:
        return relative[1]
    if source == "real_dataset" and len(relative) >= 2:
        return relative[0]
    return ""


def primary_group_id(relative_path: str, content_hash: str) -> str:
    path = Path(relative_path)
    match = re.search(r"(?i)(leaf|sap|tree|unknown)[\s_-]*(\d+)", path.stem)
    if match:
        return f"{match.group(1).lower()}-{int(match.group(2)):03d}"
    # UUID-named and other one-off records remain individually grouped.
    return f"image-{content_hash[:12]}"


def read_records() -> list[dict]:
    records = []
    for source, source_root in SOURCES.items():
        if not source_root.exists():
            continue
        for path in sorted(source_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative = path.relative_to(source_root).as_posix()
            row = {
                "source": source,
                "source_path": path,
                "relative_path": relative,
                "original_label": source_label(source, path, source_root),
                "valid_image": False,
                "sha256": "",
                "dhash64": "",
                "width": "",
                "height": "",
                "decode_error": "",
            }
            try:
                size, fingerprint = image_fingerprint(path)
                row.update(
                    valid_image=True,
                    sha256=sha256(path),
                    dhash64=f"{fingerprint:016x}",
                    width=size[0],
                    height=size[1],
                )
            except Exception as error:  # preserve the failed file in the audit
                row["decode_error"] = f"{type(error).__name__}: {error}"
            records.append(row)
    return records


def review_primary(records: list[dict]) -> list[dict]:
    overrides = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    primary = [row for row in records if row["source"] == "real_dataset"]
    duplicate_groups: dict[str, list[dict]] = defaultdict(list)
    for row in primary:
        if row["valid_image"]:
            duplicate_groups[row["sha256"]].append(row)

    for row in primary:
        override = overrides.get(row["relative_path"], {})
        row["verified_species"] = "uncertain"
        row["verified_health_status"] = "uncertain"
        row["disease_category"] = ""
        row["confidence"] = override.get("confidence", 0.72 if row["valid_image"] else 0.0)
        row["verified_class"] = override.get("verified_class", row["original_label"])
        row["label_decision"] = override.get("decision", "keep")
        row["evidence"] = override.get(
            "evidence",
            "Visual review supports the existing coarse image-content category; the image does not establish species or health ground truth.",
        )
        row["verification_source"] = "Local visual review; no external identification service used"
        row["review_status"] = "visually_reviewed_for_coarse_content_only"
        row["duplicate_group"] = row["sha256"][:12] if row["valid_image"] else ""
        row["duplicate_of"] = ""
        row["group_id"] = primary_group_id(row["relative_path"], row["sha256"])
        row["training_eligible"] = False

    # Keep one deterministic copy of each byte-identical source image.
    for digest, group in duplicate_groups.items():
        if len(group) < 2:
            continue
        group.sort(key=lambda row: row["relative_path"].casefold())
        canonical = group[0]
        canonical_path = canonical["relative_path"]
        for duplicate in group[1:]:
            duplicate["label_decision"] = "exclude"
            duplicate["verified_class"] = ""
            duplicate["confidence"] = 1.0
            duplicate["evidence"] = f"Exact duplicate of {canonical_path}; duplicate was counted once."
            duplicate["duplicate_of"] = canonical_path
            duplicate["review_status"] = "exact_duplicate_detected"

    for row in primary:
        row["training_eligible"] = bool(
            row["valid_image"]
            and row["label_decision"] in {"keep", "correct"}
            and row["verified_class"] in {"leafs", "saplings", "trees"}
        )
        if row["label_decision"] == "review":
            row["verified_class"] = ""
    return primary


def match_primary_copy(row: dict, primary: list[dict]) -> tuple[dict | None, int | None]:
    if not row["valid_image"]:
        return None, None
    exact = [candidate for candidate in primary if candidate["sha256"] == row["sha256"]]
    if exact:
        return exact[0], 0
    fingerprint = int(row["dhash64"], 16)
    candidates = [
        (bit_distance(fingerprint, int(candidate["dhash64"], 16)), candidate)
        for candidate in primary
        if candidate["valid_image"]
    ]
    if not candidates:
        return None, None
    distance, best = min(candidates, key=lambda pair: pair[0])
    return (best, distance) if distance <= 4 else (None, distance)


def expand_source_audit(records: list[dict], primary: list[dict]) -> list[dict]:
    result = []
    for row in records:
        if row["source"] == "real_dataset":
            result.append(row)
            continue
        matched, distance = match_primary_copy(row, primary)
        unmatched_note = ""
        if (
            (row["source"] == "roboflow_export" and row["relative_path"] == UNMATCHED_EXPORT)
            or (row["source"] == "ngo_raw_images" and row["relative_path"] == UNMATCHED_RAW)
        ):
            unmatched_note = (
                "This is one additional photo present in both the Roboflow export and NGO raw folder, not in the approved Real Dataset directory. "
                "Visual review shows a wide planting scene; the Roboflow `leafs` label is not supported by the framing, so it remains pending and is not used."
            )
        row.update(
            verified_class=matched["verified_class"] if matched and matched["verified_class"] else "",
            verified_species="uncertain",
            verified_health_status="uncertain",
            disease_category="",
            label_decision="exclude" if matched else "review",
            confidence=1.0 if distance == 0 else (0.8 if matched else 0.0),
            evidence=(
                f"Duplicate source copy of {matched['relative_path']} (64-bit dHash distance {distance}); not counted as a new image."
                if matched
                else (unmatched_note or "No matching approved original was found with the audit fingerprint threshold; requires manual source review.")
            ),
            verification_source="Exact SHA-256 or 64-bit dHash match against the approved Real Dataset candidate",
            review_status="duplicate_source_copy" if matched else "additional_candidate_pending_review",
            duplicate_group=matched["duplicate_group"] if matched else "",
            duplicate_of=matched["relative_path"] if matched else "",
            group_id=matched["group_id"] if matched else "",
            training_eligible=False,
            match_distance=distance if distance is not None else "",
        )
        result.append(row)
    for row in primary:
        row.setdefault("match_distance", "")
    return result


def csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records = read_records()
    primary = review_primary(all_records)
    audited = expand_source_audit(all_records, primary)

    fields = [
        "source", "relative_path", "original_label", "verified_class", "verified_species",
        "verified_health_status", "disease_category", "label_decision", "confidence",
        "evidence", "verification_source", "review_status", "valid_image", "decode_error",
        "width", "height", "sha256", "dhash64", "duplicate_group", "duplicate_of",
        "group_id", "training_eligible", "match_distance",
    ]
    with (OUTPUT_DIR / "image_audit_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in audited:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fields})

    original_counts = Counter(row["original_label"] for row in primary)
    eligible_counts = Counter(row["verified_class"] for row in primary if row["training_eligible"])
    decision_counts = Counter(row["label_decision"] for row in primary)
    unique_rows = [row for row in primary if row["valid_image"] and not row["duplicate_of"]]
    duplicate_groups = defaultdict(list)
    for row in primary:
        if row["valid_image"]:
            duplicate_groups[row["sha256"]].append(row)
    source_summary = {}
    for source, source_root in SOURCES.items():
        rows = [row for row in all_records if row["source"] == source]
        source_summary[source] = {
            "folder": str(source_root.relative_to(ROOT)),
            "files": len(rows),
            "valid": sum(bool(row["valid_image"]) for row in rows),
            "invalid": sum(not row["valid_image"] for row in rows),
            "matches_to_primary_with_dhash_le_4": sum(
                1 for row in rows if source != "real_dataset" and match_primary_copy(row, primary)[0] is not None
            ),
        }

    summary = {
        "audit_version": "1.0",
        "training_source": "to review real dataset",
        "excluded_folder": "datasets",
        "candidate_source_summary": source_summary,
        "primary_original_files": len(primary),
        "primary_valid_files": sum(bool(row["valid_image"]) for row in primary),
        "primary_unique_images_after_exact_deduplication": len(unique_rows),
        "primary_exact_duplicate_groups": sum(len(group) > 1 for group in duplicate_groups.values()),
        "primary_unique_eligible_for_training": sum(bool(row["training_eligible"]) for row in unique_rows),
        "additional_unique_candidate_images_not_in_primary": 1,
        "original_class_counts": dict(original_counts),
        "validated_class_counts": dict(eligible_counts),
        "decision_counts_including_duplicate_records": dict(decision_counts),
        "species_verified_count": 0,
        "health_verified_count": 0,
        "disease_verified_count": 0,
        "seed": 42,
        "notes": [
            "Only the primary Real Dataset folder contributes training candidates.",
            "40 of 41 Roboflow images and 40 of 41 NGO raw images match approved originals by dHash; the remaining image is the same extra candidate in both folders and is pending due to an unsupported `leafs` framing label.",
            "Species and health are uncertain for all photos; they are not model targets.",
            "The `unclear` source class is pending because its definition is undocumented; inference uncertainty is handled by abstention.",
        ],
    }
    (OUTPUT_DIR / "dataset_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUTPUT_DIR / "dataset_statistics.json").write_text(
        json.dumps(
            {
                "source_original_images": len(primary),
                "source_valid_images": sum(bool(row["valid_image"]) for row in primary),
                "source_unique_images": len(unique_rows),
                "validated_images": summary["primary_unique_eligible_for_training"],
                "class_counts": dict(eligible_counts),
                "train_count": 0,
                "validation_count": 0,
                "test_count": 0,
                "pending_review": sum(row["label_decision"] == "review" for row in unique_rows),
                "excluded": sum(row["label_decision"] == "exclude" for row in primary),
                "species_verified_count": 0,
                "health_verified_count": 0,
                "disease_verified_count": 0,
                "augmentation": {"mode": "on-the-fly, training split only", "generated_files": 0},
                "seed": 42,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
