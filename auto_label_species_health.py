"""Deprecated entry point: species and health pseudo-labels are not ground truth.

Use audit_real_dataset.py to build the reviewed image-category manifest, then
train_audited_model.py to train only on accepted image-category labels.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    print("This legacy auto-labeler is disabled: filename hashes do not identify species, and color/contour heuristics do not diagnose plant health.")
    print(f"Reviewed category manifest: {ROOT / 'reports' / 'image_audit_manifest.csv'}")
    print("Run audit_real_dataset.py and train_audited_model.py for the supported workflow.")
