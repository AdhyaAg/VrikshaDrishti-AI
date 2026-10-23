# Project Audit and Model Validation

Audit date: 2026-10-10

## Findings from the original project

- The application is Streamlit, with the home/prediction/admin views in `app.py` and separate pages under `pages/`.
- The previous active model was `models/Real_TreeClassifier.keras`, an EfficientNetB0 224×224 RGB classifier with four directory labels: `leafs`, `saplings`, `trees`, and `unclear`.
- The saved legacy test report covered 13 images and had 0.769 test accuracy and 0.568 macro F1. The test was small, and related filename-numbered photos crossed its splits.
- App copy described species and disease diagnosis, and the predictor produced health scores from color/contour heuristics. `auto_label_species_health.py` assigned species using a filename hash and health using unvalidated thresholds. Neither is ground truth.
- The prior dashboard displayed hardcoded category counts, a stale 84.21% accuracy claim, and invented inspection records.

## Dataset audit

The primary source is the project’s `to review real dataset` folder. NGO Raw images, the Roboflow export, and the prepared NGO split were audited for overlap and decoding, but were not used as training sources.

- Primary source: 128 readable files; 126 unique photos after removing two exact duplicate pairs.
- Reviewed labels: 116 unique photos accepted for the three visually reviewable framing/content categories: 54 leaf-focused, 21 sapling-focused, and 41 tree-focused.
- Decisions: 7 unique primary photos remain for review; 3 unique photos are excluded, plus 2 duplicate copies. One conflicting exact duplicate was retained once.
- NGO Raw images and Roboflow each contain 41 readable images; 40 from each match approved primary photos by dHash. The remaining photo is the same candidate in both exports and remains pending because its folder label is unsupported by the visible framing.
- The 126 prepared NGO split files match the deduplicated primary photos. Their old split separates same-number filename variants and is not used by the audited training workflow.
- Species, health, and disease labels are unverified for every photo. The undocumented `unclear` class is not a training target; inference uses a low-score abstention result instead.
- The `datasets/` folder is excluded from all training, validation, test, and dashboard statistics.

Manual visual review used contact sheets of all primary photos. The reviewed labels describe photo framing/content only; borderline images remain pending or excluded. Reviewed decisions and evidence are in `reports/audit/label_overrides.json`; every candidate-source image and its provenance is in `reports/image_audit_manifest.csv`.

## Audited training design

- Training candidates come only from the 116 accepted, deduplicated primary-source rows.
- A fixed seed (42) and nested `StratifiedGroupKFold` choose the split by class-count balance only. Same-number filename variants are kept together; exact duplicates do not cross splits.
- Split: 72 train, 22 validation, and 22 held-out test images. Class supports are recorded in `reports/evaluation_v2/run_metadata.json` and `split_manifest.csv`.
- Both candidates start from the same cached ImageNet EfficientNetB0 initialization and train a regularized classification head with class-balanced weights. The feature extractor stays frozen because only 72 reviewed training images are available; early stopping uses validation loss (maximum 10 epochs).
- Baseline and augmented candidates use the same split. Mild horizontal flip, rotation, zoom, translation, brightness, and contrast augmentation runs inside the model on training images only. Validation and test images remain unaugmented; no generated image files are added to the source data.
- Validation macro F1 selects the final candidate (ties use validation accuracy). The grouped test is evaluated afterward and does not select the candidate.

## Application changes

The app now uses `models/TreeCategory_EfficientNetB0.h5` and its paired metadata when present. Existing model files remain as fallback. Predictions describe leaf-focused, sapling-focused, or tree-focused images only; low top-class scores return an uncertain result. Species identification, health scoring, disease diagnosis, and treatment recommendations have been removed from the live inference path and PDF copy.

Both existing analytics views read counts and evaluation output from the audit artifacts. Hardcoded sample inspection logs and unsupported accuracy claims were removed. The existing CSS, page structure, and card/chart layout were preserved. The legacy pseudo-labeling, leaky training, and old evaluator scripts are disabled; previous pseudo-label records and baseline metrics are retained with explicit legacy/unverified names.

## Verification

- `audit_real_dataset.py` completed: all 128 primary images decode; the audit manifest contains 116 eligible primary-source rows and zero verified species/health labels.
- Split validation: 72/22/22; no exact image hashes or filename groups cross splits.
- Python compilation succeeded for the app, audit/training scripts, pages, and utilities.
- The category-report PDF smoke check produced a valid PDF.
- `test_inference.py` ran the app inference path on all 22 untouched grouped test images; its 11/22 argmax agreement matches the selected model's 50.0% held-out accuracy. Probabilities were finite and no health score was returned.
- A fresh loopback-only Streamlit process rendered Home, Analytics, predictor, and About at 1280×720. Uploading held-out `leaf 001.jpeg` returned “Sapling-focused” at 48.19%, matching the standalone app inference path; the result included no species or health diagnosis. The browser console contained no errors. Before/after screenshots showed the existing theme and layout intact while the content and data updated.

## Limitations

- The dataset is small and category boundaries are subjective; even the grouped test has only 22 images.
- Filename-number groups approximate related captures. Original tree/session IDs were not supplied, so this split cannot guarantee independence by physical tree or capture session.
- Softmax scores are uncalibrated; the 45% threshold is a fixed abstention rule, not a clinical or plant-health confidence measure.
- The native environment has no GPU, so candidate training runs on CPU.
- The workspace has no Git metadata, so yesterday’s changes could not be compared with a repository diff.

## Model comparison

The baseline candidate had validation accuracy **63.6%** and macro F1 **63.2%**. The augmented candidate had validation accuracy **45.5%** and macro F1 **33.8%**. Baseline was selected by the predeclared validation macro-F1 rule.

On the grouped holdout of 22 images, baseline scored **50.0% accuracy** and **49.5% macro F1** (11/22 correct). Augmented scored **68.2% accuracy** and **60.8% macro F1** (15/22 correct). The higher augmented holdout score is reported for comparison only; it did not select the model because validation scores selected baseline. Class supports are 10 leaf-focused, 4 sapling-focused, and 8 tree-focused images. The result is unstable at this sample size and should not be read as broad deployment accuracy.

Detailed per-class metrics and confusion matrices are in `reports/evaluation_v2/model_comparison.json`, `reports/evaluation_v2/classification_report_test.json`, and `reports/evaluation_v2/confusion_matrix.png`. The UI uses the validation-selected baseline, with measured test accuracy **50.0%** and macro F1 **49.5%**.
