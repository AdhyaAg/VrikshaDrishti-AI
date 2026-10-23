import os
import shutil
import random
from pathlib import Path
import hashlib
from PIL import Image

# Set random seed for reproducibility
random.seed(42)

base_dir = Path(r"d:\AI Tree Predictor")
review_dir = base_dir / "to review real dataset"
target_dir = base_dir / "NGO Tree Dataset" / "labelled dataset"

# Clean target directory if exists
if target_dir.exists():
    shutil.rmtree(target_dir)

splits = ['train', 'valid', 'test']
classes = ['leafs', 'saplings', 'trees', 'unclear']

for sp in splits:
    for c in classes:
        (target_dir / sp / c).mkdir(parents=True, exist_ok=True)

# Collect all valid unique images per category from 'to review real dataset'
cat_images = {c: [] for c in classes}
seen_hashes = set()

for c in classes:
    src_folder = review_dir / c
    if not src_folder.exists():
        continue
    for f in src_folder.glob("*"):
        if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']:
            try:
                with Image.open(f) as img:
                    img.verify()
                with open(f, 'rb') as file_obj:
                    h = hashlib.md5(file_obj.read()).hexdigest()
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    cat_images[c].append(f)
            except Exception as e:
                print(f"Skipping corrupt file {f.name}: {e}")

print("=== Image Counts per Class ===")
total_imgs = 0
for c, img_list in cat_images.items():
    print(f"  Class '{c}': {len(img_list)} unique valid images")
    total_imgs += len(img_list)
print(f"Total Unique Verified Images: {total_imgs}")

# Stratified Split (Train: ~75%, Valid: ~15%, Test: ~10%)
# For small classes (like 4 images), ensure at least 2 in train, 1 in valid, 1 in test
split_stats = {sp: {c: 0 for c in classes} for sp in splits}

for c, img_list in cat_images.items():
    random.shuffle(img_list)
    n = len(img_list)
    if n <= 4:
        # Custom split for tiny class: 2 train, 1 valid, 1 test
        train_files = img_list[:2]
        valid_files = img_list[2:3]
        test_files = img_list[3:]
    else:
        n_test = max(1, int(n * 0.10))
        n_val = max(1, int(n * 0.15))
        n_train = n - n_val - n_test
        train_files = img_list[:n_train]
        valid_files = img_list[n_train:n_train + n_val]
        test_files = img_list[n_train + n_val:]

    mapping = [('train', train_files), ('valid', valid_files), ('test', test_files)]
    for sp_name, files in mapping:
        for idx, f in enumerate(files):
            dst = target_dir / sp_name / c / f"{c}_{idx+1:03d}{f.suffix.lower()}"
            shutil.copy2(f, dst)
            split_stats[sp_name][c] += 1

print("\n=== Dataset Organization Summary ===")
for sp_name in splits:
    print(f"\nSplit '{sp_name}':")
    for c in classes:
        print(f"  {c}: {split_stats[sp_name][c]} files")

print("\nDataset successfully organized at:", target_dir)
