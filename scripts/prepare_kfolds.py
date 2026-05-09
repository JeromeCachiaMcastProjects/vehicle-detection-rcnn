import json, random
from pathlib import Path

random.seed(42)

RAW = Path("data/raw")
ALL_IMAGES = []
ALL_ANNOTATIONS = []
CATEGORIES = None
image_id_offset = 0
ann_id_offset = 0

for split in ["train", "valid", "test"]:
    ann_file = RAW / split / "_annotations.coco.json"
    if not ann_file.exists():
        continue
    with open(ann_file) as f:
        coco = json.load(f)
    if CATEGORIES is None:
        CATEGORIES = coco["categories"]
    for img in coco["images"]:
        img["split_source"] = split
        img["id"] = img["id"] + image_id_offset
        ALL_IMAGES.append(img)
    for ann in coco["annotations"]:
        ann["image_id"] = ann["image_id"] + image_id_offset
        ann["id"] = ann["id"] + ann_id_offset
        ALL_ANNOTATIONS.append(ann)
    image_id_offset += 100000
    ann_id_offset += 1000000

print(f"Total images pooled: {len(ALL_IMAGES)}")
print(f"Total annotations pooled: {len(ALL_ANNOTATIONS)}")

random.shuffle(ALL_IMAGES)
n = len(ALL_IMAGES)
fold_size = n // 5
folds = []
for i in range(5):
    start = i * fold_size
    end = (i + 1) * fold_size if i < 4 else n
    folds.append(ALL_IMAGES[start:end])

ann_by_image = {}
for ann in ALL_ANNOTATIONS:
    ann_by_image.setdefault(ann["image_id"], []).append(ann)

FOLDS_DIR = Path("data/folds")
FOLDS_DIR.mkdir(parents=True, exist_ok=True)

for val_idx in range(5):
    fold_dir = FOLDS_DIR / f"fold_{val_idx}"
    fold_dir.mkdir(exist_ok=True)

    val_images = folds[val_idx]
    train_images = [img for i, f in enumerate(folds) if i != val_idx for img in f]

    for split_name, split_imgs in [("train", train_images), ("val", val_images)]:
        split_anns = []
        for img in split_imgs:
            split_anns.extend(ann_by_image.get(img["id"], []))
        out = {"images": split_imgs, "annotations": split_anns, "categories": CATEGORIES}
        with open(fold_dir / f"{split_name}_annotations.json", "w") as f:
            json.dump(out, f)
        print(f"Fold {val_idx} | {split_name}: {len(split_imgs)} images, {len(split_anns)} annotations")

print("\nAll folds written to data/folds/")