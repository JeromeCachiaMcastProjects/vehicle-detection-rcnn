import argparse, json, time
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# ── Dataset ───────────────────────────────────────────────────────────────────
class VehicleDataset(Dataset):
    def __init__(self, ann_file, img_root):
        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        self.img_root = Path(img_root)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = None

        # Try the split_source recorded during fold preparation first
        split_source = img_info.get("split_source")
        if split_source:
            candidate = self.img_root / split_source / img_info["file_name"]
            if candidate.exists():
                img_path = candidate

        # Fall back: search all split folders
        if img_path is None:
            for split in ["train", "valid", "test"]:
                candidate = self.img_root / split / img_info["file_name"]
                if candidate.exists():
                    img_path = candidate
                    break

        # Last resort: recursive search by filename only
        if img_path is None:
            matches = list(self.img_root.rglob(img_info["file_name"]))
            if matches:
                img_path = matches[0]

        if img_path is None:
            raise FileNotFoundError(
                f"Could not find image: {img_info['file_name']} anywhere under {self.img_root}"
            )

        image = Image.open(img_path).convert("RGB")
        image = TF.to_tensor(image)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w > 1 and h > 1:
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

# ── Model ─────────────────────────────────────────────────────────────────────
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ── Train one epoch ───────────────────────────────────────────────────────────
def train_one_epoch(model, optimizer, loader, device, fold, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    bar = tqdm(loader, desc=f"Fold {fold} | Epoch {epoch}/{total_epochs} [Train]",
               unit="batch", dynamic_ncols=True)
    for i, (images, targets) in enumerate(bar, start=1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss/i:.4f}")
    return total_loss / len(loader)

# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model, loader, device, coco_gt, fold, epoch, total_epochs):
    model.eval()
    results = []
    inference_times = []

    bar = tqdm(loader, desc=f"Fold {fold} | Epoch {epoch}/{total_epochs} [Val]  ",
               unit="batch", dynamic_ncols=True)
    with torch.no_grad():
        for images, targets in bar:
            images = [img.to(device) for img in images]
            t0 = time.perf_counter()
            outputs = model(images)
            t1 = time.perf_counter()
            inference_times.append((t1 - t0) / len(images))

            for output, target in zip(outputs, targets):
                img_id = target["image_id"].item()
                for box, label, score in zip(
                    output["boxes"].cpu(),
                    output["labels"].cpu(),
                    output["scores"].cpu()
                ):
                    x1, y1, x2, y2 = box.tolist()
                    results.append({
                        "image_id": img_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score)
                    })

    avg_ms = (sum(inference_times) / len(inference_times)) * 1000

    if len(results) == 0:
        return 0.0, 0.0, 0.0, avg_ms

    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    return ev.stats[0], ev.stats[1], ev.stats[8], avg_ms
# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",       type=int,   required=True)
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=0.005)
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_dir  = Path(f"data/folds/fold_{args.fold}")
    img_root  = Path("data/raw")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print(f"Fold {args.fold} | Device: {device}")

    train_ds = VehicleDataset(fold_dir / "train_annotations.json", img_root)
    val_ds   = VehicleDataset(fold_dir / "val_annotations.json",   img_root)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, collate_fn=collate_fn)

    # num_classes = 4 vehicle classes + 1 background
    model = get_model(num_classes=5).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    rows = []
    for epoch in range(args.epochs):
        print(f"\n[Fold {args.fold}] Epoch {epoch+1}/{args.epochs}")
        loss = train_one_epoch(model, optimizer, train_loader, device, args.fold, epoch + 1, args.epochs)
        scheduler.step()

        coco_gt = COCO(str(fold_dir / "val_annotations.json"))
        mAP, mAP50, mAR, ms = evaluate(model, val_loader, device, coco_gt, args.fold, epoch + 1, args.epochs)

        print(f"  Loss={loss:.4f}  mAP={mAP:.4f}  mAP@50={mAP50:.4f}  "
              f"mAR={mAR:.4f}  Inference={ms:.1f}ms/img")

        rows.append({
            "fold": args.fold, "epoch": epoch + 1,
            "train_loss": loss, "mAP": mAP,
            "mAP_50": mAP50,   "mAR": mAR,
            "inference_ms": ms
        })

    pd.DataFrame(rows).to_csv(
        results_dir / f"fold_{args.fold}_results.csv", index=False
    )
    torch.save(model.state_dict(), results_dir / f"fold_{args.fold}_model.pth")
    print(f"\nFold {args.fold} done. Results saved to results/")

if __name__ == "__main__":
    main()