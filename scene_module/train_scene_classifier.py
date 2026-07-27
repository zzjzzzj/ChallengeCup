from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


SCENES = ["air", "sea", "urban", "forest"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a lightweight scene classifier")
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--output", type=Path, default=Path("runs/resnet18_baseline"))
    p.add_argument("--model", choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"], default="resnet18")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SceneDataset(Dataset):
    def __init__(self, csv_path: Path, transform) -> None:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            self.rows = list(csv.DictReader(f))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["image_path"]) as im:
            image = im.convert("RGB")
        image = self.transform(image)
        return image, int(row["scene_id"]), row["sensor"], row["image_path"]


def build_transforms(image_size: int):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.86, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(3),
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluation = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train, evaluation


def build_model(name: str, pretrained: bool) -> nn.Module:
    try:
        if name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, len(SCENES))
        elif name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v3_small(weights=weights)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(SCENES))
        else:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(SCENES))
    except Exception as exc:  # noqa: BLE001
        if not pretrained:
            raise
        print(f"[warning] pretrained weights unavailable ({exc}); falling back to random initialization")
        return build_model(name, pretrained=False)
    return model


@torch.inference_mode()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    losses = []
    true, pred, sensors, paths, confidences = [], [], [], [], []
    for images, labels, batch_sensors, batch_paths in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        if criterion is not None:
            losses.append(float(criterion(logits, labels).item()))
        probs = logits.softmax(dim=1)
        conf, guesses = probs.max(dim=1)
        true.extend(labels.cpu().tolist())
        pred.extend(guesses.cpu().tolist())
        confidences.extend(conf.cpu().tolist())
        sensors.extend(list(batch_sensors))
        paths.extend(list(batch_paths))
    metrics = {
        "loss": float(np.mean(losses)) if losses else None,
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true, pred, average="macro", zero_division=0)),
        "per_class_recall": {
            scene: float(value)
            for scene, value in zip(SCENES, recall_score(true, pred, labels=range(len(SCENES)), average=None, zero_division=0))
        },
    }
    for sensor in sorted(set(sensors)):
        idx = [i for i, x in enumerate(sensors) if x == sensor]
        y_t = [true[i] for i in idx]
        y_p = [pred[i] for i in idx]
        metrics[f"{sensor}_accuracy"] = float(accuracy_score(y_t, y_p))
        metrics[f"{sensor}_macro_f1"] = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    return metrics, true, pred, sensors, paths, confidences


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    train_tf, eval_tf = build_transforms(args.image_size)
    train_ds = SceneDataset(args.artifacts / "splits" / "train.csv", train_tf)
    val_ds = SceneDataset(args.artifacts / "splits" / "val.csv", eval_tf)
    test_ds = SceneDataset(args.artifacts / "splits" / "test.csv", eval_tf)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    counts = Counter(int(x["scene_id"]) for x in train_ds.rows)
    weights = torch.tensor([len(train_ds) / (len(SCENES) * counts[i]) for i in range(len(SCENES))], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history = []
    best_f1 = -1.0
    print(f"device={device} model={args.model} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.perf_counter()
        running_loss = 0.0
        seen = 0
        for images, labels, _, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * images.size(0)
            seen += images.size(0)
        scheduler.step()
        val_metrics, *_ = evaluate(model, val_loader, device, criterion)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, seen),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - start,
            **{f"val_{k}": v for k, v in val_metrics.items() if not isinstance(v, dict)},
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_name": args.model,
                    "state_dict": model.state_dict(),
                    "scene_names": SCENES,
                    "image_size": args.image_size,
                    "best_val_macro_f1": best_f1,
                    "args": vars(args),
                },
                args.output / "best.pt",
            )

    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    val_metrics, *_ = evaluate(model, val_loader, device, criterion)
    test_metrics, true, pred, sensors, paths, confidences = evaluate(model, test_loader, device, criterion)
    report = classification_report(true, pred, labels=range(len(SCENES)), target_names=SCENES, zero_division=0, output_dict=True)
    matrix = confusion_matrix(true, pred, labels=range(len(SCENES)))
    result = {
        "device": str(device),
        "model": args.model,
        "best_val_macro_f1": best_f1,
        "validation": val_metrics,
        "test": test_metrics,
        "classification_report": report,
        "split_warning": "sequential split is intentionally stricter than random frame splitting",
    }
    (args.output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    with (args.output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["actual/predicted", *SCENES])
        for scene, row in zip(SCENES, matrix.tolist()):
            writer.writerow([scene, *row])
    with (args.output / "test_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "sensor", "actual", "predicted", "confidence", "correct"])
        writer.writeheader()
        for path, sensor, y, p, conf in zip(paths, sensors, true, pred, confidences):
            writer.writerow(
                {
                    "image_path": path,
                    "sensor": sensor,
                    "actual": SCENES[y],
                    "predicted": SCENES[p],
                    "confidence": round(conf, 6),
                    "correct": int(y == p),
                }
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
