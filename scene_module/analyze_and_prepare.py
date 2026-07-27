from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat


NAME_RE = re.compile(
    r"^(?P<sensor>ir|sar)_r1_base_(?P<scene>air|sea|urban|forest)_(?P<index>\d{6})$"
)
SCENES = ["air", "sea", "urban", "forest"]
SENSORS = ["ir", "sar"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and prepare the scene-classification dataset")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_for_position(pos: int, total: int, train_ratio: float, val_ratio: float) -> str:
    # Sequential split within every sensor/scene subgroup. This is deliberately
    # more conservative than a random image split for likely adjacent frames.
    train_end = max(1, int(round(total * train_ratio)))
    val_count = max(1, int(round(total * val_ratio)))
    val_end = min(total - 1, train_end + val_count)
    if pos < train_end:
        return "train"
    if pos < val_end:
        return "val"
    return "test"


def label_stats(label_path: Path, class_count: int) -> tuple[int, Counter, list[str]]:
    boxes = 0
    classes: Counter = Counter()
    errors: list[str] = []
    for lineno, raw in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 5:
            errors.append(f"{label_path.name}:{lineno}: expected 5 fields, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            errors.append(f"{label_path.name}:{lineno}: non-numeric label")
            continue
        if not 0 <= cls < class_count:
            errors.append(f"{label_path.name}:{lineno}: class {cls} out of range")
        if not all(0.0 <= x <= 1.0 for x in coords):
            errors.append(f"{label_path.name}:{lineno}: normalized coordinate out of range")
        if coords[2] <= 0 or coords[3] <= 0:
            errors.append(f"{label_path.name}:{lineno}: non-positive box size")
        boxes += 1
        classes[cls] += 1
    return boxes, classes, errors


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "splits").mkdir(exist_ok=True)

    classes_path = dataset / "classes.txt"
    classes = [x.strip() for x in classes_path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    image_paths = sorted(dataset.glob("*.png"))
    records: list[dict] = []
    invalid_names: list[str] = []
    missing_labels: list[str] = []
    corrupt_images: list[str] = []
    label_errors: list[str] = []
    object_counts = Counter()
    hashes: defaultdict[str, list[str]] = defaultdict(list)
    dimensions = Counter()
    modes = Counter()

    parsed_by_group: defaultdict[tuple[str, str], list[tuple[int, Path]]] = defaultdict(list)
    for path in image_paths:
        match = NAME_RE.match(path.stem)
        if not match:
            invalid_names.append(path.name)
            continue
        sensor, scene, index = match.group("sensor"), match.group("scene"), int(match.group("index"))
        parsed_by_group[(sensor, scene)].append((index, path))

    positions: dict[Path, tuple[int, int]] = {}
    for items in parsed_by_group.values():
        items.sort(key=lambda x: x[0])
        for pos, (_, path) in enumerate(items):
            positions[path] = (pos, len(items))

    for path in image_paths:
        match = NAME_RE.match(path.stem)
        if not match:
            continue
        sensor, scene, index = match.group("sensor"), match.group("scene"), int(match.group("index"))
        label_path = path.with_suffix(".txt")
        if not label_path.exists():
            missing_labels.append(path.name)
        try:
            with Image.open(path) as im:
                im.load()
                width, height = im.size
                mode = im.mode
                gray = np.asarray(im.convert("L"), dtype=np.float32)
                mean = float(gray.mean())
                std = float(gray.std())
                q01, q99 = np.percentile(gray, [1, 99]).tolist()
                dimensions[f"{width}x{height}"] += 1
                modes[mode] += 1
        except Exception as exc:  # noqa: BLE001
            corrupt_images.append(f"{path.name}: {exc}")
            continue
        boxes, cls_counts = 0, Counter()
        if label_path.exists():
            boxes, cls_counts, errs = label_stats(label_path, len(classes))
            label_errors.extend(errs)
            object_counts.update(cls_counts)
        digest = sha256(path)
        hashes[digest].append(path.name)
        pos, total = positions[path]
        split = split_for_position(pos, total, args.train_ratio, args.val_ratio)
        records.append(
            {
                "image_path": str(path),
                "image_name": path.name,
                "sensor": sensor,
                "scene": scene,
                "scene_id": SCENES.index(scene),
                "sequence_index": index,
                "split": split,
                "width": width,
                "height": height,
                "mode": mode,
                "mean_gray": round(mean, 4),
                "std_gray": round(std, 4),
                "p01_gray": round(float(q01), 4),
                "p99_gray": round(float(q99), 4),
                "box_count": boxes,
                "sha256": digest,
            }
        )

    fieldnames = list(records[0].keys())
    with (output / "scene_index.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    for split in ("train", "val", "test"):
        subset = [r for r in records if r["split"] == split]
        with (output / "splits" / f"{split}.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(subset)

    exact_duplicate_groups = [names for names in hashes.values() if len(names) > 1]
    stems = {(r["sensor"], r["scene"], r["sequence_index"]): r for r in records}
    pair_rows: list[dict] = []
    for scene in SCENES:
        ids = sorted(
            {idx for sensor, sc, idx in stems if sc == scene and ("ir", scene, idx) in stems and ("sar", scene, idx) in stems}
        )
        for idx in ids:
            pair_rows.append(
                {
                    "scene": scene,
                    "sequence_index": idx,
                    "ir_image": stems[("ir", scene, idx)]["image_path"],
                    "sar_image": stems[("sar", scene, idx)]["image_path"],
                }
            )
    with (output / "candidate_ir_sar_pairs.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["scene", "sequence_index", "ir_image", "sar_image"])
        writer.writeheader()
        writer.writerows(pair_rows)

    def nested_counts(key1: str, key2: str | None = None) -> dict:
        if key2 is None:
            return dict(sorted(Counter(r[key1] for r in records).items()))
        result: dict[str, dict] = {}
        for a in sorted({r[key1] for r in records}):
            result[a] = dict(sorted(Counter(r[key2] for r in records if r[key1] == a).items()))
        return result

    audit = {
        "dataset": str(dataset),
        "image_count": len(image_paths),
        "valid_record_count": len(records),
        "class_names": classes,
        "scene_counts": nested_counts("scene"),
        "sensor_counts": nested_counts("sensor"),
        "sensor_scene_counts": nested_counts("sensor", "scene"),
        "split_counts": nested_counts("split"),
        "split_scene_counts": nested_counts("split", "scene"),
        "dimensions": dict(dimensions),
        "image_modes": dict(modes),
        "total_boxes": int(sum(object_counts.values())),
        "object_counts": {classes[k]: int(v) for k, v in sorted(object_counts.items())},
        "candidate_pair_count": len(pair_rows),
        "candidate_pair_scene_counts": dict(Counter(x["scene"] for x in pair_rows)),
        "exact_duplicate_group_count": len(exact_duplicate_groups),
        "exact_duplicate_groups": exact_duplicate_groups,
        "invalid_filenames": invalid_names,
        "missing_labels": missing_labels,
        "corrupt_images": corrupt_images,
        "label_errors": label_errors,
        "split_method": "sequential 70/15/15 inside each sensor-scene subgroup",
        "pair_warning": "same scene/index is only a candidate pair; alignment is not proven by filename alone",
    }
    (output / "dataset_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 数据集审计结果",
        "",
        f"- 图像：{len(records)} 幅；目标框：{audit['total_boxes']} 个。",
        f"- 场景：{audit['scene_counts']}。",
        f"- 模态：{audit['sensor_counts']}。",
        f"- 模态×场景：{audit['sensor_scene_counts']}。",
        f"- 划分：{audit['split_counts']}，方法为每个模态×场景子组内按序列前70%/中15%/后15%。",
        f"- 候选IR/SAR同编号对：{len(pair_rows)} 对；这只能说明文件名对应，不能证明时空配准。",
        f"- 精确重复图像组：{len(exact_duplicate_groups)} 组。",
        f"- 损坏图像：{len(corrupt_images)}；缺失标签：{len(missing_labels)}；非法YOLO标签：{len(label_errors)}。",
        "",
        "## 关键判断",
        "",
        "1. 场景标签可直接由文件名可靠解析，无需先人工补四类场景标签。",
        "2. 文件序号很可能含连续帧关系，因此不采用普通随机图片划分。",
        "3. 同编号IR/SAR只作为候选配对，训练首版场景分类器时仍按单图输入处理。",
        "4. 场景类别与传感器分布需要联合检查，防止模型把模态差异误学成场景差异。",
    ]
    (output / "dataset_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
