"""检验整图/裁剪识别结果是否来自"场景背景捷径"而非目标本身。

本模块回答一个问题：现有的高指标，究竟是模型认出了目标，还是认出了背景？

包含三项互相独立的检验：

1. label-degeneracy  标签退化检验
   统计每个场景下出现过的四维存在向量。若每个场景只对应唯一向量，
   则"整图多标签目标识别"在信息上等价于场景分类，Exact Match 不能作为目标识别证据。

2. trivial-baselines 平凡基线
   完全不看像素，只用 (场景, 框宽高) 训练浅决策树，给出裁剪四分类的平凡上限。
   ResNet18 只有显著超过该上限，才说明它学到了目标的形状与纹理。

3. pixel-ablation   像素消融（需要整图 checkpoint）
   在测试集上分别抹掉"全部目标框内像素"和"全部框外像素"，比较 Exact Match。
   目标被抹除后若指标几乎不降，说明模型依赖背景。

用法::

    python -m target_classifier_module.diagnose_shortcut \
      --manifest-dir detector_module/artifacts/detection_dataset/manifests \
      --checkpoint target_classifier_module/runs/<run>/best.pt \
      --output target_classifier_module/runs/shortcut_diagnosis

`--checkpoint` 可省略，省略时只跑前两项（纯 CPU，无需 GPU）。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier, export_text

from detector_module.boxes import parse_yolo_boxes
from target_classifier_module import CLASS_NAMES

SCENE_NAMES = ("air", "sea", "urban", "forest")
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #


def _parse_context(image_path: Path) -> tuple[str, str, int | None]:
    """从文件名解析 (模态, 场景, 序列号)。命名约定见 whole_image.parse_image_context。"""

    parts = image_path.stem.split("_")
    if not parts or parts[0] not in {"ir", "sar"}:
        raise ValueError(f"无法从文件名识别模态: {image_path}")
    scenes = [part for part in parts if part in SCENE_NAMES]
    if len(scenes) != 1:
        raise ValueError(f"无法从文件名唯一识别场景: {image_path}")
    index = next((int(part) for part in parts if part.isdigit()), None)
    return parts[0], scenes[0], index


def read_manifest_samples(manifest_dir: Path) -> list[dict]:
    """读取三个划分清单，返回每张图的划分、模态、场景、存在向量与像素框尺寸。"""

    samples: list[dict] = []
    for split in SPLITS:
        manifest_path = manifest_dir / f"{split}.txt"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"清单不存在: {manifest_path}")
        for raw in manifest_path.read_text(encoding="utf-8-sig").splitlines():
            text = raw.strip()
            if not text:
                continue
            image_path = Path(text)
            if not image_path.is_absolute():
                image_path = (manifest_path.parent / image_path).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"整图不存在: {image_path}")
            sensor, scene, index = _parse_context(image_path)
            boxes = parse_yolo_boxes(image_path.with_suffix(".txt"), len(CLASS_NAMES))
            presence = [0] * len(CLASS_NAMES)
            for box in boxes:
                presence[box.class_id] = 1
            samples.append(
                {
                    "split": split,
                    "image_path": str(image_path),
                    "sensor": sensor,
                    "scene": scene,
                    "sequence_index": index,
                    "presence": tuple(presence),
                    "boxes": boxes,
                }
            )
    if not samples:
        raise ValueError(f"清单目录没有任何样本: {manifest_dir}")
    return samples


def _image_size(sample: dict) -> tuple[int, int]:
    from PIL import Image

    with Image.open(sample["image_path"]) as opened:
        return opened.size


# --------------------------------------------------------------------------- #
# 检验 1：标签退化
# --------------------------------------------------------------------------- #


def check_label_degeneracy(samples: list[dict]) -> dict:
    """每个场景是否只对应唯一的四维存在向量。"""

    scene_vectors: dict[str, Counter] = defaultdict(Counter)
    split_vectors: dict[str, Counter] = defaultdict(Counter)
    for sample in samples:
        scene_vectors[sample["scene"]][sample["presence"]] += 1
        split_vectors[sample["split"]][sample["presence"]] += 1

    deterministic = all(len(counter) == 1 for counter in scene_vectors.values())
    distinct: Counter = Counter()
    for counter in scene_vectors.values():
        distinct.update(counter)

    trivial: dict[str, dict] = {}
    for split, counter in split_vectors.items():
        total = sum(counter.values())
        vector, hits = counter.most_common(1)[0]
        trivial[split] = {
            "sample_count": total,
            "constant_prediction": list(vector),
            "constant_exact_match": hits / total,
            "perfect_scene_oracle_exact_match": 1.0 if deterministic else None,
        }

    return {
        "label_is_deterministic_function_of_scene": deterministic,
        "distinct_presence_vectors": len(distinct),
        "theoretical_maximum_vectors": 2 ** len(CLASS_NAMES),
        "scene_to_vectors": {
            scene: [
                {"vector": list(vector), "image_count": count}
                for vector, count in counter.most_common()
            ]
            for scene, counter in sorted(scene_vectors.items())
        },
        "trivial_baselines_by_split": trivial,
        "interpretation": (
            "每个场景只对应唯一存在向量：整图多标签识别与场景分类信息等价，"
            "完美场景分类器即可得到100% Exact Match，该指标不能证明模型识别了目标。"
            if deterministic
            else "存在场景内标签变化，整图多标签任务不完全退化为场景分类。"
        ),
    }


# --------------------------------------------------------------------------- #
# 检验 2：不看像素的平凡基线
# --------------------------------------------------------------------------- #


def _box_features(samples: list[dict], with_scene: bool) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    labels: list[int] = []
    for sample in samples:
        width, height = _image_size(sample)
        for box in sample["boxes"]:
            box_width = box.width * width
            box_height = box.height * height
            feature = [
                box_width,
                box_height,
                box_width * box_height,
                max(box_width, box_height),
                box_width / max(box_height, 1e-6),
            ]
            if with_scene:
                feature += [1.0 if sample["scene"] == name else 0.0 for name in SCENE_NAMES]
            rows.append(feature)
            labels.append(box.class_id)
    if not rows:
        raise ValueError("没有任何标注框，无法构建平凡基线")
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _feature_names(with_scene: bool) -> list[str]:
    names = ["w_px", "h_px", "area_px", "max_side_px", "aspect"]
    if with_scene:
        names += [f"is_{name}" for name in SCENE_NAMES]
    return names


def trivial_baselines(samples: list[dict]) -> dict:
    """裁剪四分类的"不看像素"上限：场景查表、框尺寸树、场景+尺寸树。"""

    train = [s for s in samples if s["split"] == "train"]
    test = [s for s in samples if s["split"] == "test"]
    if not train or not test:
        raise ValueError("训练集或测试集为空，无法评估平凡基线")

    # 场景 -> 训练集中该场景最常见类别
    scene_counter: dict[str, Counter] = defaultdict(Counter)
    for sample in train:
        for box in sample["boxes"]:
            scene_counter[sample["scene"]][box.class_id] += 1
    scene_lookup = {
        scene: counter.most_common(1)[0][0] for scene, counter in scene_counter.items()
    }

    true_ids: list[int] = []
    scene_only: list[int] = []
    for sample in test:
        for box in sample["boxes"]:
            true_ids.append(box.class_id)
            scene_only.append(scene_lookup.get(sample["scene"], 0))

    results = {
        "scene_lookup": {
            scene: CLASS_NAMES[class_id] for scene, class_id in sorted(scene_lookup.items())
        },
        "scene_only": {
            "accuracy": float(accuracy_score(true_ids, scene_only)),
            "macro_f1": float(f1_score(true_ids, scene_only, average="macro", zero_division=0)),
        },
    }

    for key, with_scene, depth in (("size_only", False, 3), ("scene_and_size", True, 4)):
        x_train, y_train = _box_features(train, with_scene)
        x_test, y_test = _box_features(test, with_scene)
        tree = DecisionTreeClassifier(
            max_depth=depth, random_state=0, class_weight="balanced"
        ).fit(x_train, y_train)
        predicted = tree.predict(x_test)
        results[key] = {
            "max_depth": depth,
            "accuracy": float(accuracy_score(y_test, predicted)),
            "macro_f1": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
            "rules": export_text(tree, feature_names=_feature_names(with_scene)),
        }

    # 唯一无法靠场景解决的子问题：urban/forest 内的 soldier vs tank
    soldier_id, tank_id = CLASS_NAMES.index("soldier"), CLASS_NAMES.index("tank")
    sub_train = [
        {**s, "boxes": [b for b in s["boxes"] if b.class_id in (soldier_id, tank_id)]}
        for s in train
    ]
    sub_test = [
        {**s, "boxes": [b for b in s["boxes"] if b.class_id in (soldier_id, tank_id)]}
        for s in test
    ]
    sub_train = [s for s in sub_train if s["boxes"]]
    sub_test = [s for s in sub_test if s["boxes"]]
    if sub_train and sub_test:
        x_train, y_train = _box_features(sub_train, False)
        x_test, y_test = _box_features(sub_test, False)
        stump = DecisionTreeClassifier(
            max_depth=1, random_state=0, class_weight="balanced"
        ).fit(x_train, y_train)
        predicted = stump.predict(x_test)
        results["soldier_vs_tank_size_stump"] = {
            "test_box_count": int(len(y_test)),
            "accuracy": float(accuracy_score(y_test, predicted)),
            "rules": export_text(stump, feature_names=_feature_names(False)),
        }

    best = max(
        results["scene_only"]["accuracy"],
        results["size_only"]["accuracy"],
        results["scene_and_size"]["accuracy"],
    )
    results["best_pixel_free_accuracy"] = best
    results["interpretation"] = (
        f"完全不看像素即可达到 {best:.2%} 的裁剪分类 Accuracy；"
        "ResNet18 的裁剪分类结果必须减去该上限后才代表真正的视觉识别增益。"
    )
    return results


# --------------------------------------------------------------------------- #
# 检验 3：像素消融
# --------------------------------------------------------------------------- #


def pixel_ablation(
    samples: list[dict], checkpoint_path: Path, device_name: str = "auto", batch_size: int = 32
) -> dict:
    """抹掉目标 / 抹掉背景后，整图模型的 Exact Match 变化。"""

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from target_classifier_module.training import (
        build_resnet18,
        build_transforms,
        resolve_device,
    )
    from target_classifier_module.whole_image import compute_multilabel_metrics

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    image_size = int(checkpoint.get("image_size", 224))
    thresholds = [float(v) for v in checkpoint.get("thresholds", [0.5] * len(CLASS_NAMES))]
    _, evaluation_transform = build_transforms(image_size, "none")
    device = resolve_device(device_name)
    model = build_resnet18(len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    test = [s for s in samples if s["split"] == "test"]
    if not test:
        raise ValueError("测试集为空，无法做像素消融")

    class _Ablated(Dataset):
        def __init__(self, rows: list[dict], mode: str, fill: str) -> None:
            self.rows = rows
            self.mode = mode
            self.fill = fill

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            with Image.open(row["image_path"]) as opened:
                image = opened.convert("RGB")
            if self.mode != "original":
                array = np.array(image)
                height, width = array.shape[:2]
                mask = np.zeros((height, width), dtype=bool)
                for box in row["boxes"]:
                    x1, y1, x2, y2 = box.xyxy
                    left = max(0, int(round(x1 * width)))
                    top = max(0, int(round(y1 * height)))
                    right = min(width, int(round(x2 * width)))
                    bottom = min(height, int(round(y2 * height)))
                    if right > left and bottom > top:
                        mask[top:bottom, left:right] = True
                erase = mask if self.mode == "targets_masked" else ~mask
                keep = ~erase
                if self.fill == "mean" and keep.any():
                    value = array[keep].reshape(-1, 3).mean(axis=0).astype(np.uint8)
                else:
                    value = np.zeros(3, dtype=np.uint8)
                array[erase] = value
                image = Image.fromarray(array)
            return (
                evaluation_transform(image),
                torch.tensor(row["presence"], dtype=torch.float32),
                row["sensor"],
                row["scene"],
            )

    @torch.inference_mode()
    def _evaluate(mode: str, fill: str) -> dict:
        loader = DataLoader(
            _Ablated(test, mode, fill),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        trues, probabilities, sensors, scenes = [], [], [], []
        for images, targets, batch_sensors, batch_scenes in loader:
            logits = model(images.to(device, non_blocking=True))
            trues.append(targets.numpy())
            probabilities.append(logits.sigmoid().cpu().numpy())
            sensors.extend(list(batch_sensors))
            scenes.extend(list(batch_scenes))
        metrics = compute_multilabel_metrics(
            np.concatenate(trues).astype(np.int64),
            np.concatenate(probabilities),
            thresholds,
            sensors,
            scenes,
            CLASS_NAMES,
        )
        return {
            "exact_match_accuracy": metrics["exact_match_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "micro_f1": metrics["micro_f1"],
            "mean_label_average_precision": metrics["mean_label_average_precision"],
        }

    conditions = {
        "original": ("original", "mean"),
        "targets_masked_mean": ("targets_masked", "mean"),
        "targets_masked_black": ("targets_masked", "black"),
        "targets_only_mean": ("targets_only", "mean"),
        "targets_only_black": ("targets_only", "black"),
    }
    results = {name: _evaluate(mode, fill) for name, (mode, fill) in conditions.items()}

    original = results["original"]["exact_match_accuracy"]
    masked = results["targets_masked_mean"]["exact_match_accuracy"]
    only = results["targets_only_mean"]["exact_match_accuracy"]
    if original > 0 and masked >= 0.9 * original:
        verdict = (
            "抹掉全部目标像素后 Exact Match 几乎不下降：模型依赖背景场景而非目标本身，"
            "场景捷径成立，该结果不能作为目标识别能力的证据。"
        )
    elif masked > 0.6:
        verdict = "抹掉目标后仍明显高于恒定预测基线：存在实质性背景依赖。"
    else:
        verdict = "抹掉目标后大幅下降：模型确实依赖目标像素。"

    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "thresholds": {name: value for name, value in zip(CLASS_NAMES, thresholds)},
        "test_image_count": len(test),
        "conditions": results,
        "targets_masked_retention": masked / original if original else None,
        "targets_only_retention": only / original if original else None,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检验整图/裁剪识别指标是否来自场景背景捷径"
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="目录内必须包含train.txt、val.txt、test.txt",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="整图模型best.pt；省略则只做标签退化与平凡基线检验（纯CPU）",
    )
    parser.add_argument("--output", type=Path, help="输出目录，写入shortcut_report.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = read_manifest_samples(args.manifest_dir)
    report = {
        "manifest_dir": str(args.manifest_dir.resolve()),
        "image_count": len(samples),
        "label_degeneracy": check_label_degeneracy(samples),
        "trivial_baselines": trivial_baselines(samples),
    }
    if args.checkpoint is not None:
        report["pixel_ablation"] = pixel_ablation(
            samples, args.checkpoint, args.device, args.batch_size
        )

    degeneracy = report["label_degeneracy"]
    print("=" * 74)
    print("检验1  标签退化")
    print(f"  标签是否为场景的确定性函数: {degeneracy['label_is_deterministic_function_of_scene']}")
    print(
        f"  全数据集不同存在向量数: {degeneracy['distinct_presence_vectors']} / "
        f"{degeneracy['theoretical_maximum_vectors']}"
    )
    for scene, vectors in degeneracy["scene_to_vectors"].items():
        shown = "  ".join(f"{v['vector']}x{v['image_count']}" for v in vectors)
        print(f"    {scene:<8} {shown}")
    for split, values in degeneracy["trivial_baselines_by_split"].items():
        print(
            f"    {split:<6} 恒定预测{values['constant_prediction']} -> "
            f"Exact Match {values['constant_exact_match']:.2%}"
        )
    print(f"  {degeneracy['interpretation']}")

    trivial = report["trivial_baselines"]
    print("=" * 74)
    print("检验2  不看像素的平凡基线（裁剪四分类）")
    print(f"  scene_only      Accuracy={trivial['scene_only']['accuracy']:.2%}")
    print(f"  size_only       Accuracy={trivial['size_only']['accuracy']:.2%}")
    print(f"  scene_and_size  Accuracy={trivial['scene_and_size']['accuracy']:.2%}")
    if "soldier_vs_tank_size_stump" in trivial:
        stump = trivial["soldier_vs_tank_size_stump"]
        print(
            f"  soldier vs tank 仅用框尺寸单节点树 Accuracy={stump['accuracy']:.2%} "
            f"（{stump['test_box_count']}个测试框）"
        )
    print(f"  {trivial['interpretation']}")

    if "pixel_ablation" in report:
        ablation = report["pixel_ablation"]
        print("=" * 74)
        print("检验3  像素消融（测试集）")
        for name, values in ablation["conditions"].items():
            print(
                f"  {name:<22} Exact Match={values['exact_match_accuracy']:.2%}  "
                f"Macro-F1={values['macro_f1']:.2%}"
            )
        print(f"  {ablation['verdict']}")

    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / "shortcut_report.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("=" * 74)
        print(f"报告已写入: {path}")


if __name__ == "__main__":
    main()
