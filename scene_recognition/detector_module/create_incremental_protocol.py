from __future__ import annotations

import argparse
import json
from pathlib import Path

from scene_recognition.detector_module import ALL_CLASS_NAMES, BASE_CLASS_NAMES, CLASS_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_class_list(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def validate_partition(
    base_classes: list[str],
    rounds: list[list[str]],
    class_order: list[str] | None = None,
) -> None:
    available_classes = class_order or CLASS_NAMES
    flattened = base_classes + [name for current_round in rounds for name in current_round]
    unknown = sorted(set(flattened) - set(available_classes))
    if unknown:
        raise ValueError(f"未知目标类别: {', '.join(unknown)}")
    duplicates = sorted({name for name in flattened if flattened.count(name) > 1})
    if duplicates:
        raise ValueError(f"类别被重复分配: {', '.join(duplicates)}")
    missing = sorted(set(available_classes) - set(flattened))
    if missing:
        raise ValueError(f"以下类别未纳入协议: {', '.join(missing)}")
    if not base_classes or not rounds or any(not current_round for current_round in rounds):
        raise ValueError("协议必须包含非空基础类别和至少一轮非空增量类别")


def build_protocol(
    base_classes: list[str],
    rounds: list[list[str]],
    class_order: list[str] | None = None,
) -> dict:
    available_classes = list(class_order or CLASS_NAMES)
    validate_partition(base_classes, rounds, available_classes)
    learned = list(base_classes)
    stages = [
        {
            "stage": 0,
            "name": "base",
            "new_classes": list(base_classes),
            "old_classes": [],
            "all_learned_classes": list(base_classes),
        }
    ]
    for stage_index, new_classes in enumerate(rounds, start=1):
        old_classes = list(learned)
        learned.extend(new_classes)
        stages.append(
            {
                "stage": stage_index,
                "name": f"increment_{stage_index}",
                "new_classes": list(new_classes),
                "old_classes": old_classes,
                "all_learned_classes": list(learned),
            }
        )
    return {
        "protocol_version": "2.0",
        "scenario": "class_incremental",
        "head_policy": "single_expanding_head",
        "purpose": "按协议顺序逐类训练，验证多轮 Class-IL、回放和抗遗忘能力。",
        "class_order": available_classes,
        "stages": stages,
        "metrics": {
            "new_map": "当前轮新增类别的 mAP",
            "old_map_before": "进入当前轮前旧类别 mAP",
            "old_map_after": "完成当前轮后旧类别 mAP",
            "krr": "old_map_after / old_map_before",
            "all_map": "所有已学习类别的综合 mAP",
            "update_seconds": "从样本注入到模型更新结束的时间",
        },
        "score_targets": {"new_map": 0.60, "krr": 0.95},
        "compliance_warning": "正式实验前必须向主办方确认是否允许保存旧类样本、特征或回放缓冲区。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成可复现的多轮类别增量测试协议")
    parser.add_argument("--base", help="逗号分隔的基础类别")
    parser.add_argument(
        "--classes-file",
        type=Path,
        help="类别文件；r2 使用包含六类的 classes.txt",
    )
    parser.add_argument(
        "--round",
        action="append",
        dest="rounds",
        default=None,
        help="一轮新增类别，可重复传入；默认依次加入 small_aircraft 和 warship",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "scene_recognition" / "detector_module" / "configs" / "incremental_protocol.json",
    )
    args = parser.parse_args()

    if args.classes_file:
        class_order = [
            value.strip()
            for value in args.classes_file.read_text(encoding="utf-8-sig").splitlines()
            if value.strip()
        ]
    else:
        class_order = list(CLASS_NAMES)

    is_official_r2 = class_order == ALL_CLASS_NAMES
    default_base = [ALL_CLASS_NAMES[0]] if is_official_r2 else ["soldier", "tank"]
    default_rounds = (
        list(ALL_CLASS_NAMES[1:])
        if is_official_r2
        else ["small_aircraft", "warship"]
    )
    base_classes = parse_class_list(args.base) if args.base else list(default_base)
    round_values = args.rounds or default_rounds
    protocol = build_protocol(
        base_classes,
        [parse_class_list(x) for x in round_values],
        class_order,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
