from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .config import AgentConfig
from .detection import TargetDetector
from .image_ops import build_augmentation_plan, ensure_image, extract_handcrafted_features, quality_levels, quality_metrics
from .losses import estimate_runtime_losses
from .memory import EpisodeMemory
from .modality import ModalityRecognizer, normalize_modality
from .reasoning import (
    build_consistency_report,
    build_decision,
    describe_scene,
    resolve_final_scene,
    summarize_detection_confidence,
)
from .scene import SceneRecognizer, build_environment_state
from .schemas import AgentReport, PipelineStage, ProbabilityResult
from .target import TargetClassifier


class IntelligentRecognitionAgent:
    """智能识别流程的总控智能体。

    这个类对应流程图中浅蓝色大框里的完整逻辑链：
    输入图像 -> 图像处理/增强建议 -> 模态识别 -> 场景分类 ->
    框选目标 -> 目标分类 -> 场景/目标一致性推理 -> 输出报告。

    设计上刻意把每个能力封装成独立适配器：
    - ModalityRecognizer: 判断可见光/红外/SAR；
    - SceneRecognizer: 调用场景模型，失败时走可解释规则；
    - TargetDetector: 调用YOLO检测器，未提供权重时读取同名YOLO标签；
    - TargetClassifier: 对目标框裁剪再分类，未提供权重时保留检测类别；
    - reasoning/losses/memory: 做组合约束、决策输出和持续学习记录。

    这样汇报时可以讲清楚“智能体不是单个模型”，而是一个会组织多个
    模型/规则/记忆模块协同工作的调度层。
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        # AgentConfig集中管理所有可替换组件的路径和阈值。
        # 后续如果训练出新的YOLO、ResNet或场景SVM，只需要换配置，
        # 不需要改总控流程代码。
        self.config = config or AgentConfig()

        # 记忆模块用JSONL保存历史推理和人工反馈，服务于流程图中的
        # “各个目标历史学习基础”和持续学习/经验回放。
        self.memory = EpisodeMemory(self.config.memory_path)

        # 以下四个对象对应流程图中的四条核心识别支路。
        self.modality_recognizer = ModalityRecognizer()
        self.scene_recognizer = SceneRecognizer(self.config.scene_model, self.config.scene_metadata)
        self.detector = TargetDetector(
            self.config.detector_model,
            self.config.class_names,
            confidence=self.config.detector_confidence,
            image_size=self.config.image_size,
            device=self.config.device,
            allow_label_fallback=self.config.allow_label_fallback,
        )
        self.target_classifier = TargetClassifier(
            self.config.target_checkpoint,
            self.config.class_names,
            device=self.config.device,
            use_scene_prior=self.config.use_scene_prior_for_unknown_targets,
        )

    def run(
        self,
        image: str | Path,
        *,
        sensor_hint: str | None = None,
        modality_images: dict[str, str | Path] | None = None,
        remember: bool | None = None,
    ) -> AgentReport:
        """对单张主图像运行完整智能识别流程。

        Args:
            image: 主图像路径。可以是IR/SAR/可见光中的任意一种。
            sensor_hint: 可选模态提示，例如"ir"或"sar"。正式部署时可由传感器
                通道直接提供，演示时也可从命令行传入。
            modality_images: 可选的多模态配准图像路径，如{"ir": "...", "sar": "..."}。
                当前版本先记录对齐关系，后续可在这里接入真实的多模态融合。
            remember: 是否把本次结果写入记忆。批量试跑时通常关闭，联调/在线运行时开启。

        Returns:
            AgentReport: 统一JSON报告对象，包含识别结果、决策、损失代理和阶段耗时。
        """

        warnings = self.config.validate()
        stages: list[PipelineStage] = []
        primary_image = ensure_image(image)
        input_modalities = self._resolve_input_modalities(primary_image, sensor_hint, modality_images)

        # 1. 图像处理与增强建议。
        # 这里不直接改写原图，而是先抽取质量指标和手工特征：
        # - 质量指标用于判断是否需要对比度拉伸、去噪、锐化；
        # - 手工特征用于传统场景分类器，也方便文档中解释模型依据。
        started = time.perf_counter()
        metrics = quality_metrics(primary_image)
        features = extract_handcrafted_features(primary_image)
        levels = quality_levels(metrics)
        stages.append(
            PipelineStage(
                "image_processing",
                "ok",
                time.perf_counter() - started,
                "quality metrics and handcrafted features extracted",
                {"feature_count": len(features)},
            )
        )

        # 2. 模态识别。
        # 优先使用外部传感器提示；没有提示时根据文件名和图像统计特征判断。
        # 对应流程图中的“模态分类”。
        started = time.perf_counter()
        modality = self.modality_recognizer.predict(primary_image, sensor_hint=sensor_hint)
        stages.append(PipelineStage("modality_recognition", "ok", time.perf_counter() - started, modality.source))

        # 3. 场景分类。
        # 有训练好的joblib/SVM模型时调用模型；模型不可用时回退到文件名/统计规则。
        # 输出 air/sea/urban/forest 概率，后面会和目标检测结果再融合一次。
        started = time.perf_counter()
        scene = self.scene_recognizer.predict(primary_image, modality.label)
        warnings.extend(self.scene_recognizer.warnings)
        stages.append(PipelineStage("scene_classification", "ok", time.perf_counter() - started, scene.source))

        # 4. 环境状态向量与增强计划。
        # 环境状态向量是给决策模块看的结构化描述；增强计划对应流程图左侧的
        # “图像处理和增广”，也能直接写进实验说明。
        environment = build_environment_state(primary_image, modality, scene)
        augmentation_plan = build_augmentation_plan(modality.label, scene.label, metrics, levels)

        # 5. 目标定位。
        # 正式模型路径存在时调用YOLO；没有YOLO权重时读取图片同名txt标签。
        # 这个回退让我们在没有训练权重的阶段也能演示完整智能体链路。
        started = time.perf_counter()
        detections = self.detector.detect(primary_image)
        warnings.extend(self.detector.warnings)
        stages.append(
            PipelineStage(
                "target_localization",
                "ok",
                time.perf_counter() - started,
                f"{len(detections)} target boxes",
                summarize_detection_confidence(detections),
            )
        )

        # 6. 目标分类/类别确认。
        # 如果提供了ResNet18裁剪分类器，就对每个框裁剪后再分类；
        # 否则保留YOLO或标签中的类别。输出依然统一为DetectionBox。
        started = time.perf_counter()
        detections = self.target_classifier.refine(primary_image, detections, scene.label)
        warnings.extend(self.target_classifier.warnings)
        stages.append(
            PipelineStage(
                "target_classification",
                "ok",
                time.perf_counter() - started,
                f"{len(detections)} target labels",
                summarize_detection_confidence(detections),
            )
        )

        # 7. 场景-目标一致性推理。
        # 流程图中“模态+场景+目标=>唯一确定场景”的逻辑在这里完成：
        # 先用目标类别给场景概率投票，再检查不合理组合，例如“坦克+海洋”。
        started = time.perf_counter()
        final_scene = resolve_final_scene(scene, modality, detections)
        consistency = build_consistency_report(scene, final_scene, detections)
        decision = build_decision(modality, final_scene, environment, detections)
        decision["description"] = describe_scene(modality, final_scene, detections, consistency)
        stages.append(
            PipelineStage(
                "scene_target_reasoning",
                "ok",
                time.perf_counter() - started,
                consistency["status"],
                {"final_scene": final_scene.label},
            )
        )

        # 8. 损失代理值。
        # 在线推理没有真实标签，因此这里给的是“运行时代理损失”：
        # 低置信度、图像质量差、场景目标冲突都会提高损失，用于二次评估能力。
        losses = estimate_runtime_losses(modality, final_scene, detections, consistency, environment)

        # 9. 汇总为统一报告。
        # 报告字段尽量和流程图的模块一一对应，方便直接用于答辩展示或调试。
        preprocessing = {
            "augmentation_plan": augmentation_plan,
            "feature_summary": {
                "feature_count": len(features),
                "sample_features": {key: round(float(features[key]), 6) for key in list(features)[:8]},
            },
            "aligned_modalities": input_modalities,
        }
        memory_summary = self.memory.summary(limit=200)

        report = AgentReport(
            image=str(primary_image),
            input_modalities=input_modalities,
            modality=modality,
            scene=scene,
            final_scene=final_scene,
            environment=environment,
            preprocessing=preprocessing,
            detections=detections,
            consistency=consistency,
            decision=decision,
            losses=losses,
            memory=memory_summary,
            warnings=sorted(set(warnings)),
            stages=stages,
        )

        # 10. 持续学习记忆。
        # 默认会写入JSONL；试跑时可通过--no-memory关闭，避免污染历史记录。
        should_remember = self.config.remember_runs if remember is None else remember
        if should_remember:
            self.memory.append_report(report.to_dict())
            report.memory = self.memory.summary(limit=200)
        return report

    def _resolve_input_modalities(
        self,
        primary_image: Path,
        sensor_hint: str | None,
        modality_images: dict[str, str | Path] | None,
    ) -> dict[str, str]:
        """整理多模态输入路径。

        当前数据集多数情况下是一张主图像配同名标签；但流程图设计的是
        可见光/红外/SAR多模态协同。这个函数先把可能出现的多模态路径
        统一成{"ir": "...", "sar": "..."}格式，为后续真正的特征对齐
        和多模态融合留接口。
        """

        resolved: dict[str, str] = {}
        for key, value in (modality_images or {}).items():
            modality = normalize_modality(key)
            if modality and value:
                resolved[modality] = str(ensure_image(value))
        hinted = normalize_modality(sensor_hint)
        if hinted:
            resolved.setdefault(hinted, str(primary_image))
        if not resolved:
            predicted = self.modality_recognizer.predict(primary_image)
            resolved[predicted.label] = str(primary_image)
        return resolved


def run_single_image(
    image: str | Path,
    config: AgentConfig | None = None,
    sensor_hint: str | None = None,
    modality_images: dict[str, str | Path] | None = None,
    remember: bool | None = None,
) -> dict[str, Any]:
    """便捷函数：给脚本或Notebook直接调用，返回可序列化字典。"""

    agent = IntelligentRecognitionAgent(config)
    return agent.run(
        image,
        sensor_hint=sensor_hint,
        modality_images=modality_images,
        remember=remember,
    ).to_dict()
