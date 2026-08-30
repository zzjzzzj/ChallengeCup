# 第三阶段 Sparse-MoE v1

本阶段在现有 Ultralytics 8.4.100 / `yolo26n` Class-IL、ER、DER 链路中加入可选的稀疏多专家模块。默认不传 `--sparse-moe` 时，训练、checkpoint 和推理仍使用原有路径。

## 实现

- 5 个中性专家 `expert_0`～`expert_4`，图片级共享 Top-2 路由；P3/P4/P5 从实际 Detect 输入动态读取，不写死层号或尺度通道。
- 每个尺度和专家使用 `1x1 down → 3x3 depthwise → SiLU → 1x1 up` 残差适配器，`up` 零初始化。
- 路由查询由多尺度 GAP、停止梯度的 IR/SAR 与四场景辅助概率、均值/标准差/梯度能量/高频能量组成。辅助头只影响软路由与辅助损失，不覆盖检测结果。
- ER 总损失为 YOLO + 模态/场景 masked CE + Switch-style balance + router z-loss + 可微 expert anchor；DER 在此基础上保留现有冻结上一阶段完整模型的 dark class/box response loss。
- T1 无历史 anchor；每个阶段的 best checkpoint 用 EMA anchor 更新，后续阶段继续训练专家，不永久冻结。

## CLI

```powershell
python train.py class-il-yolo --help
```

至少可配置：`--sparse-moe`、`--expert-count`、`--top-k`、`--expert-bottleneck`、`--router-hidden`、`--aux-hidden`、`--modality-loss-weight`、`--scene-loss-weight`、`--balance-loss-weight`、`--router-z-loss-weight`、`--anchor-loss-weight`、`--anchor-rho` 和路由温度的 `--router-temperature[-start|-end|-warmup-epochs]`。

## 数据与输出

Class-IL 准备器为每个物化的 current/replay/val/test 视图生成 `context_index.csv`，字段为：

```text
materialized_image_path,source_image,sensor,scene,split,stage,
sample_role,augmentation_operation,metadata_source
```

优先使用 provenance 中的 `sensor`/`scene`，否则使用文件名兼容解析；无法解析的值为 `unknown`，训练辅助损失会 mask 掉这些行。
若使用旧的 prepared 目录且尚未生成索引，训练器仍会从物化文件名做同样的兼容回退；新准备器生成的显式索引优先级更高。

启用 Sparse-MoE 的每阶段 run 目录包含：

```text
sparse_moe_config.json
expert_usage.json
expert_anchors.pt
context_metadata_summary.json
class_incremental_stage_summary.json
```

模型 checkpoint 自身保留配置和 anchor 状态；`load_sparse_moe_checkpoint()` 可检查并加载项目自定义模型。Agent 对支持 Sparse-MoE 的检测器额外输出 `sparse_moe`，以及每个检测框的专家 ID/权重和路由熵元数据。

## 限制

首版不包含 P2、类别—模态原型、Copy-Paste、切片二次检测、Task-IL、路由分布蒸馏，也不修改 Ultralytics 源码。训练结果、数据、模型和生成 run 目录保持本地，不加入 Git。辅助标签来自显式上下文元数据；unknown 样本不会贡献辅助 CE，但仍参与 YOLO 和无标签依赖的路由正则项。
