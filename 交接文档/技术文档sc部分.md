# ChallengeCup 技术文档（SC 部分）

> 文档日期：2026-09-03
>
> 代码基线：`main` 分支，提交 `3b70c49` 及此前 SC 相关实现
>
> 适用范围：Scene Classification / Scene Recognition（以下简称 SC），覆盖场景分类、目标检测、Agent 接入、四类到六类的 Class-IL、ER/DER 经验回放、Sparse-MoE，以及训练与验收流程。

## 1. 文档目的与边界

本文用于交接当前仓库中已经实现并上传的 SC 代码，回答以下问题：

1. 当前 SC 系统由哪些模块组成，各模块如何协作；
2. 四分类基础模型如何过渡到六分类模型；
3. 多批次、小样本、按类到达的持续学习如何实现；
4. ER、DER 和稀疏多专家分别在代码中如何落地；
5. 如何准备本地数据、启动完整训练、读取结果并判断训练是否有效；
6. 哪些能力已经实现，哪些只完成了最小流程验证，哪些仍属于后续工作。

需要特别说明：本项目的六类主任务是**目标检测**，不仅是整图分类。因此，Class-IL 的主要评估指标是每类检测 `mAP@0.5` 和 `mAP@0.5:0.95`，不能只用分类 Accuracy 描述模型效果。

本文不记录任何真实数据集的本机绝对路径，也不提交数据、权重和训练运行目录。文中的路径均为占位示例，实际执行时应替换成本机路径。

## 2. 当前系统总览

当前 SC 系统可以概括为四层：

```text
输入图像
   │
   ├─ 图像质量 / 模态判断（IR、SAR）
   ├─ 场景分类（air、sea、urban、forest）
   │
   └─ 六类目标检测模型
        ├─ YOLO 检测主干与检测头
        ├─ 可选 Sparse-MoE 特征适配层
        ├─ 可选 ER / DER 类增量训练
        └─ 每图专家路由诊断
   │
Agent 编排与结果格式化
   ├─ 目标框、类别、置信度
   ├─ 场景/模态/一致性信息
   ├─ Sparse-MoE 专家编号、权重、熵
   └─ JSON / CSV / 中文描述
```

训练侧的主流程是：

```text
原始四类数据
  → 仅增广 train
  → 训练四类基础模型
  → 原始六类增量数据
  → 仅增广 train
  → 按类和来源族划分为多个小批次
  → 构建容量为 200 或 500 的回放池
  → ER 或 DER 逐批训练
  → 可选 Sparse-MoE
  → 每批次在 val 上评估与选模
  → 最后一批结束后仅在 test 上做最终评估
  → 六类最终模型与持续学习指标
```

## 3. 目录与代码职责

| 路径 | 主要职责 |
| --- | --- |
| `train.py` | 仓库统一 CLI 入口，将子命令分发到各训练模块 |
| `scene_recognition/` | SC 主包，包含分类、检测、增量学习和路由训练 |
| `scene_recognition/detector_module/` | YOLO 数据处理、基础训练、Class-IL、ER/DER、Sparse-MoE、指标计算 |
| `scene_recognition/detector_module/augment_yolo_dataset.py` | IR/SAR 离线确定性增广 |
| `scene_recognition/detector_module/prepare_batch_incremental_dataset.py` | 多批次小样本增量数据和回放视图准备 |
| `scene_recognition/detector_module/train_batch_incremental_yolo.py` | 任意批次 ER/DER 训练、逐阶段验证、最终测试 |
| `scene_recognition/detector_module/run_four_to_six_pipeline.py` | 四类基础训练到六类增量训练的一键编排 |
| `scene_recognition/detector_module/dark_experience_replay.py` | DER 教师响应匹配损失 |
| `scene_recognition/detector_module/sparse_moe_model.py` | Sparse-MoE 注入、路由、专家、辅助损失和诊断 |
| `scene_recognition/detector_module/sparse_moe_trainer.py` | 将上下文标签和 MoE 损失接入 Ultralytics Trainer |
| `scene_recognition/detector_module/sparse_moe_checkpoint.py` | MoE checkpoint、专家使用量和锚点读写 |
| `scene_recognition/detector_module/context_metadata.py` | 模态、场景、阶段、样本角色等上下文索引 |
| `scene_recognition/detector_module/metrics.py` | 逐阶段、逐类别 mAP、遗忘和 BWT 统计 |
| `scene_recognition/route_training/` | 场景路由、简单六类、困难三类的独立训练脚本 |
| `Agent/` | 推理编排、检测适配、推理解释、结构化结果输出 |
| `Agent/models/experts/torch_router.py` | 可训练 Top-K 路由器与专家使用统计 |
| `Agent/models/experts/sparse_adapter.py` | 多尺度稀疏专家适配单元 |
| `docs/四类到六类多批次小样本增量训练.md` | 增量流程的专项操作说明 |
| `交流文档/第三阶段稀疏多专家架构方案-20260830.md` | Sparse-MoE 第三阶段的设计依据与边界 |

## 4. 类别体系与数据协议

### 4.1 场景类别

场景分类统一使用以下四类：

| ID | 名称 |
| ---: | --- |
| 0 | `air` |
| 1 | `sea` |
| 2 | `urban` |
| 3 | `forest` |

### 4.2 目标类别

基础阶段为四类，增量完成后扩展到六类。类别 ID 和顺序是训练协议的一部分，不允许任意调整。

| ID | 类名 | 阶段 |
| ---: | --- | --- |
| 0 | `soldier` | 基础四类 |
| 1 | `small_aircraft` | 基础四类 |
| 2 | `warship` | 基础四类 |
| 3 | `tank` | 基础四类 |
| 4 | `patrol_boat` | 新增类 |
| 5 | `armored_vehicle` | 新增类 |

四类 checkpoint 的 `names` 必须严格等于前四类及上述顺序。六类数据 YAML 必须严格包含全部六类且顺序一致。代码会据此进行检测头扩展和旧类权重迁移；名称或顺序错误会直接导致训练语义错误，因此应在启动前阻断。

### 4.3 数据目录要求

基础数据和增量数据均使用 YOLO 目录协议：

```text
dataset_root/
├─ data.yaml
├─ images/
│  ├─ train/
│  ├─ val/
│  └─ test/
└─ labels/
   ├─ train/
   ├─ val/
   └─ test/
```

每行检测标签格式为：

```text
class_id x_center y_center width height
```

坐标必须归一化到 `[0, 1]`。训练图像文件名建议以 `ir_` 或 `sar_` 开头，使系统能够生成可靠的模态上下文。若历史数据无法改名，可以显式使用 `--default-modality ir` 或 `--default-modality sar`，但这会把缺失模态的样本统一视为同一模态。

## 5. 场景分类与目标识别现状

### 5.1 场景分类

仓库保留两条场景分类路线：

1. 正式传统路线：手工图像特征 → `StandardScaler` → ANOVA `SelectKBest(k=30)` → RBF-SVM；
2. 可选路由路线：使用 YOLOv8n-cls 训练四场景分类器，为后续按场景选择检测分支提供路由信号。

项目 README 中已有的历史结果为：111 张独立测试图像上 Accuracy 92.79%、Macro-F1 92.25%。该数值是已有实验记录，不代表本次 Class-IL 训练结果，也不能直接替代目标检测 mAP。

### 5.2 目标分类与目标检测

仓库中存在基于真实框裁剪的 ResNet18 目标分类路线，历史测试 Accuracy 为 99.79%。该结果评价的是“给定目标框后的类别判定”，不包含目标定位能力。

六类增量学习采用 YOLO 检测路线，同时学习目标位置和类别。评估时应以检测指标为主：

- `precision`、`recall`；
- 总体 `mAP@0.5`；
- 总体 `mAP@0.5:0.95`；
- 每个阶段、每个已见类别的两种 mAP；
- 遗忘、BWT 等持续学习指标。

## 6. Agent 推理架构

Agent 层负责把多个识别组件组合成一条可调用流程。主要步骤包括图像读取与质量分析、模态判断、场景识别、目标检测、可选目标分类、一致性检查、记忆/报告组织和结果格式化。

`Agent/detection.py` 的检测策略为：

1. 如果配置的本地 YOLO 权重存在，则使用 Ultralytics 加载并推理；
2. 如果模型加载或推理失败且允许 fallback，则读取图像同名的 YOLO 标签；
3. 两者都不可用时返回空目标列表并记录 warning。

这意味着 sidecar 标签回退只适合联调和数据检查，不应当被当作真实模型能力。

当加载的是 Sparse-MoE 模型时，Agent 会读取该次推理的 MoE 诊断信息，并附加到检测框和最终结果中，包括：

- `expert_ids`：Top-K 被选中的专家；
- `expert_weights`：对应的归一化路由权重；
- `router_entropy`：路由分布熵；
- 模态和场景辅助预测；
- MoE 配置和当前路由诊断。

结构化结果可输出 JSON 和 UTF-8 BOM CSV。主要 CSV 字段包括：

```text
image, scene, modality, target_type_count, target_total_count,
target_details, max_confidence, description
```

中文 `description` 由确定性格式化代码生成，不依赖在线大语言模型，因此可以在离线环境中复现。

## 7. 离线增广实现

### 7.1 基本原则

增广工具只扩充 `train`，`val` 和 `test` 只复制原图与原标签。这样可以避免评估集包含训练图像的变体，也可以避免测试结果因增广策略变化而失真。

增广默认保留原始训练样本，并为每个样本生成三个独立变体。操作由图像文件名和操作名的 SHA 哈希决定，因此在相同输入和代码版本下是确定性的。

### 7.2 IR 与 SAR 操作

IR 样本使用：

- `ir_gamma_bright`：确定性 gamma，范围约为 `[0.50, 0.70]`；
- `invert_255`：灰度/颜色反相；
- `rot180`：旋转 180°。

SAR 样本使用：

- `rot180`；
- `sar_rot90_cw`：顺时针旋转 90°；
- `sar_gamma`：确定性 gamma，从 `[0.45, 0.68]` 或 `[1.55, 1.90]` 取值。

几何变换会同步修改 YOLO 框坐标。工具还会校验图片、标签、类别范围、`nc/names` 一致性和重复 basename。

增广输出包括：

- `data.yaml`；
- `classes.txt`；
- `augmentation_manifest.csv`；
- `augmentation_summary.json`。

输出目录必须不存在或为空。当前实现故意不提供危险的递归覆盖逻辑，以避免误删本机数据。

### 7.3 已增广数据的注意事项

一键 `four-to-six-yolo` 会先执行基础数据和增量数据的增广。因此，已经增广并完成 train/val/test 划分的数据不能再次直接送入该命令，否则会产生二次增广。对于已增广数据，应从“批次准备 + 批次训练”两个阶段开始执行，具体命令见第 14 节。

## 8. 四类到六类的多批次 Class-IL

### 8.1 问题定义

当前实现针对以下持续学习场景：

- 起点是一个已经训练好的四类检测模型；
- 后续总类别空间固定扩展为六类；
- 增量数据分成多个批次到达；
- 一个批次只含任意非空类别子集，某些类别可以暂时缺失；
- 同一类别可以跨多个批次重复出现；
- 可使用每类最多 K 个当前样本模拟小样本；
- 不保存全部历史数据，只维护容量为 200 或 500 的经验回放池；
- 训练完成后使用一个六类头直接预测全部已见类别，不依赖任务 ID。

因此当前实现属于 **Class-Incremental Learning（Class-IL）**，不是 Task-IL。

### 8.2 固定六类头与权重迁移

从第一批增量训练开始，学生模型即使用固定六类检测头。迁移逻辑按类别名称把旧四类分类通道映射到六类头对应位置：

```text
旧类 0..3：从四类 checkpoint 按名称复制
新类 4..5：保留六类检测头的新初始化参数
```

该实现不依赖 Ultralytics 内部易变化的 `_remap_cls_by_names`。代码同时处理不同版本检测头属性为空、输出尺度不同和 AMP dtype 不一致等边界情况。

初始 checkpoint 必须是真实、可加载的本地 `.pt`，而且 `names` 必须是严格四类前缀。只有 YAML 或伪造权重不能进入正式训练。

### 8.3 批次计划

系统支持两种批次配置：

1. `--num-batches N`：由代码确定性地生成 N 个批次；
2. `--batch-plan plan.json`：显式声明每批次允许到达的类别。

推荐使用显式计划，示例：

```json
{
  "batches": [
    {"id": "batch_01", "classes": ["soldier"]},
    {"id": "batch_02", "classes": ["small_aircraft"]},
    {"id": "batch_03", "classes": ["warship"]},
    {"id": "batch_04", "classes": ["tank"]},
    {"id": "batch_05", "classes": ["patrol_boat"]},
    {"id": "batch_06", "classes": ["armored_vehicle"]}
  ]
}
```

也可以在同一批放入多个类别，或让类别在后续批次再次出现。计划必须覆盖两个新增类；实际增量训练集也必须出现这两个新类，否则无法形成完整六类模型。

### 8.4 来源族隔离

离线增广会产生同一原图的多个变体。准备器将“原图 + 它的全部增广变体”视为同一个 source family，并保证整个来源族只进入一个增量批次，防止同源图像跨批次泄漏。

### 8.5 小样本限制

参数 `--max-current-images-per-class K` 限制每个批次、每个当前类别最多使用 K 个来源样本族。限制发生在来源族层面，不会把一个原图的不同增广版本错误地计作多个独立原始样本。

### 8.6 已见类与未来类隔离

每批次生成的 YAML 固定为 `nc=6` 和六类 `names`，保证检测头维度不变；但训练标签、验证视图和测试视图只暴露截至当前阶段已经到达的类别。未来类别在首次到达前不会出现在该阶段的监督和指标中。

## 9. 经验回放池

### 9.1 容量

代码只接受两种固定容量：

- 200；
- 500。

准备阶段可以同时传入两次 `--buffer-size`，一次生成两套可比实验数据；训练阶段每次选择其中一种容量。

### 9.2 更新规则

第一批开始前，回放池只从基础四类训练集采样。每批次完成后，候选集合由以下部分组成：

```text
基础四类历史样本 + 截止当前已经消费的增量样本
```

随后按已见类别尽量均衡地重新构建池。下一批的 `buffer_before` 必须严格等于上一批的 `buffer_after`，准备器会记录并检查该连续性。

### 9.3 容量的准确含义

回放容量统计的是“按类别物化的 replay entry”，不一定等于不同原始图像的数量。多标签原图可能为不同类别贡献多个 entry，每个物化标签视图只保留该 entry 对应类别的标注。这使类别均衡更直接，但分析样本独立性时必须以 `source_image` 和 source family 去重。

数据物化优先使用硬链接；当文件系统或权限不支持时自动回退为复制。

## 10. ER 与 DER 的实现

### 10.1 ER

ER（Experience Replay）在第 `t` 个批次将当前数据 `C_t` 与上一阶段回放池 `B_(t-1)` 合并：

```text
D_t = C_t ∪ B_(t-1)
```

模型在该混合视图上使用标准 YOLO 监督损失训练。回放样本持续提供旧类的框和类别监督，从而减轻灾难性遗忘。

### 10.2 DER

当前项目的 DER 在 ER 数据混合基础上增加冻结教师模型。教师来源为：

- 第一批：原始四类 checkpoint；
- 后续批次：上一批选出的 best checkpoint。

暗知识匹配只作用于回放样本，并只比较教师已知的旧类别。项目实现的损失为：

```text
L_dark = λ_cls · L_cls_dark + λ_box · L_box_dark

L_total = L_YOLO + λ_DER · L_dark
```

如果同时启用 Sparse-MoE，则 MoE 辅助项继续加入 `L_total`。

其中：

- `L_cls_dark`：学生旧类原始分类 logits 与教师 logits 的置信度加权 MSE；
- `L_box_dark`：学生和教师原始框/分布输出的置信度加权 MSE；
- 教师置信度来自旧类 sigmoid 输出；
- `der_min_confidence` 可过滤过低置信度位置。

默认超参数：

| 参数 | 默认值 |
| --- | ---: |
| `der_weight` | 1.0 |
| `der_cls_weight` | 1.0 |
| `der_box_weight` | 0.25 |
| `der_min_confidence` | 0.0 |

### 10.3 与原始 DER 论文公式的关系

经典 DER 通常把历史样本及其当时 logits 一并存入 memory，并在后续训练中回放已保存 logits。当前项目没有为每个回放样本持久化大尺寸检测特征图，而是在每个阶段加载冻结的上一阶段教师，在线重新计算 dark target。

因此这里是适配密集目标检测和工程存储约束的 DER 变体，而不是逐字复现分类论文公式。它的优势是无需保存庞大的多尺度检测 logits，并能兼容动态重建的均衡回放池；代价是教师前向增加计算量，而且知识目标代表“上一阶段模型”而不是“样本首次进入池时的模型”。实验报告中应明确这一实现差异。

## 11. Sparse-MoE 第三阶段架构

### 11.1 插入位置

Sparse-MoE 被插入到 YOLO 的 Detect 头之前，对送入检测头的多尺度特征进行残差适配。输入通道从实际模型动态读取，而不是写死。当前 YOLO 示例通常对应 P3/P4/P5 三个尺度。

```text
Backbone / Neck 多尺度特征
          │
          ├─ 上下文辅助头：模态 + 场景
          ├─ 图像质量统计
          └─ 共享 Router（每张图一次）
                     │
                   Top-K
                     │
       每尺度稀疏专家残差适配
                     │
                 Detect Head
```

### 11.2 Router 输入

每张图使用一个跨尺度共享路由结果。Router query 由以下信息拼接而成：

- 多尺度特征全局池化后的嵌入；
- stop-gradient 的 IR/SAR 模态概率；
- stop-gradient 的 air/sea/urban/forest 场景概率；
- 输入质量统计：均值、标准差、梯度能量和高频信息。

辅助概率在进入 Router 时停止梯度，避免 Router 反向捷径破坏模态/场景辅助头本身的监督语义。

### 11.3 Top-2 的含义

默认 `top_k=2`。对每张图，Router 先给所有专家计算概率，只选择概率最高的两个专家实际执行，再把这两个概率归一化为和为 1 的混合权重。

因此 Top-2 不是“两个固定模型投票”，而是“每张图动态选出两个专家并加权融合”。未进入 Top-2 的专家不会执行该图的特征适配路径，这也是稀疏计算的来源。

### 11.4 专家结构

每个专家是轻量残差瓶颈：

```text
1×1 降维 → depthwise 3×3 → SiLU → 1×1 升维 → 残差相加
```

最后的升维层采用零初始化，使新插入的专家在训练起点接近恒等映射，减少对已训练检测器的瞬时扰动。

### 11.5 训练目标

Sparse-MoE 开启时，目标函数为：

```text
L = L_YOLO
  + λ_m   · L_modality
  + λ_s   · L_scene
  + λ_bal · L_balance
  + λ_z   · L_router-z
  + λ_a   · L_anchor
```

DER 同时开启时，再加 `λ_DER · L_dark`。

各项作用如下：

| 损失 | 作用 |
| --- | --- |
| `L_modality` | 让上下文头识别 IR/SAR |
| `L_scene` | 让上下文头识别四种场景 |
| `L_balance` | 避免所有样本塌缩到少数专家 |
| `L_router-z` | 控制 Router logits 数值规模 |
| `L_anchor` | 限制专家参数偏离上一阶段 EMA 锚点 |

上下文元数据缺失时对应目标标记为 unknown，并在辅助损失中掩码，不会把未知值当成错误类别参与训练。

### 11.6 默认超参数

| 参数 | 默认值 |
| --- | ---: |
| `expert_count` | 5 |
| `top_k` | 2 |
| `expert_bottleneck` | 0.25 |
| `router_hidden` | 128 |
| `aux_hidden` | 128 |
| `modality_loss_weight` | 0.10 |
| `scene_loss_weight` | 0.10 |
| `balance_loss_weight` | 0.01 |
| `router_z_loss_weight` | 0.001 |
| `anchor_loss_weight` | 0.001 |
| `anchor_rho` | 0.95 |
| `router_temperature_start` | 2.0 |
| `router_temperature_end` | 1.0 |
| `router_temperature_warmup_epochs` | 3 |

温度在 warmup 周期内由 2.0 逐步下降到 1.0，使训练前期路由较平滑，后期逐渐形成更明确的专家选择。也可以用 `--router-temperature` 固定温度。

### 11.7 专家锚点与使用量

每阶段结束后，系统使用 `anchor_rho` 维护专家参数的 EMA 锚点，并在下一阶段通过 `L_anchor` 做软约束。它不是完全冻结旧专家，因此仍保留对新数据的适应能力。

专家使用统计包括 Top-K 次数/频率、平均概率、路由熵、最大占用率和变异系数。每个阶段开始时统计重新累计；重载 best checkpoint 后，训练期累积的使用状态会恢复，避免最终报告被错误清零。

每阶段可输出：

- `sparse_moe_config.json`；
- `expert_usage.json`；
- `expert_anchors.pt`；
- `context_metadata_summary.json`。

自定义 MoE checkpoint 应通过 `load_sparse_moe_checkpoint` 恢复结构和参数。普通 `YOLO(path)` 不一定能够自动重建自定义模块。

## 12. 上下文元数据

批次准备阶段会生成上下文索引。典型字段包括：

```text
materialized_image_path
source_image
sensor
scene
split
stage
sample_role
augmentation_operation
metadata_source
```

其中 `sample_role` 用于区分 current、replay、validation、test 等角色；DER 正是据此把 dark loss 约束到 replay 样本。模态优先来自增广 manifest 或明确元数据，缺失时可以根据文件名回退；场景缺失保持 unknown 并被掩码。

## 13. 指标、选模与结果解释

### 13.1 每阶段输出

每个批次训练后在该阶段的 val 视图上计算：

- 总体 `precision`；
- 总体 `recall`；
- 总体 `mAP@0.5`；
- 总体 `mAP@0.5:0.95`；
- 每个已见类的 `mAP@0.5` 和 `mAP@0.5:0.95`。

逐类别指标按照 Ultralytics 返回的 `ap_class_index` 正确映射。若某个类别在该验证阶段没有有效样本，则该类指标写为 `null`，不会错误地用总体均值补齐。

### 13.2 两种 mAP 的区别

`mAP@0.5` 只要求预测框与真实框 IoU 达到 0.5，主要反映是否大体找到了目标。

`mAP@0.5:0.95` 对 IoU 0.50、0.55、……、0.95 共十个阈值取平均，对定位精度严格得多。持续学习报告同时保留两张“阶段 × 类别”矩阵：前者便于观察类别是否仍可识别，后者更适合判断框定位质量是否退化。

### 13.3 持续学习指标

对每个类别，系统记录：

- `first_arrival`：类别首次进入训练的阶段；
- `final`：最终阶段得分；
- `forgetting = max(history) - final`；
- `BWT = final - first_arrival_score`。

同时计算已见类别的阶段平均值（Average Seen Accuracy 的检测版本）。

解释时应注意：

- forgetting 越小越好；
- BWT 为负通常表示遗忘，为正表示后续训练对旧类有正迁移；
- 只有 1 epoch 的冒烟测试用于证明流程可运行，不能用来评价 ER、DER 或 MoE 优劣；
- ER-200、ER-500、DER-200、DER-500 必须使用相同数据划分、随机种子和训练预算才能公平比较。

### 13.4 val 与 test 隔离

每个批次只使用 val 选取 best checkpoint。test 只在最后一批完成后评估一次，不能用于调参、早停或阶段选模。

主要汇总文件：

- `batch_incremental_stage_summary.json`：单阶段指标与产物；
- `batch_incremental_training_summary.json`：全部阶段、每类矩阵、遗忘、BWT、最终测试；
- `pipeline_plan.json`：一键流程的命令、基础 checkpoint、最终模型和汇总路径。

## 14. 完整训练命令

以下命令为 PowerShell 示例。所有路径必须替换成本机路径；数据和模型不得放入 Git 工作区的待提交范围。

### 14.1 先做离线计划检查

建议先使用一个全新的空工作目录执行：

```powershell
python train.py four-to-six-yolo `
  --base-data "D:\local_data\base_four_raw\data.yaml" `
  --increment-data "D:\local_data\increment_six_raw\data.yaml" `
  --generic-model "D:\local_models\yolo_base.pt" `
  --workspace "D:\local_runs\four_to_six_plan" `
  --batch-plan "D:\local_configs\six_batch_plan.json" `
  --max-current-images-per-class 20 `
  --method der `
  --buffer-size 200 `
  --buffer-size 500 `
  --base-epochs 100 `
  --increment-epochs 30 `
  --device 0 `
  --seed 42 `
  --sparse-moe `
  --expert-count 5 `
  --top-k 2 `
  --expert-bottleneck 0.25 `
  --router-hidden 128 `
  --aux-hidden 128 `
  --anchor-rho 0.95 `
  --plan-only
```

该命令只验证协议并生成计划，不导入 Ultralytics、不训练、不访问网络。检查 `pipeline_plan.json` 中的目录和命令后，换一个新的空 workspace 运行正式流程，或清空仅含计划文件的临时计划目录后再使用。

### 14.2 原始四类 + 原始六类的一键 DER 完整流程

```powershell
python train.py four-to-six-yolo `
  --base-data "D:\local_data\base_four_raw\data.yaml" `
  --increment-data "D:\local_data\increment_six_raw\data.yaml" `
  --generic-model "D:\local_models\yolo_base.pt" `
  --workspace "D:\local_runs\four_to_six_der" `
  --batch-plan "D:\local_configs\six_batch_plan.json" `
  --max-current-images-per-class 20 `
  --method der `
  --buffer-size 200 `
  --buffer-size 500 `
  --base-epochs 100 `
  --increment-epochs 30 `
  --device 0 `
  --seed 42 `
  --sparse-moe `
  --expert-count 5 `
  --top-k 2 `
  --expert-bottleneck 0.25 `
  --router-hidden 128 `
  --aux-hidden 128 `
  --modality-loss-weight 0.10 `
  --scene-loss-weight 0.10 `
  --balance-loss-weight 0.01 `
  --router-z-loss-weight 0.001 `
  --anchor-loss-weight 0.001 `
  --anchor-rho 0.95 `
  --router-temperature-start 2.0 `
  --router-temperature-end 1.0 `
  --router-temperature-warmup-epochs 3
```

一键流程会依次执行：两套数据离线增广、四类基础训练、批次准备、DER-200 和 DER-500 两组增量训练。内部会设置离线模式，并为训练命令追加 `--no-builtin-aug --no-amp`，避免 Ultralytics 内置随机增广破坏离线增广协议，也避免 AMP 环境探测触发额外下载。

如果只需要一个容量，保留一个 `--buffer-size` 即可。工作目录必须不存在或为空。

### 14.3 ER 对照组

使用与 DER 完全相同的路径、批次计划、种子和 epoch，只把：

```powershell
--method der
```

改为：

```powershell
--method er
```

推荐形成四组主对照：ER-200、ER-500、DER-200、DER-500。如果 GPU 预算允许，再分别增加是否启用 Sparse-MoE 的消融组。

### 14.4 已增广六类数据：分阶段执行

如果现有六类目录已经是增广后的 train/val/test 合体版，不要再运行一键增广。先准备批次：

```powershell
python train.py prepare-batch-il `
  --base-data "D:\local_data\base_four_augmented\data.yaml" `
  --increment-data "D:\local_data\six_class_augmented\data.yaml" `
  --output "D:\local_runs\prepared_batch_il" `
  --batch-plan "D:\local_configs\six_batch_plan.json" `
  --max-current-images-per-class 20 `
  --buffer-size 200 `
  --buffer-size 500 `
  --seed 42
```

然后以已经完成的四类 best checkpoint 启动 DER-200：

```powershell
python train.py batch-il-yolo `
  --prepared "D:\local_runs\prepared_batch_il" `
  --initial-checkpoint "D:\local_models\four_class_best.pt" `
  --method der `
  --buffer-size 200 `
  --output "D:\local_runs\der_200_sparse_moe" `
  --epochs 30 `
  --patience 10 `
  --image-size 640 `
  --batch-size 8 `
  --workers 0 `
  --device 0 `
  --seed 42 `
  --learning-rate 0.001 `
  --der-weight 1.0 `
  --der-cls-weight 1.0 `
  --der-box-weight 0.25 `
  --der-min-confidence 0.0 `
  --sparse-moe `
  --expert-count 5 `
  --top-k 2 `
  --expert-bottleneck 0.25 `
  --router-hidden 128 `
  --aux-hidden 128 `
  --modality-loss-weight 0.10 `
  --scene-loss-weight 0.10 `
  --balance-loss-weight 0.01 `
  --router-z-loss-weight 0.001 `
  --anchor-loss-weight 0.001 `
  --anchor-rho 0.95 `
  --router-temperature-start 2.0 `
  --router-temperature-end 1.0 `
  --router-temperature-warmup-epochs 3 `
  --no-builtin-aug `
  --no-amp
```

训练 DER-500 时，将 `--buffer-size 200` 和输出目录分别改为 500 对应值。ER 对照同理修改 `--method er`。

### 14.5 最小流程测试

正式训练前可以使用小 batch、低分辨率和 1 epoch 验证环境：

```powershell
python train.py batch-il-yolo `
  --prepared "D:\local_runs\prepared_batch_il" `
  --initial-checkpoint "D:\local_models\four_class_best.pt" `
  --method der `
  --buffer-size 200 `
  --output "D:\local_runs\smoke_der_200" `
  --epochs 1 `
  --image-size 320 `
  --batch-size 2 `
  --workers 0 `
  --device 0 `
  --seed 42 `
  --sparse-moe `
  --no-builtin-aug `
  --no-amp
```

冒烟测试只检查数据、模型、损失、checkpoint、逐阶段验证和最终测试能否连通，不用于得出模型精度结论。

## 15. 路由训练脚本

`scene_recognition/route_training/` 中包含三条可独立训练的路线：

| 脚本 | 任务 | 主要默认值 |
| --- | --- | --- |
| `train_scene_router_incremental.py` | 四场景 YOLOv8n-cls Router | 224、50 epoch、batch 32、patience 10 |
| `train_easy_6class_yolov10_incremental.py` | 简单六类 YOLOv10n 检测器 | train 960、export 640、50 epoch、batch 8 |
| `train_hard_3class_yolov8n_corrected.py` | soldier/tank/armored_vehicle 困难三类检测器 | 960、150 epoch、batch 8、patience 20 |

场景 Router 的普通 Top-1 Accuracy 不等于检测路由正确率。正式部署前应在检测验证集上计算“场景路由后是否选择正确检测分支”，并依据该指标选择 Router checkpoint。

这组三分支路线与 Sparse-MoE 是两个层次：前者是模型级粗路由，后者是在单个检测模型内部进行特征级动态专家选择。是否同时启用，应通过消融实验决定。

## 16. 产物与模型加载

正式运行后，重点检查：

```text
workspace/
├─ base_augmented/
├─ increment_augmented/
├─ prepared_batch_il/
├─ runs/base_four/weights/best.pt
├─ batch_il_der_buffer_200/
├─ batch_il_der_buffer_500/
└─ pipeline_plan.json
```

具体目录名以 `pipeline_plan.json` 为准。最终六类模型、每阶段 best 模型和训练汇总路径均会写入计划文件。

普通 YOLO checkpoint 可按 Ultralytics 方式加载。包含自定义 Sparse-MoE 的 checkpoint 应优先使用项目提供的加载器，以便先重建自定义模块再恢复参数：

```python
from scene_recognition.detector_module.sparse_moe_checkpoint import load_sparse_moe_checkpoint

model = load_sparse_moe_checkpoint("D:/local_models/final_sparse_moe.pt")
```

当前 `TargetDetector` 默认仍通过 `ultralytics.YOLO(path)` 加载权重；只有该 checkpoint 能在当前 Python 环境中完整反序列化自定义 MoE 模块时，Agent 才能直接读取路由诊断。正式部署前必须用目标 checkpoint 做一次 Agent 端到端加载验证；若标准加载不能重建 MoE，则还需要为 `TargetDetector` 增加项目加载器或已加载模型注入接口，不能静默退化为 sidecar 标签回退。

## 17. ONNX 与部署边界

普通 Ultralytics YOLO 模型可以按相应版本的 export API 导出 ONNX。Sparse-MoE 模型包含项目自定义模块和动态 Top-K 稀疏执行，当前训练代码已完成 PyTorch 训练、验证、checkpoint 恢复和 Agent 诊断，但**不能据此直接承诺动态路由 ONNX 或 Ascend OM 完整可用**。

正式导出第三阶段模型前还需要：

1. 明确导出的是动态 Top-K，还是固定/静态专家组合；
2. 为自定义模块补充独立 export 路径；
3. 使用 ONNX Runtime 对比 PyTorch 的框、类别和路由输出；
4. 若上昇腾，继续验证算子支持、动态索引和精度漂移；
5. 将部署模型的 `names`、输入尺寸、归一化、置信度阈值和 NMS 配置一并固化。

因此，当前完成状态是“Sparse-MoE 的 PyTorch 训练与推理链路可用”，不是“所有部署后端已经验收”。

## 18. 已完成验证

当前代码已经完成以下层级的验证：

- 批次准备、类别协议、回放连续性和指标映射的单元测试；
- 四类到六类检测头迁移的专项测试；
- Sparse-MoE 路由、专家、checkpoint、usage 恢复和损失权重测试；
- 相关模块 `compileall`；
- 本地 NVIDIA GPU 上两批次、DER-200、Sparse-MoE、每批 1 epoch 的最小训练；
- 最小训练产生了阶段 best、最终 checkpoint、逐阶段 mAP、最终 test 和专家使用统计；
- 第一阶段尚未到达的类别在指标矩阵中正确显示为 `null`。

最近一次专项测试集为 16 项通过。全仓测试发现 119 项中 118 项通过，另有 1 项既有收集错误：`Agent.tests.unit.test_architecture_components` 引用尚不存在的 `Agent.data.copy_paste`。该问题与本次批次 Class-IL 和 Sparse-MoE 主流程无直接关系，但在宣称“全仓测试完全通过”之前仍需处理。

最小训练的数值只证明流程连通。由于 epoch、样本量和分辨率都被刻意压低，不能把其 mAP 当作模型性能结论。

## 19. 当前已实现与尚未实现

### 19.1 已实现

- 四类基础数据离线增广和基础模型训练编排；
- 六类增量数据离线增广；
- 任意多批次、任意非空类子集和类别重复到达；
- 每类 K-shot 小样本限制；
- 容量 200/500 的类别均衡回放池；
- ER 与检测版在线教师 DER；
- 四类检测头按名称扩展为固定六类头；
- 每阶段、每类别 `mAP@0.5` 与 `mAP@0.5:0.95`；
- forgetting、BWT 和最终 test；
- Sparse-MoE Top-K 路由、五专家默认配置、上下文辅助监督和专家锚点；
- Agent 读取检测结果和 MoE 路由诊断；
- 一键流程、分阶段流程、dry-run/plan-only 和本地离线运行约束。

### 19.2 尚未完成或未作正式验收

- Task-IL 的任务 ID、任务专属头和任务级评估协议；
- `docs/关键修正.md` 中提出的旧类伪标签补全；
- P2/P3 ROI 级特征蒸馏；
- 原型库、prototype rehearsal 或 prototype classifier；
- copy-paste 数据增强模块，当前相关全仓测试仍引用缺失模块；
- Sparse-MoE 动态 Top-K 的通用 ONNX/Ascend 导出与端侧性能验收；
- 大规模正式训练下 ER/DER、200/500、是否 MoE 的完整对照结论；
- 第二评测集切片后的最终挑战赛式盲测报告。

这些项目不应在论文或答辩中描述为已经完成。

## 20. 正式实验建议

建议固定同一数据划分、批次计划、K-shot 数量、seed、基础 checkpoint 和训练预算，至少运行：

| 组别 | 方法 | Buffer | Sparse-MoE |
| --- | --- | ---: | --- |
| A | ER | 200 | 关 |
| B | ER | 500 | 关 |
| C | DER | 200 | 关 |
| D | DER | 500 | 关 |
| E | DER | 200 | 开 |
| F | DER | 500 | 开 |

如果资源允许，对关键组使用至少 3 个随机种子，报告均值和标准差。除最终 mAP 外，还应同时报告：

- 每批次每类两种 mAP 矩阵；
- old-class average、new-class average；
- 每类 forgetting 和 BWT；
- 训练时间、显存、推理延迟；
- 专家占用率、路由熵和负载变异系数；
- unknown 上下文比例；
- 每类实际来源族数量，而不仅是增广后图像数。

如果专家长期只使用 1～2 个，应优先检查 balance loss、温度、上下文标签质量和数据不均衡，而不是直接增加专家数。

## 21. 数据安全与提交规范

本项目的数据集仅允许本地使用。执行训练和提交时必须遵守：

1. 数据集、图片、标签、权重、ONNX/OM、训练 runs 和 manifest 不上传远端；
2. 不在文档、日志或配置中提交真实本机绝对数据路径；
3. 运行目录尽量放在仓库外的本地磁盘；
4. 提交前执行 `git status --short` 和 `git diff --cached --name-only`；
5. 只应看到代码、测试和经过审查的说明文档；
6. 如果生成文件意外被跟踪，先从 Git 暂存区移除，再检查 `.gitignore`，不要删除唯一的数据副本。

仓库 `.gitignore` 已覆盖常见数据目录、图片/标签、runs 以及 `.pt/.onnx/.om/.joblib` 等模型文件，但提交者仍需人工复核。

## 22. 交接检查清单

正式训练前：

- [ ] 本机 Python、PyTorch、CUDA、Ultralytics 版本可用；
- [ ] 通用 YOLO 初始模型是本地文件，不依赖在线下载；
- [ ] 四类和六类 `names` 顺序通过检查；
- [ ] train/val/test 来源无泄漏；
- [ ] 图像模态能从 manifest、元数据或文件名可靠获得；
- [ ] 批次计划覆盖两个新增类；
- [ ] 新增类在增量 train 中确实有标注；
- [ ] 计划输出目录为空；
- [ ] `--plan-only` 结果已经人工核对；
- [ ] 冒烟训练可以生成最终 checkpoint 和 summary。

正式训练后：

- [ ] 每一批都有 best checkpoint；
- [ ] `buffer_before`/`buffer_after` 连续；
- [ ] 未来类在首次到达前的指标为 `null`；
- [ ] 最终 test 只执行一次；
- [ ] mAP 类别索引与 YAML `names` 一致；
- [ ] expert usage 不为空且未严重塌缩；
- [ ] ER/DER 对照使用相同协议；
- [ ] 训练产物仍位于本地且未被 Git 跟踪；
- [ ] 对外报告明确区分冒烟结果和正式结果。

## 23. 结论

当前 SC 部分已经具备一条可执行的“四类基础模型 → 六类、多批次、小样本 Class-IL”主链路。系统能够对原始数据做离线确定性增广，按类到达构建多个批次，以 200/500 容量运行 ER 或检测版 DER，并可在检测头前启用 Top-2 Sparse-MoE。训练过程能够输出逐阶段、逐类别的严格检测指标和持续学习指标，Agent 也能读取最终检测与专家路由诊断。

当前最重要的下一步不是继续堆叠模块，而是在固定协议下完成足量 epoch、多随机种子的正式对照实验，确认 DER 和 Sparse-MoE 是否真实改善旧类保持、新类学习与定位精度。随后再补 Task-IL、伪标签/ROI 蒸馏、原型库和 Sparse-MoE 部署导出，才能形成从训练理论、实验结论到端侧部署的完整闭环。
