# Agent 持续学习完善说明

## 1. 结论

本次完善把原先只能处理四类、只能生成模拟增量协议的 Agent，扩展为可执行的
`r1 四类基础模型 → r2 新增两类 → 增量微调/回放 → New-mAP 与 KRR 评测`闭环。

当前已经完成并验证：

- r2 六类标签、文件名和场景/模态解析；
- Agent 单图和批量推理中的新增类别处理；
- 新类别与场景的一致性推理；
- Agent 到 `train.py` 的参数桥接；
- 本地增量、回放、val、test 清单及六类 YAML 生成；
- 本地 checkpoint 增量微调/回放训练入口；
- old-mAP、New-mAP、all-mAP、KRR 的统一评测入口；
- 数据目录 Git 忽略和无云端依赖约束；
- 81 项自动化测试全部通过。

当前尚未产生正式持续学习分数，因为本机缺少四类基础 checkpoint、基础数据索引，
且现有 r2 目录只有 `inc_train`，没有覆盖全部六类的独立固定测试集。程序会将此状态
标记为 `evaluation_ready=false`，不会用训练集成绩伪装成正式 KRR。

## 2. 数据与隐私边界

本次只读检查的 r2 数据结果如下：

| 项目 | 数量 |
|---|---:|
| 图像 | 140 |
| YOLO 标签 | 140 |
| 目标框 | 1006 |
| IR / SAR | 100 / 40 |
| forest / sea / urban | 35 / 70 / 35 |
| 损坏图像、缺失标签、非法框 | 0 |

类别顺序必须固定：

| ID | 类别 | r2 目标框 | 角色 |
|---:|---|---:|---|
| 0 | soldier | 0 | 旧类 |
| 1 | small_aircraft | 0 | 旧类 |
| 2 | warship | 70 | 旧类 |
| 3 | tank | 65 | 旧类 |
| 4 | patrol_boat | 419 | 新类 |
| 5 | armored_vehicle | 452 | 新类 |

隐私措施：

1. `.gitignore` 显式排除 `数据集（不上传git）/`；
2. 清单只保存本机绝对路径，且输出目录已被 Git 忽略；
3. 数据准备只写清单和 YAML，不复制源图像；
4. 增量训练命令强制使用本地 checkpoint，不接受远程模型名称；
5. 训练入口设置离线模式，缺少模型或依赖时本地失败；
6. 本次未执行 Git add、commit、push，也未上传任何数据。

## 3. 持续学习任务定义

本轮属于类增量检测：

```text
阶段 0：soldier / small_aircraft / warship / tank
阶段 1：在保留阶段 0 能力的同时，加入 patrol_boat / armored_vehicle
```

必须同时回答两个问题：

1. 新类别是否学会：`New-mAP`；
2. 旧类别是否遗忘：`old-mAP-before`、`old-mAP-after` 和 `KRR`。

定义：

```text
KRR = old-mAP-after / old-mAP-before
```

评测器同时输出 mAP@0.5 与 mAP@0.5:0.95 两套口径。目标值沿用项目协议：

```text
New-mAP@0.5 >= 0.60
KRR@0.5 >= 0.95
```

## 4. 完善后的架构

```text
r1 基础数据 ──> 四类训练 ──> base checkpoint ───────────────┐
      │                                                      │
      └─> 固定 val/test ──────────────────────────────────┐  │
                                                         │  │
r2 inc_train ──> 六类校验 ──> train_increment ─────────┐ │  │
                                                       │ │  │
r1 train ──> 分场景/模态回放采样 ──> train_replay ─────┼─┘  │
                                                       │    │
                                                       ▼    ▼
                                              六类增量微调/回放
                                                       │
                                                       ▼
                                                 updated checkpoint
                                                       │
             base checkpoint ──────────────────────────┤
             固定六类 test ────────────────────────────┘
                                                       │
                                                       ▼
                                      old/new/all mAP + KRR 报告
```

Agent 在线推理路径仍为：

```text
图像质量 → 模态 → 场景 → 六类检测 → 目标确认
→ 场景/目标一致性 → 专家路由 → 结构化报告与本地记忆
```

## 5. 代码改动

### 5.1 六类与推理

- `scene_recognition/detector_module/__init__.py`
  - 区分基础四类、r2 新类和完整六类；
  - 保留旧代码使用的四类别名，避免破坏基线复现。
- `Agent/schemas.py`
  - Agent 默认使用完整六类；
  - 增加巡逻艇、装甲车辆中文名称。
- `Agent/reasoning.py`
  - `patrol_boat` 与 sea 相容；
  - `armored_vehicle` 与 urban/forest 相容。
- `Agent/target.py`
  - 扩展六类场景先验；
  - 四类裁剪分类器遇到新类时保留检测器结果，不再错误覆盖成旧类。

### 5.2 数据与协议

- `image_processing/analyze_and_prepare.py`
  - 文件名从只支持 `r1_base` 扩展为任意 `r<轮次>_<base|inc>`；
  - 没有合法记录时给出明确错误，不再出现 `records[0]` 越界；
  - IR/SAR 候选配对加入轮次和阶段隔离。
- `create_incremental_protocol.py`
  - 支持从 r2 `classes.txt` 生成`四类基础 + 两类新增`协议。
- `build_incremental_dataset.py`
  - 类别顺序由协议提供，不再强制四类。
- `prepare_continual_dataset.py`
  - 校验基础类别前缀与新增类别；
  - 生成仅增量和回放训练清单；
  - 支持独立新增类 val/test 目录；
  - 支持仅用于冒烟测试的本地 holdout；
  - 检查固定 test 是否覆盖全部旧类和新类。

### 5.3 训练与评测

- `train_continual_yolo.py`
  - 只接受本地四类 checkpoint；
  - 支持 `increment_only` 与 `replay` 两种基线；
  - 从 val 选择六类最优 checkpoint；
  - 记录版本、CUDA、耗时、类别顺序和方法边界。
- `evaluate_continual.py`
  - 使用同一套 COCO-style 101 点 AP 实现评估更新前后 checkpoint；
  - 分别聚合旧类、新类和全部类别；
  - test 类别支持不足时拒绝宣称正式闭环。
- `Agent/cli.py` 与 `train.py`
  - 新增准备、训练、评测命令；
  - 修复文档所用 `--` 分隔符被错误转发的问题。

### 5.4 可选依赖

`ultralytics` 仅在真正训练或 YOLO 推理时导入。数据、协议、指标单测和其他模块不再因
本机未安装 `ultralytics` 而在导入阶段整体失败。

## 6. 标准运行流程

### 6.1 环境变量

```powershell
$env:R2_DATASET = "<datasets_r2_inc_train 的本机路径>"
$env:BASE_DATASET = "<r1 基础数据集本机路径>"
$env:BASE_CHECKPOINT = "<四类基础检测模型 best.pt>"
```

数据变量只存在于当前 PowerShell 进程，不写入仓库。

### 6.2 生成基础索引

```powershell
python -m Agent.cli prepare-scene -- `
  --dataset "$env:BASE_DATASET" `
  --output image_processing\artifacts
```

### 6.3 生成正式增量/回放清单

```powershell
python -m Agent.cli prepare-continual -- `
  --increment-dataset "$env:R2_DATASET" `
  --base-index image_processing\artifacts\scene_index.csv `
  --replay-limit 200 `
  --output scene_recognition\detector_module\artifacts\continual_r2
```

如果另有独立新增类验证集和测试集：

```powershell
python -m Agent.cli prepare-continual -- `
  --increment-dataset "$env:R2_DATASET" `
  --increment-val-dataset "<r2_inc_val>" `
  --increment-test-dataset "<r2_inc_test>" `
  --base-index image_processing\artifacts\scene_index.csv `
  --output scene_recognition\detector_module\artifacts\continual_r2
```

关键输出：

```text
train_increment.txt          仅 r2 训练注入
train_replay.txt             从 r1 train 选择的旧样本
train_mixed.txt              r2 + replay
val.txt / test.txt           固定评测清单
data_increment_only.yaml     普通微调基线
data_replay.yaml             回放基线
continual_dataset_summary.json
```

### 6.4 本地冒烟协议

当前只有 `inc_train` 时，可以验证代码链路：

```powershell
python -m Agent.cli prepare-continual -- `
  --increment-dataset "$env:R2_DATASET" `
  --increment-val-ratio 0.1 `
  --increment-test-ratio 0.1 `
  --output scene_recognition\detector_module\artifacts\continual_r2_smoke
```

该命令会清楚标记 `official=false`。所得分数只能用于程序冒烟，不能写入答辩主表。

### 6.5 运行两条持续学习基线

普通增量微调：

```powershell
python -m Agent.cli train-continual -- `
  --data scene_recognition\detector_module\artifacts\continual_r2\data_increment_only.yaml `
  --base-model "$env:BASE_CHECKPOINT" `
  --strategy increment_only `
  --output scene_recognition\detector_module\runs\continual_r2_increment_only
```

旧样本回放：

```powershell
python -m Agent.cli train-continual -- `
  --data scene_recognition\detector_module\artifacts\continual_r2\data_replay.yaml `
  --base-model "$env:BASE_CHECKPOINT" `
  --strategy replay `
  --output scene_recognition\detector_module\runs\continual_r2_replay
```

### 6.6 计算持续学习指标

```powershell
python -m Agent.cli evaluate-continual -- `
  --data scene_recognition\detector_module\artifacts\continual_r2\data_replay.yaml `
  --before "$env:BASE_CHECKPOINT" `
  --after scene_recognition\detector_module\runs\continual_r2_replay\weights\best.pt `
  --output scene_recognition\detector_module\runs\continual_r2_replay\continual_evaluation.json
```

必须对 `increment_only` 与 `replay` 分别评测，才能判断回放是否降低遗忘。

## 7. 本次真实验证结果

| 检查 | 结果 |
|---|---|
| Python 语法编译 | 通过 |
| 自动化测试 | 81/81 通过 |
| r2 数据审计 | 140/140 合法，1006 个框 |
| Agent CLI 场景准备 | 成功，140 条索引 |
| Agent r2 批量推理 | 成功，140 条报告 |
| 模态结果 | IR 100、SAR 40 |
| 最终场景 | forest 35、sea 70、urban 35 |
| 场景—目标一致性 | 140/140 `consistent` |
| 本地冒烟清单 | train 114、val 13、test 13 |
| 冒烟 test 新类覆盖 | 完整 |
| 冒烟 test 旧类覆盖 | 缺 soldier、small_aircraft |
| 正式 `evaluation_ready` | `false` |

验证产物位于：

```text
Agent/runs/continual_validation_20260823/
scene_recognition/detector_module/artifacts/continual_r2_smoke/
```

这些目录均为本地产物，不提交 Git。

## 8. 结果表建议

正式实验至少运行：

| 方法 | 旧类 before | 旧类 after | KRR | New-mAP | All-mAP | 更新时间 |
|---|---:|---:|---:|---:|---:|---:|
| 仅增量微调 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| r2 + 旧样本回放 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 回放 + 蒸馏（后续） | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

不要只报告六类最终平均 mAP。若旧类显著下降，即使 All-mAP 较高，也不能证明持续学习成功。

## 9. 当前方法边界

已经实现的是可执行、可对照的普通微调和旧样本回放基线。理论文档中的以下机制尚未实现：

- 可靠检测响应的分类/定位蒸馏；
- P2/P3 目标区域特征蒸馏；
- 类别—模态原型损失；
- 多专家锚点式软巩固；
- 教师伪标签补全不完整旧类标注。

原因是这些机制需要改写 YOLO Trainer、检测头输出和训练损失，而不能通过普通
`model.train()` 参数真实实现。本次实现没有把这些理论项写进日志后冒充已生效。

下一阶段最合理的实验顺序：

```text
仅增量微调
→ 加旧样本回放
→ 加旧模型伪标签
→ 加分类/定位蒸馏
→ 加 P2/P3 特征蒸馏
```

每增加一项都应使用相同数据、随机种子、训练预算和固定 test，报告对 New-mAP 与 KRR
的独立影响。

## 10. 验收条件

只有同时满足以下条件，才能称为“完整持续学习实验已跑通”：

1. 有可追溯的四类基础 checkpoint；
2. 有 r1 train 回放来源或明确禁止回放的规则；
3. 有覆盖四个旧类和两个新类的独立固定 test；
4. 更新前后 checkpoint 均在同一 test 上评估；
5. 输出 old-mAP-before、old-mAP-after、New-mAP、All-mAP、KRR 和更新时间；
6. 不使用 r2 训练留出分数替代正式测试结果；
7. 原始数据、清单、权重和运行日志均只保存在本地。
