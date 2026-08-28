# 六阶段类增量学习：ER 与 DER

## 1. 当前正式协议

原来的“四类基础模型一次加入两个新类”降级为历史预实验。当前正式 Class-IL
使用单一扩展检测头，六种类别严格按照 YOLO 类别编号依次训练：

```text
T1 soldier
T2 small_aircraft
T3 warship
T4 tank
T5 patrol_boat
T6 armored_vehicle
```

阶段 `Tt` 的普通训练标注只包含第 `t` 个类别；验证与测试标注包含 `T1..Tt` 的全部
已见类别。未来类别标签不会进入当前训练视图。检测头从 1 类逐步扩展到 6 类，上一
阶段 checkpoint 是下一阶段的初始化模型。

初始模型不能使用已经训练过项目四类的 checkpoint，否则在 T1 就已经见过
`small_aircraft`、`warship` 和 `tank`，构成类别泄漏。允许使用本地通用预训练模型或
本地 YOLO 架构 YAML；训练入口不会下载模型。

## 2. 经验回放缓冲池

实验固定比较两种容量：200 和 500 张图像。缓冲池按已学习类别轮转抽样，优先覆盖
不同原始帧，再选择同一帧的增广版本，避免缓冲池被少量原图的多个增广副本占满。

每个回放槽保存：

- 本地图像路径；
- 获取该样本时的类别；
- 原始帧来源键；
- 对应单类监督标签。

每个阶段结束后，程序在全部已见类别之间重新平衡固定容量。训练阶段使用的是上一
阶段结束时的缓冲池，缓冲样本数永远不超过指定容量。

## 3. ER

ER 将当前类别样本与旧类缓冲样本混合，使用标准 YOLO 检测损失训练：

```text
L_ER = L_box + L_cls + L_dfl
```

ER 不增加蒸馏项，是后续方法的主要回放基线。

## 4. DER

DER 使用与 ER 完全相同的监督回放样本，并冻结上一阶段 checkpoint 作为教师。仅对
batch 中来自 replay manifest 的样本计算暗响应约束：

```text
L_DER = L_box + L_cls + L_dfl
      + lambda_der * (L_dark_cls + 0.25 * L_dark_box)
```

其中：

- `L_dark_cls` 匹配教师已有类别的原始分类 logits；
- `L_dark_box` 匹配 YOLO 检测头的边框分布 logits；
- 教师置信度用于降低无目标位置的背景权重；默认阈值为 0，避免弱教师阶段把 DER
  整体关闭，也可通过 `--der-min-confidence` 提高阈值；
- 新增类别通道不参与暗响应匹配；
- 暗响应由固定教师在线计算，不写入大体积云端或磁盘 logit 缓存。

因此当前 DER 是面向目标检测的在线暗响应回放实现，不是对全部新类图像施加蒸馏的
普通 LwF。

## 5. 本地运行

原始合体数据集只有 train/val 时，先保留 train 不动，并将未增广留出集按原图组分层
拆成 val/test。默认 `--test-fraction 0.5` 指 test 占原留出集的一半；同一原图及其
增广版本不会跨集合：

```powershell
python train.py split-yolo `
  --data "<原始六类合体数据集>\data.yaml" `
  --output "<本地新数据目录>" `
  --test-fraction 0.5 `
  --seed 42
```

当前本机按 `seed=42` 生成的划分为 train 2828 张（707 个原图组）、val 92 张、
test 91 张。`patrol_boat` 和 `armored_vehicle` 在 val/test 中各有 7/7 个原图；
三个集合的原图组交集为 0。划分清单与完整分类别统计保存在新数据目录的
`split_summary.json`，原始合体数据集保持不变。

随后指定新数据 YAML 和本地初始模型：

```powershell
$env:CLASS_IL_DATA = "<带独立 test 的六类数据集 data.yaml>"
$env:CLASS_IL_INIT = "<本地通用预训练模型或架构 YAML>"
```

生成六阶段视图以及 200/500 两组缓冲池：

```powershell
python train.py prepare-class-il `
  --data "$env:CLASS_IL_DATA" `
  --output "数据集（不上传git）\class_il_prepared"
```

先执行不训练的完整协议检查：

```powershell
python train.py class-il-yolo `
  --prepared "数据集（不上传git）\class_il_prepared" `
  --initial-model "$env:CLASS_IL_INIT" `
  --method der `
  --buffer-size 200 `
  --output "数据集（不上传git）\runs\class_il_der_b200" `
  --dry-run
```

正式对比需要四组独立实验：

```text
ER  + buffer 200
ER  + buffer 500
DER + buffer 200
DER + buffer 500
```

示例：

```powershell
python train.py class-il-yolo `
  --prepared "数据集（不上传git）\class_il_prepared" `
  --initial-model "$env:CLASS_IL_INIT" `
  --method er `
  --buffer-size 200 `
  --epochs 30 `
  --output "数据集（不上传git）\runs\class_il_er_b200"
```

每组实验必须使用不同输出目录，程序拒绝覆盖非空目录。

## 6. 输出与评测

每个阶段保存：

- `weights/best.pt`；
- 当前阶段输入/输出 checkpoint；
- 新类、已见类别和回放数量；
- 分类别 mAP@0.5 与 mAP@0.5:0.95；
- DER 暗响应参数和阶段耗时。

每轮训练和早停只读取 val；阶段结束后才在 test 上评估。完整训练报告输出六行类别
性能矩阵，并计算：

- 最终六类平均 mAP；
- Average Incremental Accuracy；
- Average Forgetting；
- Backward Transfer（BWT）。

输入 YAML 含独立 test 时，报告使用 test 性能矩阵并标记 `official=true`；该字段只
表示本地实验协议具备独立留出集，同时另记 `competition_official=false`，不能冒充
主办方隐藏测试。若继续使用旧的 train/val 配置，则自动退回 val 并标记
`official=false`。测试集不得用于调参、早停或 checkpoint 选择。

## 7. Task-IL 扩展边界

协议已将 `scenario`、`task_id`、类别顺序、检测头策略和回放池接口分开记录。后续增加
Task-IL 时可复用缓冲池选择、ER 混合和 DER 暗响应损失，只需新增任务划分与多头/任务
标识调度；当前版本不会把 Class-IL 的单头假设伪装成已经实现的 Task-IL。

## 8. 隐私约束

数据准备和训练均设置为本地流程。数据、标签、硬链接视图、缓冲池索引、暗响应、模型
权重和训练报告不得上传云端，也不得加入 Git。
