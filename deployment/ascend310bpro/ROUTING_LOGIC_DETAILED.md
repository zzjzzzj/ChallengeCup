# 三模型场景路由详细逻辑说明

## 1. 方案目的

当前方案采用“场景分类器 + 二选一检测器”的互斥路由结构，主要目标是：

- 在森林、城市等士兵小目标较难识别的场景中，提高 `soldier` AP50；
- 在空中、海面等相对容易的场景中减少计算量；
- 每张图只运行一个检测器，避免两个检测器叠加产生额外计算和结果融合问题；
- 场景分类不确定时优先保护士兵召回率。

## 2. 总体流程

```text
输入图像
   │
   ▼
224×224 场景分类器
   │
   ├─ air / sea ─────────► easy 六分类检测器，640×640
   │
   ├─ forest / urban ────► hard 三分类检测器，960×960
   │
   └─ 场景置信度 < 0.60 ─► hard 三分类检测器，960×960
```

## 3. 第一级：场景分类

模型文件：

`models/01_scene_router_224.onnx`

模型输入：

- 输入张量：`[1, 3, 224, 224]`
- 数据排列：NCHW
- 通道顺序：RGB
- 数据类型：float32
- 输入值归一化到 `[0, 1]`

场景类别顺序：

| 场景编号 | 场景名称 | 所属路由组 |
|---:|---|---|
| 0 | air | easy |
| 1 | forest | hard |
| 2 | sea | easy |
| 3 | urban | hard |

模型输出为 `[1,4]`。取分数最高的类别作为场景预测，同时保留最高分作为场景置信度：

```python
scene_id = argmax(scene_scores)
scene_confidence = max(scene_scores)
```

## 4. 路由判定规则

当前安全阈值为：

```yaml
route_confidence: 0.60
```

完整判定逻辑：

```python
if scene_confidence < 0.60:
    route = "hard"
elif scene_id in (0, 2):       # air、sea
    route = "easy"
else:                          # forest、urban
    route = "hard"
```

也可以概括为：

> 只有能够较确定地判断为 air 或 sea 时才进入 easy；forest、urban 和不确定场景全部进入 hard。

`0.60` 是场景路由置信度阈值，不是目标检测置信度阈值。两种阈值必须分开配置。

## 5. 为什么不确定场景进入 hard

路由的主要风险是把真正的森林或城市图像错误送入 easy 分支，这可能增加小目标士兵漏检。

当前选用的安全 checkpoint 曾出现 1 张 hard 图被原始 Top-1 判断为 easy，但该图最高置信度只有约 `0.589`，低于 `0.60`，最终会被安全规则重新送入 hard。因此在183张原始验证图上，阈值修正后的 easy/hard 有效路由准确率达到100%。

这种规则的代价是少量不确定图片会使用计算量更高的 hard 检测器，但不会同时运行两个检测器。

## 6. easy 分支：六分类检测器

模型文件：

`models/02_easy_detector_6class_640.onnx`

触发条件：

- 场景分类为 `air` 或 `sea`；
- 场景置信度不低于 `0.60`。

输入输出：

| 项目 | 数值 |
|---|---|
| 输入 | `[1,3,640,640]` |
| 输出 | `[1,300,6]` |
| 检测头 | YOLOv10 端到端检测头 |

六分类顺序：

| 最终类别编号 | 类别名称 |
|---:|---|
| 0 | soldier |
| 1 | small_aircraft |
| 2 | warship |
| 3 | tank |
| 4 | patrol_boat |
| 5 | armored_vehicle |

每个输出目标通常按以下格式解析：

```text
[x1, y1, x2, y2, confidence, class_id]
```

该模型是端到端 YOLOv10 输出，不需要再次执行传统 NMS，只需要按照目标检测置信度过滤结果。easy 分支已经直接输出最终六分类编号，不需要类别映射。

## 7. hard 分支：三分类检测器

模型文件：

`models/03_hard_detector_3class_960.onnx`

触发条件：

- 场景分类为 `forest`；
- 场景分类为 `urban`；
- 或者场景最高置信度低于 `0.60`。

输入输出：

| 项目 | 数值 |
|---|---|
| 输入 | `[1,3,960,960]` |
| 输出 | `[1,7,18900]` |
| 检测头 | YOLOv8 原始检测头 |

hard 模型只检测陆地困难场景中需要重点加强的三个类别：

| hard 局部编号 | 类别名称 |
|---:|---|
| 0 | soldier |
| 1 | tank |
| 2 | armored_vehicle |

该输出需要完成以下后处理：

1. 将检测框从模型输出布局中解析出来；
2. 为每个候选框选择最高类别分数；
3. 按目标检测置信度过滤；
4. 执行 NMS；
5. 将 hard 局部类别编号映射为最终六分类编号。

## 8. hard 类别编号映射

hard 模型的类别编号不能直接作为最终输出，必须按下表映射：

| hard 局部编号 | 类别 | 最终六分类编号 |
|---:|---|---:|
| 0 | soldier | 0 |
| 1 | tank | 3 |
| 2 | armored_vehicle | 5 |

实现示例：

```python
hard_to_global = {
    0: 0,  # soldier
    1: 3,  # tank
    2: 5,  # armored_vehicle
}

global_class_id = hard_to_global[hard_class_id]
```

## 9. 图像预处理与坐标还原

检测图像应保持原始宽高比例，通过 letterbox 填充到对应固定输入尺寸：

- easy：`640×640`；
- hard：`960×960`。

检测完成后，需要依据 letterbox 缩放比例和两侧填充值，将检测框坐标还原到原始图像坐标系，并裁剪到原图边界。

建议场景分类预处理保持与 Ultralytics YOLOv8 分类导出流程一致，避免自定义裁剪方式造成场景准确率变化。

## 10. 最终统一输出

无论图片经过 easy 还是 hard 分支，最终均整理为统一六分类输出：

```python
{
    "box": [x1, y1, x2, y2],
    "score": confidence,
    "class_id": global_class_id
}
```

因为每张图片只运行一个检测器，所以不需要：

- 跨模型 NMS；
- 加权框融合；
- 两个模型之间的重复框去除；
- 对两个检测模型的置信度进行二次融合。

## 11. 推荐配置形式

```yaml
route_confidence: 0.60
uncertain_route: hard

scene_router:
  model: models/01_scene_router_224.onnx
  input_size: [224, 224]
  classes: [air, forest, sea, urban]

easy_branch:
  scenes: [air, sea]
  model: models/02_easy_detector_6class_640.onnx
  input_size: [640, 640]

hard_branch:
  scenes: [forest, urban]
  model: models/03_hard_detector_3class_960.onnx
  input_size: [960, 960]
  class_id_remap:
    0: 0
    1: 3
    2: 5
```

目标检测置信度和 NMS IoU 应设置为单独参数，并根据部署环境和比赛验证数据调整，不应与 `route_confidence` 混用。

## 12. 当前验证结果

验证集使用当前项目固定的183张原始验证图，不包含训练增广图。

| 指标 | 当前结果 |
|---|---:|
| 场景四分类原始 Top-1 | 约95.08% |
| 阈值修正后的 easy/hard 有效路由准确率 | 100% |
| soldier AP50 | 71.40% |
| 六分类 mAP50 | 92.14% |
| 六分类 mAP50-95 | 46.23% |
| 平均计算量 | 约14.24 GFLOPs/图 |

四分类 Top-1 低于有效路由准确率，是因为 `air` 与 `sea` 之间的误判仍然进入同一个 easy 检测器，`forest` 与 `urban` 之间的误判仍然进入同一个 hard 检测器，对最终检测路线没有影响。

## 13. 计算量说明

当前估算：

- 场景分类器：约 `0.412 GFLOPs/图`；
- easy 六分类检测器 640：约 `8.4 GFLOPs/图`；
- hard 三分类检测器 960：约 `18.44 GFLOPs/图`。

183张验证图中约84张走 easy、99张走 hard，因此包含场景分类器后的平均计算量约为：

```text
0.412 + (84 × 8.4 + 99 × 18.44) / 183 ≈ 14.24 GFLOPs/图
```

与每张图同时运行两个检测器相比，互斥路由明显降低了平均计算量。

## 14. 部署注意事项

1. 场景阈值固定使用 `0.60`，低于阈值必须进入 hard。
2. hard 输出必须进行局部类别到六分类类别的映射。
3. YOLOv10 easy 输出与 YOLOv8 hard 输出格式不同，不能共用完全相同的后处理。
4. 场景分类阈值与目标检测阈值是两个不同参数。
5. 每张图只能调用 easy 或 hard 其中之一，不要将当前方案改为双检测器叠加。
6. 修改图像预处理方式后，应重新验证183张原图上的路由准确率和最终 AP50。
