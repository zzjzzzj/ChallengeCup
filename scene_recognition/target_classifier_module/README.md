# ResNet18 目标识别基线

## 主线：整图多标签识别

当前任务要求直接用ResNet18从完整遥感图识别最终目标。由于一张图可以同时包含多个类别，输出不是四选一，而是四个独立概率：

注意：数据清单由本地数据准备命令生成，不提交仓库。换机器后必须重新生成清单，不能复用其他机器的绝对路径。

```text
完整图片 → ResNet18 → [soldier, small_aircraft, warship, tank]是否出现
```

例如同一张森林图同时有士兵和坦克，标签为`[1, 0, 0, 1]`。主线不需要先做场景分类，也不需要先裁剪目标。

运行主线：

```powershell
python -m scene_recognition.target_classifier_module.train_whole_image `
  --manifest-dir scene_recognition/detector_module/artifacts/detection_dataset/manifests `
  --epochs 12 `
  --batch-size 32 `
  --augmentation none `
  --output scene_recognition/target_classifier_module/runs/resnet18_whole_image_baseline_none
```

输出 `metrics.json` 中的 `Exact Match` 表示四个类别的存在判断是否整组全对，`Macro-F1` 表示四个类别的平均识别质量。

整图ResNet18不能输出目标位置和数量，因此它不是完整目标检测；要回答“目标在哪里”，仍需使用YOLO检测。

## 辅助实验：真实框裁剪分类

这个实验读取完整遥感图及其YOLO真实标注框，裁剪每个目标，再用ResNet18进行四分类。

```text
完整图片 + 真实YOLO框
          ↓
      目标裁剪图
          ↓
       ResNet18
          ↓
soldier / small_aircraft / warship / tank
```

它只验证“已知正确位置以后能否分对类别”，是分类上限和错误分析工具，不负责寻找目标位置，不能替代YOLO检测mAP。

## 1. 准备裁剪数据

```powershell
python -m scene_recognition.target_classifier_module.prepare_crops `
  --index image_processing/artifacts/scene_index.csv `
  --output scene_recognition/target_classifier_module/artifacts/target_crops `
  --padding-ratio 0.10
```

程序先继承原图的train/val/test划分，再裁剪目标，防止同一原图的目标泄漏到不同集合。当前实测生成2957个目标：train 2025、val 459、test 473。

## 2. 训练无增广基线

```powershell
python -m scene_recognition.target_classifier_module.train_classifier `
  --epochs 12 `
  --batch-size 32 `
  --image-size 224 `
  --augmentation none `
  --output scene_recognition/target_classifier_module/runs/resnet18_target_baseline_none
```

默认使用ImageNet预训练权重。若本机无法下载或缓存权重，必须显式添加`--no-pretrained`，程序不会静默改成随机初始化。

输出包括：

- `best.pt`：按验证集Macro-F1选择的最佳权重。
- `metrics.json`：总体、IR/SAR和场景切片指标。
- `history.csv`：逐轮Loss、Accuracy与Macro-F1。
- `confusion_matrix.csv`：测试集混淆矩阵。
- `test_predictions.csv`：每个裁剪目标的来源、预测和置信度。

## 3. 增广消融

`--augmentation`每次只能选择一种，便于独立验证：

- `none`：无增广对照组。
- `flip`：水平/垂直翻转。
- `rotate90`：0/90/180/270度旋转。
- `invert`：像素取反。
- `open`：形态学开运算。
- `close`：形态学闭运算。

每种方式必须使用不同运行目录。开闭运算可能抹掉小目标，取反可能不符合真实成像机理，不能只看视觉效果，必须看独立验证集和测试集结果。

## 4. 单图推理

```powershell
python -m scene_recognition.target_classifier_module.infer `
  --image "已裁剪目标.png" `
  --checkpoint scene_recognition/target_classifier_module/runs/resnet18_target_baseline_none/best.pt
```

## 5. ONNX与310B前置准备

```powershell
python -m scene_recognition.target_classifier_module.export_classifier `
  --checkpoint scene_recognition/target_classifier_module/runs/resnet18_target_baseline_none/best.pt `
  --output scene_recognition/target_classifier_module/runs/resnet18_target_baseline_none/exports
```

这一步只生成ONNX与ATC命令模板。只有在真实310B板卡上成功转换OM、完成精度核对并实测时延后，才能声称完成端侧部署。

ONNX不包含图片预处理。板卡侧必须复现：黑色补边成正方形、缩放到224×224、像素缩放到0～1，以及ImageNet mean/std归一化；否则同一模型也会得到不同结果。

## 当前结果与限制

- 测试Accuracy：99.79%。
- 测试Macro-F1：99.78%。
- 473个目标中错1个：11×11像素的IR森林场景坦克被判为士兵。
- `small_aircraft`只出现在air，`warship`只出现在sea，存在利用场景背景捷径的风险。

因此该结果应表述为“真实框目标裁剪分类上限”，后续需要更紧裁剪、上下文消融和跨场景数据验证。
