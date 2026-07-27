"""覆盖 train_detector.resolve_model_spec 的从零训练分支。

实测依据（ultralytics 8.4.100 + torch 2.9.1，本机 CPU 单次运行）：

  构造期第一层卷积 model.0.conv.weight (16,3,3,3)
    YOLO('yolov8n.pt')   mean=-0.002786  std=0.152257  absmax=0.510742  sum=-1.203588
    YOLO('yolov8n.yaml') mean= 0.001384  std=0.113982  absmax=0.192243  sum= 0.597958
    再构造一次 yaml       mean= 0.004472  std=0.107497  absmax=0.191849  sum= 1.931945
  yaml 分支的 absmax 精确等于 PyTorch Conv2d 默认 kaiming_uniform 上界
  sqrt(1/fan_in)=sqrt(1/27)=0.192450，且两次构造互不相同 —— 确系随机初始化；
  .pt 分支的 absmax 0.510742 远超该上界 —— 确系已加载 COCO 权重。

  进入训练循环前（on_pretrain_routine_end 回调）截取的 model.0.conv.weight.sum()
    .pt   + pretrained=True   -1.203588  （保留 COCO 权重）
    .pt   + pretrained=False  -5.031788  （本版本确实被重建为随机权重）
    .yaml + pretrained=False  -5.031788
    .yaml + pretrained=True   -5.031788  （.yaml 分支忽略 pretrained）

  结论：8.4.100 的 Model.train() 里有 `weights = None if pretrained is False else self.model`，
  所以 .pt + pretrained=False 在**本版本**是真从零。但该分支被
  `if not args.get("resume") and self.ckpt` 包着，--resume 时整条跳过；
  且这是版本相关实现细节。故 resolve_model_spec 对显式 .pt + --no-pretrained 直接报错。
"""

from __future__ import annotations

import unittest

from scene_recognition.detector_module.train_detector import resolve_model_spec


class ResolveModelSpecTest(unittest.TestCase):
    def test_default_behaviour_unchanged(self) -> None:
        self.assertEqual(resolve_model_spec(None, False), ("yolov8n.pt", True))

    def test_explicit_weights_still_pretrained(self) -> None:
        self.assertEqual(
            resolve_model_spec("runs/foo/weights/best.pt", False),
            ("runs/foo/weights/best.pt", True),
        )

    def test_scratch_switches_default_to_yaml(self) -> None:
        self.assertEqual(resolve_model_spec(None, True), ("yolov8n.yaml", False))

    def test_scratch_rejects_explicit_checkpoint(self) -> None:
        for spec in ("yolov8n.pt", "yolov8s.PT", "a/b/best.pth", "x.ckpt"):
            with self.subTest(spec=spec):
                with self.assertRaises(ValueError) as caught:
                    resolve_model_spec(spec, True)
                self.assertIn("--no-pretrained", str(caught.exception))

    def test_scratch_accepts_explicit_architecture(self) -> None:
        self.assertEqual(resolve_model_spec("yolov8s.yaml", True), ("yolov8s.yaml", False))


if __name__ == "__main__":
    unittest.main()
