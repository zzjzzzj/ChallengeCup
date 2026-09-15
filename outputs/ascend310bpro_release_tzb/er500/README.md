# Ascend 310B TZB Board Test Outputs

This folder contains blind-test prediction and FPS evidence from Ascend 310B.
The formatted attachment-1 result folder is:

/home/HwHiAiUser/Desktop/workspace/ChallengeCup/outputs/ascend310bpro_release_tzb/er500/615738_FPS指标2

Raw prediction folders:

- base_before: before model on base_test_r1
- base_after: after model on base_test_r1
- increment_after: after model on inc_test_r2
- submission_package: FPS/Agent evidence package generated with --skip-map

Key files:

- base_before/predictions.jsonl
- base_after/predictions.jsonl
- increment_after/predictions.jsonl
- base_before/summary.json
- base_after/summary.json
- increment_after/summary.json
- submission_package/tzb_metrics.json
