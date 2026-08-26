# D10：D9 无标签预测与独立评测证据

这个目录是 JSON/Markdown-only 的自包含 CPU 验收包。它把冻结的 A1 空间关联拆成两个互不混淆的边界：

- `prediction/` 只读取冻结 D8 `ObjectMemory`，不读取人工标签，也不输出 F1、`expected_same` 或失败类型；
- `evaluation/` 才读取人工 instance labels，并把指标与逐对误差写入独立结果；
- `validation/` 保存两条路径各自独立重算后的报告。

## 当前结果

冻结的 office-loop 开发样例包含 10 条 observation。Prediction 穷举 45 个 pair，得到 17 条匹配边和 3 个连通分量；其中 1 个跨 3 帧分量成为永久对象，另外 4 条 observation 保持 pending。Evaluation 标签包含 17 个正 pair 和 28 个负 pair，本次 precision/recall/F1 均为 `1.0`。

这个数字只用于小型开发样例的工程回归，**不是 held-out、跨场景或优于 FOUND-IT 的结论**。A1 仍有已测试的单链接桥接风险，后续 A2 单独处理。

## 文件

```text
prediction/
  source_memory.json
  d9_result.json
  object_memory.json
  run_manifest.json
evaluation/
  d9_evaluation.json
  pair_labels.json
  run_manifest.json
validation/
  prediction.json
  evaluation.json
```

## Clean-clone 验收

只需 Python、NumPy 和仓库源码，不需要 VGGT/PE/SAM 权重或 GPU：

```bash
python -m scripts.validate_d9_association \
  evidence/week2/d9-office-loop-trash-can/prediction

python -m scripts.validate_d9_evaluation \
  evidence/week2/d9-office-loop-trash-can/evaluation
```

## 从冻结输入重算

选择一个尚不存在的临时输出目录；下面示例使用 `/tmp/vggt-relmem-d9-repro-new`：

```bash
python -m scripts.run_d9_association \
  --memory evidence/week2/d9-office-loop-trash-can/prediction/source_memory.json \
  --output-dir /tmp/vggt-relmem-d9-repro-new/prediction

python -m scripts.evaluate_d9_association \
  --prediction-dir /tmp/vggt-relmem-d9-repro-new/prediction \
  --labels evidence/week2/d9-office-loop-trash-can/evaluation/pair_labels.json \
  --output-dir /tmp/vggt-relmem-d9-repro-new/evaluation

python -m scripts.validate_d9_association \
  /tmp/vggt-relmem-d9-repro-new/prediction
python -m scripts.validate_d9_evaluation \
  /tmp/vggt-relmem-d9-repro-new/evaluation
```
