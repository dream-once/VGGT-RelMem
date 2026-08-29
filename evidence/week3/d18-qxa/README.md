# D18 frozen Q×A evidence

本目录只保留 JSON/Markdown 轻量证据，不包含图像、点云、mask、embedding、
权重或数据集。

- `office-loop/` 使用 D11 的同一历史 partial candidate cache。Q0、Q1 可重放；
  Q2 在第 4 个选择 `frame_0061` 的 outcome 为 `unmaterialized` 时立即返回
  `BLOCKED_MISSING_OUTCOME`，没有跳过该候选，也没有生成性能数字。
- `synthetic/` 使用 D14 的 complete cache，六个冻结组合全部通过。标签只由独立
  evaluator 读取，结果仅证明接口、标签隔离和确定性重放，不代表真实性能。
- Clio `cubicle` 只记录 `CLIO_HELD_OUT_PENDING` readiness，不伪造 held-out
  样本、指标或置信区间。

复验：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m scripts.validate_d18
```
