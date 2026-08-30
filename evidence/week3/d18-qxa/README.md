# D18 frozen Q×A evidence

本目录只保留 JSON/Markdown 轻量证据，不包含图像、点云、mask、embedding、
权重或数据集。

- `office-loop/` 使用 Week 3 公开的 8/8 complete candidate outcome cache。
  Q0、Q1、Q2 的六个冻结组合都完成重放；Q2 跑满 5 帧并以
  `max_budget` 停止。该结果无人工标签，只是 development engineering replay，
  不产生 recall/F1 或性能提升结论。
- `synthetic/` 使用 D14 的 complete fixture，六个冻结组合全部通过。标签只由
  独立 evaluator 读取，结果仅证明接口、标签隔离和确定性重放，不代表真实性能；
  本 fixture 的关联 F1 为 0，也不构成涨点证据。
- 旧 Week 2 partial cache 仍由回归测试覆盖：Q2 遇到 `frame_0061` 的
  `unmaterialized` outcome 时必须返回 `BLOCKED_MISSING_OUTCOME`，不得跳过。
- Clio `cubicle` 只记录 `CLIO_HELD_OUT_PENDING` readiness，不伪造 held-out
  样本、指标或置信区间。

复验：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m scripts.validate_d18
```
