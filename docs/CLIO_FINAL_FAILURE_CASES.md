# Clio 最终失败案例

这些案例来自 Apartment/Cubicle 最终本地报告，不是人工挑出的成功展示。公开仓库不
分发 Clio 图像、task YAML、mask 或点云，因此这里只保留输入标识、预期、实际、根因
和本地报告路径。指标以 2026-09-02 的严格重算为准。

## 1. Apartment `bring me a pillow`：A1 链式误合并

- 输入：A1 对 `d6_rgb_426_001` 与 `d6_rgb_438_000`，以及
  `d6_rgb_438_000` 与 `d6_rgb_438_002` 的观测图执行生产关联。
- 预期：两对都属于 evaluator 标注的不同／背景实例，不应进入同一对象。
- 实际：直接 gate 对这两对均为 false，但其他正 gate 形成桥接路径，连通分量闭包后
  两对都被预测为 same-cluster，成为 false positive。
- 根因：A1 是 single-link 连通分量；局部可连接不保证整个簇两两一致。旧 evaluator
  只统计原始 gate，曾漏掉这两项错误。
- 证据：`runs/clio-apartment-dev-v2-lc/association_benchmark.json`。

## 2. Cubicle `get textbooks`：闭包放大误合并

- 输入：包含 `d6_rgb_192_000`、`d6_rgb_192_001`、`d6_rgb_200_007`、
  `d6_rgb_208_004` 和 `d6_rgb_216_000` 的 A1 观测图。
- 预期：不同 GT／背景观测不应因第三个 observation 间接合并。
- 实际：多对 direct gate=false 的负 pair 在闭包后变成 same-cluster；Cubicle A1 总计
  由原 gate 口径的 80 个 FP/43 个 FN 变为最终聚类口径的 137 个 FP/0 个 FN。
- 根因：single-link 提升 recall 的同时放大桥接误合并。修正后的 A1 F1 是 93.47%，
  不是旧报告的 93.85%。
- 证据：`runs/clio-cubicle-heldout-v1/association_benchmark.json`。

## 3. Cubicle `mudstone rock` reference：目标对、参考物错

- 输入：关系查询 `positive--get-condiment-packets--get-mudstone-rock`。
- 预期：target 使用 `get-condiment-packets` 的 GT 对象，reference 必须使用
  `get-mudstone-rock` 的 GT 对象，再判断方向。
- 实际：target 为 `get-condiment-packets__obj_0001`，但 reference 选成
  `get-quartz-rock__obj_0001`。旧 target-only evaluator 会把 target 命中视为正确；
  严格 v2 判为错误。
- 根因：开放词汇 reference 检索对共享词 `rock` 发生语义碰撞；预测协议保存了
  `reference_id`，但旧 evaluator 没有核对它。
- 证据：`runs/clio-cubicle-heldout-v1/relation-benchmark-v2/evaluation.json`。

## 4. Apartment `find deck of cards` reference：错误 task 对象参与方向计算

- 输入：`positive--clean-toaster--find-deck-of-cards`。
- 预期：方向应由 toaster target 与 deck-of-cards reference 的 GT 对象中心计算。
- 实际：target 为 `clean-toaster__obj_0001`，reference 却是
  `find-pile-of-hats__obj_0001`，答案被严格 evaluator 拒绝计正确。
- 根因：正确 reference 缺失或语义候选质量不足时，RelationGrounder 会在可匹配对象中
  选分数最高者；只检查 target 会掩盖 reference 侧感知失败。
- 证据：`runs/clio-apartment-dev-v2-lc/relation-benchmark-v2/evaluation.json`。

## 5. Apartment `get bottle of tide`：对象中心远离 GT

- 输入：冻结 Q1F，Top-5+A2 返回最高置信永久对象。
- 预期：预测中心落在 Tide bottle 的 oriented GT OBB 内。
- 实际：严格和 RMSE-padded 两个指标都失败，预测中心距最近 GT 中心约 `3.128 m`。
- 根因：这是上游检索／分割／lifting 后形成的错误永久对象；fallback 只在没有 A2 对象
  时触发，因此不会用 Q0 覆盖一个“存在但错误”的 A2 对象。
- 证据：`runs/clio-apartment-dev-v2-lc/grounding_benchmark.json`。

## 6. Cubicle `get quartz rock`：近中心但仍在 oriented OBB 外

- 输入：冻结 Q1F 返回 A2 永久对象。
- 预期：中心进入 quartz rock 的 oriented GT OBB。
- 实际：中心距 GT 中心约 `0.094 m`，但严格和 RMSE-padded containment 都失败。
- 根因：中心距离小不等于位于旋转 OBB 内；物体尺度、方向和残余定位误差共同造成
  边界失败。该例说明不能用最近中心距离替代 containment 指标。
- 证据：`runs/clio-cubicle-heldout-v1/grounding_benchmark.json`。

## 7. Q2 枕头诊断：停止信号不等于实例覆盖

- 输入：Apartment 24 帧开发诊断，最大预算 5，gain/patience 顺序搜索。
- 预期：若称 coverage-aware，应以新增真实实例或独立对象证据作为 gain。
- 实际：Q2 在 4 帧后因连续低 gain 停止，只产生 1 条 observation；其 gain 定义仍是
  frame-scoped observation 计数，`coverage_aware=false`。
- 根因：当前 trace 没有实例级覆盖状态，也没有带标签查询策略消融。因此 Q2 只保留为
  诊断，不进入最终 Clio 主表或简历贡献。
- 证据：完整 trace 保存在 Git tag `research-final-v1-2026-09-02`；求职版主树
  仅保留该失败边界，不保留 Q2 阶段产物。

## 汇总边界

- 关系旧 target-only 数字已经作废；v2 同时验证 target/reference。
- Q1F 的 `+11.11pp` 是 18-task fixed-confirmatory 系统差值，不是 A2 单项或严格
  held-out 涨点。
- 0.60 拒答阈值未做真实 calibration；高拒答率不能替代双端定位后的关系正确性。
