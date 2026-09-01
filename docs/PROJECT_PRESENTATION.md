# 项目对外表述、5 分钟演示讲稿与面试问答

## 一句话定位

VGGT-RelMem 是 **VGGT-SLAM 几何之上的可审计语义定位可靠性层**：它隔离查询
策略与关联策略，保存候选、观测、拒答原因和成本证据；它不是完整闭环导航系统，
也不是 FOUND-IT 官方复现。

## 当前可写结果

- Clio `apartment` development 与 `cubicle` fixed-confirmatory 已分别完成 192/172
  帧几何和 18+18 task 的 GPU 物化。冻结 Q1F 在 Cubicle 的严格中心 Acc@1 为
  `27.78% → 38.89%`（`+11.11pp`）；这是系统差值，不归因于 A2 单项。
- A1/A2 统一按最终同簇关系计分。Cubicle F1 为 `93.47% / 91.56%`，A2 是负结果；
  Apartment 为 `85.29% / 88.14%`。
- 关系正例同时要求 target 与 reference 命中各自 GT。Apartment/Cubicle 的严格／
  RMSE-padded Acc@1 为 `0%/11.76%` 与 `11.41%/48.32%`；旧 target-only 数字作废。
- 负例端到端拒答为 `100% / 98.66%`，但双端定位后关系拒答只有
  `10.29% / 44.97%`。0.60 仍是未校准工程阈值。
- Q2 是 `coverage_aware=false` 的诊断实现，不进入主贡献。没有真实 calibration、
  带标签查询策略消融、统计置信区间或完整导航结论。

## 简历定稿（当前证据边界）

1. 在 VGGT-SLAM 2.0 几何之上实现开放词汇 Top-1／固定 Top-5 检索、SAM 3 分割、稳健 3D lifting、可追溯 ObjectMemory 与显式关系定位流水线，并以缓存和 JSON 契约隔离无标签预测、GT evaluator 与确定性重放。

2. 在 Clio Apartment/Cubicle 的 18+18 task 上完成 192/172 帧 GPU 物化与固定确认：冻结 Q1F 的 Cubicle 严格中心 Acc@1 为 `27.78%→38.89%`，同时公开 A2 关联回退和 target/reference 双端关系评测下降；拒答阈值明确未校准，不宣称 SOTA、严格 held-out 或完整导航。

## 5 分钟演示讲稿

### 0:00–0:40：问题与范围

“VGGT-SLAM 提供场景几何，但开放词汇目标查询、跨帧对象身份、关系定位和什么时候
拒答是另一层问题。我在固定几何之上实现了一个可审计的语义定位层。展示范围止于
感知和定位，不包含规划、控制，也不声称复现 FOUND-IT。”

画面：README 顶部定位与范围声明。

### 0:40–1:25：方法结构

展示 Q×A 分离：Q0 是 Top-1；正式主策略 Q1F 是固定 Top-5＋A2，没有永久对象时
回退 Q0；A1 是 gate 后连通分量，A2 是证据约束 complete-link。GT、OBB 和对齐只在
evaluator 打开。Q2 只作为停止策略诊断，不进入主贡献。

画面：模块目录、query manifest、frozen Q1F policy 和 ObjectMemory JSON。

### 1:25–2:05：复现与审计

执行：

```bash
python -m scripts.verify_public_clone
python -m scripts.validate_clio_final_summary \
  evidence/final-clio/benchmark_summary.json
```

说明 clean clone 测试不读取 Clio 数据；官方 task YAML 的覆盖检查是显式 integration
命令。每份本地报告记录输入路径和 SHA-256，公开摘要只保留聚合数字与边界。

### 2:05–3:05：对象定位与关联结果

展示最终表：“Apartment 是 development，Cubicle 是 fixed-confirmatory。冻结 Q1F 在
Cubicle 的严格中心 Acc@1 从 27.78% 到 38.89%，差 11.11pp。这个数字属于完整系统，
不能归因于 A2，因为 A2 的最终同簇 F1 是 91.56%，低于 A1 的 93.47%。Apartment 上
Q1F 严格差值为 0pp，也一起保留。”

补充：A1 必须按连通分量闭包计分；只统计原始 gate 会漏掉链式误合并。

### 3:05–4:00：关系定位与拒答

展示一条 `mudstone rock` reference 案例：“旧 evaluator 只验证 target，会把使用
`quartz rock` 参考物的答案算对。现在 target 和 reference 都必须命中各自 GT。严格
修正后，Cubicle 正例 strict/padded 为 11.41%/48.32%。”

再展示负例：“端到端拒答是 98.66%，但双端定位后、理由也正确的关系拒答只有
44.97%。这说明拒答率高不等于关系推理可靠。0.60 是工程默认值，未做真实校准。”

### 4:00–4:40：失败与取舍

打开 `docs/CLIO_FINAL_FAILURE_CASES.md`：展示 A1 链式误合并、reference 语义碰撞和
Q1F 严格 OBB 边界失败。明确 Q2 的 gain 不是 instance coverage，因此把 Q2 降级为
诊断；Instance Recall、Duplicate Rate、Count Error、置信区间和完整成本表仍未测。

### 4:40–5:00：结论

“项目交付的是标签隔离、确定性重放、失败可追踪的语义定位基准。最有价值的不只是
一个正差值，而是能指出这个差值来自哪条冻结策略、哪些指标回退，以及错误发生在
target、reference、关联还是拒答层。”

## 录制状态

5 分钟讲稿和命令入口已更新；公开仓库有静态材料，本地有约 10 秒环绕视频。完整
配音屏幕录制仍为 `DEMO_RECORDING_PENDING`，不能把短视频写成最终演示。

## 面试追问与答案

### 1. 为什么不是完整导航系统？

输出止于目标／关系的 3D 定位和拒答，没有路径规划、避障、控制与闭环成功率，所以
只称语义定位可靠性层。

### 2. `+11.11pp` 能说明什么？

它只说明冻结 Q1F 在 18 个 Cubicle task 的自定义中心入 OBB 指标上高于 Q0；样本小、
无置信区间，而且包含 Top-5、lifting、A2 与 fallback 的共同作用。

### 3. 为什么不能把增益归因于 A2？

A2 的 Cubicle 最终聚类 F1 为 91.56%，低于 A1 的 93.47%。对象定位 headline 是系统
级结果，不是 association 单因素消融。

### 4. A1 评测为什么需要重算？

生产 A1 对正 gate 做连通分量闭包；A–B、B–C 通过时，A–C 即使 gate 失败也会同簇。
原 evaluator 只计 gate pair，漏掉了这类链式误合并。

### 5. 关系 evaluator 修了什么？

预测中本来就保存 `reference_id`，但旧 evaluator 只核对 target。现在正例要求 target
和 reference 都命中各自 GT，严格关系负例也要求双端定位成立。

### 6. 为什么拒答率高仍不叫可靠？

缺目标或缺 reference 也会带来拒答。Cubicle 总拒答准确率 98.66%，但双端定位且原因
正确的关系拒答只有 44.97%，二者回答的是不同问题。

### 7. 置信度校准完成了吗？

没有。0.60 是未校准工程阈值，Brier/ECE 只能作诊断；没有独立真实 calibration
split，因此不用“已校准可靠拒答”表述。

### 8. 为什么不把 Q2 当核心贡献？

当前 gain 统计新增 frame-scoped observation，不等于新增实例；`coverage_aware=false`
且没有带标签消融，所以 Q2 只保留为停止过早的诊断案例。

### 9. 为什么不用 Clio 官方指标？

当前透明指标是预测中心是否位于官方 oriented OBB 内，便于定位对齐误差，但不是官方
IoU matching。README 和报告均显式设置 `official_clio_metric_claim=false`。

### 10. 如果继续做，优先补什么？

先做独立 calibration split 和带标签查询策略消融，再补 Instance Recall、Duplicate
Rate、Count Error、完整时延／显存与 bootstrap 区间；之后才讨论扩大场景或导航闭环。
