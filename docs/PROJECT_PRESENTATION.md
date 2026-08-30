# 项目对外表述与 3 分钟演示讲稿

## 一句话定位

VGGT-RelMem 是 **VGGT-SLAM 几何之上的可审计语义定位可靠性层**：它隔离查询
策略与关联策略，保存候选、观测、拒答原因和成本证据；它不是完整闭环导航系统，
也不是 FOUND-IT 官方复现。

## 当前可写结果

- office-loop 的公开 Candidate Outcome Cache 已完整物化 8/8 帧；Q0/Q1/Q2 ×
  A1/A2 六个 development replay 均完成。该 replay 无真实标签，不声称性能提升。
- Clio `apartment` 的官方公开 RGB 开发子集已完成真实 RTX 4090 重放：24 帧
  geometry、24/24 candidate outcome、3 条 pillow 3D observation。Q0/Q1/Q2 分别
  得到 0/1/1 条 observation，A1/A2 均未形成永久对象；这是保留的真实失败，
  不声称性能提升。
- Q2 在最大预算 5 下选择 5 帧并产生 14 条新增 frame-scoped observation；这不是
  对象覆盖收益，同一物体的重复观测仍可能被重复计数。
- D15.5 使用 95 帧几何、20 个查询视角、40 条有效 3D observation 和 8 个预测
  对象；其中 3/8 满足严格对象中心多视角门槛。它是本项目的对象证据可视化，不是
  VGGT-SLAM 原生几何视频的替代或优于结论。
- D17 的 synthetic 协议有 2 个正查询和 3 个结构负查询；正确拒答计入任务正确率，
  但不进入 selective answer coverage，最大回答覆盖率为 2/5=0.4。真实 calibration
  split 尚未完成。
- Clio `apartment` 固定为 development；当前只物化 RGB＋任务元数据子集，
  `cubicle` 保持未接触 held-out。没有 cubicle 结果前不写跨场景指标、SOTA 或
  优于 FOUND-IT。

## 简历定稿（当前证据边界）

> 在 VGGT-SLAM 2.0 几何前端之上实现可审计的开放词汇 3D 目标记忆与预算查询框架，通过 Candidate Outcome Cache 隔离查询与关联策略变量，并建立标签隔离、可靠拒答、确定性重放和失败审计链路；完成 office-loop 8/8 候选重放、95 帧 3D 证据可视化及 Clio apartment 24/24 GPU 开发重放，公开保留 Top-1 空结果、顺序停止过早和跨场景关联失败，不宣称 held-out 性能提升。

## 3 分钟演示讲稿

### 0:00–0:30：问题和边界

“VGGT-SLAM 能重建场景几何，但开放词汇目标查询、跨帧对象记忆和可靠拒答不是同一
个可审计实验问题。我在固定几何之上增加了可靠性层；今天展示的是感知与定位前端，
不展示路径规划和控制。”

### 0:30–1:10：系统结构

展示 README 的 Q×A 分离：Q0 Top-1、Q1 Fixed Top-K、Q2 retrieval＋pose novelty
只决定看哪些帧；A1/A2 只决定如何关联 observation。强调所有组合读取同一个
Candidate Outcome Cache，prediction 中没有 GT 或人工答案。

### 1:10–1:45：可复现重放

执行：

```bash
python -m scripts.verify_public_clone
python -m scripts.validate_d18
python -m scripts.validate_d20
```

解释公共 clone 验收会明确排除依赖 Git-ignored 上游源码的 Q0 静态源码测试；克隆
固定上游后再跑全量测试。打开 D18 表，同时展示 office-loop 8/8 与 Clio apartment
24/24 complete cache。指出 Clio 的 Q0/Q1/Q2 只有 0/1/1 条 observation，表中是
成本、停止原因和失败结构，不是性能数字。

### 1:45–2:25：直观 3D 证据

先展示公开 `overview.png`，再播放本地约 10 秒的 D15.5 环绕视频。指出背景点云和
轨迹来自 VGGT-SLAM 几何，而彩色对象点、Top-K 相机、跨帧 observation、Object
Memory OBB 和对象中心视差审计是本项目新增工作。明确 3/8 对象通过严格多视角门槛。

### 2:25–2:50：可靠拒答和失败

展示 D17：缺目标、缺 reference、缺 anchor 都有不同拒答原因；正确拒答不被伪装成
回答覆盖。展示 Q2 的已知限制：`new_observation_count` 不是对象覆盖，因此不声称
“覆盖感知降低重复”。

### 2:50–3:00：结论

“当前交付是可复现、标签隔离、成本可审计的工程基准。apartment 已暴露 Top-1
空结果、Q2 停止过早和关联失败；下一步只在 apartment 冻结校准与查询集，再一次性在
cubicle held-out 验证，结果无提升也保留为负结果。”

## 录制状态

讲稿和命令入口已完成；公开仓库目前有缩略图，本地有约 10 秒环绕视频。真正约 3
分钟的配音屏幕录制仍为 `DEMO_RECORDING_PENDING`，不能把短视频写成完整演示。
