# VGGT-RelMem

**项目定位：VGGT-SLAM 几何之上的可审计语义定位可靠性层。**

VGGT-RelMem 在固定的 VGGT-SLAM 2.0 几何结果之上，完成开放词汇候选检索、SAM 分割、3D lifting、跨视角对象记忆、关系定位和显式拒答。它是目标定位感知前端，不包含路径规划、控制或闭环导航，因此不称“完整导航系统”。

## 一眼看懂

```text
RGB + camera geometry
        ↓
PE Top-K retrieval
        ↓
SAM mask + robust 3D lifting
        ↓
ObjectObservation
        ↓
A2 complete-link ObjectMemory
        ↓
object grounding / spatial relation / abstention
```

预测阶段不读取 task GT、世界配准或 evaluator 标签。Clio 的 task OBB 与 VGGT→Clio Sim(3) 只在独立 evaluator 中使用。

项目主线解决四个问题：

- 固定 Top-5 预算比单帧 Top-1 提供更多目标观测。
- 多帧观测通过可解释的几何和质量证据组成持久对象。
- 关系查询同时验证 target 与 reference，而不是只验证目标物体。
- 证据不足时明确拒答，并保留原因与中间产物供审计。

## 最终结果

当前结论为：**完成可复现基准、查询策略与关联策略隔离、显式拒答协议、跨场景固定确认和失败分析**。

全部任务分母来自 Clio task OBB。对象定位对同一 task 的所有 GT OBB 执行 `any(containment)`；最近中心 OBB 仅用于诊断。这不是 Clio 官方 IoU matching 指标。

| 指标 | Apartment development | Cubicle fixed-confirmatory |
|---|---:|---:|
| Q0 Top-1 严格 Acc@1 | 11.11% | 27.78% |
| Q1F Top-5+A2+Q0 fallback 严格 Acc@1 | 11.11% | 38.89% |
| Q1F−Q0 严格差值 | 0.00pp | **+11.11pp** |
| Q1F−Q0 `±RMSE` 差值 | +5.56pp | **+16.67pp** |
| A1 最终连通分量 pair F1 | 85.29% | **93.47%** |
| A2 几何+质量 complete-link pair F1 | **88.14%** | 91.56% |
| 关系 target+reference 严格 / `±RMSE` Acc@1 | 0.00% / 11.76% | 11.41% / 48.32% |
| 负例拒答 / 原因命中 / 双端定位后关系拒答 | 100.00% / 31.62% / 10.29% | 98.66% / 67.79% / 44.97% |

Cubicle 的 `+11.11pp` 是限定于冻结 Q1F 协议的系统差值，不归因于某一个组件，也不包装成完全未接触的 held-out 涨点。最终 Clio run 没有保存 `semantic_embedding`，所以 A2 只能称为“任务内几何+质量 complete-link 关联”，不能称为多模态语义关联。

轻量结果、逐项边界和哈希见 [最终 Clio 证据](evidence/final-clio/README.md)。

### Post-D21 PE 代表中心扩展

这个独立扩展保持检索、SAM、lifting、A2 聚类、物体选择和 Q0 fallback 不变，只用 PE mask-crop 相似度在已选对象内部挑选一次代表观测的 3D 中心。

| 严格 Acc@1 | Apartment development | Cubicle fixed-confirmatory |
|---|---:|---:|
| 当前 Q1F | 2/18 | 7/18 |
| 最高质量观测 | 3/18 | 7/18 |
| PE 语义代表中心 | **3/18** | **8/18** |
| 当前观测 oracle | 3/18 | 8/18 |

PE 在两个场景各带来 1 个严格正确任务、0 个严格回退，但 Apartment 与无模型质量基线打平；Cubicle 也已有历史暴露。这里只报告 fixed-confirmatory 系统观察，不声称统计显著、严格 held-out、已训练 ranker 或 SigLIP2 结果。详见 [PE 扩展证据](evidence/post-d21-pe-fusion/README.md)。

## 快速开始

公开 clone 的 CPU 验证不需要 Clio 数据、模型权重或 GPU：

```bash
git clone https://github.com/dream-once/VGGT-RelMem.git
cd VGGT-RelMem
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m scripts.verify_public_clone
```

运行最小合成示例：

```bash
python -m scripts.demo --save-memory runs/demo/object_memory.json
```

单独校验最终轻量结果：

```bash
python -m scripts.validate_clio_final_summary
python -m scripts.validate_clio_pe_semantic_fusion_summary
```

当前公开回归包含 80 项 CPU 测试。完整真实 Clio 的复现命令需要合法取得数据、两个 Python 环境、模型权重、GPU 几何和本地 run 产物，参见 [验证与复现手册](docs/REPRODUCTION.md)。

## 仓库地图

| 路径 | 用途 |
|---|---|
| `relground/` | 检索、lifting、对象记忆、关系、拒答与 Clio evaluator |
| `adapters/` | VGGT-SLAM、PE/SAM 和几何文件契约 |
| `evaluation/` | 指标、基线和失败类型 |
| `configs/` | 冻结协议、查询清单和实验 manifest |
| `scripts/` | CLI；先看 [脚本索引](scripts/README.md) |
| `tests/` | 无数据 CPU 核心回归 |
| `evidence/` | 仅保留最终轻量结果；先看 [证据索引](evidence/README.md) |
| `docs/` | 复习、复现、演示和真实失败案例 |
| `runs/` | 本地运行产物，默认不进入 Git |
| `third_party/` | 本地上游源码，默认不进入 Git |

## 推荐复习顺序

1. 阅读 [复习指南](docs/REVIEW_GUIDE.md)，先建立数据流和模块边界。
2. 依次阅读 `retrieval.py`、`observations.py`、`association.py`、`relations.py`。
3. 阅读三个 Clio evaluator，理解预测与标签为什么必须隔离。
4. 运行 `verify_public_clone` 和合成 demo。
5. 最后查看 [对外表述与讲稿](docs/PROJECT_PRESENTATION.md) 和 [真实失败案例](docs/CLIO_FINAL_FAILURE_CASES.md)。

D1–D21 的逐日命令、阶段 evidence 和诊断脚本保存在 Git tag `research-final-v1-2026-09-02`，不进入求职版主树。

## 可复现与验证入口

- CPU / clean clone 验证：[完整命令](docs/REPRODUCTION.md#cpu-clean-clone)
- 36-task GPU 批处理命令：[完整命令](docs/REPRODUCTION.md#真实-clio-36-task-重放)
- 最终 evaluator：[完整命令](docs/REPRODUCTION.md#重建最终评测)
- PE mask-crop 扩展：[PE 扩展证据](evidence/post-d21-pe-fusion/README.md)
- 全部 CLI 分类：[脚本索引](scripts/README.md)

原始 Clio 数据、YAML、mask、点云、视频、checkpoint 和大型 run 报告不随 Git 分发。tracked 摘要保存分母、聚合值、协议边界和本地源文件 SHA-256。

## 当前边界

- Q1F 每个 task 最多仍运行 5 次 SAM；尚未完成整条 36-task 延迟和峰值显存统计。
- Q2 的 `coverage_aware=false` 诊断仅保存在 research tag，不进入求职版代码或主贡献。
- 关系置信度使用未校准的 0.60 工程阈值；真实独立 calibration 仍待完成。
- 尚未提供 Instance Recall、Duplicate Rate、Count Error、带标签查询策略消融和统计置信区间。
- 正式约 3 分钟录屏和 release tag 尚未完成。
- FOUND-IT 不属于本项目范围；本项目不接入、不复现，也不做同条件优劣宣称。
- 数据许可仍未核实，禁止重新分发本地 Clio 原始数据。

这些 pending 不影响公开 CPU 回归，但不能被写成已完成贡献。
