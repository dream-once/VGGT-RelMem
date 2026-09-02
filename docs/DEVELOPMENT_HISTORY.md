# VGGT-RelMem

**项目定位：VGGT-SLAM 几何之上的可审计语义定位可靠性层。** 本仓库不是完整闭环导航系统；它在固定 VGGT-SLAM 几何之上，提供候选检索、开放词汇分割、3D lifting、对象关联、关系定位、可靠拒答、缓存重放与可审计证据。

项目已完成 D1–D21 工程链路，并在 Clio `apartment` 开发场景与 `cubicle` 固定确认场景上完成端到端 GPU 推理、对象定位、关联、方向关系和拒答评测。两套几何分别使用 192/172 个采样帧。固定的上游源码位于本机忽略目录 `third_party/VGGT-SLAM`，个人代码通过 adapters 和可验证文件契约接入，不修改上游实现。几何推理使用 `vggt_geom`，PE/SAM 3 使用隔离的 `open_vocab` 环境；权重、原始数据和大型运行产物不进入 Git。

FOUND-IT 明确不属于本项目范围：本仓库不接入、不复现，也不把它列为后续任务。它仅作为 VGGT-SLAM 2.0 README 中提到的外部背景保留，不做同条件优劣宣称。项目验收只针对在固定 VGGT-SLAM 2.0 几何之上新增的查询、3D lifting、对象记忆、关联、关系定位、拒答和可审计评测模块。修订后的基线矩阵、验收门槛与 D11–D21 路线见 [D9 后修订计划](docs/POST_D9_REVISED_PLAN.md)。

## 最终结果边界

当前结论为：**完成可复现基准、查询策略与关联策略隔离、显式拒答协议、跨场景固定确认和失败分析**。Cubicle 的 18-task 对象定位 Q1F 策略在读取其数据内容前冻结，可报告限定到该协议的系统差值；关联与关系 evaluator 是后续固定确认，不包装成完全未接触的 held-out。结果不支持 SOTA、优于 FOUND-IT、已校准可靠性或完整闭环导航的结论。定稿简历表述、真实数字边界、约 3 分钟讲稿与追问答案见
[项目对外表述与演示讲稿](docs/PROJECT_PRESENTATION.md)；当前完整录屏状态仍为
`DEMO_RECORDING_PENDING`.

## Clio Apartment → Cubicle 最终实验（2026-09-02 重算）

所有任务分母均来自 Clio 的官方 task OBB；GT 与 VGGT→Clio Sim(3) 只进入 evaluator。对象定位对一个 task 的所有官方 oriented GT OBB 执行 `any(containment)`，最近中心 GT 只作诊断；这不是 Clio 官方 IoU matching 指标。冻结主策略是 Q1F：优先使用 Top-5+A2 永久对象，没有永久对象时确定性回退 Q0。2026-09-02 修复 nearest-only bug 后逐任务重算，36 个最终判定与主数字均未变化。`±RMSE` 表示给每个 GT OBB 加入实测配准 RMSE 后的敏感性结果。

| 指标 | Apartment development | Cubicle fixed-confirmatory |
|---|---:|---:|
| Q0 Top-1 严格 Acc@1 | 11.11% | 27.78% |
| Q1F Top-5+A2＋Q0 fallback 严格 Acc@1 | 11.11% | 38.89% |
| Q1F−Q0 严格差值 | 0.00pp | **+11.11pp** |
| Q1F−Q0 `±RMSE` 差值 | +5.56pp | **+16.67pp** |
| A1 最终连通分量 pair F1 | 85.29% | **93.47%** |
| A2 任务内几何＋质量 complete-link pair F1 | **88.14%** | 91.56% |
| 关系正例 target+reference 严格 / `±RMSE` Acc@1 | 0.00% / 11.76% | 11.41% / 48.32% |
| 负例拒答 / 原因命中 / 双端定位后关系拒答 | 100.00% / 31.62% / 10.29% | 98.66% / 67.79% / 44.97% |

Cubicle 上 Q1F 相比 Q0 的 +11.11pp 是 Top-K 多帧检索/lifting、可用 A2 对象记忆、确定性 fallback 及整条策略共同产生的 fixed-confirmatory 系统差值，不能归因成“A2 关联本身更好”：最终聚类同口径下，A2 F1 比 A1 低 1.91pp。关系正确性同时要求返回 target 和用于方向计算的 reference 命中各自 GT；旧 target-only 数字已废弃。关系仍使用未校准的 0.60 工程阈值，因此 Brier/ECE 只是诊断值，真实 calibration 尚未完成。
Clio D8 运行没有保存 `semantic_embedding`：Apartment 为 `0/35`，Cubicle 为 `0/181`；同一 task 内观测的 class text 也恒定。因此这些 A2 数字只描述**任务内几何＋质量 complete-link 关联**，不能描述成多模态语义关联。代码中旧的 `A2-evidence-aware-complete-link` 名称仅作为兼容 policy ID 保留。


求职版不把 Q2 写成正式贡献：现有 Q2 仍是 `coverage_aware=false` 的停止策略诊断，没有带标签的真实查询策略消融。最终主表也不声称提供 Instance Recall、Duplicate Rate、Count Error、整条 36-task 延迟/峰值显存或统计置信区间；这些未测项目不会由工程计数替代。逐例真实失败见 [Clio 最终失败案例](docs/CLIO_FINAL_FAILURE_CASES.md)。

D21 主线没有新增推理性能优化：Q1F 每个 task 最多仍运行 5 次 SAM。完整 36-task 的端到端延迟和峰值显存仍需在有卡环境重新采集。

### Post-D21 PE mask-crop 代表中心扩展

在不重跑 VGGT、检索、SAM、lifting 或 A2 聚类的前提下，新增一个训练自由的代表中心对照：对已选中 A2 对象内的每个 observation 计算 PE-Core-L14-336 的 context crop 与 masked crop 对 `sam_query` 的平均相似度，并使用最高分 observation 的 3D 中心。A2 对象选择和无对象时的 Q0 fallback 均保持不变。

| 严格 Acc@1 | Apartment development | Cubicle fixed-confirmatory |
|---|---:|---:|
| 当前 Q1F | 2/18（11.11%） | 7/18（38.89%） |
| 最高质量 observation | 3/18（16.67%） | 7/18（38.89%） |
| PE 语义代表中心 | 3/18（16.67%） | 8/18（44.44%） |

PE 相对当前 Q1F 在两个场景各增加 1 个严格正确 task、0 个回退；Cubicle 的 RMSE-padded 结果保持 13/18。Apartment 上 PE 与无模型的最高质量基线打平，因此不声称开发集存在 PE 特异增益；Cubicle 上相对当前 Q1F 和质量基线各增加 1/18，但由于场景已有历史暴露，只报告 post-D21 fixed-confirmatory 系统观察，不声称严格 held-out、统计显著、已训练 ranker 或 SigLIP2 结果。轻量证据与复现命令见 [PE mask-crop 扩展](evidence/post-d21-pe-fusion/README.md)，clean clone 可运行 `python -m scripts.validate_clio_pe_semantic_fusion_summary`。

公开仓库仅保留轻量聚合摘要、报告/配置 SHA-256 和边界说明，见 [Clio 最终轻量证据](evidence/final-clio/README.md)。原始 Clio 数据、YAML、mask、点云、视频和 pair-level 本地报告不进入 Git。clean clone 可运行：

```bash
python -m scripts.validate_clio_final_summary
```

### Clio 最终 evaluator 重放命令

公开 clone 只能校验聚合摘要。取得 Clio 数据并物化两个 run root 后，先核对任务清单；这一步明确依赖本地 YAML，不属于单测：

```bash
python -m scripts.validate_clio_query_manifest \
  --query-manifest configs/clio_apartment_queries.json \
  --task-yaml data/clio/apartment/metadata/tasks_apartment.yaml
python -m scripts.validate_clio_query_manifest \
  --query-manifest configs/clio_cubicle_queries.json \
  --task-yaml data/clio/cubicle/tasks_cubicle.yaml
```

36-task 入口按两个 tracked manifest 各执行 18 个正式 task，固定编排 PE Top-5 → SAM 3＋3D lifting → D7 → D8 → A2。先用无卡 dry-run 审计完整计划；确认使用全新 task 输出目录后，移除 `--dry-run` 执行：

```bash
python -m scripts.run_clio_36_task_batch --open-vocab-env /root/autodl-tmp/envs/open_vocab --sam3-checkpoint /root/autodl-tmp/cache/modelscope/facebook-sam3/sam3.pt --dry-run
```

dry-run 固定产生 36 个 task、72 条 GPU 命令和最多 108 条条件 CPU 命令。D6 没有 lifted observation 时跳过该 task 的 D7/D8/A2；执行模式会先检查两套几何、图像、环境、checkpoint 和输出目录冲突。

原始 Clio 数据、几何与 run 报告仍不随 Git 分发，因此陌生人必须先合法取得数据并生成两套几何；这个入口解决批处理缺口，不改变数据许可边界。

批处理完成后，用下面的 CPU evaluator 命令重建最终 Apartment/Cubicle 报告。`relation-benchmark-v2` 必须是空目录；重复运行时请选择新的空输出目录或先归档旧目录。


```bash
python -m scripts.evaluate_clio_grounding_benchmark \
  --query-manifest configs/clio_apartment_queries.json \
  --task-yaml data/clio/apartment/metadata/tasks_apartment.yaml \
  --world-alignment runs/clio-apartment-dev-v2-lc/vggt_to_clio_world_alignment.json \
  --run-root runs/clio-apartment-dev-v2-lc \
  --output runs/clio-apartment-dev-v2-lc/grounding_benchmark.json
python -m scripts.evaluate_clio_grounding_benchmark \
  --query-manifest configs/clio_cubicle_queries.json \
  --task-yaml data/clio/cubicle/tasks_cubicle.yaml \
  --world-alignment runs/clio-cubicle-heldout-v1/vggt_to_clio_world_alignment.json \
  --run-root runs/clio-cubicle-heldout-v1 \
  --frozen-policy configs/clio_cubicle_frozen_policy.json \
  --output runs/clio-cubicle-heldout-v1/grounding_benchmark.json

python -m scripts.evaluate_clio_association \
  --query-manifest configs/clio_apartment_queries.json \
  --task-yaml data/clio/apartment/metadata/tasks_apartment.yaml \
  --world-alignment runs/clio-apartment-dev-v2-lc/vggt_to_clio_world_alignment.json \
  --run-root runs/clio-apartment-dev-v2-lc \
  --output runs/clio-apartment-dev-v2-lc/association_benchmark.json
python -m scripts.evaluate_clio_association \
  --query-manifest configs/clio_cubicle_queries.json \
  --task-yaml data/clio/cubicle/tasks_cubicle.yaml \
  --world-alignment runs/clio-cubicle-heldout-v1/vggt_to_clio_world_alignment.json \
  --run-root runs/clio-cubicle-heldout-v1 \
  --output runs/clio-cubicle-heldout-v1/association_benchmark.json

python -m scripts.run_clio_relation_benchmark \
  --query-manifest configs/clio_apartment_queries.json \
  --task-yaml data/clio/apartment/metadata/tasks_apartment.yaml \
  --world-alignment runs/clio-apartment-dev-v2-lc/vggt_to_clio_world_alignment.json \
  --geometry-anchor-poses runs/clio-apartment-dev-v2-lc/geometry.anchor_poses.json \
  --run-root runs/clio-apartment-dev-v2-lc \
  --output-dir runs/clio-apartment-dev-v2-lc/relation-benchmark-v2
python -m scripts.run_clio_relation_benchmark \
  --query-manifest configs/clio_cubicle_queries.json \
  --task-yaml data/clio/cubicle/tasks_cubicle.yaml \
  --world-alignment runs/clio-cubicle-heldout-v1/vggt_to_clio_world_alignment.json \
  --geometry-anchor-poses runs/clio-cubicle-heldout-v1/geometry.anchor_poses.json \
  --run-root runs/clio-cubicle-heldout-v1 \
  --output-dir runs/clio-cubicle-heldout-v1/relation-benchmark-v2

python -m scripts.validate_clio_relation_benchmark \
  runs/clio-apartment-dev-v2-lc/relation-benchmark-v2
python -m scripts.validate_clio_relation_benchmark \
  runs/clio-cubicle-heldout-v1/relation-benchmark-v2
python -m scripts.build_clio_final_summary
python -m scripts.validate_clio_final_summary \
  evidence/final-clio/benchmark_summary.json
```

## 目录

```text
configs/              数据、阈值与实验配置
adapters/             vggt_geom / open_vocab 文件接口
relground/
  retrieval.py        top-K 与时序/视角去冗余
  observations.py     ObjectObservation 与稳健 3D lifting
  association.py      关联、去重与持久 ObjectMemory
  relations.py        显式参考系与关系约束
  calibration.py      confidence 与 abstention
  schemas.py          JSON 数据协议与 RunManifest
evaluation/           B0-B5 定义、指标与失败 taxonomy
tests/                求职版核心 CPU 自测（完整研究回归见下方标签）
scripts/              reproduce / evaluate / demo
runs/                 manifest、日志和指标（大文件不进 Git）
```

## 快速验证

从公开仓库 clean clone 后，可用下面的 CPU 入口安装并验证；核心自测不要求下载权重、数据或启用 GPU：

```bash
git clone https://github.com/dream-once/VGGT-RelMem.git
cd VGGT-RelMem
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m scripts.verify_public_clone
python -m scripts.demo --save-memory runs/demo/object_memory.json
```

`verify_public_clone` 运行求职版主分支保留的 82 项 tracked-only CPU 核心测试，覆盖
VGGT 适配、开放词汇、Top-K/Q1F、3D lifting、对象记忆、A1/A2、关系与最终
Clio evaluator。精简前的 262 项完整研究回归保存在 Git 标签
`research-complete-2026-09-02`，不会因主分支整理而丢失。如果已经位于仓库
根目录并装好依赖，可以直接从验证命令开始。

真实 D4–D8 只保留轻量 JSON、manifest 与查询级 validator 报告，收录在 [evidence/week1](evidence/week1/README.md)。视频、预览、mask、点云、权重、数据集与几何 NPZ 均不进入 Git，可按文档命令在本地重新生成。


## 换机检查与进度记忆

仓库内置 `vggt-instance-handoff` Skill。克隆或更换 AutoDL 实例后，在开始下一项任务前运行只读审计：

```bash
python .agents/skills/vggt-instance-handoff/scripts/audit_instance.py
```

审计会区分 Git 跟踪源码与本机专属的环境、权重、数据集和运行产物，同时报告上游 commit、磁盘和 GPU。可恢复的当前进度记录在 [`PROJECT_MEMORY.md`](PROJECT_MEMORY.md)；每次用户明确要求把项目更新上传 GitHub 时，应先更新该记忆和 `.agents/vggt-instance-baseline.json`，再随代码一同提交。审计本身不会下载、删除、覆盖或推送任何内容。

## VGGT-SLAM 2.0 几何接入

几何环境与后续 SAM3 环境分离。安装固定的 Python 3.11 几何环境：

```bash
bash scripts/bootstrap_vggt_geom.sh
conda run -p /root/autodl-tmp/envs/vggt_geom \
  python -m scripts.run_vggt_geometry --check-only
```

在按数字命名的 RGB 图片上运行官方几何管线并导出：

```bash
TORCH_HOME=/root/autodl-tmp/cache/torch \
conda run -p /root/autodl-tmp/envs/vggt_geom \
  python -m scripts.run_vggt_geometry \
  --image-folder data/office_loop/images \
  --output runs/office-loop/geometry.npz \
  --max-frames 16 --max-loops 0
```

适配器同时生成 frame/submap 来源 manifest 和 `world_from_anchor` 位姿 JSON。NPZ 点图保持 VGGT submap canonical 坐标，`world_from_camera` 保存上游优化后的 SL(4) 齐次变换；它可能是投影变换，下游会执行齐次除法。几何 schema 0.2 同时保存未修改的 `raw_confidence_maps`、上游 percentile threshold 得到的布尔 `valid_masks`，并保留二值 `confidence_maps` 作为旧下游兼容别名。

### 当前状态与文档日程

核心接入对应推荐文档的 **D3**。此前两次 RTX 4090 八帧推理在 schema 0.1 下通过 validator 且 SHA-256 完全一致；geometry schema 0.2 的真实 GPU 补验也已完成，新增的原始置信度与布尔有效点掩码均通过独立 validator。旧 0.1 产物仍可读取，但不包含可分析的原始置信分数。

当前文档日程状态：

- D1：官方 VGGT-SLAM、SALAD、VGGT_SPARK 均已下载并固定 commit；SAM 3 checkpoint 已从 ModelScope 官方发布页下载、校验并成功加载。
- D2：`vggt_geom` 与 Python 3.12 `open_vocab` 两个隔离环境、geometry/mask NPZ+JSON schema、适配器和 smoke test 均已完成。
- D3：原 0.1 运行器、产物与两次 GPU 严格复现已完成；0.2 raw confidence/valid mask 源码、单测和真实 RTX 4090 产物验收也已完成。
- D4：PE top-1、SAM 3 mask、mask-to-point lifting、3D OBB、可视化、validator 和两次真实 GPU 复现均已完成。
- D5：原始 temporal 与新间隔序列 hybrid（时间 + 相机距离 + 视角）K=1/3/5 均有真实 GPU 产物，六个 hybrid 查询的 validator 均为 `PASS`。
- D6：间隔多视角 Top-K 的 SAM 3 与鲁棒逐帧 lifting 已完成，真实产物与查询级多视角 validator 均已验收。
- D7：stride-aware 场景缓存、冻结 ObjectObservation 和动态证据视频已完成。
- D8：可移动的 ObjectMemory 1.0 bundle 已完成，保留 10 条 pending 观测且不提前关联。
- D9：精确同类 + 3D 中心距离/AABB 重叠 gate、跨帧组件提升及独立人工 pairwise 评测已完成。
- D10：D9 已拆成无标签的确定性预测和独立标签评测，并补齐可移动轻量证据、无匹配合法性与排列不变回归验收。
- D11：Visual Memory 与 Candidate Outcome Cache 0.1 已冻结；2026-08-29 的 GPU 补验把原 4+4 partial cache 完整物化为 8/8 可用 outcome，并证明原四个 outcome 除选择 rank 来源外未漂移。
- D12：A1 代码与已发布哈希保持冻结；A2 实现保留可选 embedding 接口以及几何/OBB/质量 pair 证据和 complete-link 聚类。Clio 最终运行的 embedding 覆盖为 0，故只称任务内几何＋质量 complete-link。
- D13：`Q0-vggt-slam-upstream-top1` 已按固定源码和 retained D4/D5 JSON 冻结为 `upstream-aligned`；不声称 FOUND-IT 官方复现，轻量 D4 的二进制验收缺口已显式记录。
- D14：Q1 Fixed Top-K 已按冻结 hybrid 阈值在完整真实 GPU cache 上通过 prediction replay；K=1 与 Q0 一致，预测仍不读取人工标签。
- D15：Q2 gain-based sequential search 已在完整真实 GPU cache 上跑满 5 步并通过 trace 重算；`performance_claim=null`，不声称真实性能提升。
- D15.5：95 帧长轨迹的场景级 RGB 点云、轨迹、Top-K 相机、40 条 3D 观测、8 个关联对象、PLY/MP4/Viser 与对象中心多视角审计均已完成；独立 validator 为 `PASS`。
- D16：Clio 数据协议、场景角色和 fail-closed 磁盘审计已完成；本地已取得 `apartment` 与 `cubicle` 公开场景并完成 192/172 帧几何采样、COLMAP pose 对齐和 evaluator-only Clio world Sim(3)。数据许可仍为 `DATA_LICENSE_UNVERIFIED`，原始数据禁止随仓库再分发。
- D17：正式关系预测继续与标签/evaluator 隔离，冻结 anchor 坐标约定和 0.60 未校准工程阈值；Apartment/Cubicle 分别完成 272/298 个正负方向查询，selective answer risk–coverage 排除正确拒答。真实数据已评测，但独立 calibration split 仍为 `REAL_DATA_CALIBRATION_PENDING`。
- D18：原 Q0/Q1/Q2 × A1/A2 cache 矩阵保持为历史工程重放；新增的完整场景 benchmark 在 18 个 Apartment development 与 18 个 Cubicle fixed-confirmatory task 上直接评测 Q0/Q1 对象定位及 A1/A2 关联，不再标记 Cubicle `PENDING`。
- D19：Q2/A2 单因素消融与六类失败审计已完成；Clio apartment 已运行无标签工程消融，Q2 base/retrieval-only 都在 4 帧后低 gain 停止，关闭 patience 才运行到 5 帧，三者都只有 1 条 observation。没有标签的真实指标仍为 `REAL_ABLATION_PENDING`，不声称性能提升。
- D20：单入口 CPU 复现命令、三张 JSON 派生表、README 数字检查及移动目录逐字节复验均为 `PASS`；新增 Clio 聚合摘要也可在 clean clone 无原始数据条件下校验算术与声明边界。
- D21：历史结果卡与 README 高风险声明审计为 `PASS`；2026-08-31 的 post-D21 Clio supplement 进一步加入对象定位、真实关联、关系与拒答结果，仍保留 calibration 和完整演示缺口；FOUND-IT 对照明确为项目范围外。

## 历史：Clio apartment 24 帧 GPU 开发验收（2026-08-30）

本节保留 2026-08-30 的 24 帧历史验收，后续完整场景结果以上方“Clio Apartment → Cubicle 最终实验”为准。当时只使用官方公开 `apartment` 的 RGB＋任务元数据开发子集，不需要作者审批；`cubicle` 尚未下载。数据许可没有单独明确，因此原始图像、YAML、mask、点云和视频均不进入 Git。

兼容 D20 历史复现卡：D17 当时用 `2 个正查询与 3 个负查询` 验证协议；`D18 complete-cache development replay` 与 `Clio apartment GPU development replay` 均通过，但 synthetic 数值无变化。这些是旧 24 帧工程验收，不覆盖上方 192/172 帧最终实验。

真实 RTX 4090 链路为 VGGT-1B → PE-Core-L14-336 → SAM 3 → Robust3DLifter → Candidate Outcome Cache → Q×A/Ablation 重放。主要验收结果：

- 24 帧 geometry schema 0.2 validator 为 `PASS`；轨迹最大平移 `1.027912` 个重建单位、最大旋转 `179.856°`。VGGT 重建单位不冒充米；
- 公开任务元数据中的 `pillow` 被选作 development query。24/24 candidate outcome 完整物化，SAM 产生 3 个实例且 3 个均完成 3D lifting；
- 有效 evidence 来自 `rgb_128` 与 `rgb_90`，同一查询的相机跨度为 `0.195787` 个重建单位和 `80.608°`，因此不再是“连续向前走的近同视角”样例；
- Q0 Top-1 得到 0 条 observation；Q1 K=5 得到 1 条；Q2 在 4 帧后因连续两次低 gain 停止，也只有 1 条。A1/A2 均未形成永久对象；
- 上述冻结 `pillow` 结果是 development engineering replay，不是 Grounding Acc@1、held-out 或性能提升结果。后续 D21.1 受控诊断表明，当前主要观察故障来自 SAM prompt 敏感性，而不是已经证明 association 阈值必须更换。

### D21.1 枕头失败诊断

在完全相同的 24 帧、VGGT geometry 和 PE 排名上，真实 RTX 4090 sweep 只改变 SAM prompt 或阈值：

- 冻结 `pillow@0.5` 的新旧运行均为 3 个 mask/2 个 evidence frame，规范 JSON 相等且三个 mask 的 SHA-256 一致；`a pillow@0.5` 结果相同，`bring me a pillow@0.5` 为 0 个 mask；
- `dinosaur pillow@0.5` 在另外 5 帧得到 5 个 mask，经逐帧 overlay 人工核查均为同一个绿色恐龙枕头；这是利用实例外观描述定位故障的 development diagnostic，不是正式 query policy；
- `pillow@0.4/0.3` 分别产生 6/8 个 mask，但新增局部或可疑小框，未被当作人工真值；六组实验的检测帧并集为 8/24，另外 16 帧没有 mask。由于尚未完成 24 帧可见性标注，这些数字不是 segmentation recall；
- 5 个干净的 `dinosaur pillow` 跨帧观测进入冻结 A2 后，10/10 正 pair 形成 1 个永久对象、0 pending。A2.1 得到相同结果且没有负 pair/held-out 证据，因此不升级为正式方法；
- COLMAP 数据库列出 1,845 帧，但本地缺 1,059 张 RGB，且没有 sparse pose/rosbag；完整轨迹 evaluator-only Sim(3) 仍为 `BLOCKED_MISSING_SPARSE_OR_ROSBAG_POSES`。

轻量证据见 `evidence/week4/d21_1-pillow-diagnostic`，不含 RGB、mask、点云、联系表或视频。公开验收命令：

```bash
python -m scripts.validate_d21_1_pillow_diagnostic \
  evidence/week4/d21_1-pillow-diagnostic/public_report.json
```

本地紧凑对照图位于 `runs/clio-apartment-gpu/d21_1-pillow-prompt-sweep-gpu/comparison/detected_union_contact_sheet.jpg`。

公开轻量验收不依赖 Clio 原始数据：

```bash
python -m scripts.validate_clio_gpu_acceptance \
  evidence/week4/clio-apartment-gpu/gpu_acceptance_report.json
python -m scripts.validate_d18
python -m scripts.validate_d19
python -m scripts.validate_d20
python -m scripts.validate_d21
```

`--check-only` 逐模块使用独立子进程，适合无卡/小内存模式。它不会加载 VGGT、SALAD 权重，也不会执行推理：

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/vggt_geom \
  python -m scripts.run_vggt_geometry --check-only
python -m unittest discover -s tests -v
```

今天的源码接入验收标准是：check-only 所有模块无 `ERROR`；VGGT-SLAM commit 为 `35327ac28b7d193df9ccc39ba6346052bb6f1207`；全部单测通过。这个结果只说明“接线完成”，不代替 D3 的 GPU 验收。

### D3 GPU 验收与复现命令（已完成）

官方示例图片已包含在 `third_party/VGGT-SLAM/office_loop.zip`。准备数据后，用 8 帧、关闭开放词汇和回环做第一轮：

```bash
mkdir -p data
unzip -n third_party/VGGT-SLAM/office_loop.zip -d data

TORCH_HOME=/root/autodl-tmp/cache/torch \
conda run --no-capture-output -p /root/autodl-tmp/envs/vggt_geom \
  python -m scripts.run_vggt_geometry \
  --image-folder data/office_loop \
  --output runs/office-loop/geometry.npz \
  --max-frames 8 --submap-size 8 --max-loops 0 \
  --disable-flow-filter

python -m scripts.validate_geometry runs/office-loop/geometry.npz
```

首次运行会下载约 5.03 GB 的 VGGT-1B 权重；`--max-loops 0` 不需要 `dino_salad.ckpt`。验收脚本必须输出 `"status": "PASS"`，且目录中同时存在：

- `geometry.npz`：关键帧点图、置信度和 SL(4) 变换；
- `geometry.manifest.json`：frame/submap 来源和上游 commit；
- `geometry.anchor_poses.json`：每帧刚性 anchor 位姿；
- `run_manifest.json`：命令、配置、运行时间、峰值显存和三个上游提交。

再换一个输出目录重复一次，两个结果都通过 validator、帧数一致且没有 NaN/Inf，即达到文档 D3 的“单场景稳定复现 + run manifest”。

几何 0.2 已在 RTX 4090 上完成真实产物补验，并使用新目录保留历史 0.1 证据。复现命令为：

```bash
TORCH_HOME=/root/autodl-tmp/cache/torch OMP_NUM_THREADS=8 \
conda run --no-capture-output -p /root/autodl-tmp/envs/vggt_geom \
  python -m scripts.run_vggt_geometry \
  --image-folder data/office_loop \
  --output runs/office-loop-v02/geometry.npz \
  --max-frames 8 --submap-size 8 --max-loops 0 \
  --disable-flow-filter

python -m scripts.validate_geometry \
  runs/office-loop-v02/geometry.npz
```

本次验收结果为 `status=PASS`、`schema_version=0.2`、`raw_confidence_available=true`；8 帧 raw confidence 全部有限，范围为 `1.0–14.375`，平均有效点比例为 `0.749846`。总运行时间 `16.270 s`，峰值显存 `4008.549 MB`。该补验只证明 D3 新数据契约可真实运行；当前 D6 仍使用兼容的二值 `confidence_maps`，因此不会为了版本对齐而机械重跑 D6–D8。

基于已导出的几何生成彩色 PLY 和三视角 PNG，不会重新运行模型：

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/vggt_geom \
  python -m scripts.visualize_geometry runs/office-loop/geometry.npz
```

云服务器上可通过仅绑定回环地址的 Viser 查看交互点云，再由 VS Code 转发 8080 端口：

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/vggt_geom \
  python -m scripts.visualize_geometry runs/office-loop/geometry.npz \
  --serve --host 127.0.0.1 --port 8080
```

## D4 受控单视角 B0/B1（已修正并验收）

Perception Encoder 固定为 `3e352cca660658d4b5c90f42a7808b11469e4c66`，SAM 3 固定为 `8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`，使用独立的 Python 3.12 `open_vocab` 环境。

复核发现，早期 `scripts.run_open_vocab_top1` 实际执行的是“原始分辨率 SAM 3 → mask 缩放 → `Robust3DLifter`”，因此它不是严格官方 B0。旧命令和 `b0_result.json` 文件名仅为历史复现保留，其方法现明确记为 `B1-robust-single-view (legacy)`，已有两次一致结果仍是有效的工程复现证据。

新 `scripts.run_single_view_baselines` 在一次运行中只执行一次 PE 和一次 SAM 3，并让两条基线共享四项输入：Top-1 帧、VGGT 实际预处理后的图像、SAM mask、VGGT 点图与世界变换。两者唯一变量是 lifting：

- `B0-official`：mask 直接取点，只过滤 NaN/Inf，使用上游普通 PCA OBB；
- `B1-robust-single-view`：增加几何置信过滤、MAD 离群点过滤和最小点数门槛。

无推理检查会核对 8 帧 resize/crop/batch-pad 映射、固定源码和本地权重：

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/open_vocab \
  python -m scripts.run_single_view_baselines \
  --check-only --max-frames 8 \
  --sam3-checkpoint /root/autodl-tmp/cache/modelscope/facebook-sam3/sam3.pt
```

真实受控运行与严格验收：

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/open_vocab \
  python -m scripts.run_single_view_baselines \
  --query 'trash can' --max-frames 8 \
  --pe-checkpoint \
  /root/autodl-tmp/cache/huggingface/hub/models--facebook--PE-Core-L14-336/snapshots/bafb0f76541d399057e980a25947f67acec76575/PE-Core-L14-336.pt \
  --sam3-checkpoint /root/autodl-tmp/cache/modelscope/facebook-sam3/sam3.pt \
  --output-dir runs/office-loop-single-view-trash-can

python -m scripts.validate_single_view_baselines \
  runs/office-loop-single-view-trash-can
```

2026-08-24 的 RTX 4090 实跑仍选择 `frame_0004`，PE 分数为 `0.20180504907322777`；SAM 在 `294×518` VGGT 网格直接输出 6 个共享 mask，没有后处理 resize。`B0-official` 生成 6 个 OBB、共 5,899 个有限点；`B1-robust-single-view` 生成 4 个 OBB、共 5,379 个点；4 个实例可一一受控比较。运行耗时 `19.409 s`，峰值显存 `5090.538 MB`，严格 validator 为 `PASS`。

保存的 `sam_input.png` 与 `preprocess.json` 记录图像和 transform 哈希；validator 同时检查 mask 直接匹配 VGGT 网格、NaN/Inf、frame/query/class/score 一致性、共享实例和点云引用。`printer` 仍作为负查询保留：前 8 帧没有可见打印机，不降低 `0.50` 阈值制造误检。

## D5 PE top-K 与冗余抑制（已完成）

`scripts.run_pe_topk` 一次编码所有候选帧，同时导出 K=1/3/5。它保持上游 PE Top-1 的非负余弦规则，并在相同分数时按 geometry 原始帧序稳定排序；视角特征来自 D3 导出的刚性 `world_from_anchor`，相机局部 `+z` 定义为前方，不把可能为投影变换的 `world_from_camera` 当作相机姿态。

这里的 Top-K 指与文本查询最相关的前 K 个候选图像帧，不是 K 个物体。每个入选帧在 D6 中仍可能由 SAM 3 分割出零个、一个或多个实例。

无卡检查不会加载 2.68 GB PE checkpoint：

```bash
python -m scripts.run_pe_topk \
  --check-only --max-frames 8 \
  --redundancy temporal --min-frame-gap 2
```

无卡检查结果为 `SOURCE_READY`：8 帧、K=1/3/5、PE commit 和全部 anchor pose 均有效。无卡容器的 cgroup 内存上限是 2 GiB，小于 checkpoint，因此不能在该模式下加载 PE。

2026-08-23 的连续八帧 temporal GPU 验收为 `PASS`：top-1 是 `frame_0004`，K=1/3/5 保留 1/3/4 帧。2026-08-24 又在间隔十帧的八视角几何上真实运行六个查询，配置明确为 `hybrid`、`min_frame_gap=2`、`min_camera_distance=0.15`、`min_view_angle_deg=3.0`；六个目录均重新通过 `validate_topk_retrieval`。因此 viewpoint/hybrid 现在不再只有单测证据。

复现命令：

```bash
OMP_NUM_THREADS=8 \
conda run --no-capture-output -p /root/autodl-tmp/envs/open_vocab \
  python -m scripts.run_pe_topk \
  --query 'trash can' --max-frames 8 \
  --pe-checkpoint \
  /root/autodl-tmp/cache/huggingface/hub/models--facebook--PE-Core-L14-336/snapshots/bafb0f76541d399057e980a25947f67acec76575/PE-Core-L14-336.pt \
  --redundancy temporal --min-frame-gap 2 \
  --output-dir runs/office-loop-d5-trash-can

python -m scripts.validate_topk_retrieval \
  runs/office-loop-d5-trash-can
```

验收产物：

- `retrieval.json`：全部帧原始排序、位姿特征、配置、PE Top-1 兼容性和前缀一致性；
- `topk_1.json`、`topk_3.json`、`topk_5.json`：各 K 的帧索引与分数；
- `topk_preview.png`：最大 K 的候选帧预览；
- `run_manifest.json`：命令、commit、配置、耗时与峰值显存。

冗余抑制可能合理地返回少于请求 K 的独立视角，此时产物会显式标记 `exhausted_nonredundant_candidates`，不会用已判定为重复的帧静默回填。

## D6 B2 Top-K 多帧鲁棒 lifting（已修正并验收）

`scripts.run_sam_topk_lifting` 直接读取 D5 的 `topk_5.json`，不会重新运行 PE；SAM 3 只加载一次。修正后，每个入选帧先复现 VGGT 的 resize/crop/batch-pad，保存 `sam_inputs/<frame>.png` 与 `preprocess/<frame>.json`，SAM mask 直接索引同形状点图，不再做推理后 resize。之后才执行置信过滤、MAD 离群点过滤、最小点数检查和世界坐标 OBB 拟合。

无 GPU 的输入完整性检查：

```bash
python -m scripts.run_sam_topk_lifting --check-only \
  --selection runs/office-loop-d5-trash-can/topk_5.json \
  --sam3-checkpoint /root/autodl-tmp/cache/modelscope/facebook-sam3/sam3.pt
```

真实运行与独立验收：

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/open_vocab \
  python -m scripts.run_sam_topk_lifting \
  --selection runs/office-loop-d5-trash-can/topk_5.json \
  --sam3-checkpoint /root/autodl-tmp/cache/modelscope/facebook-sam3/sam3.pt \
  --output-dir runs/office-loop-d6-controlled-trash-can

python -m scripts.validate_d6 \
  runs/office-loop-d6-controlled-trash-can
```

2026-08-24 的受控 RTX 4090 实跑耗时 `13.918 s`、峰值显存 `5088.913 MB`，validator 为 `PASS`。4 帧都产生有效 3D 观测：`frame_0004` 为 6/4、`frame_0001` 为 6/4、`frame_0006` 为 5/4、`frame_0008` 为 4/3（SAM 实例/有效观测）。总计 21 个 SAM 实例，15 个通过鲁棒 lifting，6 个因有效 VGGT 点少于 30 被显式拒绝。

validator 对新 `B2-topk-multiframe` 强制检查每帧 SAM 输入哈希、transform、布尔 mask dtype、mask 与点图形状一致、禁止 resize、observation 元数据和有限点云；同时保留对 2026-08-23 历史 D6 产物的读取兼容。D7 已发布缓存仍用于证明 schema/cache/视频链路，不冒充这次受控重跑的性能证据。

上述历史样例只能称为“多帧管线验收”：前 8 帧最大位姿变化仅为 `0.0669` 个未标定重建单位和 `0.630°`，不能支持多视角互补或融合结论。

### D8 前真实多视角补充验收（已完成）

D3 运行器新增确定性的 `--frame-start/--frame-stride`，并把最终帧名写入 manifest。完整 473 帧序列中选择 `frame_0001, 0011, …, 0071`：

```bash
TORCH_HOME=/root/autodl-tmp/cache/torch OMP_NUM_THREADS=8 \
conda run --no-capture-output -p /root/autodl-tmp/envs/vggt_geom \
  python -m scripts.run_vggt_geometry \
  --image-folder data/office_loop \
  --output runs/office-loop-multiview-s10/geometry.npz \
  --frame-start 0 --frame-stride 10 --max-frames 8 \
  --submap-size 8 --max-loops 0 --disable-flow-filter

python -m scripts.validate_multiview_geometry \
  runs/office-loop-multiview-s10/geometry.npz
```

相同门槛（最大平移至少 `0.5` 个未标定重建单位、最大旋转至少 `3°`）下，历史前 8 帧明确 `FAIL`；新间隔帧达到 `1.11249` 和 `4.456°`，validator 为 `PASS`。这只是固定数据集的视角跨度门槛，不把未标定平移解释成米或厘米。

新几何上的 D5/D6 查询矩阵及查询级同帧对验收：

| 查询 | 角色 | SAM 实例 | 有效 3D 观测 | 覆盖帧 | 查询级证据 |
| --- | --- | ---: | ---: | ---: | --- |
| `trash can` | 正例 | 15 | 10 | 4 | `TRUE_MULTIVIEW`，3 对帧通过 |
| `poster` | 正例 | 8 | 7 | 2 | `MULTIFRAME_ONLY` |
| `blue recycling bin` | 正例 | 7 | 5 | 3 | `MULTIFRAME_ONLY` |
| `printer` | 正例 | 2 | 2 | 2 | `MULTIFRAME_ONLY` |
| `dog` | 负例 | 0 | 0 | 0 | `NEGATIVE_CONTROL` |
| `bed` | 负例 | 0 | 0 | 0 | `NEGATIVE_CONTROL` |

查询级 validator 只查看 `frames_with_lifted_observations`，并要求同一帧对同时满足平移至少 `0.5` 个未标定重建单位和旋转至少 `3°`；不再用可能来自不同帧对的两个最大值拼成“通过”。例如：

```bash
python -m scripts.validate_query_multiview \
  runs/office-loop-mv-d6-trash-can \
  --anchor-poses runs/office-loop-multiview-s10/geometry.anchor_poses.json
```

`trash can` 的 `0001–0071`、`0071–0041`、`0071–0021` 三对通过。`poster`/`printer` 的唯一有效对只有 `0.3704` 平移，`blue recycling bin` 的有效对旋转均低于 `3°`，所以三者只能称多帧证据，不能称真正多视角证据。

`printer` 原计划作为负例，但预览确认 `frame_0051/0071` 的柜顶确有打印机，因此按真实正例记录。`dog` 和 `bed` 则是故意选择的办公室场景负对照：不是期待检出它们，而是检验 Top-K 排名之后 SAM/3D 链路能否在目标不存在时保持零证据、不制造幻觉。它们的 D6 状态是 `INSUFFICIENT_MULTIFRAME_3D_EVIDENCE`，只证明原始证据不足，不提前冒充 D10 的可靠拒答。

## D7 冻结 ObjectObservation 与场景缓存（已完成）

`ObjectObservation` 已冻结为 schema `1.0`。D7 cache loader 严格检查字段集合、schema 版本、观测 ID、查询文本和多帧覆盖；D6 的旧 `0.1` observation 可在读取时升级，但 D7 缓存中的未知或缺失字段会被拒绝。

以下连续八帧命令保留为历史 D7 证据。这里使用 `vggt_geom` 只因为环境内已有 OpenCV 视频编码器，不会加载 VGGT、PE、SAM 3，也不使用 GPU：

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/vggt_geom \
  python -m scripts.cache_scene_observations \
  --d6-dir runs/office-loop-d6-trash-can \
  --output-dir runs/office-loop-d7-trash-can \
  --image-folder data/office_loop \
  --video-duration 40 \
  --scene-id office-loop-trash-can

conda run --no-capture-output -p /root/autodl-tmp/envs/vggt_geom \
  python -m scripts.validate_d7_cache \
  runs/office-loop-d7-trash-can
```

缓存内容：

- `observations.json`：4 帧共 15 条冻结的 `ObjectObservation`；
- `masks/` 与 `points/`：每条有效观测对应的 mask 和世界坐标点云；
- `previews/`：D6 的 4 张逐帧分割预览；
- `stage_video.mp4`：40 秒、10 FPS 的动态流水线视频；
- `scene_cache.json`：36 个缓存文件的大小与 SHA-256 清单；
- `run_manifest.json`：命令、Git SHA、配置和耗时。

阶段视频不是静态图片轮播，而是按时间连续展示：

- 10 秒：D3 的 8 帧输入序列连续过渡；
- 8 秒：D5 的 PE Top-K 排名、分数条和当前候选；
- 10 秒：D6 的 SAM 3 mask 从原图渐显，并标注有效 3D 观测数；
- 12 秒：D7 缓存的真实点云与 OBB 按帧着色并旋转一周。

2026-08-23 重做后的真实缓存耗时 `8.484 s`，峰值显存为 `null`；独立 validator 为 `PASS`，确认 15 条观测包含 `20,814` 个有限 3D 点、mask 形状均为 `294×518`、视频实际时长为 `40.0 s`。逐 0.5 秒采样的动态比例为 `0.949`，高于验收门槛 `0.250`；旧静态轮播的动态比例只有 `0.048`，现在会被 validator 拒绝。缓存总大小约 `11 MiB`。

基础 Python 可直接重载缓存，不需要任何模型：

```bash
python -c "from relground.observation_cache import load_observation_cache; \
c=load_observation_cache('runs/office-loop-d7-trash-can/observations.json'); \
print(c.schema_version, len(c.frame_ids), len(c.observations))"
```


### D7 当前标准：真实多视角缓存与视频

间隔抽帧不能直接用 `geometry_index` 索引原始连续图片目录；生成器现在优先读取 D6 run manifest 中固定的 geometry manifest，逐项核对帧名，并把全部输入帧和 Top-K 入选帧写回视频 manifest。重制命令：

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/vggt_geom \
  python -m scripts.cache_scene_observations \
  --d6-dir runs/office-loop-mv-d6-trash-can \
  --geometry-manifest runs/office-loop-multiview-s10/geometry.manifest.json \
  --output-dir runs/office-loop-mv-d7-trash-can \
  --image-folder data/office_loop --video-duration 40 \
  --scene-id office-loop-multiview-s10-trash-can

conda run --no-capture-output -p /root/autodl-tmp/envs/vggt_geom \
  python -m scripts.validate_d7_cache runs/office-loop-mv-d7-trash-can
```

新视频使用 `frame_0001, 0011, …, 0071` 八个输入视角，Top-K/SAM/3D 阶段使用 `0001/0071/0041/0021`。独立 validator 为 `PASS`：10 条观测、19,062 个有限点、40 秒，动态比例 `0.975`；MP4 大小为 `6,038,230` bytes，SHA-256 为 `aa924ff9913fcca8475b16209a950004f74f20b6774d3903a9a9756eafae0dea`。

## D8 冻结 ObjectMemory 与 evidence（已完成）

D8 是 D7 与 D9 之间的稳定数据边界：它冻结 `ObjectMemory 1.0` 的字段、evidence 语义和规范 JSON round-trip，把昂贵模型产生的 D7 观测放入 `pending_observations`。它故意不执行跨帧关联、不生成永久对象，使 D9 能在不加载 VGGT/PE/SAM 3 的情况下，对确定且可复现的输入实现和评测关联策略。

```bash
python -m scripts.prepare_object_memory \
  --cache runs/office-loop-mv-d7-trash-can/observations.json \
  --output-dir runs/office-loop-mv-d8-trash-can

python -m scripts.validate_d8_memory \
  runs/office-loop-mv-d8-trash-can
```

当前标准产物来自新多视角 D7：10 条 pending 观测覆盖 `0001/0071/0041/0021` 四帧，保存/重载后的规范 JSON 完全一致；永久对象和关联决策均为 0。缺字段、未知字段、篡改 evidence 或绝对来源路径都会被拒绝。

D8 现在把 D7 来源写成相对路径 `../office-loop-mv-d7-trash-can/observations.json`；目录整体移动后 validator 单测仍通过。因此最终 D6 → D7 → D8 链路已连续，D9 可直接消费这个多视角 trash-can bundle。

## D9 可解释空间 gate 与初始对象关联（已完成）

D9 直接读取 D8 的 `pending_observations`，全程不加载 VGGT、PE 或 SAM 3。当前 gate 严格限定为文档当天范围：

1. 类别文本仅做大小写/分隔符归一化后的精确同类判断，不使用语义 embedding；
2. 同类 pair 满足“3D 中心距离不超过 `0.15`，或由观测 OBB 转成的 AABB IoU 大于 `0`”即建立无向边；
3. 对 pair 图取确定性连通分量；
4. 只有覆盖至少两个不同帧的分量才能成为永久对象，单帧重复候选继续留在 pending。

`0.15` 是当前未标定 VGGT 重建坐标单位，不是米。人工标签来自四张 D6 mask/box 预览，明确把同一实体的重复或局部 SAM mask 放在同一组；标签只进入独立评测，不进入预测命令、prediction bundle 或预测状态判定。

先生成并验证无标签预测 bundle：

```bash
python -m scripts.run_d9_association \
  --memory runs/office-loop-mv-d8-trash-can/object_memory.json \
  --output-dir runs/office-loop-mv-d9-trash-can/prediction \
  --center-distance-threshold 0.15 \
  --min-overlap-iou 0.0 \
  --min-distinct-frames 2

python -m scripts.validate_d9_association \
  runs/office-loop-mv-d9-trash-can/prediction
```

再对冻结预测进行独立标签评测和验证：

```bash
python -m scripts.evaluate_d9_association \
  --prediction-dir runs/office-loop-mv-d9-trash-can/prediction \
  --labels configs/d9_office_loop_trash_can_labels.json \
  --output-dir runs/office-loop-mv-d9-trash-can/evaluation \
  --min-pairwise-f1 0.95

python -m scripts.validate_d9_evaluation \
  runs/office-loop-mv-d9-trash-can/evaluation
```

无标签预测得到 3 个空间组件；中间 trash can 的 6 条观测覆盖 `frame_0001/0041/0021`，被提升为 `obj_0001`，另外两个各含 2 条单帧重复观测的组件仍保持 pending。因此预测产物是 1 个永久对象、4 条 pending、6 条带证据的关联决策，规范 JSON round-trip 与独立重算 validator 均为 `PASS`。如果输入中确实没有足够的跨帧匹配，生成 0 个永久对象也是合法结果，不会被当成管线失败。

`prediction/` 仅包含 `source_memory.json`、`d9_result.json`、`object_memory.json` 和 `run_manifest.json`，不含人工标签、F1 或失败案例；`evaluation/` 才包含 `pair_labels.json`、`d9_evaluation.json` 和评测 `run_manifest.json`。真实 D8 的 10 条观测对应全部 45 个 pair，其中 17 个同实例正对和 28 个不同实例负对，独立评测的 precision/recall/F1 为 `1.0`。这只是小型人工开发样例的回归基线，不是 held-out 测试，也不代表跨场景泛化。


## D11–D15 GPU 补验（2026-08-29）

无卡阶段冻结的源码和文件合同现已使用真实 RTX 4090 outcome 做完增量补验。最终总验收必须在包含 OpenCV 的 `vggt_geom` 环境运行：

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/vggt_geom \
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 \
  python -m scripts.validate_gpu_acceptance \
  runs/gpu-acceptance-20260829
```

总报告为 `status=PASS`、`gpu_acceptance=COMPLETE`，范围固定为
`ENGINEERING_REPLAY_NO_NEW_MANUAL_LABELS`。其中 D3/D5/D6/D13 实际执行了
VGGT、PE 或 SAM 3 GPU 推理；D7/D8/D11/D12/D14/D15 是基于这些真实 GPU
outcome 的 CPU 组装或确定性重放，不能写成策略算法本身在 GPU 上运行。

- D11 的 8 个候选全部变为 `available`，共 21 条 observation 和 8 条明确拒绝；原有 4 条 available outcome 内容不变，新增物化 `0061/0031/0011/0051`。PE embedding 二进制仍标记为 `not_retained`。
- D12 只运行无标签 A2 prediction，未运行新的人工标签 evaluator，因此没有新增 F1 数字。
- D13 的 Q0 Top-1 仍是 `frame_0001`，`B0-official` 与 `B1-robust-single-view` 均通过真实单视角 GPU 验收；Q0 仍只称 `upstream-aligned`。
- D14 在完整 cache 上通过 Q1 prediction replay，K=1 与 Q0 一致；没有新增 recall 结论。
- D15 从原先的 `BLOCKED_MISSING_OUTCOME` 变为完整 trace `PASS`，依次选择 `0001/0071/0041/0061/0031`，以 `max_budget_reached` 停止，`new_observation_count=14`；schema 0.1 仍使用 legacy JSON 字段 `observed_gain`，`performance_claim` 保持 `null`。

本地 bundle 内少数阶段 manifest 保留的是产物创建当时的
`GPU_ACCEPTANCE_PENDING` 快照；最终增量总报告和上述重算 validator 才是
本次补验状态。完整 mask、点云和视频仍是忽略的本地产物；公开轻量包保留
完整 D15 trace/cache、去绝对路径的 GPU report、D15.5 validator report、
manifest/audit/result table 引用与一个 530 KB 缩略图：

```bash
python -m scripts.validate_public_gpu_evidence
```

公开包位于 `evidence/week3/d15-gpu-public`。缩略图是
`configs/evidence_policy.json` 中唯一 hash-pinned PNG 例外；MP4/PLY 仅保存
大小和 SHA-256，准备作为可选 GitHub Release，不进入 Git。

## D12 A2 complete-link association（旧 policy ID 保持兼容）

D12 不修改 A1；已发布的 A1 prediction/result 和 ObjectMemory SHA-256 继续由回归测试固定。通用实现是独立的 label-free predictor：

- 双方都有 embedding 时使用 cosine，否则退化为规范化 exact-class；
- 每条观测质量为 retrieval、SAM、valid-point ratio 的几何均值，pair 取两者较小值，门槛固定为 `0.25`；
- 空间门槛固定为中心距离不超过 `0.15`，或 AABB IoU 大于 `0`；
- pair score 按 semantic/center/overlap/OBB-shape/quality 的 `0.25/0.25/0.20/0.15/0.15` 加权；
- cluster 只有所有跨 cluster pair 均通过 gate 才能合并。相同帧重复 mask 可以聚类，但永久对象仍要求至少两个不同帧。

Clio 的最终 18+18 task 运行中，`semantic_embedding` 覆盖为 `0/35` 与 `0/181`，且同 task class text 恒定。因此 Clio A2 的实际名称是“任务内几何＋质量 complete-link”，不作多模态语义关联声明。

无标签预测和独立评测命令：

```bash
python -m scripts.run_a2_association \
  --memory evidence/week1/runs/office-loop-mv-d8-trash-can/object_memory.json \
  --output-dir runs/office-loop-mv-d12-a2-trash-can/prediction

python -m scripts.validate_a2_association \
  runs/office-loop-mv-d12-a2-trash-can/prediction

python -m scripts.evaluate_a2_association \
  --prediction-dir runs/office-loop-mv-d12-a2-trash-can/prediction \
  --labels configs/d9_office_loop_trash_can_labels.json \
  --output-dir runs/office-loop-mv-d12-a2-trash-can/evaluation

python -m scripts.validate_a2_evaluation \
  runs/office-loop-mv-d12-a2-trash-can/evaluation
```

历史 D8 CPU 重放保存了 45 个完整 pair 特征和 7 次 complete-link merge，最终仍为 3 个 cluster、1 个永久对象和 4 条 pending。独立开发评测仍是 17/28 个正/负 pair、precision/recall/F1=`1.0`，与 A1 持平；桥接修复的增益只由合成 A–B–C 回归证明，不能用这个无桥接真实样例宣称性能提升。阈值在评测前冻结，评测分数不会反向改变 prediction 状态。2026-08-29 又在完整真实 GPU outcome cache 上通过无标签 prediction 重放，但未运行新 evaluator，因此状态为 `GPU_MATERIALIZATION_COMPLETE / CPU_REPLAY_PASS`，不新增 F1 结论。


## D13 Q0 upstream-aligned Top-1 协议（GPU 单视角补验完成）

正式名称固定为 `Q0-vggt-slam-upstream-top1`，状态固定为 `upstream-aligned`。这里的 `B0-official` 只保留为本仓库已审计的 VGGT-SLAM 单视角 lifting 标签；Q0 不称 FOUND-IT 官方实现，也不称 VGGT-SLAM 交互流程的逐字节官方复现。

冻结协议为：

- PE cosine 使用上游零分下界/本地 `[0,1]` clip 后选择 Top-1，同分保持 geometry 原始顺序；
- 图像按 VGGT crop 模式把宽缩放到 518，按 14 对齐高度，必要时中心裁高，并以白色对 batch 做对称 padding；
- SAM 3 threshold 固定为 `0.5`，输入是 VGGT 实际保存的预处理帧；
- mask 必须直接索引同网格 3D 点，只剔除 NaN/Inf；
- OBB 使用普通 covariance PCA；
- 禁止 confidence gate、MAD、minimum-point gate、robust PCA 和 SAM 后 mask resize。

静态冻结与验收：

```bash
python -m scripts.freeze_q0_protocol \
  --output-dir evidence/week2/d13-q0-protocol

python -m scripts.validate_q0_protocol \
  evidence/week2/d13-q0-protocol
```

validator 对九个固定源码文件做 SHA-256 和 10 项语义检查，并核对 retained D4/D5 JSON。当前 10/10 检查通过，D5 `trash can` 的 Q0 和 raw rank-1 都是 `frame_0001`。轻量发布曾按政策删除 D4 的 `masks.json` 和 `preview.png`，所以旧严格 D4 validator 当前只因这两个文件缺失而 FAIL；历史保存的 validator report 是 PASS。D13 将两种状态同时写入 limitation，未伪造文件或把缺口包装成通过。2026-08-29 的真实 Q0 单视角补验中 B0/B1 均通过，状态为 `GPU_SINGLE_VIEW_PASS`。

## D14 Q1 Fixed Top-K 预算曲线（完整 GPU outcome 重放通过）

`Q1-fixed-topk-hybrid` 固定使用 D5 已验收参数：`K=1/3/5`、`min_frame_gap=2`、`min_camera_distance=0.15`、`min_view_angle_deg=3.0`。策略只接收 rank、frame、pose 和 retrieval score 等候选 metadata；帧被选中后才揭示对应 cached outcome，人工 instance label 只进入独立 evaluator。

```bash
python -m scripts.run_fixed_topk_replay \
  --cache evidence/week2/d11-candidate-cache/candidate_cache.json \
  --output-dir runs/d14-fixed-topk --prefix real

python -m scripts.evaluate_fixed_topk_replay \
  --prediction runs/d14-fixed-topk/real_prediction.json \
  --cache evidence/week2/d11-candidate-cache/candidate_cache.json \
  --labels configs/d9_office_loop_trash_can_labels.json \
  --output-dir runs/d14-fixed-topk --prefix real
```

历史 office-loop 开发重放在请求预算 `1/3/5` 时分别选择 `1/3/4` 帧；`K=5` 因 hybrid 去冗余后候选耗尽而明确记录 `nonredundant_candidates_exhausted`，不是把未物化帧当作零检测。四个选中帧均有历史 outcome，`K=1` 与 Q0 都是 `frame_0001`。独立 evaluator 的 observed-instance recall 为 `0.667/1.0/1.0`，仅表示原单查询 development replay，不是 held-out 性能。2026-08-29 的完整真实 GPU outcome cache prediction replay 同样选择 `1/3/4` 帧并保持 K=1=Q0，但没有运行新 evaluator；状态为 `GPU_MATERIALIZATION_COMPLETE / CPU_REPLAY_PASS`。

```bash
python -m scripts.validate_d14 evidence/week2/d14-fixed-topk
```

## D15 Q2 retrieval＋pose-novelty sequential search（完整重放通过）

Q2 首步只按 retrieval score 选择，因此 budget=1 严格退化为 Q0。后续候选使用
`0.65 × candidate-universe min-max retrieval + 0.35 × pose novelty`；pose
novelty 是最小平移按 `0.15 m` 截断归一化与最小视角差按 `3°` 截断归一化
的均值。同分按 D5 原始 rank 和 frame ID 确定性排序。历史 policy ID
`Q2-gain-based-sequential-search` 仅为 schema 0.1 兼容名。

```bash
python -m scripts.run_sequential_search \
  --cache evidence/week2/d11-candidate-cache/candidate_cache.json \
  --output-dir runs/d15-sequential-search --prefix real \
  --allow-blocked
```

已发布的无卡 partial-cache trace 先揭示 `frame_0001/0071/0041` 的 8 条新 3D observation；第 4 步按 metadata 选择 `frame_0061` 后才发现 outcome 未物化，于是立即返回 `BLOCKED_MISSING_OUTCOME`，不偷看或跳过候选。它保留为策略 readiness 的历史证据。

complete synthetic cache 跑满五步并以 `max_budget_reached` 停止。Q0/Q1/Q2
同预算文件只比较 selected frames、SAM calls、lifted/rejected counts；当前
synthetic fixture 上 Q1 与 Q2 同序，既不代表提升，也不使用实例标签。
`new_observation_count` 仅指此前未出现的 frame-scoped observation ID 数；
legacy JSON 字段名为 `observed_gain`。它不是对象数、空间覆盖、instance
recall，也不参与下一候选排序；当前停止规则实质是连续两个空 observation 集。
2026-08-29 的完整真实 cache 重放跑满 5 步，`new_observation_count=14`，
`performance_claim=null`，不声称相对 Q0/Q1 有性能提升。


```bash
python -m scripts.validate_d15 evidence/week2/d15-sequential-search
```
python -m scripts.validate_public_gpu_evidence


D15 validator 对结构和离散字段保持严格比较，对有限浮点使用 `1e-12`
容差；1 ULP 跨环境漂移通过，明显数值篡改和 NaN/Inf 仍失败。

## D15.5 长轨迹场景记忆可视化（已验收）

D15.5 是进入 D16 前插入的展示与证据审计里程碑，不改变 Q0/Q1/Q2 或 A2
算法定义。真实 office-loop 长序列按 stride 5 导出 95 帧几何，覆盖转弯与返回；
随后从 95 帧做 Top-24 检索、SAM 3 与 Robust3DLifter，得到 49 个 SAM 实例、
40 条有效 3D observation 和 9 条明确拒绝。A2 将 40 条观测组成 13 个 cluster，
其中 8 个满足跨帧支持并成为预测对象。

```bash
OMP_NUM_THREADS=8 \
conda run --no-capture-output -p /root/autodl-tmp/envs/vggt_geom \
  python -m scripts.visualize_scene_memory \
  --geometry runs/office-loop-d15_5-s5/geometry.npz \
  --geometry-manifest runs/office-loop-d15_5-s5/geometry.manifest.json \
  --anchor-poses runs/office-loop-d15_5-s5/geometry.anchor_poses.json \
  --memory runs/office-loop-d15_5-s5/d12-trash-can-k24/object_memory.json \
  --observation-root runs/office-loop-d15_5-s5/d7-trash-can-k24 \
  --output-dir runs/office-loop-d15_5-s5/d15_5-trash-can-k24 \
  --max-background-points 60000 --video-seconds 10 --video-fps 12

conda run --no-capture-output -p /root/autodl-tmp/envs/vggt_geom \
  python -m scripts.validate_d15_5_visualization \
  runs/office-loop-d15_5-s5/d15_5-trash-can-k24 \
  --report evidence/week3/d15-gpu-public/d15_5_validation.json
```

独立 validator 的结果为 `PASS`：三视图 PNG 为 2931×1010；MP4 为 10 秒、
12 FPS、120 帧，动态帧比例为 1.0；彩色 PLY 有 132,204 个顶点。生成器在写出前检查点与颜色的有限性，validator 核对文件 hash、PLY header 和顶点数。对象中心审计
没有把整段轨迹跨度冒充每个对象的证据：8 个对象中 3 个为
`STRICT_MULTIVIEW`，5 个为 `DIAGNOSTIC_PARALLAX`。严格门槛要求至少
3 个不同观测帧，并且至少 2 个同一帧对同时满足 object-ray angle ≥15° 与
baseline/mean-depth ≥0.20，且覆盖至少 3 帧。

`scene_memory.ply` 是 binary little-endian 彩色 PLY。下载到本地后可直接用
CloudCompare 或 MeshLab 打开；普通文本预览无法显示二进制点云。云服务器上推荐
查看完整 D15.5 Viser 场景，它除 PLY 外还显示红色相机轨迹、20 个选中相机、
40 条 observation cloud/OBB、8 个融合对象 OBB 与标签：

```bash
cd /root/autodl-tmp/VGGT-RelMem
OMP_NUM_THREADS=8 \
conda run --no-capture-output -p /root/autodl-tmp/envs/vggt_geom \
  python -m scripts.visualize_scene_memory \
  --geometry runs/office-loop-d15_5-s5/geometry.npz \
  --geometry-manifest runs/office-loop-d15_5-s5/geometry.manifest.json \
  --anchor-poses runs/office-loop-d15_5-s5/geometry.anchor_poses.json \
  --memory runs/office-loop-d15_5-s5/d12-trash-can-k24/object_memory.json \
  --observation-root runs/office-loop-d15_5-s5/d7-trash-can-k24 \
  --output-dir /tmp/vggt-relmem-d15_5-view \
  --max-background-points 60000 --video-seconds 1 --video-fps 1 \
  --serve --host 127.0.0.1 --port 8080
```

第一行 `cd /root/autodl-tmp/VGGT-RelMem` 是查看命令的一部分；后续相对路径
都按当前工作目录解析。若从其他目录执行，viewer 会明确提示先进入仓库根目录或
为每个输入使用绝对路径，不再只返回难以判断的 `FileNotFoundError`。

保持终端运行，在 VS Code 的 **Ports/端口** 面板转发 `8080`，再打开本地地址；
`Ctrl-C` 结束。该命令会重建显示数据并生成最短预览，但不会重新运行
VGGT、PE 或 SAM 3。

固定上游 VGGT-SLAM 2.0 提供 Viser 增量地图、真实相机轨迹 walkthrough、
Top-1 单帧 OBB 和 PCD 导出；当前 checkout 没有内置 MP4 导出器。D15.5 的新增
工程工作量是把 Top-K 多帧观测、A2 对象记忆、轨迹和对象中心视差证据组织成可落盘
的 PLY/MP4/JSON/Viser 包，并提供 hash manifest、独立 validator 和反例测试。
单纯的 360° Matplotlib 编码只是展示层，不称算法创新；视频相机绕最终静态场景
旋转，不是机器人真实轨迹，也不代表物体被真实 360° 观测。当前长序列关闭了回环，
因此不声称消除漂移或提升上游几何精度。

如果需要开发安装：

```bash
python -m pip install -e '.[dev]'
```

Demo 使用两个 chair 和一个 desk 的合成 ObjectMemory，查询协议为：

```json
{
  "query_id": "demo_left_chair",
  "target": "chair",
  "relation": "left_of",
  "reference": "desk",
  "anchor_frame": "frame_0001"
}
```

输出包含 `ranked_ids`、关系分数、证据帧、置信度、解释和拒答原因。

## 两环境交换协议

`vggt_geom` 环境的新 schema 0.2 NPZ 包含：

- `frame_ids`: `(N,)` 字符串；
- `point_maps`: `(N,H,W,3)`；
- `raw_confidence_maps`: `(N,H,W)`，未修改的 VGGT confidence；
- `valid_masks`: `(N,H,W)` 布尔数组，上游 percentile threshold 的结果；
- `confidence_maps`: `(N,H,W)`，`valid_masks` 的 float32 兼容别名；
- `world_from_camera`: `(N,4,4)`。

旧 schema 0.1 文件仍可读取；它没有 raw confidence，loader 只能从二值 `confidence_maps` 恢复 `valid_masks`。因此 Robust3DLifter 中对这类文件使用的 `confidence_threshold=0.5` 本质是有效点开关，不是对原始 VGGT confidence 再做概率阈值。

`open_vocab` 环境导出 mask manifest JSON。每条记录包含 `obs_id`、`frame_id`、`class_text`、`mask_ref`、`retrieval_score`、`sam_score`，mask 本体保存为 NPY 或包含 `mask` 数组的 NPZ。所有加载都设置 `allow_pickle=False`。

准备好这两份文件后运行：

```bash
python -m scripts.reproduce \
  --geometry data/apartment/geometry.npz \
  --masks data/apartment/masks.json \
  --output runs/apartment-dev \
  --config configs/default.yaml \
  --split apartment-dev
```

输出包括逐观测点云、`observations.json`、`object_memory.json`、提升失败列表和 `manifest.json`。

冻结 ObjectMemory 后，可用 JSONL 结构化查询评测：

```bash
python -m scripts.evaluate \
  --memory runs/apartment-dev/object_memory.json \
  --queries configs/queries.example.jsonl \
  --anchor-poses data/apartment/anchor_poses.json \
  --output runs/apartment-dev/evaluation.json
```

`anchor_poses.json` 的值是 `world_from_anchor` 4×4 矩阵。坐标约定为 anchor 的 `+x` 向右、`+y` 向上、`+z` 向前；方向关系缺少 anchor 时系统会明确拒答，不会静默使用世界轴。

## 当前边界与下一步

- 已实现：确定性 top-K 去冗余、置信过滤/MAD 离群剔除/PCA OBB、任务内几何＋质量 complete-link、证据融合、JSON round-trip、左右/前后关系、逻辑回归校准和选择性预测指标；通用关联接口可接收 embedding，但 Clio run 未保存。
- D3 状态：0.1 的两次真实几何产物已通过且哈希一致；0.2 raw confidence/valid mask 源码、单测及一次真实 RTX 4090 产物均已通过验收。
- 已完成 D4 修正：历史误标 B0 已明确归为 legacy B1；严格 `B0-official` 与 `B1-robust-single-view` 共用一次 PE/SAM 和同一 VGGT 网格 mask，真实对照与严格 validator 均通过。
- 已完成 D5：temporal 与间隔序列 hybrid 都有真实 GPU K=1/3/5 产物；六个 hybrid 查询重新通过独立 validator。
- 已完成 D6 修正：`B2-topk-multiframe` 已移除 SAM 后 mask resize；4 帧真实受控重跑得到 21/15/6 个 SAM/有效/拒绝实例，validator 为 `PASS`。
- 已完成 D7：stride 图像按 geometry manifest 精确解析；新多视角缓存含 10 条观测，40 秒动态视频通过独立验收。
- 查询级结论：`trash can` 有 3 对观测帧通过同帧对门槛；`poster`、`blue recycling bin`、`printer` 仅为多帧证据；`dog/bed` 是零证据负对照。
- 已完成 D8：新多视角 D7 已生成可移动的相对路径 ObjectMemory；真实产物为 10 pending、0 永久对象、0 关联决策。
- 已完成 D9：无标签预测得到 3 个空间组件，仅跨 3 帧的 6-observation 组件成为永久对象；独立开发集评测的 45 个人工标注 pair 达到 precision/recall/F1=1.0。
- 已完成 D10：D9 预测 API 已去除标签依赖，预测与评测产物/状态完全分离，并补上确定性、顺序不变、bridge 已知失败和零匹配合法性回归验收。
- 已完成 D11：候选协议仍保留可审计的历史 4+4 partial evidence；GPU 补验已把同一 8 候选 cache 完整物化为 8/8 available，原有 outcome 内容和 raw ranking 保持不变。embedding 二进制仍不保留。
- 已完成 D12：A2 complete-link、证据记录和反桥接测试保持冻结；Clio 输入审计确认没有 semantic embedding，相关结果只按任务内几何＋质量关联解释。
- 已完成 D13：Q0 继续保持 `upstream-aligned`；10/10 静态语义检查与真实 `frame_0001` B0/B1 单视角 GPU 补验均通过，仍不称 FOUND-IT 官方复现。
- 已完成 D14：Q1 在完整真实 GPU outcome cache 上按 metadata 先选后揭示，K=1 与 Q0 一致；本次只验收 prediction replay，不新增 recall 数字。
- D15/Q2 归入诊断：retrieval＋pose-novelty trace 可重算，但 `coverage_aware=false`、`performance_claim=null`，不进入求职版主贡献或最终 Clio 主表。
- 已完成 D15.5：95 帧长轨迹生成可审计的场景级 RGB 点云、轨迹、Top-K 视角、对象记忆、PLY/MP4/Viser；3/8 个预测对象满足严格对象中心多视角门槛。
- 已完成 D16＋GPU 数据验收：`apartment=development` 与 `cubicle=fixed-confirmatory` 均已本地物化和对齐；192/172 帧几何、18+18 官方 task 分母及 evaluator-only Sim(3) 通过重放。
- 已完成 D17 严格重算：Apartment/Cubicle 分别为 136+136 与 149+149 个方向正负查询；正例同时要求 target/reference 命中 GT，负例同时报告端到端拒答、原因命中和双端定位后的关系拒答；0.60 阈值仍明确未校准。
- 已完成 D18＋Clio 完整场景补验：冻结 Q1F 在 Cubicle 的严格 Acc@1 为 27.78%→38.89%；多 GT 指标按任一 OBB containment 重算，A1/A2 均按最终聚类同簇关系计分，A2 F1 仍低于 A1。
- 已完成 D19＋Clio 工程消融：Q2 的 pose-novelty/gain-patience 与通用 A2 组件变体均可确定性重放；Clio 最终关联没有 informative semantic input，真实标签查询策略指标仍待后续。
- 已完成 D20 CPU/source：单个 CPU 命令可重跑 D16–D19 validator、从规范 JSON 重建 Q×A/关系/消融表、核对 README 数字，并在移动目录中逐字节复验 retained outputs。
- 已完成 D21＋post-D21 Clio 结果集成：轻量最终摘要绑定本地完整报告 hash、冻结关系配置、样本量、分母和适用边界；Cubicle 不再是未运行状态。
- 仍未完成的是独立真实 calibration、统计置信区间、带标签查询策略消融、Instance Recall/Duplicate Rate/Count Error、整条最终延迟/峰值显存和约 3 分钟配音录屏；FOUND-IT 同条件对照明确不在项目范围，这些缺口不被写成已完成贡献。
- GT depth、pose、OBB 只应进入 evaluator 或 geometry oracle，不得进入主推理输入。
- 当前是目标定位感知前端，不包含路径规划、控制或闭环导航，因此不称“完整导航系统”。
