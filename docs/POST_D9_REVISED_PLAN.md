# VGGT-RelMem：D9 后修订路线（D10–D21）

> 状态：工作计划，不是实验结论；实现与验收状态更新至 2026-08-29。
> 修订原因：VGGT-SLAM 2.0 的上游说明已经指向 FOUND-IT 所代表的可查询视觉记忆方向，因此本项目不再把“开放词汇检索、SAM 分割、3D 提升、对象记忆”本身包装成首创。后续价值集中在**可审计复现、预算受控的主动查询、可靠对象关联、关系定位与拒答评测**。

## 1. 项目重新定位

### 1.1 最终应该是什么

VGGT-RelMem 是建立在 VGGT-SLAM 2.0 几何输出之上的**可复现、可评测的语义定位可靠性层**。它回答的是：

1. 在相同候选帧和 SAM 预算下，应查哪几帧、何时停止，才能获得更多有效实例证据；
2. 多帧观测怎样在不使用真值的情况下合并成对象，同时避免重复、碎片化和错误桥接；
3. 对“某物在某物左边/前方”等查询，怎样给出证据、置信度和可靠拒答；
4. 每一个性能变化能否被归因到检索策略（Q 轴）或关联策略（A 轴），而不是由输入、预算或标签泄漏造成。

当前项目仍是语义导航的**感知与目标定位前端**，不包含路径规划、运动控制和闭环导航，因此最终也不应写成“完整导航系统”。

### 1.2 三类实现必须分清

| 类别 | 含义 | 命名和宣传边界 |
| --- | --- | --- |
| 上游官方复现 | 严格按公开上游代码、输入预处理、阈值和输出规则运行；版本和 commit 固定 | 只有逐项审计通过后才能写 `official`；任何本地稳健过滤都不能混入 |
| 论文/说明启发实现 | 根据 FOUND-IT 的论文或上游说明实现相近思想，但没有完整官方源码、配置或评测协议可逐行核对 | 写 `paper-inspired` 或“受 FOUND-IT 启发”，不得写“复现 FOUND-IT”或“优于 FOUND-IT” |
| 本仓库历史实现 | 已有 Top-K、SAM 3、Robust3DLifter、D7–D9 ObjectMemory 等本地代码 | 统一写 `legacy`/`ours` 并保留来源、差异和兼容路径，不回写成官方基线 |

FOUND-IT 在本项目中的角色首先是**相关工作与设计参照**。在没有获得其完整可执行实现、固定数据协议和同预算结果之前，不能把它当成本仓库已经复现的 baseline，也不能声称本项目替代或超过它。

## 2. 两条实验轴：查询策略 Q 与关联策略 A

查询策略与对象关联必须分开比较。不得用更强的 A 策略掩盖 Q 策略没有带来额外证据，也不得让不同 Q 使用不同 SAM 阈值、候选帧或几何输入。

### 2.1 查询策略矩阵

| 编号 | 名称 | 选择规则 | 预算 | 严格控制项 | 当前状态 |
| --- | --- | --- | --- | --- | --- |
| Q0 | `official-top1` | PE 全序列排序后只取 Top-1；使用上游预处理图像、SAM 直接取点和普通 PCA OBB | 1 次候选帧 SAM | 相同 PE 排序、SAM 权重/阈值、VGGT 几何、候选全集 | D13 已冻结为 `upstream-aligned`；2026-08-29 B0/B1 单视角 GPU 补验通过 |
| Q1 | `fixed-topk` | 按固定 K 和预先声明的时序/视角去冗余规则选择 | K 次或至候选耗尽 | 与 Q0/Q2 共用候选结果缓存；报告 K=1/3/5 等完整预算曲线 | D14 已实现；完整真实 GPU outcome cache 的无标签 CPU replay 通过 |
| Q2 | `gain-based-sequential` | 每步根据尚未使用的可观测信号预测边际收益，选择下一候选；达到预算或收益阈值即停止 | 最大 B 次，实际次数可小于 B | 策略不得读取真值、人工标签或未来候选的实际成败；必须保存逐步 policy trace | D15 已实现；完整真实 GPU outcome cache trace 通过，尚无性能提升结论 |

Q2 暂时只称**基于收益的序贯搜索**。当前 geometry schema 0.2 没有可审计的相机内参和完整 camera model，无法精确投影视锥、判断可见体积，因此在补齐内参及坐标约定并通过投影单测前，不称“精确 frustum search”或“视锥覆盖算法”。相机位姿新颖度可以作为近似特征，但不能冒充几何可见性。

### 2.2 关联策略矩阵

| 编号 | 名称 | 定义 | 用途 |
| --- | --- | --- | --- |
| A1 | `legacy-spatial-components` | 归一化后的类别严格相等，且中心距离不超过阈值或 AABB 有重叠；对无向图取连通分量 | 冻结 D9 回归基线，不再悄悄增加语义、置信或 OBB 特征 |
| A2 | `evidence-aware-association` | 显式使用语义一致性、OBB/中心几何、观测质量、跨帧支持和拒绝原因；每次 merge 保存可重算证据 | D12 起的新方法，独立于 Q0/Q1/Q2 比较 |

A1 的单链接连通分量存在已知“桥接”风险：A–B、B–C 通过但 A–C 不通过时，三个观测仍会进入同一分量。D10 用测试把该行为固定为**已知失败案例**，不能在 D9 结果里静默修复；D12 的 A2 再用完整链接、一致性约束或显式冲突拒绝解决。

最终主表至少应是 Q×A 的因子实验，而不是一串逐日版本：

| 查询策略 | A1 legacy | A2 evidence-aware |
| --- | --- | --- |
| Q0 official Top-1 | 必做 | 必做 |
| Q1 fixed Top-K | 必做 | 必做 |
| Q2 gain-based sequential | 可做诊断 | 主结果 |

## 3. D10：先拆预测与评测（今天）

D10 的目的不是增加模型能力，而是建立后续所有数字都可信的实验边界。

### 3.1 必做事项

1. **无标签预测**：`run_d9_association` 只读取 D8 ObjectMemory 和 A1 配置，输出 pair 预测、组件、永久对象、pending 观测及 merge 证据；命令和结果中均不要求人工标签。
2. **独立评测**：单独命令读取冻结的预测结果与人工 pair 标签，计算 precision/recall/F1/accuracy 和失败案例；标签文件、标签哈希和评测指标只出现在 evaluation bundle。
3. **独立 validator**：prediction validator 只检查守恒、确定性、证据可重算、路径安全和 JSON round-trip；evaluation validator 才检查标签覆盖、指标重算和篡改。
4. **反泄漏测试**：改变标签不能改变任何预测字节；输入 observation 顺序打乱不能改变规范化预测；零匹配/零永久对象是合法结果；桥接案例作为 A1 已知失败被明确测试。
5. **轻量证据**：公开 self-contained 的 JSON/Markdown bundle，包含冻结 D8 输入副本、prediction、evaluation、manifest 和 validator report；不提交权重、点云、mask、图片或视频。

### 3.2 D10 验收门槛

- 无 `--labels` 也能生成完整 prediction bundle；
- prediction 产物不含 `expected_same`、`error_type`、F1、failure cases 或标签哈希；
- 同一输入在标签变化前后预测文件 SHA-256 一致；
- 输入排列变化后规范 JSON 一致；
- prediction/evaluation 两套 validator 均为 `PASS`；
- 全量 CPU 单测通过，Git 中新增 evidence 只含 `.json`/`.md`；
- 现有 10 条 office-loop 观测仍产生 45 个 pair；已测得的 F1=`1.0` 只写为**开发样例回归基线**，不写成 held-out 性能。

### 3.3 D10 产物与算力

- 产物：`d9_result.json`（预测）、`object_memory.json`、prediction manifest、`d9_evaluation.json`、冻结标签副本、evaluation manifest、两份 validation report；
- 算力：CPU；不加载 VGGT、PE 或 SAM 3，不需要 GPU；
- 不在 D10 下载 Clio，也不实现 Q2。

## 4. D11–D21 逐日执行计划

每一天的“门槛”是进入下一项实验的必要条件；若门槛未过，优先修复，不把失败状态包装成“已完成”。实际运行时间可因云实例和数据下载调整，但任务顺序不变。

### D11：冻结候选全集与 Candidate Outcome Cache

**任务**

- 设计候选结果缓存，使 Q0/Q1/Q2 在同一个查询上复用完全相同的 PE 排序、SAM 推理和 3D lifting 结果；
- 将策略决策（policy trace）与候选真实运行结果分开，避免 Q2 “偷看”尚未选择帧的实际成败；
- 建立 Visual Memory manifest：帧 ID、图像哈希、位姿、PE embedding 行号和源 commit 可追溯；大 embedding 文件保持忽略。

**门槛**

- 候选覆盖全集，不只保存被某个策略选中的成功样本；
- 相同候选在不同 Q 策略下的 outcome hash 一致；
- cache 中不含人工 instance label、query answer 或 test metric；
- CPU schema/round-trip/篡改单测通过。

**产物与算力**：schema、读写器、validator、合成缓存和 manifest 使用 CPU；真实候选缓存需要一次 GPU PE/SAM 运行，可推迟到有卡时补齐。

### D12：实现 A2 evidence-aware association

**任务**

- 在不改变 A1 的前提下新增 A2；
- 特征至少分组记录：类别/语义一致性、中心与 OBB 几何、跨帧支持、retrieval/SAM/lifting 质量；
- 合并必须保存阈值、特征值、通过/拒绝原因和组件冲突；
- 显式处理 A1 的桥接案例，并保留“证据不足不合并”。

**门槛**

- A1 输出与 D10 冻结哈希一致；
- A2 在合成桥接、类别冲突、低质量观测和顺序置换测试中行为确定；
- 所有 merge 均能由保存证据独立重算；
- 关联失败不会丢失 observation，守恒检查通过。

**产物与算力**：A2 配置、预测结果、关联审计报告；CPU 为主，不需要重新跑视觉模型。

### D13：冻结 Q0 official Top-1 正式协议

**任务**

- 对照固定上游 commit，逐项核对图像预处理、Top-1 选择、SAM 阈值、mask 网格、直接取点和 PCA OBB；
- 建立 Q0 provenance manifest，并列出与 legacy Robust3DLifter 的每一项差异；
- 若任何官方细节无法核对，将该实现降级命名为 `paper-aligned-top1`，不强行保留 official 标签。

**门槛**

- Top-1 与 PE 排序首项严格一致；
- 无 mask resize、MAD、confidence gate 或最小点数等本地逻辑混入 Q0；
- 同一输入重复运行的轻量结果一致；
- validator 能发现预处理、阈值或帧号篡改。

**产物与算力**：Q0 审计清单、manifest、验证报告；真实验收需要 GPU，协议与静态检查可先在 CPU 完成。

### D14：Q1 fixed Top-K 同预算曲线

**任务**

- 从 D11 的候选全集生成 K=1/3/5（或数据允许的预注册 K）结果；
- 固定 temporal/viewpoint/hybrid 中实际使用的一种正式去冗余策略，其他策略只做消融；
- 对每个预算报告有效实例证据、重复率、失败原因、SAM 调用数和耗时。

**门槛**

- K=1 与 Q0 的候选选择一致；若 lifting 定义不同，必须分列而不是比较成同一个 baseline；
- K 的增加保持候选前缀或明确记录去冗余导致的替换；
- 所有 K 复用相同 candidate outcomes，不按结果重新选帧；
- 预算耗尽和候选耗尽明确区分。

**产物与算力**：Q1 budget curve JSON/表格和消融报告；缓存齐全后 CPU 可重放，首次生成 candidate outcomes 需要 GPU。

### D15：Q2 gain-based sequential search

**任务**

- 定义每一步可用的 policy state：PE 分数、已选帧位姿新颖度、历史可见成功/失败、已有对象覆盖和预测观测质量；
- 定义预注册的边际收益与停止规则；
- 输出逐步 candidate scores、选择、停止原因和累计成本；
- 和 Q0/Q1 在相同最大 SAM 预算 B 下比较。

**门槛**

- policy 不能读取未选择候选的 SAM/lifting outcome，也不能读取 GT；
- policy trace 可在 candidate cache 上确定性重放；
- B=1 时有清晰退化行为；
- 无内参时文档、变量和结果均不出现“精确 frustum”结论。

**产物与算力**：Q2 policy、trace、预算曲线；策略开发/重放使用 CPU，补充候选 outcome 时使用 GPU。

### D15.5：长轨迹场景记忆可视化与对象中心证据审计

D15.5 是 2026-08-29 插入 D16 前的工程验收里程碑，不新增查询策略 Q 或关联
策略 A。它使用真实长轨迹 GPU 产物，把最终 RGB 地图、相机轨迹、Top-K 选中视角、
逐帧 3D observation、A2 融合对象和证据等级组织成可复现的场景包。

**已验收结果**

- 95 个相机 anchor、20 个有有效观测的选中相机、40 条 3D observation、8 个预测对象；
- 60,000 个背景采样点与 observation clouds 合成 132,204 点彩色 PLY；
- 2931×1010 三视图 PNG、10 秒/120 帧的动态 360° 虚拟相机环绕 MP4；
- 3 个对象满足 strict object-centric multiview，5 个仅为 diagnostic parallax；
- 生成器从 anchor 与 observation center 计算对象帧对；独立 validator 按已记录的 pair metrics 重新分类，并核对冻结阈值、文件 hash、PNG/MP4 解码、视频动态性以及 PLY header/点数，最终为 `PASS`。

**门槛与边界**

- strict 对象必须至少覆盖 3 帧，并由至少 2 个同一帧对同时满足
  object-ray angle ≥15°、baseline/mean-depth ≥0.20，且覆盖至少 3 帧；
- 原地相机转向、沿同一对象射线平移、同帧重复 mask 和“角度/基线来自不同 pair”
  都有负向单测，不能误升级为严格多视角；
- MP4 是围绕最终静态场景的展示相机，不是机器人真实轨迹，也不是物体真实 360°
  表面覆盖；长序列关闭回环，坐标仍是 reconstruction units；
- 大型 geometry、point cloud、PNG 和 MP4 只保留在本地 `runs/`，Git 只发布源码、
  validator、测试和可恢复说明。

### D16：Clio 可行性、数据协议与分割冻结

**任务**

- 只从官方来源核对下载方式、许可、压缩/解压大小、场景内容、时间同步、坐标约定、相机内参、GT 类型和评价许可；
- 先评估 `apartment` 开发场景，再决定是否下载 `cubicle` held-out 场景；场景命名和用途以实际元数据为准；
- 写清 VGGT 坐标与数据坐标的对齐方式。允许 evaluator 中使用 Sim(3)/GT 对齐，主推理不得读取；
- 冻结场景/query split 和查询清单后再运行正式实验。

**下载门槛**

1. 已记录官方 URL、许可、校验和、压缩大小、预计解压大小和峰值临时空间；
2. 下载前重新执行磁盘审计，必须满足 `可用空间 - 下载/解压峰值 >= 10 GiB`；
3. 当前实例曾记录约 `24.56 GiB` 可用空间，模型权重约 `10.39 GiB`，但这些只是快照，不能据此盲下完整数据；
4. 未知大小时不下载；优先单场景、可恢复、校验和可验证的最小包；
5. 若门槛不满足，先保留 office-loop 工程验收，不删除权重或用户文件换空间。

**产物与算力**：`CLIO_FEASIBILITY.md`、dataset manifest、split manifest、磁盘预算；元数据阶段 CPU/网络，几何预处理和 candidate outcomes 可能需要 GPU。

### D17：关系定位、校准与可靠拒答协议

**任务**

- 冻结 left/right/front/behind 等关系的 anchor 坐标约定和 query schema；
- 正例、歧义例、目标缺失和 reference 缺失分开标注；
- 修正选择性预测评测，使“正确拒答的负样本”不会被当成必然错误；
- 校准器只能在 calibration split 拟合，阈值只能在 development split 选择。

**门槛**

- 缺 anchor、目标或 reference 时有稳定且可解释的拒答原因；
- 左右/前后在坐标变换单测下保持一致；
- Brier/ECE/coverage-risk/AURC 的标签定义由人工小例验证；
- evaluator 可读取 GT，主查询函数和 memory 不可读取。

**产物与算力**：查询集、关系 evaluator、校准 manifest、拒答报告；CPU。

**当前状态（2026-08-29）**：D17 已按 CPU/source 口径完成。正式 prediction
拒绝 query 内嵌答案，labels 只由 evaluator 读取；synthetic 正负例、ECE/AURC、
anchor 旋转和拒答原因均通过重算。真实校准保持 `REAL_DATA_CALIBRATION_PENDING`。

### D18：冻结协议后的开发集与 held-out 运行

**任务**

- 先在开发场景完成 Q×A 全矩阵和预算选择；
- 固定配置、代码 commit、query 文件 hash 和停止规则；
- held-out 场景只运行冻结配置，不因结果回调阈值；
- 若 held-out 数据尚不可用，明确标为 `development-only`，不得用 office-loop 替代 test 宣传。

**门槛**

- 每行结果包含查询数、正负样本数、场景数、预算和置信区间/逐查询明细；
- test 输入 manifest 与开发配置冻结时间可审计；
- 无 test label、metric 或人工答案进入 prediction/candidate cache；
- 一次性 held-out 运行失败只能修工程 bug，修复原因和是否重跑必须记录。

**产物与算力**：开发/held-out 原始 JSON、冻结配置和主结果表；有缓存时评测 CPU，端到端补跑需要 GPU。

### D19：消融、鲁棒性与失败案例审计

**任务**

- Q2 去掉位姿新颖度/历史成功/停止规则的消融；
- A2 去掉语义/OBB/质量权重和桥接约束的消融；
- 检查视角不足、相似物体、错误 SAM mask、尺度漂移和目标缺失；
- 把失败分为检索、分割、lifting、关联、关系判断和拒答六类。

**门槛**

- 每个声称的增益至少有一个只改变单因素的对照；
- 失败案例来自预先冻结查询或完整集合，不挑选“好看样例”；
- 报告均值时同时给样本量、离散程度或置信区间；
- 已知失败不从分母中删除。

**产物与算力**：消融表、失败分类 JSON 和轻量预览索引；主要 CPU，缺失候选时 GPU。

### D20：结果固化、可复现包与轻量发布

**任务**

- 从原始 JSON 自动生成表格，禁止手填最终数字；
- clean clone 只用 tracked JSON/Markdown 就能重算 D9 和正式评测表；
- 权重、数据、embedding、点云、mask、图片、视频仍不进入 Git；大演示产物放可选 Release，并保存 hash；
- 运行所有 validator、单测、路径移动和篡改测试。

**门槛**

- evidence allowlist 只允许 `.json`/`.md`（若确需其他小文本格式，先修改并说明政策）；
- 所有表格能由一个 CPU 命令从 evidence 重建；
- README 数字与 JSON 来源逐项对应；
- 新实例恢复说明不依赖本机绝对路径。

**产物与算力**：`evidence/week2+`、自动表格、复现报告、Release 清单；CPU。

### D21：最终审阅与对外表述

**任务**

- 逐句检查 README、简历和演示材料中的“官方、复现、改进、导航、SOTA”等词；
- 补齐相关工作和上游 attribution；
- 给出可复现实验命令、环境/权重/数据大小和资源成本；
- 冻结最终 commit，并区分已测结果与未来工作。

**门槛**

- 不以 development regression 冒充 held-out 结果；
- 不以 paper-inspired 实现冒充官方 FOUND-IT；
- 不以感知前端冒充闭环导航系统；
- 所有最终数字有 tracked evidence、配置 hash 和样本量。

**产物与算力**：最终 README、结果卡、复现卡和演示脚本；CPU，演示补录可选 GPU。

## 5. Candidate Outcome Cache 最小契约

候选缓存是公平比较 Q0/Q1/Q2 的核心。建议顶层至少包含：

```text
schema_version
scene_id
query_id
query_text
candidate_universe
sources
inference_config
candidates
counts
costs
artifacts
source_commits
created_at
```

其中：

- `candidate_universe`：完整候选帧 ID、有序规则、图像/位姿/geometry hash，不能只保存成功候选；
- `sources`：geometry、image manifest、PE/SAM checkpoint 和输入来源的相对路径及 hash；
- `inference_config`：PE 预处理与排序、SAM 阈值、lifting 版本、随机种子和数值精度；
- `candidates[]`：每帧的 PE rank/score、图像 hash、pose、SAM 结果、lifted observations、显式 rejection、运行时间和显存/调用成本；
- `counts`/`costs`：候选总数、成功/拒绝数、PE 次数、SAM 次数、GPU 秒、峰值显存；
- `artifacts`：大文件只保存可验证引用和 hash，不把二进制嵌入 Git JSON；
- `source_commits`：本仓库与所有上游 commit。

`policy_trace` 应另存，内容只允许包含该步之前已经可见的状态、候选打分、选择和停止原因。候选缓存可以为离线公平重放保存“所有 outcome”，但 Q2 执行器必须通过接口隔离，不能读取未选择候选的 outcome。

Visual Memory 建议把可追踪 manifest 与大 embedding 分开：Git 保存 frame→embedding row、shape/dtype、模型 revision、图像 hash；embedding NPZ/张量继续忽略，并通过 hash 校验。

## 6. 数据分割与防泄漏规则

1. **office-loop**：当前只作为工程开发与回归样例。10 条观测、45 个 pair、F1=`1.0` 可用于证明 D9 实现没被改坏，不能证明跨场景泛化。
2. **训练/校准**：如果没有训练新模型，应明确写“无训练”；若拟合校准器，只能使用独立 calibration split，保存训练样本 ID 和参数。
3. **开发集**：用于选择阈值、K/B、Q2 停止规则和 A2 配置；所有尝试都属于 development。
4. **held-out test**：在协议和配置 hash 冻结后运行。禁止查看 test 标签后调参；若数据没有官方 split，应在查看结果前按场景或 query group 做确定性分割并提交 manifest。
5. **标签隔离**：GT depth/pose/OBB、instance ID、关系答案和 answerability 只能由 evaluator/oracle 路径读取；prediction、ObjectMemory、candidate cache 和 policy trace 不得包含。
6. **对齐隔离**：如需 Sim(3) 或 GT pose 对齐，只能在 evaluator 内计算定位误差，不能把对齐结果反馈给主推理。

## 7. 最终指标体系

所有指标都要同时报告分母、场景数、查询数、预算和逐查询明细，避免一个平均数掩盖空查询或失败运行。

| 层级 | 主指标 | 辅助诊断 |
| --- | --- | --- |
| 查询策略 | Instance Recall@B、Hit/Success@B、达到首个有效证据的平均 SAM 次数 | PE/SAM 调用、GPU 秒、停止原因、候选耗尽率 |
| 2D→3D 观测 | 有效 observation 数/率、目标证据召回 | SAM 拒绝、lifting 拒绝、有限点比例、点数/置信分布 |
| 对象关联 | pair precision/recall/F1、object recall/precision | duplicate rate、fragmentation、错误 merge、桥接失败、pending 比例 |
| 关系定位 | Acc@1、MRR、relation accuracy | 各关系混淆、anchor 缺失、目标/reference 缺失 |
| 可靠拒答 | coverage–risk、AURC、answerability AUROC/AUPRC | Brier、ECE、各拒答原因；先用人工例验证正负定义 |
| 效率与复现 | 相同预算下的效果、运行时间、峰值显存、缓存命中 | 输出 hash、一致性、clean-clone 重算状态 |

只有存在合法 GT 且仅在 evaluator 中使用时，才报告 3D center error、OBB IoU 或定位误差。没有真值时不要用内部置信度代替准确率。

## 8. 三阶段结果写法模板

### 8.1 当前 D9：现在可以写

> 基于固定版本的 VGGT-SLAM 2.0、Perception Encoder 与 SAM 3，独立实现了从 Top-1/Top-K 检索、多帧 2D 分割、稳健 3D lifting 到可追溯 ObjectMemory 的无界面流水线，并将空间关联预测与人工标签评测拆分。当前 office-loop 开发样例包含 10 条观测和 45 个标注 pair，A1 回归基线的 pairwise F1 为 1.0；该数字仅用于小样例工程回归，不代表 held-out 或跨场景性能。

可以把“可复现、证据可追溯、预测/评测隔离”作为成果；不能写“超过 FOUND-IT”“实现完整导航”或“多场景达到 100%”。

### 8.2 D15 左右：中期可以写（数字必须来自届时 JSON）

> 构建了共享 Candidate Outcome Cache，在相同候选全集、视觉模型和最大 SAM 预算下，对比 official Top-1、fixed Top-K 与 gain-based sequential search。序贯策略在 `[开发场景数]` 个场景、`[查询数]` 个预注册查询上，以平均 `[SAM 调用数]` 次获得 Instance Recall@B=`[数值]`；相对 fixed Top-K 的差异为 `[数值及置信区间]`。关联部分单独比较 A1/A2，因此查询收益不由后处理变化混入。

若只有 development 数据，标题和正文必须写 `development result`；若差异不显著，也如实写“未观察到稳定提升”，转而报告效率或失败分析。

### 8.3 D21：最终结果模板（占位符不能提前填）

> 在冻结的 `[development split]` 与 held-out `[test split]` 上，我们以 Q0 official Top-1、Q1 fixed Top-K 和 Q2 gain-based sequential 为查询轴，以 A1 legacy 与 A2 evidence-aware 为关联轴进行同预算比较。Q2+A2 在 test 的 `[N]` 个查询上取得 Instance Recall@`[B]`=`[X]`、duplicate rate=`[Y]`、关系 Acc@1=`[Z]`；在 coverage=`[C]` 时 selective risk=`[R]`，平均使用 `[S]` 次 SAM 调用、耗时 `[T]`。相对最强同预算 baseline 的差异为 `[Δ 与区间]`。所有预测不读取 GT，校准与阈值在 test 前冻结。

简历式写法可压缩为：

> 在 VGGT-SLAM 2.0 几何前端之上实现可审计的开放词汇 3D 目标记忆与预算自适应查询；通过候选 outcome 缓存隔离策略变量，支持关系定位、证据追踪和可靠拒答，并在冻结 held-out 协议上以 `[预算/指标/增益]` 完成同预算验证。

若最终没有 held-out 数据或没有稳定增益，应改写为“完成可复现基准与失败分析”，不要保留上述 test/增益占位句。

## 9. 停止条件与决策原则

- 若 FOUND-IT 官方实现、数据协议和权重后来公开：先做接口/许可/预算审计，再决定接入；不直接替换现有代码，也不回写历史结果。
- 若 Q2 在严格同预算下没有优于 Q1：保留负结果，分析收益预测或候选多样性，不通过换数据/阈值掩盖。
- 若 Clio 不满足许可、空间或同步门槛：暂停下载，继续用 office-loop 做工程回归，并把泛化结论留空。
- 若 A2 只在当前 10-observation 样例变好：不能晋级为主方法，必须等待开发集和 held-out 证据。
- 每次提交只包含能由命令重算的轻量证据；任何宣传性结论都必须晚于 validator 和原始 JSON。

这条路线的核心不是继续扩大功能清单，而是让最终每个数字都能回答三个问题：**与谁比、预算是否相同、提升到底来自查询还是关联。**
