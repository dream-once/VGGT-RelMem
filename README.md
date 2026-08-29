# VGGT-RelMem

**项目定位：VGGT-SLAM 几何之上的可审计语义定位可靠性层。** 本仓库不是完整闭环导航系统；它在固定 VGGT-SLAM 几何之上，提供候选检索、开放词汇分割、3D lifting、对象关联、关系定位、可靠拒答、缓存重放与可审计证据。

项目已完成 D1–D21 工程链路，其中 D16–D21 按 CPU/source 口径验收；固定的 VGGT-SLAM 2.0、Perception Encoder 与 SAM 3 上游源码位于本机忽略目录 `third_party/VGGT-SLAM`，个人代码通过 adapters 和可验证文件契约接入，不修改上游实现。几何推理使用 `vggt_geom`，PE/SAM 3 使用隔离的 `open_vocab` 环境；权重、数据集和大型运行产物不进入 Git。

VGGT-SLAM 2.0 README 中提到的 FOUND-IT 现作为设计参照和后续优先评估的强上游候选；本仓库尚未接入或复现 FOUND-IT 官方代码，也不在缺少同条件实验时宣称优劣。修订后的基线矩阵、验收门槛与 D11–D21 路线见 [D9 后修订计划](docs/POST_D9_REVISED_PLAN.md)。

## 最终结果边界

在没有 Clio held-out 的当前阶段，结论固定为：**完成可复现基准、查询策略与关联策略隔离、可靠拒答协议和失败分析**。office-loop 是开发工程样例，
synthetic fixture 只验证正确性；二者都不支持跨场景性能提升、SOTA、优于
FOUND-IT 或完整闭环导航的结论。

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
tests/                几何、关联、序列化、坐标系、拒答测试
scripts/              reproduce / evaluate / demo
runs/                 manifest、日志和指标（大文件不进 Git）
```

## 快速验证

最低依赖为 Python 3.10+、NumPy 和 PyYAML。当前环境无需安装项目也可在仓库根目录运行：

```bash
python -m unittest discover -s tests -v
python -m scripts.demo --save-memory runs/demo/object_memory.json
```

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
- D12：A1 代码与已发布哈希保持冻结；A2 新增语义/几何/OBB/质量 pair 证据和 complete-link 聚类，完整 GPU outcome cache 上的无标签 prediction 已通过独立重算。
- D13：`Q0-vggt-slam-upstream-top1` 已按固定源码和 retained D4/D5 JSON 冻结为 `upstream-aligned`；不声称 FOUND-IT 官方复现，轻量 D4 的二进制验收缺口已显式记录。
- D14：Q1 Fixed Top-K 已按冻结 hybrid 阈值在完整真实 GPU cache 上通过 prediction replay；K=1 与 Q0 一致，预测仍不读取人工标签。
- D15：Q2 gain-based sequential search 已在完整真实 GPU cache 上跑满 5 步并通过 trace 重算；`performance_claim=null`，不声称真实性能提升。
- D15.5：95 帧长轨迹的场景级 RGB 点云、轨迹、Top-K 相机、40 条 3D 观测、8 个关联对象、PLY/MP4/Viser 与对象中心多视角审计均已完成；独立 validator 为 `PASS`。
- D16：Clio 数据协议、场景角色和 fail-closed 磁盘审计已在 CPU 完成；官方大小和数据许可仍未确认，因此状态为 `DATA_DOWNLOAD_BLOCKED_SIZE_UNKNOWN / DATA_LICENSE_UNVERIFIED`，没有下载数据。
- D17：正式关系预测已与标签/evaluator 隔离，冻结 anchor 坐标约定和 0.60 未校准工程阈值；synthetic 正负查询、ECE/AURC 和拒答重算为 `PASS`，真实数据校准仍为 `REAL_DATA_CALIBRATION_PENDING`。
- D18：Q0/Q1/Q2 × A1/A2 协议已冻结；office-loop 是 development replay，complete synthetic 只验证标签隔离和完整评测路径，Clio held-out 仍为 `PENDING`。
- D19：Q2/A2 单因素消融与六类失败审计已完成；当前 synthetic 数值变化为 0，真实消融仍为 `REAL_ABLATION_PENDING`。
- D20：单入口 CPU 复现命令、三张 JSON 派生表、README 数字检查及移动目录逐字节复验均为 `PASS`；可选二进制 Release 仍为 `PENDING`。
- D21：最终结果卡与 README 高风险声明审计均为 `PASS`；所有结果均绑定 evidence/config 哈希、样本量、预算、验证状态和适用边界。

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
- D15 从原先的 `BLOCKED_MISSING_OUTCOME` 变为完整 trace `PASS`，依次选择 `0001/0071/0041/0061/0031`，以 `max_budget_reached` 停止，observed gain 为 14；`performance_claim` 仍为 `null`。

本地 bundle 内少数阶段 manifest 保留的是产物创建当时的
`GPU_ACCEPTANCE_PENDING` 快照；最终增量总报告和上述重算 validator 才是
本次补验状态。完整 bundle、mask、点云和视频属于忽略的本地产物，不进入 Git。

## D12 A2 evidence-aware association（完整 GPU outcome 重放通过）

D12 不修改 A1；已发布的 A1 prediction/result 和 ObjectMemory SHA-256 继续由回归测试固定。A2 是独立的 label-free predictor：

- 双方都有 embedding 时使用 cosine，否则退化为规范化 exact-class；
- 每条观测质量为 retrieval、SAM、valid-point ratio 的几何均值，pair 取两者较小值，门槛固定为 `0.25`；
- 空间门槛固定为中心距离不超过 `0.15`，或 AABB IoU 大于 `0`；
- pair score 按 semantic/center/overlap/OBB-shape/quality 的 `0.25/0.25/0.20/0.15/0.15` 加权；
- cluster 只有所有跨 cluster pair 均通过 gate 才能合并。相同帧重复 mask 可以聚类，但永久对象仍要求至少两个不同帧。

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

## D15 Q2 gain-based sequential search（完整 GPU outcome 重放通过）

Q2 首步只按 retrieval score 选择，因此 budget=1 严格退化为 Q0。后续候选使用 `0.65 × candidate-universe min-max retrieval + 0.35 × pose novelty`；pose novelty 是最小平移按 `0.15 m` 截断归一化与最小视角差按 `3°` 截断归一化的均值。同分按 D5 原始 rank 和 frame ID 确定性排序。

```bash
python -m scripts.run_sequential_search \
  --cache evidence/week2/d11-candidate-cache/candidate_cache.json \
  --output-dir runs/d15-sequential-search --prefix real \
  --allow-blocked
```

已发布的无卡 partial-cache trace 先揭示 `frame_0001/0071/0041` 的 8 条新 3D observation；第 4 步按 metadata 选择 `frame_0061` 后才发现 outcome 未物化，于是立即返回 `BLOCKED_MISSING_OUTCOME`，不偷看或跳过候选。它保留为策略 readiness 的历史证据。

complete synthetic cache 跑满五步并以 `max_budget_reached` 停止。Q0/Q1/Q2 同预算文件只比较 selected frames、SAM calls、lifted/rejected counts；当前 synthetic fixture 上 Q1 与 Q2 同序，既不代表提升，也不使用实例标签。observed gain 仅指新 observation ID 数，不称精确 frustum coverage。
2026-08-29 的完整真实 GPU outcome cache 重放依次选择 `frame_0001/0071/0041/0061/0031`，跑满 5 步并以 `max_budget_reached` 停止，observed gain 为 14。该 trace 已通过确定性重算，但 `performance_claim=null`，未运行新的人工标签评测，也不声称相对 Q0/Q1 有性能提升。


```bash
python -m scripts.validate_d15 evidence/week2/d15-sequential-search
```

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
  runs/office-loop-d15_5-s5/d15_5-trash-can-k24
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

- 已实现：确定性 top-K 去冗余、置信过滤/MAD 离群剔除/PCA OBB、空间与语义关联、证据融合、JSON round-trip、左右/前后关系、逻辑回归校准、选择性预测指标。
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
- 已完成 D12：A2 complete-link、证据记录和反桥接测试保持冻结；完整 GPU outcome 驱动的无标签 CPU prediction 重放为 PASS，没有新增标签评测或 F1 结论。
- 已完成 D13：Q0 继续保持 `upstream-aligned`；10/10 静态语义检查与真实 `frame_0001` B0/B1 单视角 GPU 补验均通过，仍不称 FOUND-IT 官方复现。
- 已完成 D14：Q1 在完整真实 GPU outcome cache 上按 metadata 先选后揭示，K=1 与 Q0 一致；本次只验收 prediction replay，不新增 recall 数字。
- 已完成 D15：Q2 在完整真实 GPU outcome cache 上跑满 5 步并通过 trace 重算，observed gain=14、`performance_claim=null`；历史 blocked trace 继续作为诚实 partial-cache 证据。
- 已完成 D15.5：95 帧长轨迹生成可审计的场景级 RGB 点云、轨迹、Top-K 视角、对象记忆、PLY/MP4/Viser；3/8 个预测对象满足严格对象中心多视角门槛。
- 已完成 D16 CPU/source：Clio `apartment=development`、`cubicle=held-out` 场景角色和下载门槛已冻结；query 清单仍为 `PENDING_DATA_METADATA`。
- 已完成 D17 CPU/source：synthetic 的 2 个正查询与 3 个负查询验证标签隔离和正确拒答；本地 office-loop 无标签 replay 对 8 个同类对象给出 `ambiguous_candidates`，没有产生人工指标。
- 已完成 D18 CPU/source：冻结 Q0/Q1/Q2 × A1/A2 正式矩阵与 Q2×A1 诊断项；office-loop partial cache 中 Q2 遇到未物化 outcome 后明确阻塞，complete synthetic 的六项标签分离重放全部通过。
- 已完成 D19 CPU/source：Q2 的 pose-novelty/gain-patience 与 A2 的 semantic/OBB/quality/complete-link 单因素消融均可确定性重放；synthetic 数值无变化，历史成功特征明确为 `NOT_IMPLEMENTED`。
- 已完成 D20 CPU/source：单个 CPU 命令可重跑 D16–D19 validator、从规范 JSON 重建 Q×A/关系/消融表、核对 README 数字，并在移动目录中逐字节复验 retained outputs。
- 已完成 D21 CPU/source：最终结果卡逐项绑定 tracked evidence、配置哈希、样本量、预算和验证状态，并完成 README 中“官方、复现、改进、导航、优于、SOTA”等表述的上下文审计。
- Clio 仍未下载，真实 calibration、held-out、ablation 与新 GPU inference 仍待后续外部验收；这些缺口不影响 D16–D21 源码与 CPU 正确性验收，也不被写成性能结论。
- GT depth、pose、OBB 只应进入 evaluator 或 geometry oracle，不得进入主推理输入。
- 当前是目标定位感知前端，不包含路径规划、控制或闭环导航，因此不称“完整导航系统”。
