# VGGT-RelGround

面向语义导航的感知前端：在 VGGT-SLAM 2.0 与开放词汇分割输出之上，构建 top-K 多视角对象观察、稳健 3D 提升、跨帧对象记忆、关系约束定位及置信度拒答。

当前提交是项目的“源码起点”。核心后端、标准文件接口、合成 Demo 和单元测试已经可运行。VGGT-SLAM 2.0 官方源码现已作为本地、非跟踪的第三方 checkout 放在 `third_party/VGGT-SLAM`，锁定 commit `35327ac28b7d193df9ccc39ba6346052bb6f1207`；个人代码不复制或修改上游实现，而是由 `adapters/vggt_slam.py` 从官方 `Solver` 导出 NPZ/JSON。文本图像编码器和 SAM 3 后续放在独立的开放词汇环境。

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
python -m unittest discover -v
python -m scripts.demo --save-memory runs/demo/object_memory.json
```

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

适配器同时生成 frame/submap 来源 manifest 和 `world_from_anchor` 位姿 JSON。NPZ 点图保持 VGGT submap canonical 坐标，`world_from_camera` 保存上游优化后的 SL(4) 齐次变换；它可能是投影变换，下游会执行齐次除法。置信度按上游 percentile threshold 导出为二值有效性图，不把原始置信分数误称为概率。

### 当前状态与文档日程

核心接入对应推荐文档的 **D3**：“关闭开放词汇路径，跑通 VGGT-SLAM 几何；保存关键帧、点图、置信度与变换”。目前已在 RTX 4090 上完成两次独立的 8 帧 CUDA 推理，两个产物均通过 validator 且 SHA-256 完全一致，因此 D3 已验收完成。

当前文档日程状态：

- D1：官方 VGGT-SLAM、SALAD、VGGT_SPARK 均已下载并固定 commit；SAM 3 checkpoint 已从 ModelScope 官方发布页下载、校验并成功加载。
- D2：`vggt_geom` 与 Python 3.12 `open_vocab` 两个隔离环境、geometry/mask NPZ+JSON schema、适配器和 smoke test 均已完成。
- D3：运行器、无界面模式、产物导出、run manifest、两次 GPU 实跑及严格复现均已完成。
- D4：PE top-1、SAM 3 mask、mask-to-point lifting、3D OBB、可视化、validator 和两次真实 GPU 复现均已完成。

`--check-only` 逐模块使用独立子进程，适合无卡/小内存模式。它不会加载 VGGT、SALAD 权重，也不会执行推理：

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/vggt_geom \
  python -m scripts.run_vggt_geometry --check-only
python -m unittest discover -v
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

## D4 开放词汇 top-1 基线（已完成）

Perception Encoder 固定为 `3e352cca660658d4b5c90f42a7808b11469e4c66`，SAM 3 固定为 `8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`。它们使用独立的 Python 3.12 环境，不修改 D3 的 `vggt_geom`：

```bash
bash scripts/bootstrap_open_vocab.sh

conda run --no-capture-output -p /root/autodl-tmp/envs/open_vocab \
  python -m scripts.run_open_vocab_top1 --check-only
```

当前已经用官方 `PE-Core-L14-336` 在 D3 的 8 个关键帧上两次运行查询 `printer`，两次都选择 `frame_0006`，余弦分数均为 `0.1550685453894363`。缓存后运行约 `10.91 s`，峰值显存约 `2797 MB`：

```bash
HF_HOME=/root/autodl-tmp/cache/huggingface \
conda run --no-capture-output -p /root/autodl-tmp/envs/open_vocab \
  python -m scripts.run_pe_top1 \
  --query printer --max-frames 8 \
  --output-dir runs/office-loop-pe-printer
```

SAM 3 checkpoint 已从 ModelScope 的 `facebook/sam3` 官方发布页下载到数据盘。完整 B0 使用两个显式本地 checkpoint，不访问 Hugging Face，也不会重复下载：

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/open_vocab \
  python -m scripts.run_open_vocab_top1 \
  --query 'trash can' --max-frames 8 \
  --pe-checkpoint \
  /root/autodl-tmp/cache/huggingface/hub/models--facebook--PE-Core-L14-336/snapshots/bafb0f76541d399057e980a25947f67acec76575/PE-Core-L14-336.pt \
  --sam3-checkpoint /root/autodl-tmp/cache/modelscope/facebook-sam3/sam3.pt \
  --output-dir runs/office-loop-b0-trash-can

python -m scripts.validate_open_vocab runs/office-loop-b0-trash-can
```

两次真实运行都选择 `frame_0004`，SAM 3 输出 6 个实例，4 个具有有效 VGGT 点并成功生成 3D OBB；validator 均为 `PASS`。核心 `masks.json`、`observations.json` 和 `preview.png` 的 SHA-256 在两次运行间完全一致，峰值显存约 `5090 MB`。

`printer` 的 PE top-1 仍稳定选择 `frame_0006`，但最前 8 帧没有可见打印机，因此 SAM 3 在官方阈值 `0.50` 下返回 0 个实例；这作为负例保留，不用降低阈值制造误检。


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

`vggt_geom` 环境导出一个 NPZ，必要数组为：

- `frame_ids`: `(N,)` 字符串；
- `point_maps`: `(N,H,W,3)`；
- `confidence_maps`: `(N,H,W)`；
- `world_from_camera`: `(N,4,4)`。

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
- 已完成 D3：锁定 commit 的 VGGT-SLAM 2.0 已在 `office_loop` 前 8 帧独立运行两次，两个几何产物均通过 validator 且哈希一致。
- 已完成 D4：实际 PE 文本-帧检索、SAM 3 mask 导出、VGGT 点图抬升和 3D OBB 产物均已接入并通过真实运行验收。
- 待完成 D5：跨视角观测关联、持久化对象记忆，以及后续 Clio 标注与关系评估。
- GT depth、pose、OBB 只应进入 evaluator 或 geometry oracle，不得进入主推理输入。
- 当前是目标定位感知前端，不包含路径规划、控制或闭环导航，因此不称“完整导航系统”。
