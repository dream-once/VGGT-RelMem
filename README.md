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

核心接入对应推荐文档的 **D3**：“关闭开放词汇路径，跑通 VGGT-SLAM 几何；保存关键帧、点图、置信度与变换”。当前无卡模式下已完成 D3 的源码和环境准备，但尚未执行 CUDA 推理，因此不能把 D3 标为验收通过。

同时完成了部分前置工作：

- D1：官方 VGGT-SLAM、SALAD、VGGT_SPARK 均已下载并固定 commit，磁盘已检查；SAM 3 权限和有卡 GPU 状态尚未验收。
- D2：`vggt_geom` 环境、geometry NPZ/JSON schema、适配器和 smoke test 已准备；`open_vocab` 环境尚未建立，因此完整 D2 仍未结束。
- D3：运行器、无界面模式、关闭回环时跳过 SALAD 权重、产物导出和 run manifest 已接好；等待 GPU 实跑。

`--check-only` 逐模块使用独立子进程，适合无卡/小内存模式。它不会加载 VGGT、SALAD 权重，也不会执行推理：

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/vggt_geom \
  python -m scripts.run_vggt_geometry --check-only
python -m unittest discover -v
```

今天的源码接入验收标准是：check-only 所有模块无 `ERROR`；VGGT-SLAM commit 为 `35327ac28b7d193df9ccc39ba6346052bb6f1207`；全部单测通过。这个结果只说明“接线完成”，不代替 D3 的 GPU 验收。

### 恢复 GPU 后的 D3 验收

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
- 已完成 D3 基线：锁定 commit 的 VGGT-SLAM 2.0 已在 office_loop 前 8 帧跑通，几何产物通过 validator；严格的稳定复现仍需第二次独立运行。
- 待接入：实际文本-帧编码器、SAM 3 mask 导出器和 Clio 标注。
- GT depth、pose、OBB 只应进入 evaluator 或 geometry oracle，不得进入主推理输入。
- 当前是目标定位感知前端，不包含路径规划、控制或闭环导航，因此不称“完整导航系统”。
