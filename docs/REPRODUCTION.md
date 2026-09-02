# 复现手册

本手册覆盖从公开 CPU 验证、Clio 几何与对齐，到 36-task 推理和最终 evaluator。公开 CPU 验证与真实 Clio GPU 重放是两个不同层级。

## CPU clean clone

CPU 回归不读取 Clio YAML、原始图像、权重或本地 run：

```bash
git clone https://github.com/dream-once/VGGT-RelMem.git
cd VGGT-RelMem
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m scripts.verify_public_clone
python -m scripts.demo --save-memory runs/demo/object_memory.json
```

轻量结果验证：

```bash
python -m scripts.validate_clio_final_summary
python -m scripts.validate_clio_pe_semantic_fusion_summary
```

## 本地依赖

真实流程使用两个隔离环境：

- `vggt_geom`：VGGT-SLAM 几何。
- `open_vocab`：PE 与 SAM 3。

本地还需要：

- `third_party/VGGT-SLAM` 及冻结的上游 commit；
- PE-Core-L14-336 和 SAM 3 checkpoint；
- 合法取得的 Clio Apartment/Cubicle 数据；
- 两套场景 geometry NPZ 与 anchor poses；
- 足够空间保存不进入 Git 的 `runs/` 产物。

环境和 GPU 可用下面的命令核对：

```bash
nvidia-smi
bash scripts/bootstrap_vggt_geom.sh
bash scripts/bootstrap_open_vocab.sh
```

## Clio 几何与 evaluator-only 对齐

先审计本地数据，再分别按照最终实验使用的 stride 生成 192/172 帧几何：

```bash
python -m scripts.audit_clio_feasibility \
  --filesystem data/clio \
  --output runs/clio-feasibility.json

TORCH_HOME=/root/autodl-tmp/cache/torch \
conda run -p /root/autodl-tmp/envs/vggt_geom \
  python -m scripts.run_vggt_geometry \
  --image-folder data/clio/apartment/images \
  --output runs/clio-apartment-dev-v2-lc/geometry.npz \
  --max-frames 192 --frame-stride 6 --max-loops 0

TORCH_HOME=/root/autodl-tmp/cache/torch \
conda run -p /root/autodl-tmp/envs/vggt_geom \
  python -m scripts.run_vggt_geometry \
  --image-folder data/clio/cubicle/images \
  --output runs/clio-cubicle-heldout-v1/geometry.npz \
  --max-frames 172 --frame-stride 4 --max-loops 0
```

使用本地 COLMAP reconstruction 建立仅供 evaluator 使用的 Sim(3)。主推理不得读取下面两个 world-alignment 输出：

```bash
python -m scripts.align_clio_colmap_poses \
  --anchor-poses runs/clio-apartment-dev-v2-lc/geometry.anchor_poses.json \
  --colmap-images data/clio/apartment/sparse/0/images.bin \
  --output runs/clio-apartment-dev-v2-lc/vggt_to_colmap_alignment.json \
  --scene-transform configs/clio_scene_transforms.json \
  --scene-id apartment \
  --world-output runs/clio-apartment-dev-v2-lc/vggt_to_clio_world_alignment.json \
  --max-rmse-m 0.15

python -m scripts.align_clio_colmap_poses \
  --anchor-poses runs/clio-cubicle-heldout-v1/geometry.anchor_poses.json \
  --colmap-images data/clio/cubicle/sparse/1/images.bin \
  --output runs/clio-cubicle-heldout-v1/vggt_to_colmap_alignment.json \
  --scene-transform configs/clio_cubicle_scene_transform.json \
  --scene-id cubicle \
  --world-output runs/clio-cubicle-heldout-v1/vggt_to_clio_world_alignment.json \
  --max-rmse-m 0.15
```

## 真实 Clio 36-task 重放

先核对两个 tracked 查询清单与本地 task YAML：

```bash
python -m scripts.validate_clio_query_manifest \
  --query-manifest configs/clio_apartment_queries.json \
  --task-yaml data/clio/apartment/metadata/tasks_apartment.yaml

python -m scripts.validate_clio_query_manifest \
  --query-manifest configs/clio_cubicle_queries.json \
  --task-yaml data/clio/cubicle/tasks_cubicle.yaml
```

先 dry-run 审计 36 个 task 的完整计划：

```bash
python -m scripts.run_clio_36_task_batch \
  --open-vocab-env /root/autodl-tmp/envs/open_vocab \
  --sam3-checkpoint /root/autodl-tmp/cache/modelscope/facebook-sam3/sam3.pt \
  --dry-run
```

dry-run 应生成 36 个 task、72 条 GPU 命令和最多 108 条条件 CPU 命令。确认两套 geometry、图片、环境、checkpoint 和输出目录后，移除 `--dry-run` 执行。入口会在 D6 无有效 3D observation 时跳过该 task 的 D7/D8/A2。

## 重建最终评测

以下命令假设两个 run root 已完整物化。

### 对象定位

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
```

### 对象关联

```bash
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
```

### 关系与拒答

输出目录必须为空：

```bash
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
```

最后重建轻量公开摘要：

```bash
python -m scripts.build_clio_final_summary
python -m scripts.validate_clio_final_summary
```

## PE mask-crop 扩展

训练自由 PE 代表中心实验的 GPU 提取、CPU evaluator 和摘要构建命令集中在 [post-D21 证据说明](../evidence/post-d21-pe-fusion/README.md)。

## 产物边界

- 原始数据、checkpoint、mask、点云、视频和完整 task 报告保留在本机。
- Git 只保留配置、轻量摘要、失败案例、聚合数字和源产物哈希。
- 不要在预测阶段读取 GT、world alignment 或 evaluator 输出。
- 新的真实运行应使用新目录，避免覆盖冻结证据。
