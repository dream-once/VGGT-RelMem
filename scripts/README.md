# CLI 索引

日常复习只需要下面的主线入口。其余以 D 编号命名的脚本是研究阶段构建器或 validator，为历史 evidence 的可审计重放保留。

## 最常用

| 入口 | 用途 | GPU |
|---|---|---:|
| `python -m scripts.verify_public_clone` | 公开 clone 全部 CPU 回归 | 否 |
| `python -m scripts.demo` | 最小合成 ObjectMemory 示例 | 否 |
| `python -m scripts.reproduce_d20` | 重建 D20 派生表与 validator 集 | 否 |
| `python -m scripts.validate_d21` | 校验最终结果卡与声明边界 | 否 |

## 主推理链路

| 阶段 | 入口 |
|---|---|
| VGGT 几何 | `run_vggt_geometry.py` |
| PE Top-K 检索 | `run_pe_topk.py` |
| SAM + 3D lifting | `run_sam_topk_lifting.py` |
| Observation cache | `cache_scene_observations.py` |
| ObjectMemory | `prepare_object_memory.py` |
| A2 关联 | `run_a2_association.py`, `evaluate_a2_association.py` |
| 关系查询 | `run_relation_protocol.py`, `evaluate_relation_protocol.py` |
| 结构化查询 | `evaluate.py` |
| 可视化 | `visualize_geometry.py`, `visualize_scene_memory.py` |

## Clio 最终实验

| 入口 | 用途 |
|---|---|
| `run_clio_36_task_batch.py` | 编排 Apartment + Cubicle 36-task 主链 |
| `evaluate_clio_grounding_benchmark.py` | 多 GT OBB 对象定位 |
| `evaluate_clio_association.py` | A1/A2 最终聚类同口径评测 |
| `run_clio_relation_benchmark.py` | target+reference 关系与拒答 |
| `build_clio_final_summary.py` | 从本地报告构建轻量摘要 |
| `validate_clio_final_summary.py` | clean clone 校验最终摘要 |

完整参数见 [复现手册](../docs/REPRODUCTION.md)。

## Post-D21 PE 扩展

| 入口 | 用途 |
|---|---|
| `run_clio_pe_semantic_fusion.py` | GPU、无标签 mask-crop PE 打分 |
| `evaluate_clio_pe_semantic_fusion.py` | CPU、独立 GT evaluator |
| `build_clio_pe_semantic_fusion_summary.py` | 构建 tracked 聚合摘要 |
| `validate_clio_pe_semantic_fusion_summary.py` | clean clone 声明边界校验 |

## 历史脚本

`build_d*`、`run_d*`、`validate_d*` 以及旧的单阶段审计脚本继续保留，因为 week1–week4 evidence 的 manifest 绑定了路径和 SHA-256。它们不是新读者的推荐入口，也不应在没有对应冻结配置时随意重跑。
