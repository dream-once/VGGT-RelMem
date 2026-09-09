# 关系定位改进的轻量证据

完整说明：[关键模块文档](../../docs/RELATION_MODULES.md)。本目录保存 2026-09-09～10 的事后受控实验结果；Apartment/Cubicle 已有历史暴露，不是新 held-out 结果。

| 比较 | Cubicle 严格正例正确 | 负例误答 | 边界 |
|---|---:|---:|---|
| 同一适配器，Top-1 PCA→A2 PE 代表中心 | 10/149→28/149 | 0/149→1/149 | 系统差值；medoid 也达到 28/149 |
| 固定 A2 fused，选后检查→关系先过滤 | 0/10→3/10 | 0/36→0/36 | 独立同类消歧诊断；三题共享 textbooks |
| 固定 A2 PE，加实体证据下限 | 28/149→28/149 | 1/149→1/149 | 回答数 104→91，少 13 个错误输出，覆盖率下降 |

Apartment 主集严格为 1/136→1/136，容差 45/136→28/136；同类消歧严格为 0/40，没有严格提升。候选组分差拒答会过度拒答，完整保留为失败消融。

- [benchmark_summary.json](benchmark_summary.json)：两场景 6 候选池×5 策略的全部聚合值、固定分母、family 子组、严格增减项、三个消歧案例、源文件及本地逐查询产物 SHA-256。
- [validation_report.json](validation_report.json)：候选来源重建、上游 PCA 数值核对、20,610 条预测/评分重放及逐行计数的校验报告。
- [diagnostics.json](diagnostics.json)：left/right/front/behind 分项、双端位置错误、各查询跨视角方向一致性；只做诊断，不改变预测。

无需 Clio 数据检查摘要完整性及算术：

```bash
python -m scripts.validate_clio_relation_improvement --summary-only evidence/relation-improvement
```

有本地原始缓存时执行完整重放：

```bash
python -m scripts.run_clio_relation_improvement
python -m scripts.validate_clio_relation_improvement --evidence evidence/relation-improvement
python -m scripts.analyze_relation_improvement --evidence evidence/relation-improvement
```

本次以严格双端中心命中为主指标，OBB 配准 RMSE 扩张仅为敏感性分析。baseline 是上游式 Top-1 候选加自建统一关系适配器，不是上游官方关系系统。GT 只用于出题和评估，不供 grounder 排序；显式 taxonomy 也不是学习到的语义分类器。源码重放与轻量算术验证的证据强度不同。

[固定候选上限审计](candidate_ceiling.json) 单独标记 `gt_read=true`：Cubicle 当前 PE 固定中心 oracle 为 28/149，Top-1 PCA oracle 为 21/149；不属于可部署的预测性能。复现命令：`python -m scripts.audit_relation_candidate_ceiling`。
