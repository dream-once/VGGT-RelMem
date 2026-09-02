# Evidence 索引

求职版只保留能够验证 README 最终结论的轻量 JSON/Markdown，不分发原始 Clio 数据、checkpoint、mask、点云、视频或完整 task run。

## 当前结果

| 路径 | 内容 |
|---|---|
| [final-clio](final-clio/README.md) | Apartment/Cubicle 对象定位、A1/A2、关系和拒答最终摘要 |
| [post-d21-pe-fusion](post-d21-pe-fusion/README.md) | PE mask-crop 代表中心扩展 |

两份摘要都保存固定分母、聚合结果、协议边界和本地源报告 SHA-256，并提供不依赖原始数据的 validator。

## 完整研究快照

week1–week4、D20/D21 结果卡、synthetic fixture 和诊断产物保存在 Git tag `research-final-v1-2026-09-02`，不进入求职版主树。

## 阅读建议

1. [最终 Clio 摘要](final-clio/README.md)
2. [PE 扩展摘要](post-d21-pe-fusion/README.md)
3. [真实失败案例](../docs/CLIO_FINAL_FAILURE_CASES.md)
