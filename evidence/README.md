# Evidence 索引

公开仓库保留轻量 JSON/Markdown 证据，不分发原始 Clio 数据、checkpoint、mask、点云、视频或大型 run。

## 当前结果

| 路径 | 内容 |
|---|---|
| [final-clio](final-clio/README.md) | Apartment/Cubicle 对象定位、A1/A2、关系和拒答最终摘要 |
| [post-d21-pe-fusion](post-d21-pe-fusion/README.md) | PE mask-crop 代表中心扩展 |
| [D21 final](week3/d21-final/README.md) | 冻结结果卡、claim audit 和显式缺口 |
| [D20 reproduction](week3/d20-reproduction/README.md) | CPU 派生表与移动目录复验 |

## 研究历史

这些目录按开发阶段保留，用于追溯而不是日常导航：

- `week1/`：单视角、Top-K、多视角 lifting、Observation cache。
- `week2/`：Candidate cache、Q0/Q1/Q2、A1/A2 与早期关联。
- `week3/`：关系、消融、D20 可复现包和 D21 结果卡。
- `week4/`：Clio Apartment GPU 补验与 D21.1 pillow 诊断。

不要移动或改名这些历史文件：多个 manifest 和 validator 通过相对路径及 SHA-256 绑定它们。

## 阅读建议

求职或复习时只需依次查看：

1. [最终 Clio 摘要](final-clio/README.md)
2. [PE 扩展摘要](post-d21-pe-fusion/README.md)
3. [最终结果卡](week3/d21-final/result_card.md)
4. [真实失败案例](../docs/CLIO_FINAL_FAILURE_CASES.md)

需要追查某个结论时，再通过 result card 的 source reference 回到 week1–week4。
