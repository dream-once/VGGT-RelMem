# D19 one-factor ablation evidence

- Q2 只执行 `retrieval_only` 和 `no_gain_patience` 两个真实存在的单因素变体。
  “历史成功作为候选打分特征”在当前 Q2 中不存在，因此明确记录为
  `NOT_IMPLEMENTED`。
- A2 固定 Q1 K=5 输入，分别移除 semantic、OBB shape、quality 和
  complete-link。特征权重移除后其余权重确定性归一化；semantic/quality 的工程
  gate 同时关闭，避免名义移除但仍参与拒绝。
- synthetic 指标仅用于正确性消融；本 fixture 上所有数值变化为 0，不声称提升。
- office-loop 只报告结构变化；Q2 各变体均在同一未物化 outcome 处阻塞。
- 失败审计覆盖完整冻结 synthetic 查询，分类固定为 retrieval、segmentation、
  lifting、association、relation、abstention，不删除已知失败。

复验：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m scripts.validate_d19
```
