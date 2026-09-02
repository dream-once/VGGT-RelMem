# VGGT-RelMem 复习指南

这份文档用于快速恢复项目思路。先理解数据如何流动，再看实验编号；不需要从 D1 顺序读到 D21。

## 1. 核心问题

输入是一段带相机几何的 RGB 序列和自然语言目标，例如“tape measure”。系统需要回答：

1. 哪些帧可能看到目标？
2. 每帧中哪个 mask 对应目标？
3. mask 中的像素如何变成可靠的 3D 观测？
4. 不同帧的观测是否属于同一物体？
5. 哪个持久对象应该作为最终答案？
6. 关系查询中的 target 和 reference 是否都可靠？
7. 证据不足时为什么拒答？

## 2. 数据流

```text
query + RGB sequence + VGGT geometry
                 │
                 ▼
retrieval.py / q1_fixed_topk.py
选择 Top-K 且抑制相邻重复视角
                 │
                 ▼
PE + SAM adapters
得到 query-conditioned mask
                 │
                 ▼
observations.py
置信过滤、MAD 离群剔除、3D 中心和 PCA OBB
                 │
                 ▼
association.py
pair gate + complete-link → ObjectMemory
                 │
        ┌────────┴────────┐
        ▼                 ▼
relations.py          calibration.py
显式参考系关系         confidence / abstention
        └────────┬────────┘
                 ▼
独立 Clio evaluator
```

最重要的工程约束是：预测代码不能读取 GT、世界配准或 evaluator 标签。预测先落盘，evaluator 再读取标签计分。

## 3. 建议阅读顺序

| 顺序 | 文件 | 要回答的问题 |
|---:|---|---|
| 1 | `relground/retrieval.py` | Top-K 如何去除时间和视角冗余？ |
| 2 | `relground/q1_fixed_topk.py` | 固定预算 Q1 如何冻结并重放？ |
| 3 | `relground/observations.py` | 2D mask 如何变成稳健 3D observation？ |
| 4 | `relground/schemas.py` | Observation、Memory 和 manifest 如何序列化？ |
| 5 | `relground/association.py` | pair gate、连通分量与 complete-link 有何区别？ |
| 6 | `relground/relations.py` | 左右/前后使用哪个 anchor 坐标系？ |
| 7 | `relground/calibration.py` | 置信度与拒答如何分离？ |
| 8 | `relground/clio_grounding_benchmark.py` | 多 GT OBB 如何严格计分？ |
| 9 | `relground/clio_association_benchmark.py` | A1/A2 如何用最终同簇关系同口径比较？ |
| 10 | `relground/clio_relation_benchmark.py` | 为什么 target 和 reference 都必须命中？ |

## 4. 四个容易混淆的设计

### Q0、Q1 与 Q1F

- Q0：上游一致的 Top-1 单视角基线。
- Q1：固定 Top-5 多视角方案。
- Q1F：有 A2 永久对象时使用 Q1；没有对象时确定性回退 Q0。
- Q2：顺序搜索诊断，目前不是最终主策略。

最终 `+11.11pp` 比较的是 Q1F 和 Q0 的整条系统策略，不是“A2 单独贡献”。

### A1 与 A2

A1 对通过 pair gate 的边建立连通分量，因此 A-B、B-C 会把 A 和 C 传递合并。A2 使用 complete-link：新成员必须与簇中全部成员兼容，从而抑制桥接误合并。

最终 Clio 没有保存 observation semantic embedding，因此 A2 的有效证据只有任务内几何与质量。

### 严格指标与 ±RMSE

严格 Acc@1 要求预测中心位于任意一个该 task 的 GT OBB 内。若加入世界配准 RMSE 后命中，只能记在敏感性指标 `±RMSE`，不能替代严格指标。

### 拒答

关系查询不仅要找到 target，还要找到 reference。两端定位失败、关系证据不足或置信度低都可能拒答；报告必须区分端到端拒答、原因命中和两端定位成功后的关系拒答。

## 5. 如何理解 PE 扩展

PE 扩展没有重新选择物体。它只在当前 A2 已选对象的多次观测中，寻找与查询最相似的 mask-crop，并采用那次观测的 3D 中心。

这修复了 Apartment 和 Cubicle 各一个严格错误，但已经达到当前 observation 集合的严格 oracle。若继续提升，应先检查：

1. 正确实例是否进入 Top-K 候选；
2. SAM mask 是否覆盖正确实例且边界干净；
3. mask 内像素投影到 3D 后是否被错误深度或动态点污染；
4. A2 是否选错物体或把实例错误合并。

只有在这些步骤能提供新的正确候选后，更复杂的 ranker 才有新的上限。

## 6. 面试时的最短讲法

> 我在固定 VGGT-SLAM 几何上实现了一层可审计的开放词汇 3D 定位系统。系统用固定 Top-5 检索和 SAM 生成多视角 3D 观测，通过 complete-link 对象记忆抑制链式误合并，并在关系查询中同时验证目标和参考物。预测与 GT evaluator 完全隔离，证据不足时显式拒答。Clio Cubicle 的固定确认实验中，Q1F 相对 Q0 的严格 Acc@1 从 27.78% 提升到 38.89%，但我把它限定为小规模系统差值，不包装成 SOTA 或完全未接触的 held-out 结论。

更完整的讲稿和追问见 [PROJECT_PRESENTATION.md](PROJECT_PRESENTATION.md)。

## 7. 动手复习

```bash
python -m scripts.verify_public_clone
python -m scripts.demo --save-memory runs/demo/object_memory.json
```

然后打开生成的 `object_memory.json`，沿着 observation、association evidence 和 object center 反查代码。真实实验命令见 [REPRODUCTION.md](REPRODUCTION.md)。
