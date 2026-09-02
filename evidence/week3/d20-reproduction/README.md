# D20 reproducibility package

> Historical D20 snapshot: this package intentionally reconstructs the pre-Cubicle
> D16–D19 state. Its `clio_held_out=PENDING` field is not current project status;
> post-D21 Apartment/Cubicle results live in `evidence/final-clio/`.

规范输入全部是 tracked JSON；`result_tables.json` 与三张 Markdown 表由源码自动
生成，没有手填结果。`sources/` 中只保存 D15.5 的轻量 manifest/audit 快照，
PLY、PNG、MP4、权重和数据仍不进入 Git。

从仓库根目录执行一个 CPU 命令，即可重跑 D16–D19 validator、重建 office-loop
complete-cache、Clio apartment 24/24 development replay 与 synthetic 表格、核对 README 数字，并逐字节比较 retained outputs：

```bash
cd /root/autodl-tmp/VGGT-RelMem
PYTHONDONTWRITEBYTECODE=1 python -m scripts.reproduce_d20 \
  --output-dir /tmp/vggt-relmem-d20-rebuild \
  --verify-retained evidence/week3/d20-reproduction
```

独立门槛：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m scripts.validate_d20
```

D20 本身的结果范围是 synthetic correctness、office-loop complete-cache 与 Clio
apartment development engineering replay；其 Cubicle pending 字段必须按历史快照
解读。后续 Cubicle fixed-confirmatory 已完成；真实 calibration、带标签查询策略
ablation 和可选二进制 Release 仍未完成。
README 的 clean-clone 入口包含创建虚拟环境、安装 `.[dev]`、运行全量单测与 CPU demo；
`Pillow` 已列入 dev 依赖，避免新环境在图像相关单测处缺包。
