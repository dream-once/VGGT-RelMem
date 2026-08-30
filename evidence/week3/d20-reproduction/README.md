# D20 reproducibility package

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

结果范围是 synthetic correctness、office-loop complete-cache 与 Clio apartment
development engineering replay；Clio 的 Q0/Q1/Q2 observation 为 0/1/1，不是性能
提升。cubicle held-out、真实 calibration、带标签真实 ablation 和可选二进制
Release 均未完成。
README 的 clean-clone 入口包含创建虚拟环境、安装 `.[dev]`、运行全量单测与 CPU demo；
`Pillow` 已列入 dev 依赖，避免新环境在图像相关单测处缺包。
