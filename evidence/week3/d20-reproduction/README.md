# D20 reproducibility package

规范输入全部是 tracked JSON；`result_tables.json` 与三张 Markdown 表由源码自动
生成，没有手填结果。`sources/` 中只保存 D15.5 的轻量 manifest/audit 快照，
PLY、PNG、MP4、权重和数据仍不进入 Git。

从仓库根目录执行一个 CPU 命令，即可重跑 D16–D19 validator、重建表格、核对
README 数字，并逐字节比较 retained outputs：

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

结果范围仍是 synthetic correctness 与 office-loop engineering replay；
Clio held-out、真实 calibration、真实 ablation 和可选二进制 Release 均未完成。
