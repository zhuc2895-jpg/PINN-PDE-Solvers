# explore/ —— 架构方案探索

主文档：[docs/architecture-exploration.md](../docs/architecture-exploration.md)

这个目录只回答两个问题，而且答案已经拿到：

1. 把 PDE 收敛成 `sample / losses / loss_weights` 之后，四个 PDE 是否真能跑在同一个训练循环上
2. 换网络架构到底值不值（相同点数、种子、优化器下的实测对比）

**原型已落地为正式包。** 原来的 `explore/pinn_proto/` 已提升为仓库根目录的 `pinn/`
（方案 B），四份自包含脚本缩成了薄入口。因此：

- `architecture_sweep.py` 现在直接 `import pinn`，重跑测量的是**生产代码**，不再是原型副本
- `sweep_log.txt` 是原型阶段留下的原始记录，数字仍然有效（同一套数学、同一套接缝）

```bash
python explore/architecture_sweep.py --device cpu              # 完整（约 5 分钟）
python explore/architecture_sweep.py --device cpu --epochs 60  # 快速
```

`--device` 默认 `cpu`：本机 torch 构建无法在 RTX 5060（sm_120）上执行 kernel，
但 `torch.cuda.is_available()` 依然返回 True，选 cuda 会挂住。详见主文档第 2.1 节
与 `pinn/device.py`。
