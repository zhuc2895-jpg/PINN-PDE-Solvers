# RANS + PINN 神经残差修正 Demo：一页论证报告

**决定：进入二维周期性山丘（带边界条件）。** 预先注册的 H1/H2/H3 均通过；但三根物理轴中只有结构约束产生了可测的边际作用，A2/A3 的单轴归因不通过。进入二维时不能把本 Demo 描述成“所有物理约束都有效”。

## 结论

**H1（残差存在且可参数化）：通过，但有可辨识性边界。** 基准-真值误差中闭合误差占 99.99% 以上；反演重放误差约 1.2e-16。中心线应力因子为零，395 的最外层 5% 也低于可辨识性下限，因此 H1 只对壁面到外层的大部分区域成立。

**H2（物理约束改善外推）：通过主判据，但归因是混合的。** 冻结超参下，full 臂在训练 Reτ=180/395 的改善为 89.4% / 92.3%；在未见 Reτ=590 为 88.1% +/- 4.3%，预注册 30% 阈值通过。纯数据 `all_off` 在第 12 档有 1 个种子不收敛，full 三个种子均收敛，说明结构约束主要买到稳定性；但逐轴消融均未通过“误差至少增加 10%”的预注册归因判据。

**H3（冻结耦合稳定）：通过。** 3 个种子 × 3 个 α 共 9/9 收敛，权重漂移为 0；α=1.0/0.5/0.2 的平均迭代次数为 10.3/19.0/56.7，壁面剪应力误差约 4.0e-11。欠松弛提高稳定性但显著增加迭代成本。

## 单轴归因（不美化）

| 轴 | 臂 | Reτ=590 改善 | 误差相对增加 | 预注册结果 | 说明 |
|---|---|---:|---:|---|---|
| structure | A1 | -298.1% | 32.4 | fail | ablated solver did not converge for every seed |
| rans_residual | A2 | 88.1% | 1.65e-07 | fail | relative error increase is below the pre-registered threshold |
| smoothness | A3 | 92.0% | -0.33 | fail | relative error increase is below the pre-registered threshold |

## 耦合扫描

| alpha | 收敛 | 迭代次数 mean +/- std | 最终剖面相对误差 | 壁面剪应力相对误差 | 权重漂移 |
|---:|---:|---:|---:|---:|---:|
| 1 | 3/3 | 10.3 +/- 2.5 | 0.0199 | 4.01e-11 | 0.00e+00 |
| 0.5 | 3/3 | 19.0 +/- 0.0 | 0.0199 | 4.01e-11 | 0.00e+00 |
| 0.2 | 3/3 | 56.7 +/- 0.6 | 0.0199 | 4.01e-11 | 0.00e+00 |

## 必出图

1. [速度剖面](../figures/01_velocity_profiles.png)
2. [涡黏修正量](../figures/02_eddy_viscosity_correction.png)
3. [数据量—误差](../figures/03_data_volume_curve.png)
4. [耦合残差历史](../figures/04_coupling_residual_history.png)

## 数据表

- [训练误差表](../results/error_table.csv)
- [外推与逐种子结果](../results/extrapolation_table.csv)
- [单轴消融表](../results/ablation_table.csv)
- [预注册判据表](../results/criteria.csv)
- [耦合扫描汇总](../results/coupling_summary.csv)

## 边界与停止条件

本 Demo 使用合成真值、平衡槽道和单套闭合形式，不能替代 DNS/LES。若进入二维，应保留“结构约束带来稳定性、RANS 残差与平滑项未显示边际收益”的原结论，并把中心线/外层不可辨识性列为后续观测设计问题，而不是事后删除。
