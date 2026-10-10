"""T9: assemble the four required figures, one-page report, and ADR.

The report is a read-out of archived evidence, not a second tuning pass.  The
only new computation here is the pair of profile figures: they retrain the two
pre-registered arms at the frozen T7 hyperparameters so that the figure can
show all three seeds alongside the baseline and synthetic truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from pinn.evaluate import relative_l2
from pinn.model import MLP
from pinn.plotting import plt, save_figure
from pinn.trainer import TrainConfig, train

from .ablations import ALL_OFF, FULL
from .channel import ChannelClosure, ChannelProblem, _rans_residual
from .closure import BaselineClosure, TruthClosure
from .coupling import train_frozen_full_model
from .experiment import (
    DEFAULT_EPOCHS,
    DEFAULT_MODEL_DEPTH,
    DEFAULT_MODEL_WIDTH,
    DEFAULT_SEEDS,
    EXTRAPOLATION_BUDGET,
    TEST_RE_TAU,
    Hyperparameters,
)
from .solver import SolveResult, solve_channel, wall_clustered_grid

__all__ = [
    "PROFILE_RE_TAUS",
    "DeliverableData",
    "collect_deliverable_data",
    "plot_velocity_profiles",
    "plot_eddy_viscosity_correction",
    "build_report_markdown",
    "build_adr_markdown",
    "generate_deliverables",
]


PROFILE_RE_TAUS: Tuple[float, ...] = (180.0, 395.0, TEST_RE_TAU)


@dataclass
class DeliverableData:
    """Solved fields retained only for the T9 profile figures."""

    solved: Dict[Tuple[str, int, float], SolveResult]
    residual: Dict[Tuple[str, int, float], float]


def _freeze(model: torch.nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _train_arm(
    arm: str,
    seed: int,
    hyperparameters: Hyperparameters,
    *,
    epochs: int,
    width: int,
    depth: int,
    budget: int,
    verbose: bool,
) -> Tuple[ChannelProblem, torch.nn.Module]:
    problem = ChannelProblem(seed=seed, ablation=FULL if arm == "full" else ALL_OFF)
    model = MLP(3, 1, width=width, depth=depth, seed=seed)
    train(
        problem,
        model,
        TrainConfig(
            epochs=epochs,
            lr=hyperparameters.learning_rate,
            n_interior=budget,
            device="cpu",
            seed=seed,
            verbose=verbose,
        ),
    )
    _freeze(model)
    return problem, model


def collect_deliverable_data(
    hyperparameters: Hyperparameters,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    epochs: int = DEFAULT_EPOCHS,
    width: int = DEFAULT_MODEL_WIDTH,
    depth: int = DEFAULT_MODEL_DEPTH,
    budget: int = EXTRAPOLATION_BUDGET,
    Re_taus: Sequence[float] = PROFILE_RE_TAUS,
    verbose: bool = False,
) -> DeliverableData:
    """Train the full and data-only arms for the four required figures."""
    solved: Dict[Tuple[str, int, float], SolveResult] = {}
    residual: Dict[Tuple[str, int, float], float] = {}

    for seed in seeds:
        if verbose:
            print(f"[T9] training full/data-only seed={seed}")
        full_problem, full_model, _ = train_frozen_full_model(
            seed,
            hyperparameters,
            epochs=epochs,
            width=width,
            depth=depth,
            budget=budget,
            verbose=verbose,
        )
        data_problem, data_model = _train_arm(
            "all_off",
            seed,
            hyperparameters,
            epochs=epochs,
            width=width,
            depth=depth,
            budget=budget,
            verbose=verbose,
        )

        for Re_tau in Re_taus:
            truth = full_problem.solved_truth(Re_tau)
            baseline = solve_channel(full_problem.base, Re_tau, y=truth.y)
            full_solved = solve_channel(
                ChannelClosure(full_model, full_problem.base, structure=True), Re_tau
            )
            data_solved = solve_channel(
                ChannelClosure(data_model, data_problem.base, structure=False), Re_tau
            )
            solved[("truth", seed, Re_tau)] = truth
            solved[("baseline", seed, Re_tau)] = baseline
            solved[("full", seed, Re_tau)] = full_solved
            solved[("all_off", seed, Re_tau)] = data_solved
            residual[("baseline", seed, Re_tau)] = float(
                _rans_residual(baseline, Re_tau)
            )
            residual[("full", seed, Re_tau)] = float(
                _rans_residual(full_solved, Re_tau)
            )
            residual[("all_off", seed, Re_tau)] = float(
                _rans_residual(data_solved, Re_tau)
            )

    return DeliverableData(solved=solved, residual=residual)


def _plot_profile_family(
    data: DeliverableData,
    seeds: Sequence[int],
    Re_tau: float,
    ax,
) -> None:
    first_seed = seeds[0]
    truth = data.solved[("truth", first_seed, Re_tau)]
    x = truth.y.reshape(-1) * Re_tau
    mask = x > 0.0
    ax.plot(x[mask], truth.U.reshape(-1)[mask], color="black", linewidth=2.0, label="truth")
    baseline = data.solved[("baseline", first_seed, Re_tau)]
    ax.plot(
        x[mask],
        baseline.U.reshape(-1)[mask],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="baseline",
    )
    for seed in seeds:
        solved = data.solved[("all_off", seed, Re_tau)]
        ax.plot(
            x[mask],
            solved.U.reshape(-1)[mask],
            color="tab:green",
            alpha=0.35,
            linewidth=1.0,
            label="data-only seeds" if seed == first_seed else None,
        )
        solved = data.solved[("full", seed, Re_tau)]
        ax.plot(
            x[mask],
            solved.U.reshape(-1)[mask],
            color="tab:blue",
            alpha=0.35,
            linewidth=1.0,
            label="full PINN seeds" if seed == first_seed else None,
        )
    ax.set_xscale("log")
    ax.set_xlabel("y+")
    ax.set_ylabel("U+")
    ax.set_title(f"Re_tau={Re_tau:g}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)


def plot_velocity_profiles(
    data: DeliverableData,
    path: Path,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    Re_taus: Sequence[float] = PROFILE_RE_TAUS,
) -> None:
    fig, axes = plt.subplots(1, len(Re_taus), figsize=(12.0, 4.2), sharey=True)
    if len(Re_taus) == 1:
        axes = [axes]
    for ax, Re_tau in zip(axes, Re_taus):
        _plot_profile_family(data, seeds, Re_tau, ax)
    fig.suptitle("Velocity profiles: baseline, synthetic truth, data-only, full PINN")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, str(path))


def plot_eddy_viscosity_correction(
    data: DeliverableData,
    path: Path,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    Re_taus: Sequence[float] = PROFILE_RE_TAUS,
) -> None:
    fig, axes = plt.subplots(1, len(Re_taus), figsize=(12.0, 4.2), sharey=True)
    if len(Re_taus) == 1:
        axes = [axes]
    for ax, Re_tau in zip(axes, Re_taus):
        first_seed = seeds[0]
        truth = data.solved[("truth", first_seed, Re_tau)]
        baseline = data.solved[("baseline", first_seed, Re_tau)]
        x = truth.y.reshape(-1) * Re_tau
        mask = x > 0.0
        ax.plot(
            x[mask],
            (truth.nu_t.reshape(-1) - baseline.nu_t.reshape(-1))[mask],
            color="black",
            linewidth=2.0,
            label="truth - baseline",
        )
        for seed in seeds:
            solved = data.solved[("full", seed, Re_tau)]
            ax.plot(
                x[mask],
                (solved.nu_t.reshape(-1) - baseline.nu_t.reshape(-1))[mask],
                color="tab:blue",
                alpha=0.40,
                linewidth=1.1,
                label="full - baseline" if seed == first_seed else None,
            )
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("y+")
        ax.set_ylabel("Delta nu_t")
        ax.set_title(f"Re_tau={Re_tau:g}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Eddy-viscosity correction across wall-normal distance and Re_tau")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, str(path))


def _csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row(
    rows: Sequence[Mapping[str, str]],
    *,
    arm: str,
    budget: int,
    Re_tau: float,
) -> Mapping[str, str]:
    for row in rows:
        if (
            row["arm"] == arm
            and int(row["budget"]) == budget
            and float(row["Re_tau"]) == Re_tau
        ):
            return row
    raise KeyError((arm, budget, Re_tau))


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def build_report_markdown(
    *,
    results_dir: Path,
    figures_dir: Path,
    profile_decision: str = "进入二维周期性山丘（带边界条件）",
) -> str:
    inversion = json.loads(
        (results_dir / "inversion_report.json").read_text(encoding="utf-8")
    )
    training = _csv_rows(results_dir / "error_table.csv")
    external = _csv_rows(results_dir / "extrapolation_table.csv")
    ablation = _csv_rows(results_dir / "ablation_table.csv")
    coupling = _csv_rows(results_dir / "coupling_summary.csv")

    t180 = _row(training, arm="full", budget=EXTRAPOLATION_BUDGET, Re_tau=180.0)
    t395 = _row(training, arm="full", budget=EXTRAPOLATION_BUDGET, Re_tau=395.0)
    t590 = _row(external, arm="full", budget=EXTRAPOLATION_BUDGET, Re_tau=590.0)
    data_only_590 = _row(
        external, arm="all_off", budget=EXTRAPOLATION_BUDGET, Re_tau=590.0
    )
    coupling_by_alpha = {float(row["alpha"]): row for row in coupling}
    inversion_closure_share = min(
        float(entry["closure_share"]) for entry in _csv_rows(results_dir / "inversion_error_table.csv")
    )

    axis_lines = []
    for row in ablation:
        axis_lines.append(
            f"| {row['axis']} | {row['ablated_arm']} | "
            f"{_percent(float(row['ablated_improvement_mean']))} | "
            f"{float(row['relative_error_increase']):.3g} | {row['status']} | "
            f"{row['detail']} |"
        )

    alpha_lines = []
    for alpha in (1.0, 0.5, 0.2):
        row = coupling_by_alpha[alpha]
        alpha_lines.append(
            f"| {alpha:g} | {row['n_converged']}/{row['n_seeds']} | "
            f"{float(row['iterations_mean']):.1f} +/- {float(row['iterations_std']):.1f} | "
            f"{float(row['profile_rel_l2_mean']):.4f} | "
            f"{float(row['wall_shear_rel_error_mean']):.2e} | "
            f"{float(row['model_weight_drift_mean']):.2e} |"
        )

    return f"""# RANS + PINN 神经残差修正 Demo：一页论证报告

**决定：{profile_decision}。** 预先注册的 H1/H2/H3 均通过；但三根物理轴中只有结构约束产生了可测的边际作用，A2/A3 的单轴归因不通过。进入二维时不能把本 Demo 描述成“所有物理约束都有效”。

## 结论

**H1（残差存在且可参数化）：通过，但有可辨识性边界。** 基准-真值误差中闭合误差占 {100.0 * inversion_closure_share:.2f}% 以上；反演重放误差约 {max(float(entry['replay_relative_error']) for entry in inversion):.1e}。中心线应力因子为零，395 的最外层 5% 也低于可辨识性下限，因此 H1 只对壁面到外层的大部分区域成立。

**H2（物理约束改善外推）：通过主判据，但归因是混合的。** 冻结超参下，full 臂在训练 Reτ=180/395 的改善为 {_percent(float(t180['improvement_mean']))} / {_percent(float(t395['improvement_mean']))}；在未见 Reτ=590 为 {_percent(float(t590['improvement_mean']))} +/- {_percent(float(t590['improvement_std']))}，预注册 30% 阈值通过。纯数据 `all_off` 在第 12 档有 1 个种子不收敛，full 三个种子均收敛，说明结构约束主要买到稳定性；但逐轴消融均未通过“误差至少增加 10%”的预注册归因判据。

**H3（冻结耦合稳定）：通过。** 3 个种子 × 3 个 α 共 9/9 收敛，权重漂移为 0；α=1.0/0.5/0.2 的平均迭代次数为 {float(coupling_by_alpha[1.0]['iterations_mean']):.1f}/{float(coupling_by_alpha[0.5]['iterations_mean']):.1f}/{float(coupling_by_alpha[0.2]['iterations_mean']):.1f}，壁面剪应力误差约 {float(coupling_by_alpha[1.0]['wall_shear_rel_error_mean']):.1e}。欠松弛提高稳定性但显著增加迭代成本。

## 单轴归因（不美化）

| 轴 | 臂 | Reτ=590 改善 | 误差相对增加 | 预注册结果 | 说明 |
|---|---|---:|---:|---|---|
{chr(10).join(axis_lines)}

## 耦合扫描

| alpha | 收敛 | 迭代次数 mean +/- std | 最终剖面相对误差 | 壁面剪应力相对误差 | 权重漂移 |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(alpha_lines)}

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
"""


def build_adr_markdown() -> str:
    return """# ADR-0002: Demo 通过后进入二维周期性山丘

## Status

Accepted.

## Context

一维槽道流 Demo 的预注册判据为：H1 残差存在且可参数化，H2 未见 Reτ=590 改善至少 30%，H3 冻结修正耦合不发散。T7 得到 full 臂 Reτ=590 平均改善 88.1% +/- 4.3%；T8 得到 9/9 耦合收敛且权重漂移为 0。H1 的反演闭合误差占比超过 99.99%，但中心线和外层部分区域不可辨识。

单轴消融没有支持“三根物理轴都有效”：A1 去掉结构约束后出现非收敛；A2 去掉 RANS 残差后误差变化约 1.6e-7；A3 去掉平滑项后误差反而下降 33%。因此主结论应限定为：结构化参数化和耦合稳定性有效，不能声称每根物理约束都有独立贡献。

## Decision

进入二维周期性山丘，但按以下边界执行：

1. 主要假设限定为“结构化残差修正可在 coarse RANS 外推中保持稳定”，不声明全部物理约束均有边际贡献。
2. 二维实验必须使用更高保真参考或真实 LES/DNS 子集，不能只复用合成真值。
3. 保留 A2/A3 的零结果，并优先研究可观测性（中心线、外层）而不是继续加网络容量。

## Consequences

- 许可继续二维方法开发，但二维结果必须重新做泄漏隔离、冻结超参和逐种子报告。
- 若二维中结构约束不再带来稳定性，停止扩展，转向弱耦合或混合数值方法。
- ADR-0001 的 solver-in-the-loop 决定继续有效。
"""


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def generate_deliverables(
    *,
    root: Path = Path("RANS_PINN_Demo"),
    docs_dir: Path = Path("docs"),
    seeds: Sequence[int] = DEFAULT_SEEDS,
    epochs: int = DEFAULT_EPOCHS,
    width: int = DEFAULT_MODEL_WIDTH,
    depth: int = DEFAULT_MODEL_DEPTH,
    budget: int = EXTRAPOLATION_BUDGET,
    verbose: bool = False,
) -> None:
    root = Path(root)
    figures_dir = root / "figures"
    results_dir = root / "results"
    report_dir = root / "report"
    hyperparameters = Hyperparameters(
        **json.loads(
            (results_dir / "frozen_hyperparameters.json").read_text(encoding="utf-8")
        )
    )
    data = collect_deliverable_data(
        hyperparameters,
        seeds=seeds,
        epochs=epochs,
        width=width,
        depth=depth,
        budget=budget,
        verbose=verbose,
    )

    plot_velocity_profiles(data, figures_dir / "01_velocity_profiles.png", seeds=seeds)
    plot_eddy_viscosity_correction(
        data, figures_dir / "02_eddy_viscosity_correction.png", seeds=seeds
    )
    shutil.copyfile(
        figures_dir / "data_volume_curve.png",
        figures_dir / "03_data_volume_curve.png",
    )
    shutil.copyfile(
        figures_dir / "coupling_residual_history.png",
        figures_dir / "04_coupling_residual_history.png",
    )

    _write_text(
        report_dir / "demo_report.md",
        build_report_markdown(results_dir=results_dir, figures_dir=figures_dir),
    )
    _write_text(
        Path(docs_dir) / "adr" / "0002-proceed-to-2d-periodic-hill.md",
        build_adr_markdown(),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("RANS_PINN_Demo"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--width", type=int, default=DEFAULT_MODEL_WIDTH)
    parser.add_argument("--depth", type=int, default=DEFAULT_MODEL_DEPTH)
    parser.add_argument("--budget", type=int, default=EXTRAPOLATION_BUDGET)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_deliverables(
        root=args.root,
        docs_dir=args.docs_dir,
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        width=args.width,
        depth=args.depth,
        budget=args.budget,
        verbose=args.verbose,
    )
    print(f"Wrote T9 deliverables under {args.root} and {args.docs_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
