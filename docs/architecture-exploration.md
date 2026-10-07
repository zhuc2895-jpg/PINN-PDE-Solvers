# PINN 架构方案探索

> 结论先行：把"PDE 数学"收敛成一个 `sample / losses / loss_weights` 接缝，其余
> （网络、自动微分、采样工具、训练循环、绘图）全部下沉为共享实现 —— 下面叫**方案 B**。
> 这不是纸上推演：`explore/pinn_proto/` 里有一个可运行原型，四个 PDE（含三输出的
> Navier–Stokes）已经跑在同一个训练循环上。
>
> 同时实测了网络架构：在点数、种子、优化器、学习率调度完全相同的前提下，给 tanh-MLP
> 加一层 Fourier 特征，把 Poisson 的相对 L2 误差从 **12.8% 降到 7.0%**，训练时间
> 只增加 23%。

"探索架构方案"至少有两个层次，本文都覆盖，并且每一节都有可复现的数字支撑：

- **代码架构**：四个独立脚本怎么合并，接缝应该切在哪里（第 3、4 节）
- **模型架构**：网络本身换成什么（第 5 节）

复现命令见第 8 节，原始输出留在 `explore/sweep_log.txt`。

---

## 1. 现状体检

四个脚本各自自包含，这一点在"教学示例"的定位下是优点，但四份代码已经形成事实上的重复。

| 脚本 | 总行数 | 非注释代码行 | 网络类 | `autograd.grad` 调用点 |
|---|---|---|---|---|
| `pinn_2d_poisson.py` | 318 | 194 | `PINN` | 5 |
| `pinn_2d_convection_diffusion.py` | 324 | 206 | `PINN`（复制） | 4 |
| `pinn_2d_navier_stokes.py` | 385 | 257 | `NavierStokesNet` | 10 |
| `pinn_burgers_sample.py` | 227 | 146 | `PINN`（复制） | 2 |
| **合计** | **1254** | **803** | 4 | **21** |

把空行、注释、行内空白规范化之后再逐行比对：`pinn_2d_poisson.py` 与
`pinn_2d_convection_diffusion.py` 有 **120 行完全相同**，分别占两者非注释代码的
**68%** 和 **62%**。这两个文件在数学上只差一个残差表达式和一个 `source_term`。

重复清单：

| 被重复的东西 | 份数 | 形态 |
|---|---|---|
| MLP 定义 | 4 | 3 个同名 `PINN` + 1 个 `NavierStokesNet` |
| `torch.autograd.grad` 链 | 21 处 | 全部内联在各 `pde_residual` 里 |
| 训练循环 | 4 | Adam + StepLR + history 累积 + 打印 |
| 采样 | 4 | 2 份 `sample_interior`/`sample_boundary` 复制 + 2 处内联 |
| 可视化 | 4 | 每个脚本一套独立的 matplotlib 面板 |
| 超参数 | 4 | 同一概念三种命名：`HIDDEN_LAYERS` / `HIDDEN` / `LAYERS` |

还有一个接口漂移：`PINN.forward` 在 Poisson / convection-diffusion / Navier–Stokes 里是
`forward(x, y)`，在 Burgers 里是 `forward(x)`（内部自己 `torch.cat`）。同名同类，两种调用约定，
调用方必须记住自己拿的是哪一份。

---

## 2. 顺带发现的 5 个现存问题

这些不是重构引入的，是**现在就在仓库里**的。前三个会直接影响你现在能不能跑出结果。

### 2.1 GPU 路径在这台机器上会挂死（影响最大）

四个脚本都用同一个模式选设备：

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

这台机器上 `torch.cuda.is_available()` 返回 **True**（RTX 5060 Laptop，算力 `sm_120`），
但 `mappo` 环境里装的是 `torch 1.11.0+cu113`，编译时最高只支持到 `sm_86`。
结果是 CUDA kernel 不是"报错退出"，而是**挂起**：

- 一个 256×256 的 matmul 在 GPU 上运行 35 秒无返回，只能杀进程
- 同一份 `pinn_2d_poisson.py`：误走 GPU 时 8 分钟没跑完；强制 CPU 后 **30.3 秒**跑完

所以现在直接 `python pinn_2d_poisson.py` 有大概率卡住而不是给你一个错误信息。
两种修法，任选：

- 装匹配 `sm_120` 的 torch（需要 CUDA 12.8+ 的构建，即 torch ≥ 2.7），或
- 在探测设备时做一次**真实 kernel 自检**，失败就退回 CPU（原型里已采用这个思路，见 4.3）

### 2.2 Poisson 的学习率调度是死代码

```python
EPOCHS = 1500
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.5)
```

`step_size` 等于总 epoch 数，所以衰减只在第 1500 步（训练结束后）触发一次。实测：

```
lr after epoch 1499: 0.001
lr after epoch 1500 (training already over): 0.0005
```

整个训练过程学习率恒为 `1e-3`。对照另外三个脚本：convection-diffusion 用
`step_size=800`、Burgers 用 `3000`、Navier–Stokes 用 `2000`——只有 Poisson 这个是空的。

### 2.3 Burgers 会在无界面环境卡住

`pinn_burgers_sample.py` 是四个脚本里**唯一**没有 `matplotlib.use("Agg")`、却**唯一**调用
`plt.show()` 的。在服务器 / CI / 无 GUI 会话里跑它，会在最后一步阻塞。

### 2.4 配点固定，全程不重采样

四个脚本都在训练开始前抽一次配点，然后 1500–10000 个 epoch 一直用同一批点。
Poisson 只用 512 个内部点解二维问题；误差最大的区域（例如解的梯度陡处）不会被补点。
这是"结果不够准"时排在第一位的可调项，而不是换网络。

### 2.5 loss 权重散落在三种形态里

- Poisson / convection-diffusion / Burgers：直接 `pde + bc` 相加，隐式权重 1.0
- Navier–Stokes：模块级常量 `W_CONT/W_MOM/W_BC = 1/1/10`

同一个概念（多目标损失的加权）在四份代码里三种写法，没法统一调、也没法做对比实验。

---

## 3. 代码架构：三个候选方案

| | 方案 A：共享工具库 | **方案 B：问题即插件（推荐）** | 方案 C：配置驱动求解器 |
|---|---|---|---|
| 形态 | 保留四个脚本，把 `PINN`、`sample_*`、绘图抽到 `pinn_common.py` | 一个通用 `train()`，每个 PDE 是一个实现 `sample/losses/loss_weights` 的对象 | PDE 用 YAML/字典描述，由求解器解释 |
| 新增概念数 | 1（工具模块） | 3（Problem 协议 / Batch / Trainer） | 6+（注册表、DSL、字段类型、编译层…） |
| 能去掉多少重复 | 网络/采样/绘图；训练循环仍是 4 份 | 网络/采样/绘图/训练循环/权重，全部 | 同 B |
| 加第 5 个 PDE | 复制一个脚本，改残差 | 新增一个 30–60 行的 problem 文件 | 写一段配置（如果不需新算子） |
| 加第 5 个、且残差形态全新 | 照旧复制 | 需要新增一个 derivative 辅助函数 | 可能要改求解器内核 |
| 可测试性 | 差（逻辑散在脚本顶层） | 好（problem 可脱离训练单测残差） | 好，但测试的是 DSL 而不是数学 |
| 适合谁 | 只想先止血 | 想长期维护 4→N 个 PDE | 想做成通用求解器产品 |
| 对当前四个 PDE 是否过度 | 否，但治标 | 否 | **是** |

选 **B** 的理由很直接：唯一值得存在的接缝落在"这个 PDE 的残差是什么"上，
而这恰好是四个脚本里唯一不重复的部分。方案 A 保留了最大的那份重复（训练循环），
方案 C 对四个 PDE 而言引入的解释层比被解释的东西还多。

判断依据写在代码里：方案 B 下每个 PDE 只需要描述数学，任何公共行为（重采样、
自适应权重、L-BFGS 微调、日志、绘图）都只实现一次，自动对四个 PDE 生效。

---

## 4. 方案 B 的接缝，以及验证结果

### 4.1 接缝长什么样

```python
class PDEProblem(Protocol):
    name: str
    coords: tuple[str, ...]      # ("x","y") 或 ("x","t")
    out_dim: int                 # 1 表示 u；3 表示 (u,v,p)

    def sample(self, n_interior, n_boundary, n_initial, device) -> Data: ...
    def losses(self, model, data) -> dict[str, Tensor]: ...   # 未加权的各项损失
    def loss_weights(self) -> dict[str, float]: ...
    def exact(self, coords) -> dict[str, Tensor] | None: ...
```

训练循环因此完全不知道自己在解什么：

```python
terms = problem.losses(model, data)
total = sum(weights.get(k, 1.0) * v for k, v in terms.items())
```

### 4.2 原型目录

> 下面这份目录后来提升成了正式包 `pinn/`（同名同结构，路径从 `explore/pinn_proto/` 移到了仓库根）。
> 本节保留原型阶段的名字，作为方案 B 的设计记录。

```
explore/pinn_proto/
  derivatives.py    # d() / grad_all() / laplacian() —— 21 处 autograd 样板只剩这三个函数
  model.py          # MLP(activation=..., n_fourier=...) + SIREN 初始化 + Fourier 特征
  sampling.py       # uniform / 正方形四边 / 时空条带两端
  problem.py        # PointSet / Data / PDEProblem 协议 / inputs_from
  trainer.py        # 通用 Adam + StepLR + history 训练循环
  problems/
    poisson.py              # 稳态、1 输出、齐次 Dirichlet
    convection_diffusion.py # 稳态、一阶 + 二阶项、Pe≈10
    burgers.py              # 时变、IC + BC、无解析解
    navier_stokes.py        # 三输出 (u,v,p)、三个残差、四个壁面不同目标值
```

### 4.3 四个 PDE 跑同一个 trainer（实测）

```
[1] all four PDEs through the same sample/losses/weights seam (300 epochs each)
  2d-poisson                loss 9.28e+01 -> 1.03e-01 (0.001x)  terms=['bc', 'pde']            ( 6.4s)
  2d-convection-diffusion   loss 6.58e+00 -> 4.48e-02 (0.007x)  terms=['bc', 'pde']            (20.2s)
  1d-burgers                loss 6.16e-01 -> 1.38e-01 (0.224x)  terms=['bc', 'ic', 'pde']      (23.1s)
  2d-lid-driven-cavity      loss 3.35e+00 -> 3.22e-01 (0.096x)  terms=['bc', 'cont', 'mom_x', 'mom_y']  (21.8s)
```

注意 300 epoch 只是一次**连通性冒烟测试**，不是精度结论（Navier–Stokes 那行用的是
800 内部点，远小于原脚本的 2048，也没有跑到 5000 epoch）。它要证明的是接口成立：
损失项个数（2 / 3 / 4 项）、输出维度（1 / 3）、条件类型（BC / IC / BC+IC）、
是否有解析解 —— 四种组合全部由同一段 `train()` 驱动。

### 4.4 架构替我们挡住的一个 bug

把 Burgers 接进通用 trainer 时，第二个 epoch 就炸：

```
RuntimeError: Trying to backward through the graph a second time ...
```

原因不是训练循环，而是**初始条件的目标值带着计算图**：`u(x,0) = -sin(pi*x)` 是从
一个 `requires_grad=True` 的叶子算出来的，训练点固定复用，于是第二次 backward 又踩到
同一张已经释放的图。

原始脚本里这个坑不出现，只是因为它在循环内重新算了一遍目标值——属于"碰巧躲开"。
在方案 B 下它被**结构性**修掉：`PointSet` 在构造时就把 targets 全部 `detach()`，
并附上一行注释说明为什么：

```python
def __post_init__(self) -> None:
    # 目标是数据，永远不进计算图；否则复用时会第二次 backward 到同一张图
    self.targets = {name: t.detach() for name, t in self.targets.items()}
```

这类"让非法状态不可表达"的修正，只有在接缝存在时才有一个地方可写。

---

## 5. 模型架构：跑了才知道

实验设置：2-D Poisson，1500 epoch，同一批配点、同一优化器与调度
（`lr_step=1500`，即复刻原脚本那个"不衰减"的调度）、同一种子，**只换网络**。
误差是相对 L2，在 128×128 网格上对解析解计算。

| 网络 | 参数量 | 相对 L2 误差 | 最终 loss | 训练耗时 |
|---|---|---|---|---|
| MLP-tanh 64×4（当前实现） | 12,737 | 1.283e-01（12.8%） | 1.055e-02 | 29.8 s |
| **MLP-tanh + Fourier(16, σ=1.5)** | 14,657 | **6.953e-02（7.0%）** | 1.039e-01 | 36.7 s |
| MLP-sin（SIREN 初始化） | 12,737 | 9.876e-01（98.8%） | 7.961e+00 | 31.7 s |
| MLP-silu | 12,737 | 1.581e-01（15.8%） | 1.537e-01 | 50.0 s |

作为对照，提交里的 `pinn_2d_poisson.py` 自己跑一遍是 **0.110987（11.10%）**、30.3 s；
原型的 tanh 行给 12.8%，两者同量级，差异只来自随机数消耗顺序（建模初始化与采样点的
先后不同），不是精度差异。

三点解读：

1. **Fourier 特征在这类光滑问题上几乎免费地赢**：误差降到约 55%，代价是参数量 +15%、
   时间 +23%。它治的是 PINN 的谱偏置——网络天然先学低频，而 `sin(πx)sin(πy)`
   信息集中在 π 附近，随机傅里叶映射把那部分抬到手边。
2. **SIREN 在本问题上明显更差（98.8%）**：`sin` 激活对初始化尺度极敏感，标准的
   `w0=30` 是为图像/SDF 那类高频目标调的，直接搬到这里的频率尺度上并不合适。
   要用它必须重新调 `w0` 和第一层初始化，不要照抄。
3. **loss 低不等于误差小**：Fourier 行的最终 loss（1.04e-01）反而比 tanh 行
   （1.06e-02）高一个量级，但真实误差只有一半。PINN 的 loss 是配点上的残差均值，
   不是误差估计；只看 loss 曲线调参会被带偏，所以四个问题的"是否有解析解/基准表"
   这件事要一直保留在评测层里。

候选清单（按"值得先试"排序，不都是本次实测过的）：

| 候选 | 适用条件 | 代价 |
|---|---|---|
| **Fourier features / 随机特征映射** | 光滑、周期、含中高频的解（本次适用） | 极低，已实现 |
| **modified MLP（Wang & Perdikaris 2021，带 U/V 门控）** | 训练不稳、收敛慢时的通用增强 | 低 |
| 更宽/更深 + 残差连接 | 解的空间变化剧烈 | 中 |
| 多输出头（本仓库 NS 已在用） | 多个耦合场，共享特征 | 低 |
| SIREN / 正弦激活 | 高频、超声、波场类问题 | 需重调 `w0`，本次不适用 |
| 因果 / 时间 marching | 时变问题长区间（Burgers 这类） | 中 |
| DeepONet / FNO 等算子网络 | 要"一族 PDE"而不是单个解 | 大，是另一个课题 |

---

## 6. 训练策略：通常比换网络更值钱

按投入产出排序，前两项建议在动网络之前先做：

1. **loss 权重自适应**。现状是 NS 手工 `W_BC=10`、其余 1.0。可查的起点：
   Wang 等 2021 的 learning-rate annealing / NTK 加权、McClenny & Braga-Neto 2020 的
   self-adaptive PINN（软注意力）、Chen 等 2018 的 GradNorm。建议先实现最简的一版
   自适应权重，因为它能同时改善四个 PDE，而换网络只影响一个。
2. **配点重采样**。现在 512 个点冻结 1500 epoch。RAD / 残差自适应加密
   （可查：Wu 等 2023 的 residual-based adaptive sampling、Lu 等 2021 DeepXDE 的 RAR）
   对二维 Poisson 这种低维问题成本很低，通常比加深网络有效。
3. **优化器二阶微调**。Adam 跑完接一小段 L-BFGS，小规模问题上常常再降一个量级，
   代价是显存/时间。
4. **评测层**。Poisson / convection-diffusion / Burgers 有解析解或可造解析解，
   NS 有 Ghia 等 1982 的中心线基准表——这套评测已经是仓库里最值钱的部分，
   重构时务必原样保留，别退化成"只看 loss 曲线"。

（以上文献线索凭记忆给出，用于检索，未逐条核对卷期页码。）

---

## 7. 建议路线

**P0 — 止血（半天）**

- 修 2.1 的设备探测：加一次真实 kernel 自检，失败回退 CPU（否则这台机器上什么都跑不出来）
- 修 2.2 的 Poisson 调度（改成 `step_size=500` 之类，或直接删掉调度）
- 修 2.3：Burgers 补 `matplotlib.use("Agg")`，删掉 `plt.show()`
- 补 `requirements.txt` / `pyproject.toml`，把 torch 版本约束写清楚
- 验收：四个脚本在 CPU 上都能一条命令跑完并产出 PNG

**P1 — 落地方案 B（1–2 天）**

- 把 `explore/pinn_proto/` 提升为包（例如 `pinn/`），四个脚本各缩成一个 problem 文件
- 保留每个脚本的独立入口和输出图片名，README 的"运行任一脚本"体验不变
- 验收：四个脚本的 L2 误差 / 基准对比与重构前一致（Poisson 应仍在 11% 附近）

> **状态：已落地（分支 `codex/plan-b-package`）。** `explore/pinn_proto/` 提升为 `pinn/`
> 包，四个脚本变成薄入口（数学交给 `pinn/problems`，训练交给 `pinn.trainer`），
> 输出图片名保持不变。复现验收：`python pinn_2d_poisson.py` → **11.63%**，与提交版
> 脚本的 11.10% 同量级（差异只来自随机种子）。注意本节 P0 的 2.2 调度修复**故意没有**
> 一起做——薄入口显式保留 `lr_step=1500`，为的是让这次重构是纯结构变更、数字可比。
> 残留 P0 项（2.1 设备探测、2.2 调度、2.3 headless 阻塞）见 `pinn/device.py`
> 与各入口脚本里的注释。

**P2 — 用实验层换精度（按需）**

- 实现自适应 loss 权重与重采样，跑一组对照
- 给 Poisson / convection-diffusion 打开 Fourier 特征（本次实测约减半误差）
- 验收：每个 PDE 有一张"基线 vs 改进"的误差表，而不是只留一张图

---

## 8. 复现

```bash
# 原型依赖已装的 torch / numpy；本机请显式用 CPU
python explore/architecture_sweep.py --device cpu            # 完整：四个 PDE 冒烟 + 四网络对比（约 5 分钟）
python explore/architecture_sweep.py --device cpu --epochs 60  # 快速版本（约 1.5 分钟）
```

`--device` 默认就是 `cpu`，原因是 2.1 那个坑：一个跑不了 kernel 的 torch 构建依然会
报 `cuda.is_available() == True`，然后挂住。

本文数据来自 `explore/sweep_log.txt`（完整日志）与一次单独的原始脚本对照运行：

```bash
CUDA_VISIBLE_DEVICES=-1 python pinn_2d_poisson.py   # 提交版本自身：L2 11.10%，30.3 s
```

## 9. 风险与取舍

- **抽象过度的门槛**：如果最终只会保留一个 PDE，方案 B 也不划算。这里成立的前提是
  仓库要长期维护 4→N 个 PDE——也就是当前的定位。
- **`sin` 激活不能照抄**：第 5 节已经标出，它的失败是初始化尺度问题，不是"激活不行"。
- **原型与生产代码的区别**：`explore/` 里的东西是可抛弃的探索产物，没有测试、
  没有并行、绘图能力也刻意留白（每个 PDE 的可视化本来就该是各自定制的）。
  提升为正式包时必须补测试，尤其是残差本身。
- **结果图会被覆盖**：脚本都写死输出文件名，重构时应加 `--out` 之类的参数，
  否则对比实验只能互相覆盖。





