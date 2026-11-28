# 加权图分裂：论文复现代码

对应 Xiaofei Wu 的论文 **Weighted graph splitting for structured regression with sample and feature partitions**。

[English](README.md) · [数据与预处理](docs/DATA.md) · [实验及输出对应表](docs/EXPERIMENTS.md) · [环境说明](docs/ENVIRONMENT.md)

入口 [reproduce.py](reproduce.py) 可以从空结果目录执行原有38阶段研究及其四幅图。当前稿件还包含独立运行的联合权重实验和图5；应在原流程之后执行 [ACCELERATION_EXTENSION.md](docs/ACCELERATION_EXTENSION.md) 中的命令，再执行 [ADAPTIVE_COMPARISON.md](docs/ADAPTIVE_COMPARISON.md) 中的自适应对照与新样本敏感性实验。仅运行 `--all` 不会生成这两组新增扩展。运行不依赖已有实验结果或论文 LaTeX 源码。两个公开数据集随 scikit-learn 安装，实验过程中无需联网下载数据。

[RUN_VERIFICATION.json](RUN_VERIFICATION.json) 记录此前从空结果目录完成的原有流程、环境和校验情况；它是精简验证记录，不是实验输入，也不能替代实际重跑。

## 安装和完整运行

在 `reproduce.py` 所在目录运行。记录的环境为 **Python 3.12.14**；[requirements-lock.txt](requirements-lock.txt) 固定运行时依赖闭包，[requirements.txt](requirements.txt) 保留五个主要科学计算包的版本。

Linux/macOS：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python reproduce.py --validate
.venv/bin/python reproduce.py --all --jobs 3
```

Windows PowerShell（已安装 Python 3.12）：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe reproduce.py --validate
.venv\Scripts\python.exe reproduce.py --all --jobs 3
```

已经验证的平台是 Linux。TCP 实验采用可跨平台的 `spawn` 进程启动方式，但没有另行认证 Windows/macOS 的时间和绘图结果。执行实验、生成图和表格源码不需要 LaTeX 或 GPU。

`--validate` 执行七个数值检查阶段，**不等于重跑全部实验**。`--all` 执行 38 个阶段，包含 138 条主实验记录、40 条真实数据重复划分记录、14 次完整的双进程 TCP 运行，以及 pilot/continuation、停止准则、划分诊断和最终输出校验。完整流程成功后，执行记录才会写入 `status: passed`，同时记录结果行数检查和脚本哈希。

此前归档的 Linux 运行从空结果目录执行原有38个阶段，使用 `--jobs 3`、每个求解器一个 BLAS 线程，耗时 **888.52 秒，约 14.8 分钟**。这是该环境的实测参考，不是其他机器的耗时保证。完整实验包含大量优化拟合、严格停止阈值和 TCP 延迟模拟中的实际等待。准确耗时以本机 `logs/run_*.json` 的记录为准。`--jobs 3` 仅并行部分互相独立的实验；内存或 CPU 紧张时使用 `--jobs 1`。时间结果受硬件、操作系统调度及运行负载影响。

## 从头运行与重新生成

面向 GitHub 的代码压缩包不带已生成的实验结果。独立复现时，将代码解压到新目录后执行 `--all`。它从头执行阶段序列并覆盖同名输出，不提供自动断点续跑。论文投稿包可以另外包含已完成实验的结果，供核对使用。

完成一次**原有38阶段运行**后，可以重新生成其汇总、表格和四幅图片，不重复原有优化实验；新增联合权重实验另行使用扩展文档中的汇总命令：

```bash
.venv/bin/python reproduce.py --postprocess
```

Windows 将解释器路径改为 `.venv\Scripts\python.exe`。`--step` 可单独执行指定阶段，但必须先生成该阶段的前置输入。详见 `python reproduce.py --help` 和[实验对应表](docs/EXPERIMENTS.md)。验证模式写入带时间戳的 `results_check/`，不覆盖论文实验表格。

| 输出目录 | 内容 |
|---|---|
| `results/` | 原始记录、系数与迭代轨迹、诊断汇总、处理后的数据划分 |
| `tables/` | 数值表格及正文片段的 LaTeX 源码 |
| `figures/` | 原有四幅图（PDF、SVG、PNG）及新增图5（PDF、PNG） |
| `logs/` | 各阶段日志及 JSON 执行记录 |
| `results_check/` | 带时间戳的数值检查结果 |

图文件名保留稳定标识，不全部对应当前正文图号：

| 当前论文图号 | 文件名主体 | 生成脚本 |
|---|---|---|
| 图 1：因子结构 | `fig1_factor_structure` | `code/make_figures.py` |
| 图 2：算法流程 | `fig_alg1_workflow` | `code/make_workflow.py` |
| 图 3：收敛轨迹 | `fig2_convergence` | `code/make_figures.py` |
| 图 4：通信比较 | `fig3_communication` | `code/make_figures.py` |
| 图 5：联合权重异质性 | `fig_joint_heterogeneity` | `code/summarize_acceleration.py` |

## 结果的适用范围

主实验在单进程内执行逻辑工作节点。TCP 实验在同一台物理机器上，以两个独立操作系统进程分别运行 Worker 和 Coordinator，并通过回环 TCP 套接字交换数据；一个 Worker 聚合四个样本块。LAN/WAN 设置在应用层加入延迟和序列化等待，不是物理多机网络测量，也不是实际链路带宽整形。运行环境需要允许本机回环套接字。

真实数据重复划分使用每个数据集五个 split seed。Diabetes 在各训练划分内重新选择惩罚参数；乳腺癌数据固定使用原始验证划分选出的参数，后续测试划分可能与该选参验证集重叠。因此后者的预测指标用于描述固定参数下不同优化器的一致性，不能当作新的调参方法的独立预测性能评估。详见[数据说明](docs/DATA.md)。

可选的 `--step tcp-replay` 只是合成通信回放诊断，需要已有重复划分结果；它不属于 `--all`，也不作为正文实验依据。完整的分布式优化实验由 `tcp_distributed_solver.py` 执行并合并为 14 条运行记录。

另有一个同一划分内的重复计时核验，可运行 `python code/tcp_timing_repeat_audit.py`。该脚本在 Diabetes seed-2026 的 LAN 划分上执行三组配对运行，并在每组内随机安排 Unit/Budget 顺序，结果写入 `results/tcp_timing_repeats_focus.csv`。它只用于估计同一划分内的计时波动，不是额外的独立数据划分，也不是物理网络测试。归档汇总见 `results/tcp_timing_repeats_focus_summary.json`；不同机器和系统负载下的时间会变化。

随机种子和停止规则已固定，但舍入、BLAS 实现和字体差异可能影响逐字节结果与渲染；计时结果不要求完全相同。运行记录和集中式基准的环境文件可用于追踪差异。离线安装方法见[环境说明](docs/ENVIRONMENT.md)。

## 仓库与使用权限

代码仓库地址：https://github.com/xfwu1016/HGS-ADMM。运行不需要账号凭据或私有数据服务。

原始材料未附单独的代码许可证，此处没有新增许可证或代作者授予使用权。如需授权，请联系作者 `xfwu1016@ynu.edu.cn`。第三方软件和数据仍适用各自的条款。


## 本次修订（2026年11月27日）

本轮仅重组论文、统一表头并从已有pilot记录新增包络诊断，未改变求解器或重跑全部优化实验。原有结果文件均保持不变。`python code/pilot_uncertainty_summary.py`可重生成新增诊断；它也已接入表格生成阶段。七项数值验证重新通过，记录见`EDITORIAL_REVISION_CHECKS.json`。历史完整重跑与audit文件不代表本轮再次全量运行。


## 2026年11月28日联合权重扩展

新增一次切换的双侧图权重与rho联合选择规则、oracle实际运行、共同rho网格和异质性实验。新实验单独保存，不覆盖旧结果；运行命令、样本结构、通信账本及适用限制见docs/ACCELERATION_EXTENSION.md。

## 新增自适应对照与高维稳健性实验

本轮新增七种方法对照、高维新样本及 pilot／滞后敏感性实验，独立于原38阶段和联合权重扩展。完整命令、保护区间、失败记账与成本口径见 [ADAPTIVE_COMPARISON.md](docs/ADAPTIVE_COMPARISON.md)。未达目标精度的运行不参与成本排名。
