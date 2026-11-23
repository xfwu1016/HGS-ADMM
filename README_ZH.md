# 混合样本与特征划分下的结构回归：复现说明

在 Windows PowerShell 中进入含 reproduce.py 的目录，安装 Python 3.12，然后运行

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe reproduce.py --all
```

Linux 或 macOS 使用 python3.12 创建虚拟环境，再运行 .venv/bin/python reproduce.py --all。依赖安装以后无需联网下载数据。仅做小型验证时用 --validate；重新生成全部原始实验、图、表及先导诊断用 --all。每次完整复现应在独立目录进行。依赖版本、原始数据来源和阶段见英文 README.md。

正式比较保留原有 138、30、27、135 次求解与45个先导。新增 pilot_box.csv 有45项顶点界诊断，pilot_length.csv 有36项先导长度比较。后者收录三个模型、三个布局和四种长度的全部 seed 0 结果。所有流量为模拟协议传送的标量数；时间依机器而变。数据、预生成图表、论文和环境均不随代码包分发。
