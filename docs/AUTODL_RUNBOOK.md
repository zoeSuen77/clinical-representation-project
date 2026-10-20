# AutoDL 完整训练操作说明

Group 8 · CSI 436/536 · 更新于 2026 年 10 月 20 日

本项目已完成本地训练前准备。以下命令用于租好实例后的部署和运行；当前尚未连接 AutoDL，也未开始正式的 65 次评估。

## 租好后提供什么

把 AutoDL 控制台给出的 **SSH 命令**发到当前对话，例如 `ssh -p PORT root@HOST`，并说明实例 GPU、镜像及 Python 版本。若使用 SSH 密钥，提供本机密钥路径即可；认证信息不放入代码、报告或 GitHub。密码认证时通过终端交互完成。

收到连接信息后，将依次检查连接、GPU 与驱动、安装依赖、验证数据和模型，再启动完整实验。无需提前在本地运行正式训练。

## 资源与目录

- 本项目冻结三个 BERT-base 编码器并依次加载，只有特征提取使用 GPU；LR、linear SVM 和 TF-IDF 拟合使用 CPU。当前脚本使用一张 GPU。
- 配置目标建议为单卡至少 16 GB 显存、至少 16 GB 主机内存，以及 20 GB 以上可用磁盘。这是保守的工程配置建议，尚未测量 AutoDL 上的显存峰值或耗时；实际兼容性以 CUDA 小样本检查为准。
- 固定 Python 3.12、PyTorch 2.11.0 和锁文件中的其余依赖。不要因为镜像预装了另一版 Torch 就跳过检查。
- 项目、虚拟环境和缓存放在 `/root/autodl-tmp/clinical-representation-project`。AutoDL 文档将 `/root/autodl-tmp` 定义为数据盘；数据盘内容不随保存镜像一起保存，结果需单独回传。[目录说明](https://www.autodl.com/docs/env/)
- 首次模型和数据下载需要网络。如果实例无法访问官方源，可通过 SSH 从本地传入原始 XML 和已缓存的固定版本权重，再做校验；这些文件不通过 GitHub 分发。

## 安装和训练前检查

所有项目命令均从项目根目录执行。先根据实际驱动选择官方 Torch CUDA wheel；安装脚本支持 `cu126`、`cu128`、`cu130`，默认 `cu128`。这些通道对应 PyTorch 2.11.0 的官方发行选项。[PyTorch 安装说明](https://pytorch.org/get-started/previous-versions/#v2-11-0)

```bash
cd /root/autodl-tmp
git clone https://github.com/zoeSuen77/clinical-representation-project.git
cd clinical-representation-project
git rev-parse HEAD
nvidia-smi

# 已有目录时先检查版本及本地更改，不重复 clone 或强制覆盖。
# 若 python3.12 不在 PATH 中，PROJECT_PYTHON 指向实例内的 Python 3.12。
PROJECT_PYTHON=python3.12 TORCH_CHANNEL=cu128 bash scripts/setup_autodl.sh
source .venv-autodl/bin/activate

python -m pytest -q
python -m experiments.prepare_data --config configs/final_main.json
python -m experiments.preflight --config configs/final_main.json
python -m experiments.smoke_models --device cuda \
  --workdir build/autodl-smoke-01 --output results/pretrain/autodl_smoke.json
python -m experiments.launch_final
```

最后一条只打印计划，不训练。小样本检查使用 30 句人工数据、五种方法、10 次技术评估，不构成科学实验结果。小样本工作目录必须是新目录；再次检查时使用新编号。

安装脚本不会覆盖已有虚拟环境。首次安装失败后，应检查错误并修复该环境，或指定一个新的 `PROJECT_VENV`。`pip freeze` 和实际 CUDA 环境会保存到 `results/pretrain/`。本地验证的 macOS 锁文件不是一份已实测的 Linux/CUDA 环境证明。

## 实机放行标准

开始正式运行前，需要全部满足：

1. `autodl_environment.json` 中 `passed=true`，CUDA 张量运算成功。
2. 测试全部通过，无依赖冲突。
3. 数据指纹、9,647 个 ID 的划分清单与固定协议一致，预检显示 65 次计划评估。
4. 三种固定 revision 的模型均加载成功，输出有限的 768 维向量，缓存复用通过。
5. 人工数据的 10 次评估审计通过，恢复运行跳过已完成组。

实机条件不通过时，先解决驱动、依赖、下载或文件问题。不能为了让检查通过而修改正式数据划分或标签。

## 启动与恢复

采用 `tmux` 或 `screen` 保持进程独立于 SSH 连接。AutoDL 官方也建议长任务使用守护会话，避免断线结束程序。[后台运行说明](https://www.autodl.com/docs/daemon/)

```bash
tmux new -s group8-final
source .venv-autodl/bin/activate
python -m experiments.launch_final --device cuda --execute
```

用 `Ctrl+B` 后按 `D` 离开 tmux，`tmux attach -t group8-final` 返回。不要在已有运行进程尚存时再开一份训练。

运行顺序为 TF-IDF、BERT、BioBERT、PubMedBERT。每组单独保存日志；全部完成后自动审计 65 次评估并生成表格和图。进度记录为 `results/metrics/final_main/launch_status.json`；详细日志在 `results/logs/final_main/`。

进程确实退出后，可以恢复：

```bash
python -m experiments.launch_final --device cuda --execute --resume
```

恢复单位是特征组：通过审计且设备一致的组被跳过，不完整或损坏的组重新拟合。它不承诺从任意单次拟合或编码 batch 断点继续。完整表征缓存通过校验后可复用；中断的表征提取需重新提取该模型。CPU、GPU 和不同缓存设置使用不同键。

配置、数据与已有结果不一致时拒绝继续；需要新实验名称。文件锁防止两个 launcher 同时操作同一实验，请通过 launcher 启动正式训练。

## 验收和回传

```bash
python -m experiments.audit_results --config configs/final_main.json
python -m src.visualization --name final_main
mkdir -p build
tar -czf build/group8_final_results.tar.gz \
  configs requirements-lock.txt src experiments scripts tests \
  data/processed/final_main_manifest.json \
  results/metrics/final_main results/metrics/final_main_data_audit.json \
  results/figures/final_main results/logs/final_main results/pretrain
```

从本机通过 `scp -P PORT root@HOST:/root/autodl-tmp/clinical-representation-project/build/group8_final_results.tar.gz 本地目录` 回传，解压后重算指标并保存本次 Git commit。以上打包范围不含原始文章、模型权重或认证文件；本地模型权重可按需单独保留。

确认回传文件、65 次审计和图表一致后，再在控制台停止计费实例。不能把后台进程结束等同于通过验收，也不能把 GitHub 更新等同于课程提交。

## 当前验证边界

本地已使用全新 Python 依赖环境验证数据重建、模型 CPU 加载、缓存、人工数据拟合、审计、图表和已完成组恢复。AutoDL 的网络、驱动、CUDA 运算、GPU 显存峰值和实际运行时间，均需在租用实例后验证。后续跨文档划分、类别比例对照、人工错误分析和期末材料仍按完成计划执行。
