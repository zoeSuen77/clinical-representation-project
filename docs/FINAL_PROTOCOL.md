# 最终主实验协议

版本：2026-10-20-v1

小组：Group 8 — Yuhan Sun、Zeyu Pan、Jiayu Liao

配置：`configs/final_main.json`

## 研究问题

冻结的通用或医学预训练表征，在什么训练预算下更有助于 BioScope 句级否定与推测提示词检测？结论对文档划分和训练类别比例是否敏感？

主任务为 `no_cue`、`negation_only`、`speculation_only` 三分类。混合提示词句子暂时排除。任务不提供疾病实体的肯定、否定或不确定状态标签。

本协议在查看过 pilot 结果后确定，因此最终主实验是探索性扩展。语料和文档可能与 pilot 重叠；不能把新测试集称为完全未见的独立外部验证集。

## 数据与划分

- 来源：官方 BioScope abstracts XML；原始数据 SHA-256 固定在配置中。源文件不匹配时，准备步骤失败。
- 原始 11,871 句，保留 11,559 句。混合提示词排除及精确重复处理延用 pilot 规则。
- 在划分前连接共享精确重复句的文档，并追加保守的近重复分组。
- 近重复规则：Unicode 单词及数字转小写，忽略标点；至少 8 个 token；取连续 5-token shingle 集合；Jaccard ≥ 0.90 时连接两句所属文档组。阈值不依据模型得分调整。
- 规则检查了 6,191 对候选句，命中 3 对，涉及两篇文档。文档组由 1,257 个变成 1,256 个；最大组含 7 篇文档。105 个短句未进入近重复规则。这不保证排除全部语义改写或短模板。
- 使用五折 StratifiedGroupKFold，`split_seed=1729`；fold 0 为测试，fold 1 为验证，fold 2–4 为训练。
- 训练和测试不封顶；验证按类别分层抽取 400 句，`subset_seed=2026`。验证折中未选出的 1,912 句不移回训练集。

| 集合 | no_cue | negation_only | speculation_only | 总句数 | 文档数 | 文档组数 |
|---|---:|---:|---:|---:|---:|---:|
| 训练 | 5,076 | 780 | 1,083 | 6,939 | 760 | 754 |
| 验证 | 292 | 45 | 63 | 400 | 204 | 200 |
| 测试 | 1,686 | 260 | 362 | 2,308 | 256 | 251 |

合计使用 9,647 句。清单 `data/processed/final_main_manifest.json` 只保存 ID、标签和划分，不包含原文。

## 方法与训练预算

五个方法：TF-IDF + LR、TF-IDF + linear SVM、BERT + LR、BioBERT + LR、PubMedBERT + LR。三种模型的仓库 ID 和 commit revision 固定在配置中；与注册表不一致时拒绝运行。

- TF-IDF：word unigram/bigram、lowercase、sublinear TF、最多 30,000 维；词表和 IDF 只拟合当前训练子集。
- 冻结表征：最后一层 mean pooling，排除 padding 和特殊 token；长度上限 192，batch size 16。仅在当前训练子集拟合 StandardScaler。
- LR/SVM 均使用 balanced class weights。C 候选为 0.1、1、10，按验证 macro-F1 选择；并列时选择候选顺序中的第一个。
- 平衡训练预算：每类 10、25、50、100 句；抽样种子 13、37、73。同一种子下预算嵌套，各方法使用完全相同的训练 ID。
- 完整训练池每个方法只在 seed 13 拟合一次。全部使用固定的 400 句验证标签，因此预算是训练标签预算。
- 每个方法为 12 个平衡预算实验加 1 个完整池实验，共 **65 个计划评估**、**195 个 C 候选拟合**。这不是已完成实验数。
- 新实验使用独立的 `final_main` 目录。改变数据或协议后应采用新实验名；同协议显式重跑使用 `--overwrite`。

## 报告规则

主要指标为 macro-F1；另报 accuracy、macro precision/recall、各类 recall、固定标签顺序的混淆矩阵。平衡预算报告三个训练子集的均值和样本标准差。完整池只报告一个结果，不报告重复拟合产生的零标准差。

完整池具有自然类别比例，不连接到平衡预算学习曲线中，也不把两者差异解释为纯样本量效应。记录编码时间与分类器训练时间，模型与分词器差异也列为解释限制。

## 预先规定的后续对照

以下仍待执行，不计入第一天完成情况：

1. 使用另外两个固定的分组随机种子 **2903、4517** 重复主要比较，分别生成新配置和结果目录。训练抽样变化与文档划分变化分别报告。
2. 在主划分上固定总训练量 **300 句**，比较每类 100 句与自然比例分层抽样；采用相同的三个抽样种子、五种方法、验证和测试集。
3. 分层检查约 60 个案例，记录抽样规则与两名成员的复核情况。人工句子对继续作为定性诊断。

## 审计与执行入口

数据预检只表示可以开始训练，不能代替完成结果审计。通用审计要求完整的方法及种子预算矩阵、唯一测试预测、准确训练 ID、完整 C 搜索、匹配的个体及汇总文件，并重算全部指标。空目录、漏跑、损坏指标或收敛警告均不能通过。

已保存记录不能独立证明验证分数的计算过程或研究者从未查看测试结果，仍需要代码审查与独立复现。

```bash
python -m experiments.prepare_data --config configs/final_main.json
python -m experiments.preflight --config configs/final_main.json

# 默认只打印计划；当前尚未开始正式模型训练。
python -m experiments.launch_final

# 租好 AutoDL 并通过 CUDA 小样本验证后执行。
python -m experiments.launch_final --device cuda --execute
python -m experiments.audit_results --config configs/final_main.json
python -m src.visualization --name final_main
```

本地训练前准备与验证见 `docs/STAGE_REPORT.md`；远程环境、后台会话和恢复规则见 `docs/AUTODL_RUNBOOK.md`。缓存格式更新记录设备与校验和，不改变本协议的任务、模型 revision、数据划分或实验矩阵。

## 第一版指纹

- 配置：`b109ca55d6b6a0fbf93ea5201b6f20af724c5511d45ff171289433740d971ec5`
- 处理后数据：`99d26e5e462297a85ea970df68bc95744ae9b4b04527a8761db50cbc02b4eb8a`
- 划分清单：`63b535f3777eeec993194efcd566f2ed0fabafdd4beb7f82e3ef80304bf58626`

完整数据审计与预检记录位于 `results/metrics/final_main_data_audit.json` 和 `results/metrics/final_main/preflight.json`。
