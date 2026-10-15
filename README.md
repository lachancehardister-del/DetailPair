# DetailPair

检测模型研究项目。协作流程：Work审阅本地实验结果并给出方案，Codex实现与验证，用户在Windows运行后回传Work，继续迭代。

| 版本 | 介绍与代码 | 发布包 | 记录 |
| --- | --- | --- | --- |
| v0.2.9（当前归档） | [封存决定与精度口径修正](DetailPair_v0.2.9/README.md) | [完整归档工程ZIP](DetailPair_v0.2.9/releases/DetailPair_v0.2.9_Archive_Package.zip) | [决策](DetailPair_v0.2.9/DECISION_LOG.md) · [索引](DetailPair_v0.2.9/RESULTS_INDEX.csv) · [检查](DetailPair_v0.2.9/CHECKS.md) · [日志](DetailPair_v0.2.9/WORK_LOG.md) |
| v0.2.8 | [旧B5/B6同轮检查与GPU评价](DetailPair_v0.2.8/README.md) | [完整代码ZIP](DetailPair_v0.2.8/releases/DetailPair_v0.2.8_Experiment_Package.zip) | [检查](DetailPair_v0.2.8/CHECKS.md) · [日志](DetailPair_v0.2.8/WORK_LOG.md) |
| v0.2.7 | [GPU局部机制诊断](DetailPair_v0.2.7/README.md) | [完整代码ZIP](DetailPair_v0.2.7/releases/DetailPair_v0.2.7_Experiment_Package.zip) | [检查](DetailPair_v0.2.7/CHECKS.md) · [日志](DetailPair_v0.2.7/WORK_LOG.md) |
| v0.2.6 | [GPU复查说明](DetailPair_v0.2.6/README.md) | [完整代码ZIP](DetailPair_v0.2.6/releases/DetailPair_v0.2.6_Experiment_Package.zip) | [检查](DetailPair_v0.2.6/CHECKS.md) · [日志](DetailPair_v0.2.6/WORK_LOG.md) |
| v0.2.5 | [运行说明](DetailPair_v0.2.5/README.md) | [完整代码ZIP](DetailPair_v0.2.5/releases/DetailPair_v0.2.5_Experiment_Package.zip) | [检查](DetailPair_v0.2.5/CHECKS.md) · [日志](DetailPair_v0.2.5/WORK_LOG.md) |
| v0.2.4 | [运行说明](DetailPair_v0.2.4/README.md) | 源码保留在版本目录 | [环境与状态审阅](DetailPair_v0.2.4/onboarding/REVIEW.md) |

当前状态：`archived_no_consistent_benefit`。旧同完成38轮B5/B6未观察到删幅值项的一致收益，按Work预设规则封存当前组合及直接删项方案。v0.2.9仅离线归档与修正宏精度支持集合/零约定/micro报告，**本轮无需用户本地运行**。保留论文模型改进目标，等待Work的新瓶颈假设与最近邻审计；未实现下一模型或安排训练、GPU评价、测速、test。

仓库约定：每个新版本建立独立DetailPair_vX.Y.Z目录，其源码、配置、输入材料、报告、日志、工具和发布包全部放在该版本内，保留旧版本和失败记录。根目录只放介绍（隐藏Git配置除外）。用户已授权自动提交与同步后续工作，并记录验证与同步结果；遵守各轮方案的数据/权重上传范围。版本开发指引见[当前版本AGENTS](DetailPair_v0.2.9/AGENTS.md)。

后续收到Work方案和结果包，即按既定流程检查、实现、验证、同步并交付新版，无需重复确认执行意图；本地正式评价使用GPU，无CPU回退。
