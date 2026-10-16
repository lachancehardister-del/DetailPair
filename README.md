# DetailPair / BoxRoute

检测模型研究项目。协作流程：Work审阅本地实验结果并给出方案，Codex实现与验证，用户在Windows运行后回传Work，继续迭代。

| 版本 | 介绍与代码 | 发布包 | 记录 |
| --- | --- | --- | --- |
| BoxRoute v0.3.0（当前工程） | [G0运行说明](BoxRoute_v0.3.0/README.md) | [完整代码ZIP](BoxRoute_v0.3.0/releases/BoxRoute_v0.3.0_Code.zip) | [方案](BoxRoute_v0.3.0/MODEL_PROPOSAL.md) · [检查](BoxRoute_v0.3.0/CHECKS.md) · [日志](BoxRoute_v0.3.0/WORK_LOG.md) |
| v0.2.9（已验收归档） | [封存决定与精度口径修正](DetailPair_v0.2.9/README.md) | [完整归档工程ZIP](DetailPair_v0.2.9/releases/DetailPair_v0.2.9_Archive_Package.zip) | [决策](DetailPair_v0.2.9/DECISION_LOG.md) · [索引](DetailPair_v0.2.9/RESULTS_INDEX.csv) · [检查](DetailPair_v0.2.9/CHECKS.md) · [日志](DetailPair_v0.2.9/WORK_LOG.md) |
| v0.2.8 | [旧B5/B6同轮检查与GPU评价](DetailPair_v0.2.8/README.md) | [完整代码ZIP](DetailPair_v0.2.8/releases/DetailPair_v0.2.8_Experiment_Package.zip) | [检查](DetailPair_v0.2.8/CHECKS.md) · [日志](DetailPair_v0.2.8/WORK_LOG.md) |
| v0.2.7 | [GPU局部机制诊断](DetailPair_v0.2.7/README.md) | [完整代码ZIP](DetailPair_v0.2.7/releases/DetailPair_v0.2.7_Experiment_Package.zip) | [检查](DetailPair_v0.2.7/CHECKS.md) · [日志](DetailPair_v0.2.7/WORK_LOG.md) |
| v0.2.6 | [GPU复查说明](DetailPair_v0.2.6/README.md) | [完整代码ZIP](DetailPair_v0.2.6/releases/DetailPair_v0.2.6_Experiment_Package.zip) | [检查](DetailPair_v0.2.6/CHECKS.md) · [日志](DetailPair_v0.2.6/WORK_LOG.md) |
| v0.2.5 | [运行说明](DetailPair_v0.2.5/README.md) | [完整代码ZIP](DetailPair_v0.2.5/releases/DetailPair_v0.2.5_Experiment_Package.zip) | [检查](DetailPair_v0.2.5/CHECKS.md) · [日志](DetailPair_v0.2.5/WORK_LOG.md) |
| v0.2.4 | [运行说明](DetailPair_v0.2.4/README.md) | 源码保留在版本目录 | [环境与状态审阅](DetailPair_v0.2.4/onboarding/REVIEW.md) |

旧DetailPair状态仍为`archived_no_consistent_benefit`，v0.2.9归档修正已由Work验收通过。新BoxRoute是Work已给冻结假设/最近邻核验的独立候选；本轮交付G0工程/形态支持/两轮流程/初始开销验证，用户先在Windows GPU运行G0回传Work。G1独立门禁，不能自动接60轮；无200轮、多seed或test。本轮CPU合成检查不代表真实NEU检测收益、GPU速度或创新性成立。

仓库约定：每个新版本建立独立DetailPair_vX.Y.Z目录，其源码、配置、输入材料、报告、日志、工具和发布包全部放在该版本内，保留旧版本和失败记录。根目录只放介绍（隐藏Git配置除外）。用户已授权自动提交与同步后续工作，并记录验证与同步结果；遵守各轮方案的数据/权重上传范围。版本开发指引见[当前版本AGENTS](BoxRoute_v0.3.0/AGENTS.md)。

后续收到Work方案和结果包，即按既定流程检查、实现、验证、同步并交付新版，无需重复确认执行意图；本地正式评价使用GPU，无CPU回退。
