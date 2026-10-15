# DetailPair

检测模型研究项目。协作流程：Work审阅本地实验结果并给出方案，Codex实现与验证，用户在Windows运行后回传Work，继续迭代。

| 版本 | 介绍与代码 | 发布包 | 记录 |
| --- | --- | --- | --- |
| v0.2.7（当前） | [GPU局部机制诊断](DetailPair_v0.2.7/README.md) | [完整代码ZIP](DetailPair_v0.2.7/releases/DetailPair_v0.2.7_Experiment_Package.zip) | [检查](DetailPair_v0.2.7/CHECKS.md) · [日志](DetailPair_v0.2.7/WORK_LOG.md) |
| v0.2.6 | [GPU复查说明](DetailPair_v0.2.6/README.md) | [完整代码ZIP](DetailPair_v0.2.6/releases/DetailPair_v0.2.6_Experiment_Package.zip) | [检查](DetailPair_v0.2.6/CHECKS.md) · [日志](DetailPair_v0.2.6/WORK_LOG.md) |
| v0.2.5 | [运行说明](DetailPair_v0.2.5/README.md) | [完整代码ZIP](DetailPair_v0.2.5/releases/DetailPair_v0.2.5_Experiment_Package.zip) | [检查](DetailPair_v0.2.5/CHECKS.md) · [日志](DetailPair_v0.2.5/WORK_LOG.md) |
| v0.2.4 | [运行说明](DetailPair_v0.2.4/README.md) | 源码保留在版本目录 | [环境与状态审阅](DetailPair_v0.2.4/onboarding/REVIEW.md) |

Work已决定停止当前完整B6正式扩展。v0.2.7仅使用已有终点权重、固定64张train图，在GPU上进行冻结BN且无参数更新的局部损失/门控诊断；不训练、测速或使用val/test。CPU合成检查不代表真实局部机制结论。

仓库约定：每个新版本建立独立DetailPair_vX.Y.Z目录，其源码、配置、输入材料、报告、日志、工具和发布包全部放在该版本内，保留旧版本和失败记录。根目录只放介绍（隐藏Git配置除外）。用户已授权自动提交与同步后续工作，并记录验证与同步结果；遵守各轮方案的数据/权重上传范围。版本开发指引见[当前版本AGENTS](DetailPair_v0.2.7/AGENTS.md)。

后续收到Work方案和结果包，即按既定流程检查、实现、验证、同步并交付新版，无需重复确认执行意图；本地正式评价使用GPU，无CPU回退。
