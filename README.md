# DetailPair

检测模型研究项目。协作流程：Work审阅本地实验结果并给出方案，Codex实现与验证，用户在Windows运行后回传Work，继续迭代。

| 版本 | 介绍与代码 | 发布包 | 记录 |
| --- | --- | --- | --- |
| v0.2.8（当前） | [旧B5/B6同轮检查与GPU评价](DetailPair_v0.2.8/README.md) | [完整代码ZIP](DetailPair_v0.2.8/releases/DetailPair_v0.2.8_Experiment_Package.zip) | [检查](DetailPair_v0.2.8/CHECKS.md) · [日志](DetailPair_v0.2.8/WORK_LOG.md) |
| v0.2.7 | [GPU局部机制诊断](DetailPair_v0.2.7/README.md) | [完整代码ZIP](DetailPair_v0.2.7/releases/DetailPair_v0.2.7_Experiment_Package.zip) | [检查](DetailPair_v0.2.7/CHECKS.md) · [日志](DetailPair_v0.2.7/WORK_LOG.md) |
| v0.2.6 | [GPU复查说明](DetailPair_v0.2.6/README.md) | [完整代码ZIP](DetailPair_v0.2.6/releases/DetailPair_v0.2.6_Experiment_Package.zip) | [检查](DetailPair_v0.2.6/CHECKS.md) · [日志](DetailPair_v0.2.6/WORK_LOG.md) |
| v0.2.5 | [运行说明](DetailPair_v0.2.5/README.md) | [完整代码ZIP](DetailPair_v0.2.5/releases/DetailPair_v0.2.5_Experiment_Package.zip) | [检查](DetailPair_v0.2.5/CHECKS.md) · [日志](DetailPair_v0.2.5/WORK_LOG.md) |
| v0.2.4 | [运行说明](DetailPair_v0.2.4/README.md) | 源码保留在版本目录 | [环境与状态审阅](DetailPair_v0.2.4/onboarding/REVIEW.md) |

Work已决定停止当前完整B6正式扩展。v0.2.8只盘点旧40轮实验的B5/B6，选最新可确认共同完成轮（不超过38）后，在GPU上评价clean和Gaussian.03的val；没有配对就打包库存，不补训。64图机制结果已审阅，局部冲突不等于删损失改善AP。CPU合成检查只证明工程流程。

仓库约定：每个新版本建立独立DetailPair_vX.Y.Z目录，其源码、配置、输入材料、报告、日志、工具和发布包全部放在该版本内，保留旧版本和失败记录。根目录只放介绍（隐藏Git配置除外）。用户已授权自动提交与同步后续工作，并记录验证与同步结果；遵守各轮方案的数据/权重上传范围。版本开发指引见[当前版本AGENTS](DetailPair_v0.2.8/AGENTS.md)。

后续收到Work方案和结果包，即按既定流程检查、实现、验证、同步并交付新版，无需重复确认执行意图；本地正式评价使用GPU，无CPU回退。
