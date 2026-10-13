# DetailPair

检测模型研究项目。协作流程：Work审阅本地实验结果并给出方案，Codex实现与验证，用户在Windows运行后回传Work，继续迭代。

| 版本 | 介绍与代码 | 发布包 | 记录 |
| --- | --- | --- | --- |
| v0.2.5（当前） | [运行说明](DetailPair_v0.2.5/README.md) | [完整代码ZIP](DetailPair_v0.2.5/releases/DetailPair_v0.2.5_Experiment_Package.zip) | [检查](DetailPair_v0.2.5/CHECKS.md) · [日志](DetailPair_v0.2.5/WORK_LOG.md) |
| v0.2.4 | [运行说明](DetailPair_v0.2.4/README.md) | 源码保留在版本目录 | [环境与状态审阅](DetailPair_v0.2.4/onboarding/REVIEW.md) |

当前v0.2.5复用B2/B6权重进行评价，默认val，不重新训练；真实权重和原图保留在本地。云端合成检查不代表真实NEU精度或GPU速度。

仓库约定：每个新版本建立独立DetailPair_vX.Y.Z目录，其源码、配置、输入材料、报告、日志、工具和发布包全部放在该版本内，保留旧版本和失败记录。根目录只放介绍（隐藏Git配置除外）。用户已授权自动提交与同步后续工作，并记录验证与同步结果；遵守各轮方案的数据/权重上传范围。版本开发指引见[当前版本AGENTS](DetailPair_v0.2.5/AGENTS.md)。
