# v0.2.7 工作日志

## 2026-10-15（Asia/Shanghai）：Work局部机制诊断工程

- 按已授权迭代流程读取审阅包并直接实施。Work决定停止完整B6正式扩展，仅做终点/冻结BN局部诊断，不训练/测速/val/test/新结构/监督修改。
- 原输入DetailPair_RecheckReview_20261015.zip SHA256=2c8a417680d5558bb65e234284a27cbddeb748d4a6ac6caef8effdb58a8e9549；7项CRC/安全路径检查通过。保留Work报告、任务、audit/区组CSV/分析脚本/首次失败与最新状态。
- 本轮未收到Work引用的214项原始recheck ZIP；报告里的1440条筛查、6.4685/7.1978ms、352/320有效shape等按Work复核引用，不冒充本轮独立重算。
- 独立建立v0.2.7，model.py/engine.py/data.py字节不变。实现train-only规范排序Random178取64，冻结选样内容hash/类别覆盖/源只读；官方LetterBox正确图框变换、实际batch2/320；两条件独立Generator100003+batch_index同图同噪声。
- 未融合权重副本，BN/随机层eval、仅layer1 requires_grad；使用原loss原语和完整checkpoint train_args。每批原loss total/整个目标层梯度对齐，前/背景分解和A梯度重构，B6三组梯度/三子带门控幅值统计，正确标识zero/disconnected/N/A和cos null。
- 参数/BN/磁盘权重/选中源数据hash核验、无optimizer/backward/epoch/参数更新/pt写回，batch图及时释放、临时hook/capture/cache异常清理。保存全部批状态/分布/负cos比例/空区域和失败，不把相关批次当独立训练。
- 首轮集成发现Ultralytics会重置简化model.args，导致缺loss超参数；改从checkpoint完整train_args恢复，并缺失拒绝，不猜默认值。原失败log/run/ZIP保留。旧真实身份记录保存的box7.5/cls.5/dfl1.5已核对；真实权重仍须本地读取。
- 8项工程测试及CPU随机checkpoint/合成train-only集成通过（4选中图、两条件各2批、实际320）：original total/grad对齐，CPU最大梯度差两条件均0，源/BN/参数/权重不变，同扰动核验通过。九个标量解析/有限差分、空区域/stop-grad/零范数、标签变换、缺配置失败及ZIP排除通过。
- 发布ZIP解包后源清单/语法、help/pack-only/旧pack转发验证通过；pip check通过。科研工具箱原版资源本环境不可用，未虚构已读，实际源码/依赖及数学工程检查如实记录。
- Windows/GPU/真实64train、每条件32批机制结果pending；不能归因FP/AP或称B5消融/监督收益。本地GPU环境不升级、不CPU回退，下一步windows\22_mechanism_diagnosis.cmd，完整结果ZIP交Work判断。
- 发布包SHA256：1b3f57e02221ae01c16d7d14d3abfd1ce07a6e0ccceca261f4ebfadd881508d6
- 保留原各版本及工作区原有v0.2.4/onboarding删除，不将这些已有删除纳入提交；所有新材料和日志在本版目录，根只更新介绍。主体推送后另记远端核验。
