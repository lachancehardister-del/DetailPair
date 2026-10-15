# v0.2.9 实际离线检查

2026-10-15（Asia/Shanghai）。本次无需用户本地运行，无训练、恢复、微调、GPU评价、测速或test；未运行全工程集成。

## 已执行

`/workspace/venvs/detailpair/bin/python tests/check_reporting_archive.py`：6项通过（verification/reporting_checks.log）。仅汇总算术与序列化，没有模型调用。

1. 无预测但GT>0：逐类precision null、recall0，原available宏P排除该类；新增固定类别零约定与micro明确不同。
2. 有FP无TP：precision=0有分母且纳入支持集合，原计数不改。
3. 各类有预测：全类别宏P与micro按手算不同（.625与.4），零约定在此等于原宏P。
4. 支持数量相同但类ID不同：原宏P差值null、不可直接比较并说明原因；recall亦按真实支持集处理；固定全类别约定可比。
5. 全空预测/GT时原宏与micro未定义，零约定0仅标记为约定；类别映射不同拒绝比较。
6. 新comparison兼容旧缓存无支持字段的形式，由原逐类计数推导；AP、逐类TP/FP/FN及输入JSON深拷贝前后一致，CSV记录支持数量与不可比原因。

`/workspace/venvs/detailpair/bin/python tools/archive_matched_review.py`：11项Work原文件大小/SHA256通过；由Work派生classes_recomputed.csv的24行计数核验4条件汇总，与其scenarios表完全一致；六类零约定与micro匹配任务数值。来源和局限明确写入verification/offline_archive_checks.json。原审阅ZIP12项CRC/路径检查通过、原文件未改；原始154项结果ZIP不在环境中。

此处是**Work派生表的内部算术检查**。720条预测重匹配、1440实际输入记录检查、官方AP、Windows磁盘只读3242文件均为Work报告，不是当前Codex重跑。未执行原review_matched_20261015.py，因为所依赖原ZIP/此前材料缺失；保留脚本及原CRLF失败作为来源。官方AP值原样转存，不重新积分。

model.py/engine.py/data.py与v0.2.8逐字节一致，SHA256见verification/source_identity.json。实际依赖及环境见verification/environment_offline.json；没有安装或升级。当前代码所需科研工具箱原版模块未提供，不宣称已读，也不开展新研究审计。

发布检查：Python AST、完整ZIP CRC/逐文件SHA256、必要依赖/配置/Windows/打包入口存在性及解包后的同6项离线测试。检查结果写入verification/release_check.json。只使用离线报告测试，不执行GPU/训练入口；旧代码/原结果未改。

## 边界与纠正

生成文件后首次测试命令从仓库根目录调用了版本内相对日志路径，重定向即失败，测试没有启动；已改为版本工作目录后实际执行并通过，未把未执行当通过。任务输出仅在v0.2.9新目录。v0.2.4/onboarding九项既有未暂存删除不处理、不提交。

无需新的Windows/GPU验收来封存或修正已有指标解释；没有安排用户补跑。未来任何新模型/实验须等待Work明确新设计，现有入口和依赖保留仅供追溯与后续授权使用。

首轮包校验脚本误把实际requirements.txt命名为requirements_cpu.txt，存在性断言失败；包内实际依赖文件齐全。已纠正检查器文件名，原失败保留verification/release_check_initial.log，之后重新检查。
