# v0.2.9 工作日志

日期2026-10-15，Asia/Shanghai。依据用户持续授权，在独立版本内实现、验证、记录并自动同步。当前任务是Work给出的离线归档，不创建新本地训练或GPU评价任务。

输入DetailPair_MatchedReview_20261015.zip（12项，CRC/安全路径通过），原件与全解包保存在work_inputs/20261015。11项Work清单文件大小/SHA256一致，包括首次CRLF/LF源码哈希断言失败及修正前脚本；原字节不改。输入SHA256/大小见verification/input_provenance.json。原154项matched ZIP当前不在环境，未声称720条重匹配/AP曲线复算/真实磁盘核验或运行Work完整脚本，不要求重复上传。Work审阅为决策来源，和当前用户持续授权区分记录。

执行预设规则：旧同完成38轮B5未在clean和Gaussian.03显示去幅值项一致收益，封存当前完整B6及直接删项方案，状态archived_no_consistent_benefit。保留模型改进/论文目标，等待Work新瓶颈假设和最近邻公式/目标/开销核验；本轮没有执行这些研究，也未自行实现新模型。无训练、恢复、微调、加seed、扫μ、梯度手术、测速或test。

从v0.2.8复制完整源码、依赖、配置和原Windows/结果打包入口，旧版本/运行/原JSON/CSV/失败不改；不复制旧runs/exports/设置/旧verification。导入Work完整PROJECT_STATUS后追加实际事实。新增CURRENT_BRANCH_CLOSED、DECISION_LOG、RESULTS_INDEX和archive。model/engine/data逐字节一致。

定位实际固定指标聚合在reporting.summarize_counts，evaluation.fixed_evaluate调用它。保留逐类precision null和原available宏P/R值，新增支持类ID、数量、显式固定类别零约定宏P及pooled micro P/R。matched comparison在支持集合不同（即使数量相同）时将原宏差值置null并写原因，AP/逐类TP/FP/FN及旧输入不变；另列新定义与可比性，结果pack包含archive。

仅执行6项必要离线报告测试，均通过；Work派生24类表/4条件内部汇总算术与其场景表一致，11项审阅原文件hash通过。原始720图记录缺失，修正输出明确只以Work派生表为来源，不当新实验结果。最初日志相对路径工作目录错误在测试启动前纠正，后实际成功记录保留。没有跑完整工程集成或GPU。科研工具箱原版模块未提供，不宣称使用或做新文献/创新认证。现有云端环境满足离线操作，无安装、服务或新增环境配置。

归档状态、证据边界和实际检查见CURRENT_BRANCH_CLOSED、RESULTS_INDEX、CHECKS与verification。完整源码ZIP以releases内校验文件为准，本次无需用户运行。根目录只改介绍；原v0.2.4/onboarding九项未暂存删除保持不动。同步结果在验证远端后追加。

首轮发布校验器错误期望requirements_cpu.txt（项目实际为requirements.txt），更正检查器后重验，未改依赖声明以迎合检查；失败日志保留verification/release_check_initial.log。

最终完整归档包源码CRC/逐文件SHA256、必要文件、解包6项离线测试与仅pack入口检查通过；本地源/解包目录均未创建runs，无模型执行。包SHA256：ea3a1b5b6a9b71b376f29f899e9402d0aa2f753a5da6a79e202ed4d26ce02cd2；大小289185 bytes。

## GitHub同步验证

2026-10-15T16:56:34.804387+08:00：完整源码归档包、报告口径修正、原Work审阅材料和实际离线检查已推送origin/main，远端commit `28269dfc23bd05533a886f9b043938c48d4a6080` 与本地HEAD一致。本轮无真实原图/权重上传及新实验；后续仅追加本同步记录，不改发布包。v0.2.4/onboarding九项既有删除未暂存。
