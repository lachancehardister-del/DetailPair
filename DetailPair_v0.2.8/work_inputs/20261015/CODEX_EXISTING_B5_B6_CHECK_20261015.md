# Codex任务：复用旧同轮B5/B6检查点的有限对照，不新增训练

日期：2026-10-15。先读MECHANISM_RESULT_REVIEW_20261015.md。当前已有64图诊断支持门控处局部冲突，但A平均norm约为D的2%，不证明删A提高AP；旧第38轮CSV反而B5略低于B6。完整B6的正式扩展保持停止。本轮只盘点和利用现有材料，避免再增加长训练。

## 1. 已有材料

实际最新v0.2.7工程及DetailPair_mechanism_results_20261015_141454_447971.zip（92项）内完整源码。下一版独立目录，建议D:\ModelImprovement\DetailPair_v0.2.8，不覆盖v0.2.2/.4/.5/.6/.7或旧结果。

旧结果DetailPair_results_20261011_164903_059647.zip已在当前对话；其中pilot__20261011_112235_644745为B2/B6各40轮、B5完成38轮后第39轮CUDA中断的运行。旧CSV第38轮AP50–95：B5 .26453、B6 .26884。不是本次200轮B6，也不是其精确恢复。

本地候选权重目录：
- D:\ModelImprovement\DetailPair_v0.2.2\runs\pilot__20261011_112235_644745\B5_s0\training\weights
- D:\ModelImprovement\DetailPair_v0.2.2\runs\pilot__20261011_112235_644745\B6_s0\training\weights

先用这些显式目录及用户已有归档位置，不扫描整个D盘；目录迁移允许配置重定位并核验保存的原run身份。文件是否仍存在未知，不要求用户上传权重或重复上传已有结果。

数据仍为D:\datasets\NEU-DET_DetailPair_s0_20261011_104211_157196\data.yaml；train1438/val180/test180不改变。环境仍为D:\ModelImprovement\envs\detailpair_cuda_v021，Torch2.5.1+cu121/vision.20.1+cu121/Ultralytics8.3.221，启动实查，不自动升级、重建或CPU回退。

## 2. 第一阶段：检查点库存和确定性配对

只读列出epoch*.pt以及元数据可解释的last.pt/best.pt；保存路径/大小/SHA256/模型与EMA类型/variant/pair_config/train_args/保存epoch/参数有限性。torch.load仅针对可信本地既有文件。若旧CUDA失败前的检查点损坏，明确记录，不静默跳过为“成功”。

查已安装8.3.221的save_model源码确认checkpoint epoch字段如何对应CSV epoch，保存这段源码。不能把epoch37.pt文件名直接写成第37轮；通常是零起算，但必须从实际语义与元数据核验。

从B5/B6都实际存在、模型身份正确、元数据epoch可确认、各自CSV已经完整写入且不超过第38轮的检查点中，取**最新共同完成轮**。不看AP选epoch，不选各自best，不用B6_200代替。若38缺失可按同一规则选择更早最新共同轮，记录原因；若不存在可确认的共同轮，写status=blocked_missing_matched_checkpoint并正常打包库存，后续不自动训练/恢复/微调。

核对两检查点同原run、seed0、scratch、320配置、训练batch2、计划epochs40/patience50、AdamW/lr0.001/lrf.01、扰动/增强/优化器/选择政策一致及训练选样哈希一致；原实际训练结束轮数可以不同，但选中的共同轮必须相同。记录无法从旧包证明的初始化张量哈希等项，不凭同seed宣称历史轨迹完全相同。

B5/B6模型都应为DetailDown(gated=True)，同层1和参数规模，差异是pair_config.energy=0/.02，其余pair/cons=.05相同。现有validate_checkpoint为B2/B6、200轮身份写了特定检查，不能直接套用导致B5被误拒绝，也不能全局放松原入口；新增专用old-matched验证器，仅允许这个40轮运行和B5/B6条件。

## 3. 第二阶段：配对检查点只读评价

只有库存配对成功才执行。两个独立训练权重均按相同EMA/model选择规则加载，float32、eval、device0、实际320×320、batch1。核验自定义类和names，权重/源数据前后哈希不变；不写.pt、不optimizer、不resume。

仅val180图、429GT，两种场景：clean、Gaussian std=.03，seed178，按已有派生代码构造同一份共享图，原图/标签不改。原生输入解码、clip/量化、图/标注对应和哈希记录；与前轮同样规则。只进行4组模型×场景评价，不重新跑完整噪声谱或性能测速。

- 官方AP：conf=.001、NMS IoU=.7、max_det300、rect=False、half=False、batch1，记录每次实际shape应为[1,3,320,320]；AP50–95主指标，AP50和各类AP辅助。
- 固定P/R：conf=.25、single-label NMS、match IoU=.5、一对一匹配，输出每类TP/FP/FN/GT、宏P/R与FPPI以及逐图预测/匹配。
- 每项保存checkpoint SHA256、实际共同训练轮、原预算40、配置/数据/噪声身份、原始指标和状态。旧CSV的验证actual shape未逐步记录，不把旧CSV当新320指标。
- 结果表统一B5−B6差值；不要把B5_38的AP直接比较B2_162或B6_200做最终优劣主张。不同预算历史只列参考且明确不可作本轮单因素效应。

关闭幅值损失只影响训练目标，在相同B6权重上把A系数置0再推理不会形成无幅值项训练消融；禁止用这种做法代替独立B5权重。

## 4. 边界与决策

这是已有短预算单种子对照，不是新方法、完整因果证明、正式多种子消融或最终论文性能。B5和B6架构相同，删训练损失不会移除Haar门控推理开销，因此即使AP改善也不等于更轻或更快。

若B5未在clean主AP和Gaussian .03上显示一致收益，保持封存当前组合，不自动调μ、补训或换seed寻找正结果。若方向一致也只回传Work判断是否值得新的明确假设/受控实验，不自动长训。若权重缺失只回传库存与影响，Work再决定是否另行成对训练；本任务没有这项授权。

## 5. 需要修改和交付

新增独立matched_checkpoint_eval.py/配置/库存模块及Windows入口，复用已有evaluation/reporting/data与结果pack；保持原模型/损失/训练入口不变。按实际任务读取科研工具箱适用原版模块，实查import/版本/设备。保留原生进度条，禁止PIPE逐行tee；缺失、损坏、身份错误、评价失败都写状态并打包，不把blocked说成实验成功。

完整独立工程ZIP：源码、依赖、配置、README、Windows激活/运行.cmd、结果打包入口、协议、CHANGELOG、实际CHECKS及PROJECT_STATUS.md；先交完整ZIP再安排本地执行。不以片段或目录清单交付。

可执行的工程检查：错variant/不同run/不同训练轮/只凭文件名推轮次/匹配权重缺失应拒绝评价；检测记录可人工复核、数据不改、只有两场景、实际输入shape和权重哈希可复查。Windows/GPU/真实旧检查点未执行就标pending。

结果ZIP包含库存、配对理由、原始指标/预测、身份/环境和失败，默认排除原图/标签/派生全图/权重。原数据、旧版本、旧CUDA失败与当前64图诊断保留；test封存，无付费API、外部账号、投稿、编辑通信或外传原数据/权重。
