# Codex下一轮任务：复用B2/B6权重做统一评价，不重新训练

日期：2026-10-13。先阅读LONG_RESULT_REVIEW_20261013.md及既有CODEX_TASK_DetailPair_B2_B6_v0.2.5.md。本说明更新下一步顺序；两者不一致时本轮说明优先。科研判断由Work给出，Codex实现完整工程，用户Windows执行后把结果ZIP回传Work。

## 目标

本次真实结果B2=162轮早停、B6=200轮上限，均成功；独立val mAP50–95为34.8363%和34.6994%。完整B6未显示整体精度提升。先复用已训练权重验证精度、类别召回、端到端速度及合成扰动鲁棒性，禁止自动追加训练、换模型、调损失权重或启动全消融。

## 已有材料

- 完整DetailPair_v0.2.4_Experiment_Package.zip及源码；已交付任务说明指定的下一版v0.2.5如果已经存在，先读取其实际实现及检查记录，不假定已完成，不覆盖现有代码版本。
- 新结果包DetailPair_results_20261013_054850_726320.zip（133项，SHA256=5d65fa0e74691994f2efd6332f4c0e28bf53614395a136065bf3db685a8d8a99）。只核验日志，不把包内缺权重误判成模型未训练。
- 数据：D:\datasets\NEU-DET_DetailPair_s0_20261011_104211_157196\data.yaml，train1438/val180/test180，沿用不重划分。
- 环境：D:\ModelImprovement\envs\detailpair_cuda_v021，旧记录Torch2.5.1+cu121/vision.20.1+cu121/Ultralytics8.3.221，当前执行时重新实查，默认不安装升级。
- 本轮B6权重日志路径：D:\ModelImprovement\DetailPair_v0.2.4\runs\pilot__20261012_125741_680519\B6_s0\training\weights\best.pt。
- 本轮B2权重日志路径：D:\ModelImprovement\DetailPair_v0.2.4\runs\pilot__20261012_230632_329133\B2_s0\training\weights\best.pt。

权重磁盘存在及内部元信息未由Work核验；本地入口启动时实查，计算SHA256并验证自定义checkpoint和variant。不能上传权重、静默换官方权重或resume。日志显示最终best/last已strip optimizer，不凭这些权重断言可以精确恢复训练。

## 修改与交付

1. 按既有完整任务说明实现原生终端进度条和batch1端到端评价测速，保持model.py/engine.py核心训练目标不变。保留旧run.py前向benchmark的原语义，增加明确新入口；禁止PIPE逐行tee训练刷屏。
2. 新的配置默认显式绑定上面两个权重、val、device0、320×320、batch1、FP32；工作点conf=.25、NMS IoU=.7、max_det300、匹配IoU=.5。这些为新增协议默认值，写入配置和报告；AP低阈值评价与固定工作点P/R区分。
3. 主测速：已解码主机图像到主机检测结果，包含预处理、H2D、推理、NMS、D2H；排除磁盘读/解码、加载、写文件、绘图。GPU前后同步，warmup30次排除、全val180图重复3轮，记录实际张量shape=1×3×320×320、融合状态与各条件模型参数。输出每图latency/预测数、mean/P50/P95和同口径串行FPS。不与任何训练或prunecheck任务同时测速，启动时读取GPU进程；发现其他计算任务给出具体诊断，不替用户结束其他进程。
4. 统一clean评价，逐类别AP及固定工作点TP/FP/FN/Precision/Recall/GT、预测和匹配记录，含crazing弱类与pitted_surface下降。原日志中的类别指标仅四舍五入且非conf=.25，不能复制为新实测。官方独立重载AP与旧metrics.json差异原样保存，不悄悄用旧值替换。
5. 新增共同合成扰动评价：Gaussian std=.01/.03/.06；impulse probability=.01（黑/白各一半），在RGB[0,1]范围生成、clip并量化保存，seed178。生成一份共享派生val副本给两条件，图与标签对应、哈希可核验；原数据不改。明确是否各档重新seed或同随机基样本，配置冻结，结果包含实际SNR及生成规则。复用原robustness实现中可用代码，先实查不要猜API。
6. 每个扰动档位报告官方AP、固定阈值各类P/R、相对各自clean下降、B6−B2差值。全档位报告，不筛除不利结果；合成扰动不是采集噪声，新增分析是clean结果后提出的探索性方案。计时主比较在clean图上，扰动评价不必扩大为所有档位完整测速。
7. test保持留出不默认执行，方法与阈值冻结后才能显式最终测试。当前不新增训练种子；将追加多种子/消融安排标成待Work判断。

## 验收

- 检查点variant、seed/run和输入配置一一对应，权重只读；没有训练、optimizer.step或改写.pt。
- 数据selection/源哈希一致，val图180，GT429框；类别names以data.yaml为准，不能把多类图数相加要求180。
- 新测速原始数量=180×3×2条件，warmup不计，记录失败和实际缺样；均值/FPS/分位数可重算。单张重复测量不当训练独立重复。
- 定义一对一匹配并检查重复预测、错类、空GT、空预测；精确报告无分母情形，不把最佳F1工作点召回混入conf=.25。
- 扰动两条件用同一份图/标签/哈希；按原图ID保存生成清单。派生数据本地保存但回传ZIP默认不含原数据/权重/整套派生图片，保留清单、参数、哈希、指标与预测。
- 源码/配置/失败路径/结果pack检查真实完成；Linux无Torch时只能写已做静态/可独立检查，Windows/GPU环节待用户运行。不能以pip可安装或技能已安装称为运行成功。
- 若FLOPs工具不支持自定义层，标记未验证；不要抄替换前YOLOsummary做B6参数结论。

## 实际交付

完整独立版本ZIP：源码、依赖、配置、README、Windows激活环境及运行.cmd、结果ZIP入口；新增CHANGELOG、协议、实际检查记录、PROJECT_STATUS.md。先交付ZIP，再让用户本地运行，不能交付片段或仅任务目录。代码版本名按已有版本递增，旧代码、结果、失败均保留。

任务结果会回传Work，下一步依据是否有稳定类别/鲁棒性价值以及延迟代价决定：追加多种子、修订机制或停止当前组合。不能为证明预期收益追加大量搜索或以AP50单指标胜出宣布可投稿。无付费API、外部账号、投稿或编辑通信。
