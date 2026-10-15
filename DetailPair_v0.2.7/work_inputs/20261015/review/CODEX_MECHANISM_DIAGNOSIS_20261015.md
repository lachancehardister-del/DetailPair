# Codex任务：现有权重的局部机制诊断，不训练新模型

日期：2026-10-15。前置阅读RECHECK_RESULT_REVIEW_20261015.md及工程现有模型/损失源码。当前循环：Work分析与研究决策，Codex实现完整工程，用户Windows运行，Work核验回传结果。

## 1. 目标与停止项

本轮平衡测速B2=6.4685ms、B6=7.1978ms，B6慢11.27%；同320输入val AP50–95为35.1279%/34.2651%，完整B6无整体收益。停止当前组合的正式扩展，不追加长训练、seed、全消融、B5从头训练、YOLO11s容量训练或反复测速。不更换数据或阈值，不执行test。

下一轮仅检验一个解释性假设：**现有细节幅值保留目标可能鼓励背景纹理通过门控，并与检测目标在局部产生冲突**。这是待验证假设，不是已确认根因或新颖性贡献。诊断结果只用于决定是否值得提出更改监督方式的下一步方案，不直接认定创新、泛化或去掉该损失必定改善。

## 2. 已有文件与资源

基于实际v0.2.6完整工程独立建立下一版（建议v0.2.7）；本轮DetailPair_recheck_results_20261015_133501_258306.zip含源码/configs/windows/tests/协议/README，可恢复。不要要求用户重传已有文件。

- D:\ModelImprovement\envs\detailpair_cuda_v021：原Torch2.5.1+cu121、torchvision.20.1+cu121、Ultralytics8.3.221；启动实查，默认不升级、不CPU回退。
- D:\datasets\NEU-DET_DetailPair_s0_20261011_104211_157196\data.yaml；仅train1438图用于本次诊断。val180/test180不用于这轮机制选样。
- B2权重：D:\ModelImprovement\DetailPair_v0.2.4\runs\pilot__20261012_230632_329133\B2_s0\training\weights\best.pt，SHA256=c234ae47e6d8257543a038abc3ea0714d0f081abf2ce5e9dbe0dcd9c73a2c21b。
- B6权重：D:\ModelImprovement\DetailPair_v0.2.4\runs\pilot__20261012_125741_680519\B6_s0\training\weights\best.pt，SHA256=2a5f5dd7152b28a3a3952f96fbeca409e231e07de2a06a6b5686d7a01c6bc324。
- 旧选择清单、train_args与环境继续绑定其run。源数据/权重只读，不上传。

按实际任务读取科研工具箱适用的原版深度学习/实验设计资源并实查依赖；无需重做全库文献检索。若涉及新的创新机制，后续由Work另核验文献，本任务不命名新算法。

## 3. 诊断设置

### 选样与扰动

从冻结train列表按源文件路径规范排序，使用本地独立random.Random(178)打乱后固定取64张；保存选样清单及图/标签SHA256，标签/类别计数实际报告。两条件使用完全相同图顺序。若某类缺失，标明覆盖不足，不自动重新抽到有利样本。64图是有限资源探索，不具代表性保证。

FP32、device0、320×320、batch2，共32批/条件；使用固定square/letterbox和正确变换后的框，禁用随机几何增强/翻转。训练标注格式保持normalized xywh及batch_idx。保存实际输入shape，不能出现旧独立val入口的352隐式padding。

使用原代码配对扰动规则noise_std=.03、brightness范围±.05。为可比，诊断副本显式设perturbation_seed=100003、perturbation_step=批次索引0..31，逐批保存实际seed、亮度偏移和噪声张量哈希。两个模型每批噪声/偏移必须相同，不依赖初始化消耗全局随机数。此为新诊断种子政策，不是重放历史终点训练批次。

### 权重与BN

两个已训练终点检查点均以未融合副本加载，验证variant/pair_config/run/seed/hash；checkpoint epoch=-1来自strip，不作精确resume。磁盘文件绝不修改，无optimizer、optimizer.step、训练epoch循环、checkpoint写回或.pt导出。

为计算原paired loss使用模型训练损失模式，但将所有BatchNorm设为eval、冻结running_mean/var/num_batches_tracked；其他随机层关闭或明确记录状态。仅在副本中为需要求梯度的层1参数开启requires_grad；strip后可能默认全部False，必须实查。使用torch.autograd.grad输出，不调用更新步骤。参数和BN状态前后哈希一致，磁盘权重SHA256前后相同。

必须在报告注明：这是**终点权重、冻结BN、无随机几何增强的局部诊断**，不等同历史训练中动态BN/参数状态，不是B5消融，不证明最终AP因果。

## 4. 修改及输出

1. 新增独立机制诊断入口（如mechanism_diagnose.py）、配置、Windows命令及结果pack，不改原model.py/engine.py训练行为。
2. 读取现有PairDetectionModel.loss、haar/box_mask/region_loss/amplitude_loss，复用其原语、stop-gradient、归一化和批量缩放，不从latest_aux浮点日志求梯度。诊断可专门重建同一个损失计算图，但必须与原loss作同副本/同输入/同BN模式/同perturbation_step数值与梯度一致性核验。不要以另写一个相似损失代替。
3. 精确分解原标量：两视图检测损失求和后均值D，C=batch_size×.05×region_consistency，A=batch_size×.02×detail_amplitude（B6）；原返回total=D+C+A。B2的A不适用，不伪造零梯度“通过”。原框架clean_loss/noisy_loss.sum()约定不得改成逐元素广播。
4. 保留梯度图的D/C/A（以及可计算时前景/背景分项）用于torch.autograd.grad，随后及时释放图，不积累32批图导致显存增长。原临时forward hook/capture/cache须在异常时清理。
5. B6按三组参数分别报告：gates、low/detail投影、整个替换层；B2报告整个原层1。只比较同一模型同一参数组内各损失梯度方向，不能把B2/B6不同维数梯度直接拼接比cos。
6. 每批输出D/C/A未加权与加权值、梯度L2 norm、相对检测梯度norm比、cos(D,C)、cos(D,A)、cos(D,C+A)、梯度内积；梯度为零或未连通时分别标zero_norm/disconnected，cos为null，不制造0或NaN。没有A的条件相关字段为not_applicable。若拆出A_fg/A_bg，重组成原A核验后才报告。
7. 输出B6三子带前景框内/框外gate分布（mean、中位数、分位数、near1/near_floor）、原始/门控后幅值、干净参考幅值、加权误差；按照GT框并集计算代理区域，类别框重叠及背景含未标缺陷的局限说明。不要将框内全部像素称为真实缺陷分割真值。
8. 汇总全部32批每项分布、有效批数、负cos比例、空区域/零梯度/失败计数，保存逐批原始记录。不要只保留最冲突批次，不能用32个相关批次作32次独立训练显著性检验。

## 5. 解析与数值边界

Work已检查简化幅值目标：固定上游特征，原始干净/扰动幅值相同A>0、0<g<1时，L(g)=|gA−A|/(A+.001)，导数−A/(A+.001)，局部梯度下降倾向增大g。9个标量有限差分已通过。这个事实只描述指定条件，不证明整个网络一定开门或增加FP。

实现时加入小型可人工核对数值/梯度验证：标量门控、空前景/空背景、0幅值、stop-gradient干净目标、batch缩放正确、诊断total及目标参数grad与原loss一致。用当前环境实跑能做的部分，未安装Torch就注明；不要把数值例子当真实诊断。

若实图梯度存在冲突，只能写“这些终点/样本/模式下存在局部冲突迹象”，不能写成已确认训练失败根因。若未观察到冲突，也不能证明幅值目标有效；如梯度对齐但AP差，记录未支持该解释，交Work重新判断。不自动换超参数寻找冲突。

## 6. 验收与交付

- 固定64train图、两条件同图/同扰动、32批记录，模型/源数据/BN/权重未更新；完整有效或失败数量可核算。
- 原loss与分解total/梯度对齐，身份/正确shape/stop-gradient/零范数和异常清理检查通过或明确失败。
- 无新训练、test、长测速、环境升级、原生进度条逐行tee；有失败也打包原始状态。
- 完整独立版本工程ZIP：全部源码、锁定依赖、配置、README、Windows激活与运行入口、结果ZIP打包入口、协议、CHANGELOG、实际CHECKS、更新PROJECT_STATUS.md。先交付ZIP再要求用户本地运行，不给片段或目录清单代替。
- 结果包包含逐批统计、选样/扰动/权重/数据身份、环境、损失一致性/状态检查、失败、汇总；默认排除原图/标签、权重、完整大张量。必要的小型数值测试记录可包含。
- Windows/GPU/真实样本未执行就标pending；云端CPU合成测试不能称为NEU/RTX4060结果。无付费API、外部账号、投稿或编辑通信。

回传Work后的决策：只有局部证据提示值得检验，才另设计单一修改或B5受控训练；本任务不自动进入该阶段，更不能在没有新实证时声称修改已完成或性能提升。
