# 原生基线与最近邻核验

日期：2026-10-16（Asia/Shanghai）。项目：NEU-DET检测模型改进，首选Journal of Real-Time Image Processing，次选The Visual Computer。当前DetailPair B5/B6组合保持封存。本轮确定的是下一轮可证伪候选与实现协议，未取得新模型实证。

## 一、基线核验结论

已实际获取官方Ultralytics **v8.3.221** 源码，解析annotated tag得到提交 **e544db43838fd058370cad216bf82a7639f29b35**。固定文件URL、实际HTTP状态、时间、字节数与SHA256见SOURCE_RETRIEVAL_LOG.json。网页检索工具读取若干GitHub固定版本页失败，随后标准库urllib直接取得源码；两种结果分别保留，没有以main版在线文档替代固定版。

| 核验项 | 源码观察 | 可以得出的结论 |
|---|---|---|
| B0网络 | PairDetectionModel继承DetectionModel；B0 module=base，构造器不替换layer1 | 静态支持B0保持原网络；尚未做实际张量等价 |
| B0目标 | pair=False直接return super().loss(batch,preds)，不进入双视图 | B0没有B2的配对及一致性目标 |
| 官方目标 | 普通Detect调用v8DetectionLoss；BCE分类、CIoU框回归、DFL，TaskAlignedAssigner | 以此为受控原生基线目标；不能把“加入IoU信息”笼统称新意 |
| 损失缩放 | criterion返回3分量乘batch_size；trainer再sum后backward | 新辅助标量须加在sum后的检测标量上，不能广播加到三个分量 |
| PairTrainer | 覆盖模型构建，增加辅助日志callback；其余继承官方trainer | 仍须运行前向/损失/梯度/BN/一步更新等价检查，不能据继承关系宣称完成复现 |
| 训练配方 | 旧工程scratch、batch2、AdamW .001/.01、nbs默认64、弱几何增强、无Mosaic | 属于本项目受控配方，未证明它是最佳或与论文默认配方一致 |
| 有效更新量 | 固定官方trainer按nbs/batch动态设置累积，warmup期间变化 | batch2不等于每2张都optimizer.step；后续必须记录实际step数 |

**现有材料不足以给出正式原生基线的最终AP。** B2是配对训练+一致性对照，不能替代原生模型基线。B0的代码身份已静态核验，数值等价与真实训练仍由下一工程验证。

### 一个新确认的输入管线细节

固定版 `DetectionTrainer.build_dataset` 明确给val传 `rect=mode == "val"`；`build_yolo_dataset`使用 `cfg.rect or rect`，val pad=.5。因此仅在训练配置写rect=False，**仍不能保证训练期间的val是320方形**。此前独立352/320复查结论保持；这次源码分析不是对历史每次张量形状的重新测量。

新工程对所有条件统一覆盖数据构建的rect参数，并用观察型validator记录和断言真实train/val输入为320。该管线控制须单独声明：比较的是**原生网络与原生检测损失，在同一受控数据/评价管线下**，不是未经修改的官方默认训练命令。独立最终AP与测速一律batch1、FP32、实际320、rect=False。

### 新阶段初始化决定

本轮已从官方公开release下载 `yolo11n.pt`，仅作为数据文件，未加载或执行：

- URL：https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
- 字节数：5613764。
- SHA256：`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`。

下一阶段五条件都使用这个相同初始化，六类头先统一构建，再克隆共同state。选择预训练是新阶段协议决定，不是本轮证明旧结果较低的原因；不得与旧scratch AP拼作公平提升。权重已放入交接包，用户Windows无需另建环境或提供账号。

## 二、最近邻核验

这是有记录的定向范围检索，不是全领域系统综述。执行23条完整查询，见SEARCH_LOG.json；实际提取230条检索命中记录、220个去重URL，另有原文打开/定位记录。它们不是数据库总命中数、220篇纳入论文或独立研究数。选出8项方法记录作针对性比较，其中7项全文可得，1项仅出版社预览及选定代码。没有Scopus/WoS完整导出、双人筛选或全领域召回保证。

| 工作 | 实际材料 | 核验到的机制/目标与新候选边界 |
|---|---|---|
| Strip R-CNN；arXiv:2501.03775v4 | 全文§3.2–3.3；固定提交c2d3cf4…的StripBlock及检测配置 | 小核后串行正交条带，投影为乘性注意；所读检测配置为分类CE和SmoothL1。条带特征提取已有先例。作者仓库标AAAI2026，本文以已读arXiv版本记录，不另认证会议状态。 |
| PKINet-v2；arXiv:2603.16341v1，2026-03-17 | 全文§3.3–3.4式2–10；提交a33aa22…的PKSModule/配置 | 方形与轴向/空洞卷积并行，BN后求和并投影，输出乘输入；线性分支可重参数化。所读配置为OrientedRCNN的CE/SmoothL1。不能把条带+方核融合当新意；仓库标ECCV2026，会议出版状态未另核验。 |
| LSKNet；ICCV2023 | CVF完整PDF§3；提交4671650…的LSKblock | 顺序5×5与7×7(d=3)DW；通道均值/最大值生成两个sigmoid空间选择权重。输入依赖的选核已有先例；其已读模块未提供我们的框比例KL监督，但这不等于查遍全仓库训练入口。 |
| Guided Anchoring；CVPR2019 | CVF完整PDF§3.2–3.4式2–6 | 预测框宽高，形状驱动可变形特征适配；L=λ1Lloc+λ2Lshape+Lcls+Lreg。几何辅助监督与特征适配的宽泛思想已有先例；我们的候选是固定三专家概率路由，保留原生anchor-free检测头。 |
| SPDCN；arXiv:2607.21456v1，2026-07-23 | 全文§III-C/III-D，式7–11 | 水平/垂直条带产生offset与mask，控制可变形采样；BCE+Dice分割目标。不能声称首次方向建模或首次用于钢材；分割任务、像素真值与当前框检测不同，作者仓库根链接未核验为完整可运行实现。 |
| DENet；Remote Sensing 2026；10.3390/rs18060873 | 出版社全文§3及Crossref身份 | 特征offset引导方向条带采样，配合尺度融合；文章说明保持RT-DETR匹配及标准分类/框回归协议。没有把它当新框路由监督的完整数值复现。公开代码链接存在，本轮未运行该仓库。 |
| SFF-Net；Mathematics 2025；10.3390/math13203285 | 出版社全文§3及Crossref身份 | 根据训练框建立监督范围，聚焦卷积/采样分配。用框指导特征处理已有工业先例。其范围限制和样本分配不等于本候选的概率路由目标；未作完整复现或横向AP比较。 |
| MPSL-Net；SIVP 2026；10.1007/s11760-026-05634-6 | 出版社摘要/预览；官方master提交7ab0e58…的my_model.yaml、SDetect/StripConv及v8DetectionLoss类 | 所读回归头含串行条带，所读loss类为BCE/CIoU/DFL。SIDH术语到完整训练入口的映射未全链核验，付费全文目标未取得，不能据局部代码排除所有相似监督。初查main失败后已找到master，不能写成代码不可得。 |

论文原始入口及代码路径见SOURCE_RECORDS.json。三篇期刊DOI通过Crossref身份核对，不把注册成功当论述支持；对应机制只引用实际全文/代码/摘要覆盖范围。未系统查询全体Crossmark/撤稿库，不能声称已完整排查撤稿。

### 为首轮确定一个可实现的最近邻对照

选用**PKINet-v2的PKSModule机制适配**，而不把Windows工程变成另一整套OBB框架。固定原始5分支及19×19最大范围，不缩核偷换方法；放在相同P3位置并加明确标注的共同残差包装。纯PyTorch适配须对固定原类做输出/梯度/部署等价检查，保留许可证和变更说明。

这个条件称 `S: YOLO11n + PKS适配`，**不是完整PKINet-v2论文复现**。不比较论文DOTA分数与本项目NEU分数，不把A100上的作者速度当4060速度。正式论文阶段如保留贡献，还需补充恰当的完整近期方法/第二数据集验证；本轮不提前宣称已经完成。

## 三、研究决策

确定新候选 **BoxRoute-Det（内部代号）**：YOLO11n在P3处加入一个三专家路由块，使用训练框宽高关系监督分支使用比例。完整定义见MODEL_PROPOSAL.md。

候选的可检验增量为：**在同一结构、初始化和预算下，正确框几何监督能否比自由路由、方向交换监督及已知融合模块取得更好的高IoU检测与开销取舍。**

“条带卷积”“输入依赖选核”“框监督特征适配”均不是新贡献。本轮未检得完全同式不构成首次证明；MPSL全文等仍有限制，因此创新性仍是候选判断。先验证研究假设，再决定是否进入正式实验，不以方法命名或公式改写宣布可发表。

## 四、执行边界与来源

本轮实际完成固定源码静态分析、原文/代码读取、公开预训练文件SHA256、NumPy解析/数值检查。网页无Torch/Ultralytics，未执行网络前向/反向/等价、作者模型、真实NEU训练或4060测速。训练框形态分布尚未读取，须在下一工程的G0中用train-only统计核对；没有假定所有缺陷细长或框宽高等于真实方向。

已按实际任务读取科研工具箱原版literature-review、scientific-critical-thinking、experimental-design及必要工作流/设计参考；不采用临床质量分数评价视觉模型。实际numpy/pandas可用，requests/pydoe/Torch/Ultralytics不可用；检索使用已有网页能力和标准库urllib，未安装额外软件、调用付费API、登录账号或外传未发表材料。

科研流程引用已核验arXiv当前记录为v2：Kassis, T., Agarwal, V., He, Y., Patel, D., Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065 。仅为研究流程来源，不是本模型机制创新来源。
