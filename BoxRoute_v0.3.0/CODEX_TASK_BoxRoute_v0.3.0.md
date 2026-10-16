# Codex执行任务：v0.3.0 原生基线与BoxRoute最小验证工程

本说明连同交接ZIP可直接执行。用户已授权Work核验/定方案、Codex实现、用户Windows运行、回传Work审阅的流程；直接完成工程与实际可做检查，不重复询问是否实现，不要求重复上传已有文件。

## 1. 目标与范围

创建独立 **BoxRoute_v0.3.0** 完整工程，Windows放在 `D:\ModelImprovement\BoxRoute_v0.3.0`。以YOLO11n为原生网络、原生检测头和BCE/CIoU/DFL目标，比较P3处已知PKS融合、自由方向路由与正确/交换框几何监督。

旧DetailPair v0.2.9已封存；本任务不是继续训练旧B5/B6。研究假设与公式以MODEL_PROPOSAL.md为准，文献边界以BASELINE_AND_NEAREST_AUDIT.md为准。新候选未证明有效或创新，不写“已提升”“已复现论文性能”。

默认实现并安排 **G0工程/数据/开销最小验证**。同时交付已冻结但独立的G1先导入口，G0结果回Work审阅前不得自动接着运行60轮。正式200轮、多seed、第二数据集和test均不在本轮自动执行范围。

## 2. 已有输入

- `inputs/DetailPair_v0.2.9_Archive_Package.zip`：完整旧源码/评价/报告修正工程，SHA256=ea3a1b5b6a9b71b376f29f899e9402d0aa2f753a5da6a79e202ed4d26ce02cd2。只作为可复用工程来源，解到独立目录，旧版本与原结果保持。
- 本交接内BASELINE_AND_NEAREST_AUDIT、MODEL_PROPOSAL、protocol.yaml、来源记录及PROJECT_STATUS。主状态比旧工程内状态新，须合并追加，不能用旧文件覆盖。
- `sources`中的官方v8.3.221参考源码、固定提交PKS/Strip/LSK代码与配置及各自LICENSE。原文HTML/PDF不打包，URL/哈希/读取位置见来源记录。
- `sources/official_yolo11n_v830.pt`：官方公开初始化数据文件，5613764字节，SHA256=0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1；Work只下载校验，未torch.load。优先复用此文件，不要求用户提供账号或另传权重。
- Windows冻结数据：`D:\datasets\NEU-DET_DetailPair_s0_20261011_104211_157196\data.yaml`。沿用1438train/180val/180test，1798总数；原1800的两幅冲突重复图已按旧审计隔离。按标注names识别六类，不能按文件名前缀重划分/重新去重。
- 环境复用：`D:\ModelImprovement\envs\detailpair_cuda_v021`；历史为Python3.12、Torch2.5.1+cu121、torchvision0.20.1+cu121、Ultralytics8.3.221、RTX4060Laptop。实际存在性和版本须本机检测，不假设环境仍完整；无GPU不得静默CPU跑正式本地实验。

原图/用户训练权重/旧完整结果不在本交接包里，不要求上传。统计和运行在用户Windows执行；不用其他项目的1440划分、剪枝门槛或N/S路由配置。

## 3. 新代码需要实现的部分

建议独立模块：`boxroute/model.py`、`geometry.py`、`trainer.py`、`identity.py`、`evaluate.py`、`benchmark.py`、`reporting.py`，以及G0/G1/pack入口。可按现有结构合理调整，但交付必须完整。

### 3.1 原生与旧B0核验

N直接构建固定版官方DetectionModel，不用旧PairTrainer。旧B0仅用于等价检查，不作为已完成的真实基线结果。将同一个nc6原生state严格复制给B0，比较：

- eval输出、train检测loss三个分量及总标量；
- 相同非空GT与空GT的前向、所有共同参数梯度、BN buffers；
- 在相同optimizer配置下一次step后的共同参数；
- 至少64与320输入、batch1/2合理组合，FP32，记录max abs/rel误差与容差依据。

建议float32 atol1e−6/rtol1e−5；如硬件规约需调整，只能给出具体误差和原因，不能改阈值掩盖结构差异。检查网络层/参数key、DetectionModel.init_criterion、TAL和真实loss缩放。只有AST检查不算运行等价。

### 3.2 共同输入管线，特别修复训练期val的rect强制

固定版DetectionTrainer.build_dataset给val强制rect=True，builder用cfg.rect OR rect。**仅overrides rect=False不够。** 新条件共同使用一个受控trainer基类：数据构建显式rect=False，验证只增加观察/断言，不改官方指标计算；N仍保留官方网络和loss。

记录每批实际shape/dtype，train和训练期val最后两维均320×320；训练期val批量可不同于1，独立最终AP/固定阈值/测速必须batch1。训练val、final_eval和standalone val均应覆盖，不能把配置imgsz当测得shape。出现352必须停止并保存失败，不能继续生成假“同320”表。

原data.yaml可读配置及做SHA；生成只含train/val的私有配置供本轮使用。不要枚举test图/标签、不要做test评价。源数据与旧审计副本均只读，cache/派生文件写新工程私有目录，禁止改原标注。

注意cache=False不等于禁止YOLO写标签.cache：须在新工程构建逐字节同源的train/val工作副本，标签独立复制，框架扫描/cache全部落在私有目录；记录source→working路径及图/标签SHA。不得让数据加载器在原冻结目录生成缓存。

### 3.3 共同初始化和模型接入

官方COCO权重重建nc6原生模型后，保存一个共同初始state；包括未迁移的六类头随机参数，不能各条件另初始化不同头。所有匹配公共张量逐字节相同，记录迁移/未迁移key及整体哈希。

在layer16的**原模块输出之后**加同通道、同分辨率包装；先实际探测C/H/W。原层元信息i/f/type/np和Detect输入索引16/19/22必须保持正确。不得盲目替换layer16原C3k2、重新编号漏接PAN或修改Detect通道。

U/G/W完全相同state：三DW(3×3,1×7,7×1)+BN+SiLU，1×1 C→3路由softmax，1×1 C→C+BN投影，标量γ初始0，router权重/bias初始0。P与专家Conv bias=False、router含bias。外部Y=X+γT(X)，常规分支BN初始正常值。全部分支一直计算，即使γ=0；不能做零γ捷径。

G/W只在训练loss里捕获本次α，使用finally清理hook/临时张量，日志只保存detach后的标量。model eval不留非叶tensor缓存，必须支持EMA deepcopy、save/load及融合。不得因为日志保留计算图导致再次出现深拷贝或显存问题。

### 3.4 几何目标和训练loss

逐式实现MODEL_PROPOSAL的q、框ROI平均α、epsilon后重新归一化的KL，以及每图先平均实例、再平均batch。目标来自同批原生loss使用的增强后框；不读取类别名构造方向，不把GT喂给推理。

N/S无新loss，U λ0，G正确q，W交换H/V且同λ。λ(e)=.05 min(1,(e+1)/5)。总loss=原生loss向量.sum()+BλLroute；aux不得广播计入3次。box/cls/dfl官方日志保持，route和γ/α诊断另列JSONL或CSV，不替代主loss。

ROI规则：格点中心映射、左上含/右下不含、细框中心最近格点回退，逐实例处理重叠并记数；无框图贡献0；数值稳定/非法框拒绝。实现全程PyTorch以保留梯度，不在loss内numpy转换后重建tensor。

### 3.5 最近邻S对照

固定PKINet提交a33aa22d188c9946cc83fba60e3bb8ac0ec82ff7的PKSModule，保留原5×5局部DW、5分支、最大19范围、branch_scale=1，外部Y=X+γ PKSModule(X)。共同γ=0。最小PyTorch端口可把build_norm_layer选定BN替换为等价BN2d，不为一个模块安装MMCV或整套MMRotate。

用固定原类与port在相同state下做FP32输出/梯度检查；另对eval下HKR转换前后做包括边界在内的等价检查。可在CPU测试原类中注入等价BN构建器，只执行已审阅的模块，不把端口等价写成整篇OBB论文复现。

名称固定 `YOLO11n+PKS-adapt`。不改成7小核后仍叫原PKS；不以G在NEU胜S推断胜完整PKINet-v2/DOTA结果。保留作者/许可证及适配位置、包装、训练目标差异。

## 4. G0默认本地任务

1. 实查解释器/依赖/GPU0/驱动/源码SHA/公开初始化SHA；不自动升级Windows环境、不静默CPU替代。
2. 只审计train/val真实1438/180、六类names、图/标注SHA与已冻结清单；test仅引用旧审计180，不重新枚举。缺历史selection文件时列明影响，按可独立核验部分执行，严禁重新随机划分。
3. 用完整train框计算宽高比、尺度、q分布、H/V覆盖及|qH−qV|≥.2计数。若该计数<30，标记blocked_shape_support，不启动候选pilot；这是工程筛查，不是样本量功效结论。
4. 执行原生/B0数值核验、五条件前向/损失/反向、共同state与γ0初始输出一致、几何边界/ROI重叠/空框/小框、损失标量缩放、hook/EMA/保存重载和PKS端口/部署检查。
5. 固定seed178从train规范清单选64图、val选16图，保存实际选择与覆盖；五条件最多各2轮，仅验证流程。G1不得接用这些权重。样本/图像不打包外传，示例预测默认只留框坐标和指标。
6. 单独对初始N/G的同48 val图做开销检查：batch1/FP32/fuse/实际320、无γ0跳过、预测及NMS输出一致；共用30秒预热并记录实际次数，每区组预热200，固定N/G/G/N/G/N/N/G、8×48=384条正式记录。48图Random178选取，记录类别覆盖。
7. 测量沿用“已解码host图像→预处理/H2D/网络/NMS/D2H→host框”的边界，排除磁盘解码、权重加载、预热、写盘。保存mean/P50/P95/FPS与原始记录/硬件/电源状态、1.10工程漂移筛查。初始头/γ0的速度不当最终训练模型速度；若开销或稳定性不合预期，只报告并回Work，不循环择优。
8. 汇总status、失败、未执行和打包；G0结束不自动进入G1。

## 5. G1入口实现，但须G0回传审阅后使用

完整配置在protocol.yaml。固定N/S/U/G/W，seed0，COCO初始化共同nc6 state，全1438train/180val，320/batch2/FP32，60完整轮、patience0禁早停，AdamW .001/lrf.01/nbs64，其他明确增强字段相同；实际optimizer.step数和warmup累积必须记录。运行顺序事前固定S→W→U→G→N，各条件独立子进程。

每条件保存last/best、CSV/配置/环境/源身份，不需要每轮16MB周期权重。采用on_model_save在完成60轮且final_eval strip之前，**字节复制**last到独立 `matched_epoch60_unstripped.pt`，记录SHA；后续核验epoch59/CSV60/EMA/有限性。save_period=-1，所复制终点作为主比较，不用epoch−1的剥离文件推测训练轮次。

最终clean AP50–95为主；AP50/AP75/各类AP/固定conf.25、IoU匹配.5的P/R/TP/FP/FN/FPPI及宽高比/尺度分组为诊断；Gaussian .03 seed178为次要共享扰动。官方AP用conf.001、multi_label=True、NMS.7/maxdet300；固定点single-label、conf.25。借用旧评价实现时重做N/S/U/G/W身份校验，**不要调用旧B2/B6或旧40轮B5/B6专用validator**。

几何/尺度诊断按GT框分组报告召回与匹配IoU；FP保留总体及逐类口径。首轮不另造分组AP或按预测框形状随意分配FP。路由统计同时报告专家/加权专家范数和γ，不将α大小直接解释为实际语义贡献。

报告支持类集合不同的宏P差值null，保留原available宏、逐类null、六类零约定macro和micro；沿用已修正口径。所有结果整表保留，不能只挑AP50、受益类别或best。

效果/开销继续门槛严格按MODEL_PROPOSAL：G−N≥1pp，G−U及G−W各≥.3pp，G≥S，AP75不低于N，最差类AP5095下降≤3pp，新增参数/MAC各≤5%，稳定mean延迟≤1.05×N、P95≤1.10×N。单seed门槛是工程筛查，不是显著性或期刊门槛。达不到则回Work，不自动扫λ/核/层/seed或长训。

G1训练后须用终点N/G重新平衡测速：固定完整180val图、N/G/G/N/G/N/N/G八区组、1440正式样本，共同预热30秒、每区组200预热、1.10漂移筛查；不能用G0初始测速替代最终速度。

## 6. 进度、失败和只读要求

- 子进程直接继承终端stdout/stderr，保留原生原地进度条。**禁止stdout=PIPE逐行转发/tee训练输出**。结构化CSV/JSONL、父进程退出码、异常traceback、阶段状态单独保存。
- 子进程失败保留partial与已有权重/日志，停止当前suite并打包；不自动改batch、换CPU、重置随机种子、删失败或宣称CUDA原因已修复。
- 所有新结果时间戳目录；输入数据、旧源码、用户权重及已有结果只读。模型初始化资源可复制到新weights目录并验SHA，不覆盖旧文件。
- 无付费API、外部账号、原图/未发表权重上传、投稿或编辑通信。结果包默认排除.pt/.pth、图像、标签文件和大特征tensor；提供完整工程代码与元数据，用户训练权重保留Windows。

## 7. 验收与交付内容

1. 完整可下载 `BoxRoute_v0.3.0_Code.zip`，含全部源码、固定依赖、配置、README、协议、Windows命令、结果pack入口、许可证/来源、checks及PROJECT_STATUS。不得以补丁/伪代码/目录表代替。
2. Windows入口至少覆盖：环境/身份检查、G0、打包；G1另设显式入口，默认不自动运行。路径全部D盘，复用既有环境；README写清解压路径、数据、运行顺序、预计输出与回传ZIP名称，不虚构预计时长。
3. Codex当前可用CPU环境实跑必要PyTorch数值/回归测试；若依赖不可用则如实pending，工程仍须提供Windows G0检查。不得把AST或Work的NumPy检查冒充Torch前向反向通过。
4. 必须实查模型单批读取/前向/损失/反向、EMA deepcopy/保存重载、PKS端口与部署等价、共同初始化、λ缩放与ROI数值边界；模拟样本只验工程，不写NEU收益。
5. 打包后重新解包核对完整性、路径、必要文件、SHA与可运行入口。保留所有失败/修复前版本，新结果另存。
6. 交付变更说明：哪些代码复用、与原生不同的共同数据管线、与作者PKS不同的适配、已执行/未执行、Windows/GPU仍待核验。更新状态，不宣称完整最近邻复现、创新成立或可投稿。

交接完成后由用户先运行G0回传Work，再决定G1。请直接实现并交付上述完整工程。
