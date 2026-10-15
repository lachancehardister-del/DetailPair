# DetailPair local mechanism protocol 1.0

日期：2026-10-15（Asia/Shanghai）。来源为Work的CODEX_MECHANISM_DIAGNOSIS_20261015.md和RECHECK_RESULT_REVIEW_20261015.md。目标仅是局部解释性证据，当前完整B6停止正式扩展。无新结构、训练、test、测速、阈值更改或文献创新声明。

## 身份、选样与预处理

沿用两份终点best.pt及原run/seed0/目标，SHA256 B2=c234ae47e6d8257543a038abc3ea0714d0f081abf2ce5e9dbe0dcd9c73a2c21b，B6=2a5f5dd7152b28a3a3952f96fbeca409e231e07de2a06a6b5686d7a01c6bc324。checkpoint epoch=-1仅反映strip，不证明可以resume。

train-only YAML解析器不枚举val/test目录或读取其标签。1438train路径按resolve/POSIX分隔符/casefold排序（原字符串打破同值排序；casefold冲突拒绝），本地random.Random(178).shuffle取64，不根据类别覆盖/结果重抽。两份旧selection的train成员集合相同且对应1438；选中64内容哈希与两份旧记录逐项一致，未使用样本只核验成员列表，不冒充全量内容复核。保存每类框数/缺类及原选样文件hash。此小样本不保证代表性。

官方imread BGR读取选中原图，用已安装LetterBox(auto=False,scale_fill=False,scaleup=True,center=True,stride32)同时变换图像和Instances框，再convert xywh/normalize320。图像BGR→RGB/CHW/FP32/255，batch2，断言2×3×320×320。无随机几何/翻转；记录实际shape和目标tensor身份；不重用会产生352的旧官方val入口。

## 状态与扰动

未融合模型副本，训练损失路径必须开启，但所有BN为eval，running_mean/var/num_batches_tracked不更新；Dropout/DropPath/StochasticDepth类设eval并记录。全部参数先requires_grad=False，仅layer1全部参数True（BN affine仍可求梯度，running buffers固定）。Ultralytics载入模型args会被精简，必须从checkpoint完整train_args恢复原box/cls/dfl等；缺系数拒绝，不猜默认值。已提供历史身份记录中两条件box7.5/cls.5/dfl1.5，真实执行仍读本地checkpoint核验。

FP32，关闭TF32；确定性算法开启，cudnn benchmarkFalse/deterministicTrue，CUBLAS_WORKSPACE_CONFIG默认:4096:8（已有值如实记录）。若设备不支持所需确定性计算，保留错误，不切CPU或静默改变规则。

扰动严格复用原调用顺序：Generator(device) seed=100003+batch_index；先rand(N,1,1,1)得到±.05偏移，再randn与图同shape/dtype，noise*.03+offset后clip[0,1]。每批记录raw standard noise/offset/noisy-input hash及offset实际值；reference model.loss的两次实际输入hash与重建图比较，跨模型再比较全部batch input/perturbation记录。不是历史训练batch重放。

## 损失与梯度

D=(clean_loss.sum()+noisy_loss.sum())/2，不广播loss vector。C=N*(.05*cons)，A=N*(.02*amplitude)，total=D+(C+A)；B2无A，报告not_applicable。D本身按原criterion已有batch缩放，不能再除或乘一次。

C调用原region_loss；A调用原amplitude_loss及原stop-gradient target、channel mean/归一化。前景/背景分项精确复写region_loss的代数：target.detach；err=mean_channel(abs(pred-target))、scale=mean_channel(abs(target))；每图按区域计数归一化，再除区域scale均值+.001；缺区域该图贡献0，最后对整个batch取mean；fg权重1、bg权重.1。C_fg+C_bg、A_fg+A_bg须还原原标量；A分项合成梯度也与原A梯度核验。三子带误差贡献共用原三通道scale分母，并除3重构channel mean，不能各子带独立归一化冒充原损失。

每批先同副本/同输入/同step/同BN模式执行原model.loss，取完整layer1参数梯度；释放原图后重建原语计算图，用torch.autograd.grad分项和total，rtol1e-4/atol1e-5对齐数值/全部目标参数梯度及None连接状态。保存实际误差，不匹配直接失败，不自动改容差。只从tensor计算梯度，latest_aux浮点日志不参与。

B6参数组gates、low/detail投影、whole_layer；B2 whole_layer。每个loss/group保存连接/未连接tensor数、L2、相对D比；cos(D,C)、cos(D,A)、cos(D,C+A)及分区分项、内积。未连通为disconnected，norm/dot/ratio不伪造为0；连通但norm0为zero_norm；未定义cos为null，B2相关A字段为not_applicable。混合连接组对真正未使用的坐标补零形成同一参数空间方向，同时报告连接情况。不同模型/不同组维数不拼接比较cos。

## 门控与幅值

每批B6三子带，按原box_mask从正确变换后的normalized xywh构造GT框并集代理mask。每区域输出noisy gate及clean gate的n/mean/median/min/max/q01/q05/q25/q50/q75/q95/q99，near_one=g>.99，near_floor=g<=floor+.01（floor=.1）。同时输出raw_noisy/gated_noisy/clean-reference绝对系数分布与原共享归一化误差、完整N*.02加权贡献。空区域为empty_region/null，不当0分布。

掩码来自框，不是缺陷分割真值；框内含背景，框外可能漏标，padding如存在也在代理区域内。幅值不是平方物理能量。clean目标stop-gradient，不是EMA教师。本轮不额外实施gate梯度反演、新损失或参数调优。

## 汇总、失败与只读核验

逐批保留32条/模型，失败后剩余not_run，另一模型仍尝试，不换数据/条件择优重试。汇总全部有效批次各项分布/有效数/null数/负cos比例/zero_norm/disconnected/empty_region/失败/未执行计数；相关批次不是训练独立重复，不计算显著性结论。

每批及条件前后对全部参数值和所有BN buffers hash；逐条件磁盘权重SHA前后、末尾选中图/标签SHA核验；记录.grad未累积与临时hook/capture/cache清理。requires_grad/mode/perturbation_step/criterion等诊断控制状态可在内存副本中改变，不能混同权重/BN更新。正常batch图在结束时释放；失败保留traceback和已完成记录，不导出.pt或大型tensor。

程序无optimizer/step/.backward/训练epoch循环，只有固定批次前向与autograd.grad。最终状态分开报告执行成功与参数/BN/磁盘权重实际核验；状态未知/失败不写成通过。输出ZIP默认排除原图/标签文件、权重、完整张量和缓存，保留必要哈希、身份、实际设置、统计、失败及源码。

## 证据边界

Work已在指定条件证明标量A>0、0<g<1时d|gA−A|/(A+.001)/dg=−A/(A+.001)，本版数值/自动梯度小例验证同一事实。固定上游/幅值相同的推论不证明完整网络一定开门或增加FP。

实图若有冲突，只能报告终点权重/这些样本/冻结BN模式下的局部迹象；无冲突也不证明幅值目标有效。不能等同B5从头训练、归因最终AP、宣称监督修正已完成或可投稿。是否下一步单一机制改动由Work另定。

本环境只发现云环境setup技能，未找到科研工具箱原版模块资源，未冒充已读取；实际依赖与模型/损失源码已检查，数学事实和工程实现用可运行测试核验。无需也未进行新的文献检索、付费API、额外账号、投稿或编辑通信。
