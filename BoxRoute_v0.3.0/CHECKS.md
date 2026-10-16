# 实际检查 — BoxRoute v0.3.0

2026-10-16，Asia/Shanghai。云端Python3.12.14、Torch2.5.1+cpu、torchvision0.20.1+cpu、Ultralytics8.3.221，无CUDA GPU。本轮使用已有环境，未安装MMCV/MMRotate/timm或升级依赖。Work的NumPy原型与本次实际PyTorch检查分开记录。当前技能目录未提供科研工具箱原版模块，未虚构已读；实现严格依据已交接的冻结提案/协议及审阅来源。

## 已执行的PyTorch检查

- `tests/check_native_equivalence.py`：8个64/320、batch1/2、空GT/非空GT组合。严格同state、网络key、eval输出、原生box/cls/dfl与总标量、全部共同参数梯度、BN buffers及一步AdamW参数相同；本次最大绝对梯度差0。atol1e-6/rtol1e-5未放宽。实际criterion/TAL类及原生source随JSON保存，B0不当正式真实基线。
- `tests/check_boxroute.py`：6项，PyTorch q单纯形/缩放/交换/非法输入、ROI左上含右下不含/细框回退/重叠/空图、实际autograd gradcheck、λ和不广播、固定原PKS与port输出/全部参数及输入梯度、原类与port包含全部边界的HKR转换、公开初始化和五条件真实单批前向/loss/backward、EMA deepcopy/保存重载及fuse、注入criterion异常后的hook清理。
- 五条件完整模型测试使用公开COCO初始化后共同六类头，合成B2×3×320×320；原生N/G初始输出一致，U/G/W初始state逐字节一致，实际P3=[1,64,40,40]。当前G新增6276参数/9,216,000卷积MAC；比例+0.242222%/+1.167334%。MAC是实际Conv2d hook计数，不包含BN/SiLU/softmax/乘加/传输/launch，不代表延迟。
- `tests/check_data_pipeline.py`：1项私有源复制/官方实际dataset/loader测试，train与val真实2×3×320×320 FP32、独立val构建rectFalse；352输入实际拒绝并留记录；原始树哈希/条目不变，label.cache只在私有目录，test缺失也不枚举。历史selection缺失状态及固定选样可重复验证。
- `tests/check_evaluation_timing.py`：3项独立AP/固定GT/P/R/AP75/shape、平衡记账/漂移与G1拒绝、CPU6图×8区组48样本的实际HostDetector/NMS记账；缩短预热明确标为synthetic，仅验证计数/输入/初始输出一致，不验收GPU速度。另1项G1预设阈值与停止逻辑，纯合成指标输入；小数指标单位是fraction，不把1pp写成1.0分数。

原生/PKS端口比较使用atol1e-6/rtol1e-5。整个网络Conv-BN融合比较使用atol/rtol1e-4并保存最大绝对误差，原因是FP32融合后累加顺序与像素坐标量级；它不是放宽B0或PKS原类等价阈值。梯度有限差分/gradcheck用float64进行数值验证，生产模型仍强制FP32。

## 完整CLI流程与失败

`tests/check_g0_flow.py`实际创建4train/2val程序图像，明确 `--synthetic-check --flow-check` CPU运行：先原生/B0与五条件工程检查，五个独立子进程S/W/U/G/N各2完整轮。训练val与final_eval实际320；每条件保留strip前字节复制的matched_epoch2_unstripped.pt，元数据epoch1/CSV2/EMA/有限性通过，实际optimizer步数>0。全部条件的4个增强后图像/框batch哈希一致，源字节和工作副本均未改变。G1无Work审阅记录实际exit1/失败记录/打包；合成G0不能放行G1。未执行60轮。

日志 `verification/G0_cpu_flow_initial.log`、`G0_cpu_flow_final.log`；原阶段记录均保留。结果ZIP实际CRC/排除图像、标签文件、私有副本、全部.pt/.pth/.npy/.npz通过。CPU合成流程主动跳过实机384计时，另有48条CPU记账测试，绝不当G0 GPU计时或训练后速度。

初版6项测试中λ0的Python浮点精确相等断言失败（.010000000000000002 vs .01）；改为明确数值容差断言，未改公式/模型阈值。原 `boxroute_cpu_initial.log` 保留，修正后 `boxroute_cpu_fixed.log` 6项通过。复制旧评价工具时初次相对父目录解析错误在代码生成阶段纠正，未把未执行检查当通过。所有来源/早期数学检查失败也保留Work原件。

最后增强了实机计时权重身份：真实计时写初始化/终点文件SHA256及模型state SHA；无权重文件的CPU记账标明state-only。相应实际计时记账复检在 `verification/timing_identity_final.log`。早先已通过的合成G0流程不涉及GPU计时；小改动未冒充新跑的真实G0。

## 来源和身份

验收7项ZIP及新交接48项ZIP安全路径/CRC通过；验收清单6项、交接清单47项哈希通过。官方公开权重5613764字节/SHA256正确，已在当前Torch实际加载。7个固定官方模块的安装源码与交接文本在统一换行/BOM口径下逐字一致，原文件SHA均保留。PKS原类只执行已审阅的两个定义、注入等价BN构造器，未执行整套第三方训练仓库。旧model.py字节一致副本仅用B0检查。

## 未执行 / 用户下一步

Windows.cmd/CUDA12.1/GPU0、真实1438/180数据与六类形态支持、64/16真实两轮、384条实机初始计时、G1真实60轮/终点10评价/1440条训练后计时均pending。沒有test访问、付费API、外部账号、用户原图或训练权重外传、投稿或编辑通信。原生validator自动打印的Speed不是本任务公平测速。

最终代码包会检查CRC/安全路径/逐文件SHA、依赖/协议/源码/Windows/公开权重/许可证及解包后的help/pack和必要导入，记录于verification/release_check.json。完整工程已交付后，用户只先运行Windows G0回传Work；不自动接G1。

CPU记账复检首次复用了测试固定目录，触发block目录FileExistsError；未删已有记录，改测试为时间戳输出后重验。原timing_identity_repeat_failure.log及partial保留，真实G0/G1本来每次新时间戳run不受此测试夹具问题影响。辅助环境清单最初误用系统Python（无Torch元数据），改用实际测试venv后生成；见preparation_notes.md。
