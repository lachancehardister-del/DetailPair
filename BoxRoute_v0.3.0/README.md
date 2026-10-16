# BoxRoute v0.3.0：先运行 G0 工程最小验证

这是独立的新候选工程，旧DetailPair B5/B6保持封存。当前只安排 **G0**：真实数据/环境核验、数值和序列化检查、五条件各两轮流程验证及初始N/G开销。**G0结束先把结果ZIP交给Work审阅，再决定是否启用独立G1。** 当前没有BoxRoute真实检测收益或创新性已成立的结论。

## 解压和 Windows 运行

把 `BoxRoute_v0.3.0_Code.zip` 完整解压到 `D:\ModelImprovement`，使主入口为：

`D:\ModelImprovement\BoxRoute_v0.3.0\run_boxroute.py`

PowerShell按顺序运行：

```powershell
cd D:\ModelImprovement\BoxRoute_v0.3.0
.\windows\01_check_environment.cmd
.\windows\02_G0_minimal.cmd
```

复用 `D:\ModelImprovement\envs\detailpair_cuda_v021`。脚本使用该环境绝对python路径，并在脚本内设置DLL搜索路径；不安装、升级或重建环境。正式本地任务要求GPU0、Torch2.5.1+cu121、torchvision0.20.1+cu121、Ultralytics8.3.221；缺GPU或版本/固定源码/公开权重身份不符即停止，无CPU回退。

冻结数据默认 `D:\datasets\NEU-DET_DetailPair_s0_20261011_104211_157196\data.yaml`。1438train/180val，test180只引用历史数量，不枚举/评价；按data.yaml六类names读取，不按文件名前缀重划分。实际数据和旧权重继续留在你的Windows。

如果历史版本或数据目录迁移，仅编辑 `configs/local_paths.yaml` 的路径；可明确传入另一个路径配置：

```powershell
.\windows\02_G0_minimal.cmd "D:\my configs\local_paths.yaml"
```

历史selection缺失会列明影响，并继续当前可独立核验部分，不重新划分数据。公共COCO初始化已经随包提供于 `sources/official_yolo11n_v830.pt`，启动校验SHA256，无需账号或另传权重。

## G0 实际做什么

1. 检查环境、GPU驱动、固定版源码及公开初始化；仅对train/val生成逐字节私有副本，所有YOLO标签cache写入新run内。
2. 统计全train框的宽高比、尺度、q及H/V代理支持；`|qH-qV|>=.2`不足30框时输出 `blocked_shape_support`，不启动候选流程。
3. 原生DetectionModel与旧B0的8组size/batch/空框等价、五条件的前向/损失/反向、共同初始化、PKS端口/部署、几何ROI和EMA保存重载检查。
4. 固定Random178选64train/16val，S→W→U→G→N各2轮工程流程。Windows G0入口显式传入 `--flow-check`；这不是G1的60轮效果pilot，`pilot_auto_start=false`保持，G1不自动跟跑。裸CLI省略该参数时仅做工程部分并标明flow未执行，不能冒充完整G0通过。
5. 选同48val图，初始N/G输出/NMS一致后做N/G/G/N/G/N/N/G八区组、384条计时；共同30秒预热、每区组200次预热。GPU计时前须自行停止其他GPU计算任务，程序不结束其他进程。状态/漂移/开销失败直接回传，不循环择优。

计时边界：已解码host BGR→letterbox/normalize/H2D/网络/NMS/D2H→host框，排除磁盘、加载、预热、写盘。初始γ=0时分支仍执行，G0速度不当训练后最终速度；GPU延迟和稳定性均待本地检查，不虚构时长。

原生进度条由独立子进程直接继承终端；没有PIPE逐行转发。每个条件日志记录实际optimizer.step、warmup累积、LR、EMA及增强后图像/框hash，辅助loss/γ/α/专家及加权专家范数另列，不覆盖官方box/cls/dfl。

## 回传结果

每次新建 `runs/g0_时间戳`。成功、阻断或失败后均打印：

`exports\BoxRoute_results_时间戳.zip`

把**完整结果ZIP交给Work**，不要只截最后一行。默认不含原图、标签文件、私有副本、用户训练权重或大特征张量；含配置/源身份、形态统计、数值误差、实际shape、阶段成功/失败/未执行、两轮流程、原始测速及失败。手工打包：

```powershell
.\windows\03_pack_results.cmd
```

退出码0表示当前入口完成（实际状态以status.json为准）；2表示支持/开销/稳定性筛查未通过或未完成全部G0流程；1表示环境/工程或子进程失败。任何失败不自动减batch、换CPU、重置seed或删旧运行。

## 五条件与原生边界

| 条件 | 模型与目标 |
| --- | --- |
| N | 固定官方DetectionModel、原生BCE/CIoU/DFL与TAL |
| S | `YOLO11n+PKS-adapt`，固定原PKS五分支/19范围，γ残差包装，无路由loss |
| U | 三专家自由路由，λ0 |
| G | 与U相同结构/初始state，正确框几何KL |
| W | 与G相同结构/λ，交换H/V几何目标 |

保留原layer16的计算，在其输出后包装；实际探测 `[1,64,40,40]`，Detect索引16/19/22不变。六类原生头只初始化一次，匹配的COCO张量及未迁移头均共同复制，U/G/W完整state相同。不是旧scratch B2/B6的延续，也不能与其AP拼成公平提升。

所有条件共同覆盖train/val构建为rect=False，并观察/断言实际320方形（包括训练val、final_eval和独立val）。这与官方默认训练val强制rect=True不同；N的网络、检测头、loss与官方指标算法保持原生。训练val/final_eval的批量可大于1，其best打印只作流程记录；主结果另取未剥离终点EMA、独立batch1 FP32评价。

PKS端口仅将审阅原类的默认BN构造替换为等价BN2d，不安装MMCV/MMRotate；保留原5×5局部DW、五分支、19范围、branch_scale1及HKR转换。接入位置/γ包装是适配，不是完整PKINet-v2或DOTA性能复现。源码及许可证见 `sources`、`THIRD_PARTY_NOTICES.md`。

## G1 已实现，当前不启动

`windows/90_G1_after_Work_review.cmd` 是独立门禁入口，需要Work审阅通过的真实G0记录JSON作为参数。模板 `configs/G1_review_record.example.json` 默认未批准；门禁绑定真实G0状态哈希、代码、协议、数据和共同初始化，合成G0不能放行。没有批准记录就拒绝训练，不要仅为绕过门禁修改模板。

未来获准后：重新从共同公开初始化开始，完整1438/180、seed0、60轮/patience0、batch2/FP32/320/AdamW/.001/.01/nbs64，固定S→W→U→G→N；不接G0权重。on_model_save在strip前逐字节复制 `matched_epoch60_unstripped.pt`，核验epoch59/CSV60/EMA，best不作主比较。独立clean/Gaussian.03、AP50–95/AP50/AP75/逐类/固定P/R/FPPI/GT形态分组，以及训练后N/G完整180val×8=1440条平衡计时均已接入。原available宏及null保留，不同支持集合的宏差值null，另报六类零约定和micro。

所有继续门槛按 `MODEL_PROPOSAL.md`、`protocol.yaml` 执行，结果只回Work；不自动200轮、多seed、第二数据集或test。

## 已验证与待验证

云端已实跑PyTorch CPU：原生/B0 8组、几何/PKS/全五模型/序列化检查、私有数据和实际方形输入、五独立子进程各两轮的完整合成G0及G1拒绝门禁、独立AP/固定评价与48条CPU计时记账。它们只验证工程，不是NEU或GPU结果。当前实测G模块新增6276参数、9,216,000卷积MAC；相对N约+0.2422%/+1.1673%，不代表GPU延迟达标。

Windows/CUDA、真实1438/180形态支持、G0 384条GPU计时、真实两轮流程及G1 60轮/终点评价/1440条GPU测速仍未执行。详见 `CHECKS.md`；方案、研究边界和实际交接来源原文保留，不宣称模型有效、创新成立或可投稿。
