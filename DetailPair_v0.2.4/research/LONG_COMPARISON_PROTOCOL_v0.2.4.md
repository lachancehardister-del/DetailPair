# B2/B6 长预算比较协议 v0.2.4

冻结日期：2026-10-11（Asia/Shanghai）。用户指定：B2/B6相同更长预算，200轮上限，50轮早停耐心值。该协议未运行；不是结果报告。

## 预算解释

每条件epochs=200、patience=50，含义为最多200轮，验证fitness满足原框架早停条件时提前停止。不是先200轮再额外50轮，也不是固定250轮。最终逐条件实际轮数可能不同，要同时报告停止轮数与独立best重载验证。原engine.py已经显式patience=50，未修改；config把50写出并由runner验证。保存实际安装版本EarlyStopping源码及training/args.yaml以核验执行口径。

## 固定共同设置

B2原YOLO11n双视图＋一致性；B6 Haar门控＋双视图＋一致性＋细节幅值目标。模型公式、目标系数、AdamW/初始学习率0.001/原默认warmup、增强、FP32及其余engine设置不变。两者均scratch，不接着40轮训练，不从旧best/last续训，以免旧40轮学习率调度和选择历史混入新预算。学习率调度由总epoch200决定，不能把40轮曲线视作200轮的前40轮等价轨迹。

两条件seed0，img320、batch2、workers0、全部1438train/180val，源D:\datasets\NEU-DET_DetailPair_s0_20261011_104211_157196\data.yaml，固定旧划分，不重分、不做test检测评估（test仍参与数据审计）。记录源哈希、类别框数量、完整选样清单。9个旧dHash警告经此前人工复核，保留工件独立性与近重复覆盖限制，仅探索比较显式覆盖；硬错误仍阻止训练。

每条件一个独立新Python进程顺序运行，固定seed127随机顺序记录；新输出独立v0.2.4，旧40轮结果、B5失败和诊断包保留。此前CUDA非法内存访问未定位，本轮按用户新指令转向B2/B6长预算，不宣称已修复。首条件失败后仍尝试另一独立进程，报告成功/失败，绝不把部分结果写成完整比较。完整训练不设置CUDA_LAUNCH_BLOCKING，诊断入口仍独立保留。控制台实时显示子进程进度，同时落盘console.log。

## 判断口径

统一比较各自best checkpoint独立val重载的mAP50–95（主指标）、mAP50、precision/recall，保留所有epoch曲线、实际轮数、辅助损失与门控分布。等上限/早停设置不等于等实际训练时间或总计算量，因为架构与停止轮数可能不同；单种子不支持统计显著性。B2对B6验证整个新增结构与幅值目标的组合，不能拆解门控、Haar、幅值各项贡献，后续需要消融。收益不保证：40轮中B6已经低于B2，200轮仍可出现无改善或退化。

若任一中断，保留原失败、实际轮数、检查点，先解决稳定性后做完整比较；若都完成则评估继续多种子/消融是否值得。此轮不因失败改变batch、精度、初始化、驱动或软件版本，不挑选最佳种子；无投稿、编辑通信、收费API或新增外部账号。

## 执行与限制

完整代码ZIP包含compare_long.py、原核心源码与已有诊断入口、configs/long_comparison.yaml、Windows入口、依赖、README、pack及版本哈希。复用已成功的D盘GPU环境，不安装软件或下载权重。进度可在终端查看，也可以打开本次runs/long_comparison_时间戳/B2_console.log或B6_console.log。

本轮网页Linux无Torch，实际仅语法、固定配置/顺序/错误拒绝逻辑、核心源码一致性、ZIP检查；未执行真实模型、早停、Windows批处理或200轮训练。原B2/B6 Windows40轮成功是旧实证，不冒充新版运行。官方当前train文档核实epochs/patience一般含义；固定v8.3.221 GitHub源码页读取失败，失败请求也保留于research/EARLY_STOP_RETRIEVAL.json。运行时保存安装EarlyStopping源码补上固定版本证据。
