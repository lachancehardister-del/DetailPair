# DetailPair v0.2.6：平衡区组 GPU 测速与 clean 口径复查

本版复用同一 B2/B6 best.pt，不训练、不改模型、不跑正式test，也不自动重跑四档扰动。旧v0.2.5及其成功、QQ阻止启动和漂移记录全部保留。

## Windows 下一步（GPU）

完整解压到 `D:\ModelImprovement\DetailPair_v0.2.6`。复用现有CUDA环境，不安装或升级依赖。先自行关闭QQ、prunecheck、训练及其他GPU计算；程序不会替你kill进程。

```bat
conda activate D:\ModelImprovement\envs\detailpair_cuda_v021
cd /d D:\ModelImprovement\DetailPair_v0.2.6
windows\18_recheck_all.cmd
```

默认使用GPU device0、CUDA12.1、RTX4060 Laptop GPU、FP32、320×320、batch1。没有CPU回退。默认精确路径仍指向旧v0.2.4两份best.pt及selection/metrics；两份权重SHA256已冻结，见 `configs/balanced_recheck.yaml`。移动文件时允许只修改路径，仍要通过SHA/variant/seed/run/源数据核验。路径含空格的配置可以作为.cmd第1参数传入。

只做某一项时用独立入口：

```bat
windows\19_timing_only.cmd
windows\20_clean_diagnostic.cmd
```

这两个入口不会自动启动另一个任务。默认推荐18一次完成；不要同时运行。旧15入口/1.0配置保留为历史，不能把旧三轮汇总当作本次复查结论。

## 测速如何改变

固定8区组顺序 `B2,B6,B6,B2,B6,B2,B2,B6`（ABBA+BAAB）；每区组完整val180，单条件4区组720条，总1440条。先初始化实际torchvision NMS和两条件完整推理路径，再共同预热目标30秒，B2/B6等量交替；最后完成等量一对可能略超30秒，实际秒数/次数写日志。每区组再执行对应模型完整推理200次。所有预热独立记录并排除。

GPU上只保留当前模型；在CPU保留已融合模型，区组切换才迁移设备。计时边界仍为已解码主机图片→预处理/归一化/H2D/推理/NMS/D2H→主机检测结果，GPU前后同步。加载、解码、模型迁移、预热、快照和写盘均在计时外。逐图样本先缓存在内存，区组结束写盘；异常保存已完成partial样本，不高频flush或逐图轮询nvidia-smi。

每区组前后保存GPU进程、时钟/功耗/温度/P-state/显存/驱动，以及可读取的CPU、电源计划/电池信息；读不到为unknown，不能据温度直接认定热降频。`hardware_notes`可填写实际机器/电源模式/是否插电，默认unknown。

报告全部区组、每组前60/中60/后60、各条件4组均值、ABBA/BAAB配对及相邻先/后位置效应。跨同条件区组或组内窗口的最大/最小均值比超过1.10，标记 `timing_unstable`；不删慢组，不自动重跑。执行成功、数据完整、速度比较是否可信分开记录；不稳定也可能exit0，因为工程执行完成，速度优势仍待Work判断。

## clean口径诊断

同一进程/环境依次重载同一权重：

1. 按v0.2.4原evaluate调用复现旧入口，默认参数由已安装Ultralytics8.3.221实际解析。用字节一致的源JPEG/标签私有副本，恢复原selection的working文件名和顺序；rect加载器最终按长宽比排序也原样记录。
2. 按v0.2.5显式参数，对同样的PIL派生clean PNG评价。记录官方加载器实际解码RGB/BGR像素哈希、预处理张量shape/dtype/哈希、letterbox属性、已安装loader/letterbox源码、全部有效validator参数、AP/各类AP及官方匹配矩阵与预测。
3. 仅在实测rect不同且另有非绘图参数或解码像素差异时，增加一个预先指定的rect单因素对照：同一PNG、同一新参数，只把rect改为True。不会按AP方向选因素或进行参数搜索；未触发条件则明确not executed。其他残余原因标待查。

不重跑固定conf=.25 P/R；官方多标签AP预测不充当single-label工作点预测。不把.3514改成B6新固定阈值。新旧AP并列保存，不覆盖旧指标，不预先归因JPEG、FP32或模型错误。

## 回传给Work

结束或失败时打印 `exports/DetailPair_recheck_results_*.zip`。把完整ZIP交给Work。默认排除原图/标签、派生全图、权重、缓存及官方绘图中的图像样本；保留数据清单/像素和张量哈希、有效参数、预测/AP匹配、全部区组/预热/设备记录和失败。

中断或要手工打包：

```bat
windows\21_pack_recheck.cmd
```

不要删除旧v0.2.5结果，不自动循环到稳定为止。Work依据新结果决定下一步，当前不追加训练seed、B5、YOLO11s、全消融或新结构。

## 已完成与待执行

本轮收到并独立检查187项原始评价ZIP，源码与v0.2.5一致，确认10/10成功、1080原始计时完整及B2显著轮间漂移；QQ失败记录保留。B2/B6 clean AP50:95为35.1279%/34.2651%，不是B6整体收益证据。

云端实际检查范围见 CHECKS.md。Windows/GPU、真实权重重载、1440条GPU计时和真实clean诊断仍待本地运行。CPU随机检查点/合成JPEG检查不能代替这些结果；新默认预热也不保证消除漂移。
