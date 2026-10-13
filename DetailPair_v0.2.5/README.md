# DetailPair v0.2.5：复用 B2/B6 权重的统一评价

本版不重新训练。完整保留 v0.2.4 的 model.py / engine.py 和训练目标，新增 clean、四档共同合成扰动的官方 AP、固定工作点逐类 P/R、真实图片 batch1 端到端测速。compare_long.py 改为子进程直接继承终端，不再逐行管道转发；训练入口保留供将来明确授权使用，本轮不运行。

## Windows 下一步

完整解压，使入口位于 `D:\ModelImprovement\DetailPair_v0.2.5\realtime_eval.py`。保留旧版代码、runs、原数据和权重。复用已有环境，无需安装、升级 Torch/Ultralytics/驱动。

先自行停止其他 GPU 计算任务（包括 prunecheck），不要与训练同时测速。在新 Anaconda Prompt 执行：

```bat
conda activate D:\ModelImprovement\envs\detailpair_cuda_v021
cd /d D:\ModelImprovement\DetailPair_v0.2.5
windows\15_evaluate_existing.cmd
```

该 .cmd 使用已有环境的绝对 python.exe，并检查退出码；也可直接双击运行。数据和两份 best.pt、selection.json、旧 metrics.json 的精确路径在 `configs/realtime_comparison.yaml`，默认沿用 Work 本轮明确给出的旧 v0.2.4 路径。路径变化时只修改路径字段，并保留对应原 run/variant/seed 绑定；不要静默替换权重。路径含空格时可传入配置：

```bat
windows\15_evaluate_existing.cmd "D:\My Configs\realtime_comparison.yaml"
```

找不到权重/selection、种子/条件/run/类别不一致、源图或标签哈希变化、其他 GPU 计算进程、CUDA/依赖不符合时会报告具体错误并保留失败包；不会训练、resume、自动下载权重、CPU回退或结束其他进程。数据审计沿用原训练 selection，并核验1438train/180val、val429框；不重新划分。

## 运行和输出

默认val，device0，320×320，batch1，FP32，fuse=true。AP入口conf=.001；固定工作点conf=.25、NMS IoU=.7、匹配IoU=.5、max_det300；两类P/R不能混用。主计时从已解码主机图片到主机检测结果，含预处理/H2D/推理/NMS/D2H，排除磁盘/解码/加载/绘图/写文件；warmup30排除，完整180图重复3轮，每条件540条，共1080条。

输出独立 `runs/realtime_eval_时间戳`：

- 配置/环境/权重身份与哈希、selection/源数据/共同派生数据清单。
- `B2|B6/timing/latency.jsonl` 和汇总，含逐图逐轮延迟、实际shape、预测数、模型实际参数和GPU进程记录。
- 每条件/每场景的 `official_ap.json`、逐类CSV、低阈值AP预测JSONL，以及 `fixed_metrics.json`、固定阈值预测/GT/一对一匹配JSONL。
- clean对旧metrics的原始差值；各档对各自clean下降、B6−B2全档/全类对比，`accuracy_latency.csv/.png`。
- success/partial_or_failed/failed状态、失败traceback，保留不利结果，不把部分失败当全部成功。

结束或失败时自动打印 `exports/DetailPair_eval_results_*.zip`。上传此完整结果ZIP给 Work，不要只挑有利图表。默认不含原数据、权重或整套派生图片；这些仍在本地。中断或需要重新打包时：

```bat
windows\16_pack_evaluation.cmd
```

test保持留出，默认入口不会评价test。显式最终测试需要方法/阈值/权重/数据冻结记录，见 REALTIME_EVALUATION_PROTOCOL.md；不要用test选参数。

## 证据边界

Work审阅包记载：B2=162轮早停、B6=200轮上限；独立val AP50:95为34.8363%和34.6994%，不支持B6整体精度提升。本轮收到的是审阅材料，未收到其引用的133项原始结果ZIP和真实权重；不把审阅当作本轮独立复算。

云端执行的是CPU随机检查点/合成数据工程检查，非真实NEU结果；详见 CHECKS.md。Windows、实际CUDA/权重、val180完整评价、1080条GPU测速和真实扰动结果待用户执行。FLOPs未核验，不抄替换层前summary；合成扰动不代表采集噪声，重复图片测速不等于多训练种子。未来多种子/消融或停止组合，由Work依据本次结果决定。
