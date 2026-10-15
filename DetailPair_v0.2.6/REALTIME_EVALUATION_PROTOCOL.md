# DetailPair 复用权重评价协议 1.0

日期：2026-10-13（Asia/Shanghai）。来源：CODEX_NEXT_EVAL_20261013.md；与原v0.2.5任务不一致时本轮方案优先。新增阈值与扰动属于clean长训结果后提出的探索协议，不追称预注册。配置原文SHA256、有效配置、软件/硬件环境均落盘。

## 权重与数据

B2绑定pilot__20261012_230632_329133/B2_s0；B6绑定pilot__20261012_125741_680519/B6_s0，seed0。只读各自验证选出的best.pt。检查自定义PairDetectionModel/variant/pair_config/实际门控层、checkpoint train_args里的run/seed和冻结训练配置、names，以及权重SHA256。save_dir缺失时用project/name还原记录的run；不依据文件名猜条件。原训练selection对应train/val的源路径/图标签SHA256逐项验证，两条件源身份一致；工作副本路径差异不判作样本差异。

沿用1438train/180val/180test；默认评价val180、GT429。只在明确最终测试时读取test标签进行评价；不重划分、不额外训练、不修改原图/标签/权重。新输出为独立时间戳目录。原数据许可/工件独立性/9个既有近重复候选风险不因本轮工程核验消失。

## 计时

严格batch1、FP32、320×320完整张量，rect=False/LetterBox auto=False，灰边114、BGR转RGB、CHW/255归一化。AutoBackend fuse=true两条件相同策略；记录BN剩余数、实际模型参数、dtype和实际shape。自定义Haar层BN可能仍保留，忠实记录，不宣称两个模型已完全融合。FLOPs留空/未测。

主计时：已解码主机BGR数组 -> letterbox/归一化 -> H2D -> FP32推理 -> class-aware NMS(conf=.25/IoU=.7/max_det300，single-label) -> 坐标映射到原图 -> 主机xyxy/score/class数组。含上述所有操作。排除读盘/解码、模型加载、warmup、JSON转换/写盘、画图。CUDA开始前及结束前同步；perf_counter_ns墙钟计时；eval/inference_mode。NMS工作点与固定P/R一致。

每条件warmup30次排除，val180固定源路径排序，顺序重复3轮，每条件540条/总1080条。两条件分开顺序测速，同一时刻只测一个GPU模型。启动、每轮前后读取GPU计算进程；其他进程或查询失败直接阻止，不自动终止外部任务。该检查不证明整段运行没有短时竞争或功耗漂移；环境记录和逐轮统计供Work审阅。功耗模式/插电/机器型号无法自动读时为unknown，可在hardware_notes补充实际非秘密信息。

保存每图ID/轮次/variant/run/seed/权重哈希/时戳/预测数量/shape/latency_ms。mean、P50/P95用numpy.percentile默认线性插值；串行FPS=n/总秒数=1000/mean_ms。每轮及总样本数均可重算。重复测量不是独立训练重复；不定义“30FPS合格”，两个点不叫完整Pareto曲线。

## 精度与逐类匹配

官方独立权重重载验证：Ultralytics8.3.221 validator、batch1、FP32、rect=False、conf=.001、IoU=.7、max_det300、官方multi-label NMS和IoU .50:.05:.95 AP定义；导出原始低阈值预测及各类AP。官方最佳F1的P/R单独命名，不能冒充固定conf=.25。

固定工作点：与主测速相同预测接口，conf=.25、NMS IoU=.7、max_det300、single-label；匹配IoU=.5。各类按置信度降序/同分原索引排序，对尚未匹配的同类GT选择最高IoU，达到阈值为TP，否则FP；每GT最多匹配一次；剩余GT为FN。重复预测FP、错类预测FP且其未匹配GT为FN。GT=TP+FN；P=TP/(TP+FP)、R=TP/GT。零分母写null，宏平均排除未定义类别并记录有效类别数，不以1或0冒充。

官方AP与固定工作点匹配算法/阈值/多标签策略有明确差异，不能互换。names由data.yaml读取并与checkpoint比对；多类图片的分类图像计数不相加当val180。全类/全场景输出，尤其crazing与pitted_surface，不筛除不利项。clean新官方指标与旧metrics.json原样并列/保存差值；旧文件缺失则明确missing_original_metrics，绝不补写新值或猜差异原因。

## 共同扰动

只生成一次派生数据，供两条件共用：Gaussian std .01/.03/.06与impulse .01（黑/白各.005）。RGB浮点[0,1]（float64），clip，numpy.round(x*255)转uint8无损PNG；随机种子178。每档重新创建numpy.default_rng(178)，Gaussian各档共享相同标准随机场后按std缩放；impulse用同一seed的均匀空间随机场，三个通道共用mask。原图ID/原图标签哈希/派生图标签哈希、实际MSE、SNR dB均保存。SNR=10log10((mean(original²)+1e-12)/(mean(quantized/255-original)²+1e-12))。

派生图/标签在本地runs/.../datasets中保留，每次使用前核验哈希。回传ZIP排除该目录，清单另存在run顶层。每档报告官方AP、固定P/R、相对各自clean的下降(clean−档位)及B6−B2；所有档位都报告。主测速仅clean，扰动不增加完整测速。

## 最终test显式门禁

默认配置split只能val。方法、阈值和权重固定后，Work明确决定最终test，才使用windows/17_final_test_explicit.cmd。参数1为冻结JSON，参数2可选配置；此入口不会训练。冻结JSON必须含：

```json
{
  "method_and_thresholds_frozen": true,
  "split": "test",
  "config_sha256": "原val配置文件完整字节SHA256",
  "weights_sha256": {"B2": "best.pt SHA256", "B6": "best.pt SHA256"},
  "test_records_sha256": "下面canonical test records SHA256"
}
```

test_records_sha256由detailpair.evaluation.read_records(data,'test')返回records，用json.dumps(records,sort_keys=True).encode()求SHA256。只能为身份审计生成此冻结记录，不看test预测/指标调参。最终test入口检查配置/权重/数据身份与显式冻结记录，保留test180但不要求GT=val429。所有报告写split=test；原训练train/val selection仍核验。最终测试后不依据test结果回调阈值或模型。

## 结果与失败

任何前置失败都记录request/environment/failure/status并尝试打包，退出1；单条件/单场景失败保留原始结果并尝试其他独立条件/场景，最终partial_or_failed和非零退出。成功要求2条件×5场景、完整计时和源/权重哈希末尾复核。自动包和手动pack保留结构化配置、环境、身份/派生清单、预测/匹配、计时、CSV、图、失败及源码；默认排除原图标签、权重、整套派生图片和缓存。用户原权重留在本地，本版不上传它们。
