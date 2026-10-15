# DetailPair Timing Recheck 2.0

日期：2026-10-15（Asia/Shanghai）。来源：Work的CODEX_TIMING_RECHECK_20261015.md。未收到单独EVAL_RESULT_REVIEW_20261015.md；本轮直接核验原始评价ZIP及任务说明，足以完成工程。1.0入口/配置不放松；新增独立2.0配置和all/timing-only/clean-diagnostic入口。

## 冻结身份与输入

B2 SHA256：c234ae47e6d8257543a038abc3ea0714d0f081abf2ce5e9dbe0dcd9c73a2c21b。
B6 SHA256：2a5f5dd7152b28a3a3952f96fbeca409e231e07de2a06a6b5686d7a01c6bc324。

variant/seed0/原run/训练目标及源selection train1438/val180逐项哈希核验，val429GT；test封存。核心model.py/engine.py/data.py与旧版逐字节一致。权重只读，原图和标签不改；不训练、resume或下载替代权重。依赖Torch2.5.1+cu121/vision0.20.1+cu121/Ultralytics8.3.221、GPU device0 RTX4060 Laptop，FP32，无CPU回退。CPU测试必须单独synthetic配置和显式标志，不允许用于真实结论。

## 预热、调度与计时

固定 `B2,B6,B6,B2,B6,B2,B2,B6`，8组×180=1440，单条件4组720。图序为固定source manifest ID顺序，共同clean PNG的PIL解码BGR主机数组，保持v0.2.5主计时图片处理方法。torch/torchvision CUDA NMS与两模型实际推理先初始化，初始化记录排除。

共同预热目标30秒，两条件等量交替，每对完成后检查截止时间，因此最后一对可能略超目标；记录目标/实际秒数、各次数和每次warmup。设备迁移间隙计入共同预热墙钟但不计入推理样本，不能声称整30秒GPU持续满载。两条件同规则，不因某条件慢另加次数。每区组再固定200次完整同接口预热。所有warmup有独立phase，绝不混入计时。

CPU保存两份已融合backend；每次lease只迁移当前条件到GPU，退出后卸回CPU，禁止并发lease。初始化/迁移/empty_cache不在样本内。原HostDetector.predict/timed函数保留：LetterBox auto=False固定320²，BGR→RGB→CHW/255，FP32/H2D，实际模型forward，class-aware single-label NMS(conf .25/IoU .7/max_det300)，坐标还原/D2H/主机检测数组；GPU开始前/结束前同步，perf_counter_ns。排除读盘/解码、加载、预热、写盘、JSON转换、快照、绘图。每图记录block/variant/seed/run/权重hash/imageID/time/shape/detections/latency。

区组内部仅缓存数据，不逐图flush或调用nvidia-smi。组末写完整JSONL，异常写latency_partial和完整已完成记录。硬终止进程/断电可能丢失尚在内存的当前组；正常Python异常/KeyboardInterrupt保留partial，不伪称能抗强制杀进程。每组前/后低频设备和进程快照；竞争进程由用户退出，包括QQ，不自动kill/忽略。查询不可用时unknown，无法确认GPU进程状态则阻止。功耗/电池/powercfg查询只读，不调节硬件。

## 预定义分析

报告每组/每条件总体mean/P50/P95，串行FPS=1000/同口径mean_ms。三等分窗口（60/60/60）和各条件四组均值，保留全部样本。窗口比和同条件跨组max/min比>1.10→timing_unstable；阈值是工程筛查而非期刊或显著性标准。

ABBA和BAAB分别计算两条件组均值及B6−B2；另报告四个相邻先后位置配对、faster_variant是否改变。样本/区组不完整→incomplete；位置排序变动或漂移超限→速度比较待Work评估。即便全部通过，也只标工程筛查通过，不证明消除了混杂。运行exit0=无执行失败，与稳定/速度优势分别判断；不稳定不自动重试、不择优保留某轮、不把旧全局均值画为新比较点。

## clean受控诊断

旧evaluate后续单张predict不参与AP，本轮不重复；旧入口忠实于v0.2.4 evaluate的val调用：data/split/imgsz/batch1/device/workers0/plotsTrue及输出路径，其他参数由已安装8.3.221解析。观察类只记录，不改解析/输入张量或匹配。保存call_kwargs、model overrides、有效validator args、实际loader order/rect/batch_shapes/transform属性及已安装loader/letterbox源码。

旧数据用selection对应字节一致原格式JPEG及标签副本，恢复旧working文件名排序；rect加载器自身重排长宽比照常保留，实际顺序单独记录。新数据按v0.2.5同一PIL RGB→uint8 PNG生成，显式halfFalse/rectFalse/conf.001/IoU.7/max_det300；官方AP及多标签匹配定义不改。

通过安装版BaseDataset实际imread hook记录原JPEG/PNG实际loader BGR和RGB像素hash；另记PIL源RGB和PNG RGB hash。preprocess前后记录实际shape/dtype/归一化tensor hash、值域、ori_shape/ratio_pad。哈希仅身份观测，不把BGR/RGB不同顺序当作解码差异。观测复制/哈希成本属于诊断，不用于速度统计。

两条件各跑旧JPEG和新PNG，保存全类AP/预测/官方IoU .50:.05:.95逐预测TP矩阵、GT框和类。旧metrics.json与新旧入口指标并列差值；不存在时明确missing。固定conf=.25已由Work核验，本轮不重复跑固定P/R，不更改conf。

最多一个预声明因素rect：当有效rect为旧True/新False，并存在其他非绘图参数差异或实测解码RGB差异时，对同一新PNG保持所有新参数仅rectTrue，记录AP变化及残余旧入口差值。因果解释范围限于该PNG上rect设置（含shape/padding）变化；其他差异未解不预归因。条件不满足不追加其他因素，不搜索AP更好配置，不试验正式test。

## 归档

新run独立时间戳；原始QQ失败/漂移/所有旧版本保留。返回ZIP不含原图标签、派生图片、模型权重、官方样本可视化和缓存；包含清单/必要hash、输入张量身份/有效参数、预测匹配、全部计时/预热/快照/失败、协议和源码。原始review输入及其SHA单独保留。无付费API、额外账号、投稿或编辑通信。
