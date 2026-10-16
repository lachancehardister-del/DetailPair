# BoxRoute v0.3.0 工作日志

日期2026-10-16，Asia/Shanghai。持续用户授权为Work定方案、Codex实现/验证/日志/自动Git同步、交完整工程后用户本地GPU运行、回传Work。保留旧版本/失败；v0.2.4/onboarding九项既有未暂存删除不处理/不提交。

先收到DetailPair_ArchiveAcceptance_20261016.zip：Work验收v0.2.9通过、无新归档工程任务；原件与验收报告保存在本新分支work_inputs，不修改旧发布包或反复发行归档版。随后收到CODEX_TASK_BoxRoute_v0.3.0.md，在缺配套提案时仅推进可独立核验原生/B0；向用户询问材料位置后，用户提供BoxRoute_Research_Handoff_20261016.zip并明确内容在该包。补齐后按冻结MODEL_PROPOSAL/protocol和固定来源继续，未猜几何q或替代PKS。来源SHA/大小/清单见verification/input_provenance.json与sources_manifest.json。

独立BoxRoute_v0.3.0，旧DetailPair不改。Work最新PROJECT_STATUS从新交接导入并追加实际实现，未用旧状态覆盖新决策。复用旧包的修正reporting、共享扰动/固定匹配和测速边界，重建新条件身份与原生单视图训练；没有调用旧B2/B6或旧B5/B6专用checkpoint validator。legacy model仅为B0等价；N直接官方DetectionModel。

实现共同nc6COCO初始化与完整迁移key/哈希、原layer16后同维包装、U/G/W相同state，PKS固定五分支端口。正确/交换框目标、ROI平均KL、Bλ单次加到原生loss.sum、finally清hook/无非叶缓存、detached诊断。全条件共享强制rectFalse、私有源复制/cache管线与实际320断言；test只历史计数不枚举。五子进程继承终端，不PIPE tee。G0的2轮工程流程由Windows入口显式--flow-check，G1研究pilot不自动启动。

G1入口提供冻结60轮、seed0、共同初始化全数据、AdamW与弱几何配方、实际步数/warmup/增强hash、strip前字节复制终点EMA、元数据/CSV核验、clean/noise独立官方AP/AP75与固定P/R/GT分组、修正宏支持集合、训练后1440计时与全部门槛。门禁需真实G0及Work批准记录，并绑定协议/代码/源数据/初始化；合成G0不可放行。未执行60轮或正式扩展。

实查已有venv Torch2.5.1+cpu/vision0.20.1+cpu/Ultra8.3.221。无GPU，未安装新软件、MMCV/MMRotate/timm或升级。完成8原生等价组合、6项几何/模型/PKS/序列化、1数据管线、3评价/计时记账+1决策门槛测试；完整显式CPU合成G0五独立子进程各2轮通过，实际320/终点EMA/源只读/增强哈希及G1拒绝和结果pack验证通过。错误/失败保留、纠正详见CHECKS。公开初始化已实际torch.load；Work仅下载的状态已更新为本次工程事实。

实测G相对N静态参数+6276（+0.242222%）、卷积MAC+9,216,000（+1.167334%），不当GPU延迟或检测提升。48条CPU计时只验记账，真实G0的384条及G1的1440条尚未执行。无NEU收益、创新性或可发表性结论。科研工具箱原版模块当前未提供，不宣称已读；研究资料来自明确Work交接。

所有输入、源码、来源/许可、配置、检查、日志和发布包在本版本内；根目录只更新介绍。Code ZIP明确包含公开官方初始化，结果ZIP排除原图/标签文件/私有副本/用户训练权重/特征；Git只同步提供材料与工程元数据，不同步云端合成训练.pt或图像。完成后追加发布SHA及远端核验。

最终复检捕获CPU记账测试固定输出目录复用，保留旧/partial与失败日志，测试改为时间戳目录后重验；真实suite原本使用新run目录。辅助metadata命令使用系统Python无Torch，改回实际测试venv；不安装依赖，说明保留verification/preparation_notes.md。




## GitHub同步核验

2026-10-16T21:21:18.389660+08:00：完整BoxRoute工程、Code ZIP、Work新交接与旧归档验收材料、实际CPU检查/合成流程/失败记录已推送origin/main，远端commit `5d926a6846ad318fdb72eb7130da3f049624d8f8` 与本地HEAD一致。真实用户图像/标签文件/训练权重未上传；仅代码包明确包含的固定公开初始化随源码同步。v0.2.4/onboarding九项用户已有删除未暂存。后续仅追加本同步记录，不改发布包。

交付前补全G0全train的q三分支直方图/分位数、宽高比/面积分位及逐类信息性覆盖；2项手算统计检查通过（方形/横/竖三框，及全空GT应blocked_shape_support）。仅统计元数据增强，模型、训练和测速公式未改。

正式交付：releases/BoxRoute_v0.3.0_Code.zip，SHA256=a8e8321c76d84acba7dee19567b80aa80044d1e35ca73f3e784e7434b2fe8554，5757035 bytes。完整包最终CRC/清单/解包/入口/公开资源核验通过；补充的train分布统计2项检查通过。此前构建为中间版本，交付以本值和releases/*.sha256为准。
