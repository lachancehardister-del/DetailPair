# v0.2.6 实际检查（2026-10-15 Asia/Shanghai）

已执行：

- 原输入ZIP187项CRC与安全路径检查通过，SHA256为51c5dc11ef2e4c85af7a0bf1d34496fd623bedb8d84f367a4c86a300949b7bfc。源码benchmark/realtime_eval/model/engine与v0.2.5一致。原QQ失败与成功/漂移记录保留。
- 独立重算1080条原计时、逐轮均值和clean AP/TP/FP，见verification/received_result_check.json。没有重跑用户真实权重。
- 7项单元检查：ABBA+BAAB窗口/位置/FPS统计；漂移和不完整记录不删除样本；缓冲调度和warmup排除/中途2条partial保留；新2.0冻结拒绝且旧1.0不放松；实际CPU torchvision NMS bootstrap几何；实际DetectorPool的模拟设备串行lease/卸载与禁止并发；Windows风格GPU查询N/A字段标准化为unknown。
- CPU完整工程集成使用明确随机B2/B6检查点、6张合成JPEG：八区组48条计时、旧JPEG/新PNG两条件官方AP、逐图loader像素/input tensor/匹配记录、原selection顺序恢复、源图/标签/权重hash不变。另单独执行rect控制函数。CPU测试的warmup为短配置，不等于正式30秒/200次。
- 缺配置CLI实际返回1并保存failure/status/ZIP，正常运行和中途异常打包都检查；返回ZIP排除datasets、原图/标签/样本图、权重和缓存。
- model.py/engine.py/data.py字节一致，HostDetector类AST一致，旧realtime_eval.py/1.0配置字节一致；Python源解析通过。旧run.py pack转发新结果打包器实际执行。

日志：recheck_cpu.log为首轮开发集成；recheck_cpu_final.log为增强集成；recheck_unit_release.log为最新7项单元检查；legacy_pack_forwarding.log为旧pack转发检查。预期缺文件traceback是故意的失败路径，不伪装为正常成功，也未删除。

未执行：Windows .cmd、真实CUDA kernel/RTX4060、两份真实best.pt重载、30秒共同/200次正式预热、val180/GT429的1440条GPU计时、真实clean旧新AP/因果诊断、正式test。GPU设备字段解析通过模拟响应检查，真实驱动/功耗/P-state可用性待本地运行；不凭CPU或静态检查声称这些成功。

CPU检查只能证明工程流程。速度结论必须看本地新run的data completeness、timing_status、窗口/位置变化及设备快照；即使通过1.10筛查也不是无混杂的证明。本轮不声明模型创新或可投稿。
