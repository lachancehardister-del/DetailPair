# v0.2.5 实际检查

日期：2026-10-13（Asia/Shanghai）。云端Linux/Python3.12.14，Torch2.5.1+cpu、torchvision0.20.1+cpu、Ultralytics8.3.221；无GPU。

通过：

- Python源文件编译、冻结配置解析、model.py/engine.py/data.py与v0.2.4字节一致；旧run.py benchmark函数AST一致。
- 5项unittest：重复预测/错类/空GT/空预测/未定义分母；置信度顺序/漏检；均值/P50/P95/FPS；子进程原生继承且退出7/GPU竞争诊断；已知张量驱动实际CPU NMS及host预处理/归一化/坐标映射。
- 明确CPU合成集成：两份随机初始化自定义B2/B6检查点、4张合成val图片、5个场景、10次官方AP和固定P/R评价、24条320×320计时（每条件12，warmup2排除）、40条低阈值官方AP图片记录。实际参数/shape/逐轮统计均落盘。此测试无训练，随机AP不代表真实结果。
- 变体/seed/run不一致、缺权重、选择清单标签哈希变化均拒绝；原权重/源数据hash保持；同seed派生图复现一致。
- 缺配置CLI实际退出1，保留failure/status并打包。结果ZIP CRC通过，排除datasets、.pt、cache并包含失败。保留两次开发集成及各自预期失败运行，不删记录。
- 原tests/check_invariants.py、tests/check_novelty_controls.py运行退出0；pip check无冲突。

日志：verification/realtime_cpu.log为首轮（4个单元测试），realtime_cpu_final.log为增强后完整集成（5个单元测试）。invariants_v025.log、novelty_controls_v025.log、pip_check.log是本版实际检查。

未执行：Windows .cmd和原生控制台刷新、CUDA12.1/实际GPU NMS、真实B2/B6 best.pt加载、真实val180/429GT和1080条主GPU计时、真实精度/类别召回/扰动鲁棒性、最终test、长训练、FLOPs。CPU测试为工程流程检查；不可用其延迟作模型架构效率结论。Work引用的133项原始结果ZIP未上传，本轮只保留审阅包及其分析输出。

整包哈希/源码清单由打包时生成 verification_manifest.json 与发布ZIP旁SHA256文件。历史verification/v024_checks.json仅反映旧版本检查范围（当时无Torch），不代表本轮CPU环境。
