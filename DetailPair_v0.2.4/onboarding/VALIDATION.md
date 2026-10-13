# 云端检查结果

日期：2026-10-13（Asia/Shanghai）。

- 实验包 SHA256 与附件文档一致；57 项文件哈希导入前及验证后均通过。
- Python3.12.14，Torch2.5.1+cpu、torchvision0.20.1+cpu、Ultralytics8.3.221；requirements.txt 固定版本安装成功。
- install.sh 已执行并重复执行成功；两次 pip check 均无依赖冲突。解析版本保存在 requirements_cpu_resolved.txt。
- tests/check_invariants.py 与 tests/check_novelty_controls.py 均退出0，全部既有断言执行通过。
- run.py smoke 退出0，实际检查13个条件，前向/反向数值有限与模型深拷贝通过。
- run.py synthetic --device cpu --variant B6 退出0：一轮训练，results.csv 一行；best.pt 保存、独立重载验证4张合成图片、单图预测、status=success 和 completed=completed 均确认。AP为0属短合成流程结果，不代表真实检测性能。
- run.py pack 退出0；结果ZIP完整性及排除权重/合成输入检查通过。
- Git 已跟踪文件未改动；仅新增版本目录与 onboarding 辅助文件。训练生成 runs、exports、local_settings 和 Python缓存，均位于新增版本目录。
- 未执行 Windows、CUDA、真实数据训练/评价、200轮比较、最终test、端到端测速或v0.2.5开发。

日志见 logs/；原始历史结果ZIP未提供，40轮性能仅核对既有报告，不是本轮独立复算。
