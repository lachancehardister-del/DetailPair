# v0.2.8 实际检查（2026-10-15 UTC）

云端Linux/Python3.12.14，Torch2.5.1+cpu、torchvision0.20.1+cpu、Ultralytics8.3.221；pip check通过，未安装/升级依赖。environment_cpu.json记录实查，无GPU。复用既有云端环境，本轮无新增环境草稿/服务配置。

## 已执行

`/workspace/venvs/detailpair/bin/python tests/check_matched.py --integration`（显式CPU合成配置由测试生成）：11项检查及完整四组评价通过。日志 `verification/matched_cpu_final.log`；最终设备/官方实际设置增强后复检见 `verification/matched_cpu_release.log`。

1. 安装版save_model与save_metrics原始源码核验；checkpoint37对应CSV38，缺epoch/剥离epoch=-1、文件名冲突拒绝。
2. 错variant与错误class names拒绝（检查点路径绑定正确但模型variant错误也拒绝）。
3. 不同原run和200轮预算拒绝。
4. 不同完成轮无共同配对，最新共同轮按轮次选择而非AP。
5. 优化器/增强等train_args差异、不同EMA/model角色拒绝。
6. 不完整CSV不能证明第38轮，超过38拒绝。
7. 损坏文件保留拒绝与元数据，只扫描显式目录直接文件。
8. 固定协议拒绝CPU真实入口、test、352输入、删场景。
9. 人工一GT/两重叠预测案例一TP一FP，precision=.5。
10. 错幅值目标与非有限参数拒绝。
11. 无38时选择更早共同轮37，记录回退原因。

完整CLI集成用随机未训练B5/B6自定义检查点，**只模拟旧run元数据用于工程验证**；8张合成train、4张val。clean/Gaussian.03四组官方AP和固定P/R完成；每组4张实际FP32[1,3,320,320]、官方与固定输入分别跨模型完全一致，逐图匹配/原指标/FPPI/统一差值表齐全。所有临时检查点、源图/标签/CSV/选择文件hash前后不变。未训练模型AP=0不具有科研意义。源码数据读取剔除了test字段，未评价test。

CLI缺配置实际exit1并保留failure/status/ZIP；默认旧本地目录在云端缺失，实际exit2且状态blocked_missing_matched_checkpoint、库存与原因完整。ZIP CRC和排除原图/标签文件/派生图/权重/张量检查通过。Ultralytics官方进度输出中的Speed为库自动打印，不是本任务的公平测速结果。

上传机制结果92项CRC通过；Codex由原始B2/B6各32批/64唯一图重算全部summary并一致；同图/扰动记录一致，每批记录的参数/BN前后相同、original total差0。B6门控D/A反向21/32，A/D平均.02030257。上传源码清单中69项存在且一致，23项未被原结果打包器纳入（research/tools），没有已存在文件哈希不符。见 `verification/mechanism_result_audit.json` 与可复算脚本 `tools/review_mechanism_results.py`（在仓库内、原输入存在时使用）。未收到Work机制review文档，不宣称读过；缺原图/权重，不能独立复跑GPU梯度。

model.py/engine.py/data.py/evaluation.py与v0.2.7逐字节一致，Python源解析通过。最终发布ZIP进行CRC、逐文件清单SHA256、解压后入口help/pack-only/run.py pack验证，结果见 `verification/release_extraction_check.json`（生成后随发布包纳入）。

## 纠正及留档

matched_cpu_initial.log保留首个命令工作目录错误导致未找到测试文件；纠正目录后执行。matched_cpu_first.log中9项检查的1项断言只期待variant字样，但实际先因错误run/condition拒绝；改为构造run正确、模型variant错误的输入验证相应分支。matched_cpu_second.log的9项+集成通过；增强为11项和元数据库存、启动依赖记录后matched_cpu_final.log通过。matched_cpu_release.log再验证实际device/dtype与官方有效conf/IoU/rect/half/batch/imgsz/augment/single_cls/agnostic_nms设置。均保留，不把初次失败抹去。

## 待本地运行

Windows .cmd、CUDA12.1/GPU0、真实旧B5/B6检查点是否仍存在及完整、同轮选择、NEU val180/429GT四组AP/P/R均 **pending**。旧第38轮CSV数值仅为任务文件报告，不是本次原始复算/新320评价。B5旧CUDA非法访问原因未被修复或证明。科研工具箱原版模块在本轮可用技能与工作区未提供，未虚构已读。无新增训练、长预算、多种子、测速、test或外部科研通信。
