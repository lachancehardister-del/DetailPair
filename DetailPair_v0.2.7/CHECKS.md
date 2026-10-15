# v0.2.7 实际检查（2026-10-15 Asia/Shanghai）

环境：云端Linux/Python3.12，Torch2.5.1+cpu、torchvision0.20.1+cpu、Ultralytics8.3.221，pip check通过。精确版本见verification/environment_cpu.json。

通过的8项工程测试：

1. A=.2/1/3、g=.2/.5/.8九个标量例子，自动梯度与解析−A/(A+.001)及有限差分一致，clean目标stop-gradient。
2. 空前景/空背景、零幅值、原region_loss与前/背景代数重构、零范数有限且target无梯度。
3. 原model.loss与诊断total及完整layer1目标参数梯度一致，B2/B6各batch1和batch2；batch缩放、参数/BN不变；B6投影对A未连通、B2 A不适用；分带加权误差重构。
4. 注入criterion异常后临时hook/capture/cache清理。
5. zero_norm、disconnected和未定义cos/null区分。
6. train-only读取（val/test故意不存在）、独立Random178可重复选样、已安装LetterBox图/框一致变换；人工宽图例框从(.5,.5,.5,.5)变为(.5,.5,.5,.25)，实际2×3×320×320。
7. 冻结协议拒绝改CPU/样本数/权重/训练batch，失败与未执行计数，不把相关批次当训练重复。
8. checkpoint完整train_args恢复，不猜缺失box/cls/dfl；禁止no_grad/inference模式与BN意外training模式。

完整CPU合成集成：明确随机B2/B6自定义检查点，6张合成train图固定取4，两条件各2批，实际320×320/batch2；两模型同图/同噪声/偏移，逐批original total与目标层grad对齐（本次最大绝对梯度差B2=0、B6=0），所有参数/BN/磁盘权重/源图标签hash不变。缺配置CLI实际返回1，保留failure/status/ZIP，包CRC和排除图/标签/权重/大张量检查通过。

日志：

- mechanism_cpu.log：首轮单元通过，但加载权重后原args被Ultralytics精简，集成因缺loss系数失败；该记录和失败run保留。
- mechanism_cpu_fixed_args.log：修正为从checkpoint train_args恢复后集成通过。
- mechanism_cpu_release.log、mechanism_cpu_final.log：增强后的8项单元与完整CPU集成通过；故意缺配置traceback为预期失败路径，未删记录。
- v027_checks.json：当前实际检查机器摘要；input_provenance.json：本轮Work审阅包身份与资源边界。

model.py/engine.py/data.py与v0.2.6逐字节一致，源解析通过。结果ZIP与最终交付ZIP另作解包/清单/命令入口验证。原始214项真实recheck包本轮未收到，不把Work复核数字写成本轮独立复算。

未执行：Windows .cmd、实际CUDA/RTX4060、真实两份best.pt加载、NEU64train/每条件32批机制诊断、实图是否存在局部冲突。云端随机权重/合成图不是实证，不证明B6失败根因、AP因果或监督改动收益。

科研工具箱原版模块在可用技能目录与workspace资源中未找到，未虚构已读；实际依赖、原模型/损失源码和上述数值/梯度测试已检查。没有额外文献检索、外部付费服务或投稿。
