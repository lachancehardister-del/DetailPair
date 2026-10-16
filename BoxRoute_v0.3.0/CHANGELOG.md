# BoxRoute v0.3.0 — 2026-10-16 Asia/Shanghai

- 建立独立BoxRoute分支，保存DetailPair v0.2.9验收、Work新方案/冻结协议/来源与许可证，旧分支不改。
- 原生N直接用固定DetectionModel；相同state的旧B0通过8组CPU数值检查，eval、原生三分量/总loss、全部共同梯度/BN及一步AdamW更新最大差0。
- 全条件统一私有逐字节train/val副本与rect=False构建，观察train/训练val/final_eval/独立val实际320/FP32；不枚举test，不在源目录写cache。
- COCO公开权重重建nc6共同state，完整记录迁移/未迁移key；原layer16后接模块，原计算/元信息及Detect16/19/22保持正确。U/G/W完整state相同。
- 三专家路由/训练框几何ROI KL、W交换H/V、单视图原生loss.sum()+Bλroute；finally清hook，model只保留detach标量诊断；EMA/保存重载/融合通过。
- 固定PKS五分支/19范围的BN等价端口，对原类FP32输出/梯度及含边界HKR部署转换通过；S明确为PKS-adapt。
- G0实机入口包括64/16两轮流程和初始48图384样本平衡开销；G1独立门禁、60轮未剥离终点EMA、10组评价/AP75/固定P/R/GT分组和训练后1440计时已实现。
- 保留修正的支持集合宏P/R、null、六类零约定和micro；完整表格与失败保留，不以best或合成结果宣称NEU收益。
- CPU合成五子进程两轮流程及增强哈希/320输入/终点身份/G1拒绝/结果排除范围通过；Windows/CUDA/真实NEU/GPU速度与60轮效果尚未执行。
