# DetailPair v0.2.9：离线归档与精度报告修正

当前组合状态为 **`archived_no_consistent_benefit`**。本轮只归档与修正未来报告，**不需要你在本地运行任何实验**。

Work复核同完成38轮、EMA、实际320的旧B5/B6：clean AP50–95为28.4432%/29.8228%，Gaussian.03为27.8656%/29.1564%，未观察到去幅值项的一致收益。按预设规则封存当前完整组合及直接删项方案。保留论文模型改进目标，等待Work的新瓶颈假设及最近邻公式/目标/计算成本核验；本轮未做新模型研究或实现。

## 本次交付

- [CURRENT_BRANCH_CLOSED.md](CURRENT_BRANCH_CLOSED.md)：封存范围和无需新实验的说明。
- [DECISION_LOG.md](DECISION_LOG.md)、[RESULTS_INDEX.csv](RESULTS_INDEX.csv)：各阶段证据、预算、权重轮次、输入shape及限制。
- [archive/work_review](archive/work_review)：收到的Work报告与派生汇总表的只读副本；原ZIP、完整解包及首次换行失败仍在仓库本版本work_inputs中。
- [archive/reporting_correction](archive/reporting_correction)：基于Work已有逐类汇总表的离线比较，不是重新匹配720条预测。
- [CHECKS.md](CHECKS.md)：实际离线验证；[PROJECT_STATUS.md](PROJECT_STATUS.md)：保留Work完整状态并追加本次实现事实。

## 精度报告约定

原逐类precision无预测分母时继续为null，原available-class宏P/R保留。新增支持类ID和数量；两条件支持集合不同，宏平均差值为null，注明 `different_class_support`。同为5类但具体类别不同，也不能直接比较。

另列 `macro_Precision_zero_no_predictions`：固定全部六类、无预测类计0的明确报告约定；原null不改。`micro_Precision`使用全部TP/(TP+FP)。无整体分母时仍为null。macro recall同样报告实际支持集合。JSON内对应字段为 `macro`、`macro_fixed_class_zero_no_predictions`、`micro`。

本次B5/B6原宏P支持6类/5类，不能宣称统一六类宏P优势15–16pp；六类零约定clean为.5709717356/.6049659019，noise为.5715236077/.6142768587。AP和原TP/FP/FN未更改，匹配、模型、损失、训练及噪声流程保持原样。

## 来源和边界

本环境仅收到12项审阅ZIP，缺少其引用的154项原始matched结果ZIP。原始四组评价、共同轮次/EMA、权重身份和720条重匹配为Work报告，不能写成本轮独立复算。本轮实查审阅ZIP CRC/路径/11项清单哈希，并核对Work派生逐类表的内部汇总算术。没有加载真实权重、推理、反向、GPU评价、测速或访问test；不要求重复上传已有结果。

完整源码ZIP保留依赖、配置、原Windows入口和结果打包入口，以便追溯及未来明确授权后的任务使用。旧入口保留不代表本轮安排继续跑24评价或任何训练。数据固定train1438/val180/test180，共1798；Windows原图和权重继续本地保存。

模型/engine/data与v0.2.8逐字节一致。未来报告改动只涉及 `detailpair/reporting.py`、`detailpair/evaluation.py::fixed_evaluate` 的新增汇总字段和 `matched_checkpoint_eval.py::comparison`；结果打包同时保留archive目录。旧版本、旧原JSON/CSV、CUDA失败和计时失败一律保留。
