# 离线表格口径修正

本目录只从Work提供的classes/scenarios汇总表重新组织指标。原始154项matched ZIP和720图预测在此环境缺失；没有独立重新匹配预测、重新推理或重算官方AP曲线。来源见../work_review及../../verification/input_provenance.json。

AP、逐类TP/FP/FN/GT保留Work提供值；原available-class宏值保留，但宏P支持6类/5类时差值置null并解释。另列显式固定全部六类“无预测记0”宏精度与micro precision。CSV空差值对应JSON null，非零收益。macro recall也披露实际GT支持集合。

本目录不是新实验结果，不需要用户本地运行，不修改旧JSON/CSV，不增加科研证据。
