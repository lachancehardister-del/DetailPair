# BoxRoute研究交接包（2026-10-16）

本包交给Codex实施下一版完整工程。**它是研究核验与实现任务交接，尚不是可直接训练的BoxRoute代码包；本轮用户不用本地运行。**

阅读顺序：

1. BASELINE_AND_NEAREST_AUDIT.md：原生路径、真实输入隐患、最近邻公式/目标与获取边界。
2. MODEL_PROPOSAL.md：唯一新候选、公式、对照、G0/G1预算及停止标准。
3. CODEX_TASK_BoxRoute_v0.3.0.md：可直接执行的完整实现合同。
4. protocol.yaml：同一协议的机器可读版本。

将本ZIP和任务MD一起交给Codex即可，不必重新提供旧文件。inputs内已有原始v0.2.9工程ZIP，sources含固定官方/作者参考源码和实际下载的公开初始化权重，附来源/许可证。PROJECT_STATUS保留完整项目历史并追加本轮。

检索来源：SEARCH_LOG.json含全部23条查询、平台、日期、实际返回记录标题/URL/引用ID及失败；SOURCE_RETRIEVAL_LOG.json含直接获取的成功/失败/时间/SHA；SOURCE_RECORDS.json区分7项全文方法与1项预览+选定代码。没有将检索总URL数当纳入研究数。完整网页/论文原文缓存是审阅中间材料，不在本交接重新分发；需要原文时按记录URL/哈希获取。

check_design_math.py、requirements_audit.txt及checks只用于复核本轮实际执行的NumPy/AST检查，不是新模型训练实现。未声称网络前后向、真实数据训练或Windows/GPU已通过。保留首次AST检查失败和修复前脚本。

Codex交付的下一包必须是独立完整BoxRoute_v0.3.0_Code.zip，包含真实源码、依赖、配置、README、Windows和结果打包入口；默认用户先运行G0。G1入口虽实现，须G0回传Work审阅后才使用。不自动运行200轮/多seed或test。
