# v0.2.9 (2026-10-15 Asia/Shanghai)

- 按Work预设分支封存完整B6组合及直接删幅值项B5方案，状态archived_no_consistent_benefit；无新本地步骤/训练/GPU评价/测速/test或模型研究实现。
- 导入12项Work审阅与任务材料，保留首次Windows CRLF/LF哈希断言失败、修复前脚本及所有旧版本/失败。
- 新增CURRENT_BRANCH_CLOSED、DECISION_LOG、RESULTS_INDEX及来源/预算/权重/shape边界，PROJECT_STATUS导入Work完整状态并追加本次实做。
- reporting.summarize_counts保留逐类null/原available-class宏值，新增P/R支持类ID、明确固定类别无预测计0宏P及pooled micro P/R；实际fixed_evaluate出口同时写这些字段及FPPI。
- matched comparison按实际支持集合判断宏P/R可比性，不同集合差值null并给原因；同数量不同类也拒绝直接相减。另列零约定和micro，AP及逐类TP/FP/FN输出不变。结果pack纳入archive。
- 6项离线报告回归检查通过；Work派生24类汇总/4条件算术及11项审阅文件哈希核验通过，原154项ZIP/720逐图记录未提供，不冒充重新匹配或AP重算。
- 原model/engine/data与v0.2.8逐字节相同；完整源码ZIP保留依赖、配置、Windows/结果打包入口，本次无需用户运行。

# v0.2.8 (2026-10-15)

- 导入新Work任务和92项机制结果包，重算两条件32批汇总、状态及同图/扰动身份；B6门控D/A反向21/32批，A/D范数均值2.03%，不作AP因果主张。
- 新增专用old40检查点库存与验证器：显式目录、可信本地load、SHA256/有限性/模型和EMA/原run与训练策略，安装版保存与CSV源码留档；元数据结合完整CSV映射轮次。
- 最新共同完成轮<=38，无AP选轮；无配对退出2且打包，不启动训练/恢复。不同预算、run、variant、目标、轮次与来源不允许评价。
- 配对后GPU0 FP32实际320方形batch1，仅clean/Gaussian.03共享val图；官方AP和固定P/R/FPPI、逐图匹配、实际输入哈希、统一B5−B6表和只读前后核验。
- Windows24评价、25仅库存、26结果打包；复用既有CUDA环境，不自动安装、CPU回退、测速或test。
- 原model/engine/data/evaluation逐字节保留；run.py pack指向当前结果打包。完整版本独立交付，保留旧版本与失败。

# v0.2.7 (2026-10-15 Asia/Shanghai)

- 按Work决策停止完整B6正式扩展，新增64固定train图的终点局部机制诊断，不训练/测速/test/改模型或监督。
- 未融合副本、BN和随机层eval、仅layer1 requires_grad；同Generator政策重建原D/C/A和区域分项，并逐批与原model.loss标量/整个目标层梯度核验。
- 三参数组梯度norm/相对D比/cos/内积，严格区分zero_norm/disconnected/not_applicable；三子带前景/背景代理门控/幅值分布及共享归一化误差贡献。
- 全部参数/BN/磁盘权重/源数据只读检查、同图同扰动哈希、完整有效/失败/未执行计数、失败清理及结果pack。
- 首轮集成发现Ultralytics会精简加载模型args，改为从checkpoint完整train_args恢复原损失超参数并缺失拒绝；初次失败保留。
- Windows22诊断/23打包入口，固定GPU/CUDA12.1/FP32，无CPU回退、不升级环境。model.py/engine.py/data.py保持原样。
- 8项工程测试和CPU随机checkpoint合成集成通过，真实Windows/GPU64图诊断仍pending；不把局部冲突当AP因果或B5消融。

# v0.2.6 (2026-10-15 Asia/Shanghai)

- 按Work方案复核原始187项结果，确认QQ阻止记录、10/10成功及1080条样本；原B2轮间漂移不解释为架构速度优势。
- 新增独立2.0平衡区组测速协议：ABBA+BAAB、共同等量30秒目标预热、区组200次预热、1440条样本、设备/电源/进程记录、窗口与跨组漂移筛查及partial失败保留，不自动重跑。
- 新增clean旧JPEG/新PNG受控诊断：实际validator参数、官方loader像素/归一化张量哈希、原selection/实际loader顺序、预测/AP匹配及全类差值；最多一个预声明rect单因素控制。
- 新Windows18完整/19仅测速/20仅诊断/21打包入口，GPU device0/CUDA12.1/FP32，禁止CPU回退、训练、正式test和全扰动扫。
- model/engine/data不变，HostDetector计时边界AST不变，旧1.0入口/配置不放松；新pack排除原图、样本可视化、标签、权重、缓存。
- CPU随机检查点合成集成、7项单元检查与失败路径通过；真实GPU/Windows/clean原因诊断仍pending，不把工程验证当真实速度结论。

# v0.2.5 (2026-10-13 Asia/Shanghai)

- 按Work长预算复核方案复用B2/B6权重，不追加训练，不改变模型/训练目标。
- 新增独立评价入口、冻结配置、已解码图片到主机结果的batch1 FP32计时、固定阈值逐类一对一匹配、官方AP低阈值原始预测。
- 四档共同合成扰动、副本/源身份哈希、全类/全场景差值、精度—延迟图、只读检查点与selection核验、GPU竞争阻止和失败打包。
- compare_long原生终端继承；不再stdout PIPE逐行tee。run.py原benchmark语义和实现保留，pack接入完整新源码/结构化结果。
- 新增Windows15评价/16打包/17显式最终测试入口。默认val，test有冻结门禁；无自动安装/升级/CPU回退。
- 实际CPU随机检查点/合成数据集成与失败路径通过；Windows/CUDA/真实权重与val180实测待本地运行。旧B5 CUDA问题不宣称修复。

# v0.2.4 (2026-10-11)

- 用户指定B2/B6各max200/patience50，scratch长预算比较。
- 独立进程、实时控制台与日志、失败留档、保存实际轮数/早停源码。
- 核心model/engine不改，不宣称CUDA已修复。

# v0.2.3 (2026-10-11)

- 复核B2/B6完成、B5中断、B1待跑；无正收益结论。
- 增加只读检查点和32图独立CPU/同步CUDA诊断；不是修复或精确恢复。

# v0.2.2 (2026-10-11)

- 核验实际Windows CUDA流程成功。
- 增加全量train/val四条件40轮收敛诊断入口；模型/engine未修改。
- 已有GPU环境直接复用，失败与成功都尝试打包。

# v0.2.1 (2026-10-11)

- 核验真实CPU四条件短先导；暂无检测收益证据。
- 模型与训练engine源码不变。
- pack携带subset元信息但排除输入图像标签。
- 独立D盘GPU克隆/检查入口，失败留档。

# v0.2.0（2026-10-10）

- 保持B6主要结构与目标；将“细节能量”准确表述为平均绝对幅值。
- 新增B7、E_UNIFORM、E_SHIFT、E_SIGNED、E_FFT及对应CLI/扩展先导配置。
- 门控区域均值、接近全通过比例、幅值保留比进入辅助日志。
- 扰动使用独立步次种子，减少模型初始化消耗随机数造成的扰动不一致。
- 新增Haar投影等价、符号歧义、全通过倾向及控制目标数值检查。
- D盘环境显式prefix创建脚本；旧工程/环境不覆盖。
- 新增近期方法差异与创新边界核验；不是已确认的新方法。
- 全部v0.1源码、旧运行、修复前记录保留在v0.1目录；新运行仅在v0.2目录。
