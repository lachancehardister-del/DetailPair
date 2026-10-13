# DetailPair 工作日志

## 2026-10-13（Asia/Shanghai）：云环境导入与首次同步

- 用户授权本次上传，并要求今后自动同步项目代码、数据、权重和实验结果、记录日志。授权写入 AGENTS.md。
- 导入 v0.2.4 完整源码与项目资料；原ZIP SHA256 为 a3b083cec81d5c3562caf938e9dc2a47d60a674920cad0138da26d50e81c9361；57项源码清单哈希一致。
- 上传独立状态与v0.2.5任务文档、云环境安装/启动说明、解析依赖版本和审阅报告。未实现v0.2.5。
- 云端CPU依赖安装和重复安装、pip check、两组既有数值测试、13条件smoke、B6合成一轮训练/保存重载/验证/预测、结果打包均通过。具体日志在 onboarding/logs/。
- 本次数据与权重来自 runs/synthetic_B6_20261012_170119_483383；仅为合成流程验证，非用户本地NEU-DET、真实实验权重或200轮结果。best.pt、last.pt、epoch0.pt以及结构化结果均保留上传。
- 用户本地真实数据、权重、历史结果尚未提供。GPU/Windows、真实AP、200轮最终比较、真实图端到端速度未验证。现有40轮报告B6低于B2；未收到原结果包独立复核。
- 忽略Python/数据缓存、虚拟环境、local_settings和.env凭据文件；保留研究记录、失败记录、结果包和训练权重。
- 同步目标：github.com/lachancehardister-del/DetailPair 的 main；推送完成后核验远端提交。

### 首次同步结果

- 主体提交：`f98dc887729d25725d80ca8efcc2c52dfb29cc6e`。推送到远端 main 成功，git ls-remote 已核验远端SHA一致。
- 共新增154个文件，含源码、文档、合成数据、三个合成检查点、结果与原始日志；未上传本地真实实验材料（尚未收到）。
- 本记录通过独立日志提交同步；推送后再次核验远端HEAD。

## 2026-10-13（Asia/Shanghai）：Work方案的v0.2.5工程实现

- 用户授权按CODEX_NEXT_EVAL_20261013.md实现并交付完整代码包。收到Work审阅ZIP，CRC/路径检查通过，原样保留及安全解包；未收到它引用的133项原始训练结果ZIP、原始数据或真实best.pt。
- Work审阅包SHA256：22c63c6fa69eecbef82a9174fc41319f63eb07fb0b5ba3c497610a5e9e6c95fb
- 新增独立v0.2.5：只读checkpoint/seed/run/训练目标与selection/源哈希核验；原生终端子进程；val官方AP低阈值原始预测与固定工作点逐类TP/FP/FN/P/R；已解码主机图到主机结果的batch1 FP32逐图计时；GPU竞争阻止；四档共同派生扰动和全类全档差值；失败保留与结果pack；Windows完整入口。
- model.py、engine.py、data.py字节不变，原run.py零输入benchmark函数AST不变。不重新训练、resume、换模型/损失/划分；默认val/test显式冻结。原v0.2.4及旧失败结果保留。
- 实际检查：5项unittest、随机B2/B6检查点×5场景CPU合成集成、24条计时/40个AP图片记录、错误变体/seed/run/缺权重/源标签hash、共享扰动复现、缺配置退出1及失败ZIP；原数值不变量/近邻控制/pip check均通过。两次开发集成与预期失败记录保留。
- Windows/CUDA/真实权重身份、val180/429GT及1080条GPU计时、真实扰动精度/FLOPs尚未执行。单种子长预算结论引用Work，不作本轮独立实证；B6整体提升不成立。
- 本轮已明确不上传真实训练权重/原图/整套派生图片。仓库同步源码/输入审阅材料/工程检查报告与结构化结果/发布包；忽略运行时派生全图与缓存。
- 发布包：releases/DetailPair_v0.2.5_Experiment_Package.zip；SHA256：5bf6ae2dcf8860c789da47a7e91255d6e29420272736789f536367153084fa49
- 下一步用户Windows复用已有环境与B2/B6 best.pt，执行windows/15_evaluate_existing.cmd，回传完整DetailPair_eval_results ZIP给Work。是否多种子/消融/停止组合由Work决定。
- 主体提交推送后通过git ls-remote核验远端main，并另记同步结果。

### v0.2.5同步结果

- 主体提交 `d00f0ab6c84041accd5d835406f66eab8d8f29b1` 已推送main，远端SHA独立核验一致。完整发布ZIP、Work输入、源码与实际工程验证日志同步成功。
- 本次同步无真实权重/原图/派生全图；本地真实GPU评价留待用户执行。该同步记录通过后续日志提交保存并再次核验远端HEAD。
