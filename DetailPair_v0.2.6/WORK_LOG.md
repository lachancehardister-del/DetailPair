# v0.2.6 工作日志

## 2026-10-15（Asia/Shanghai）：Work测速复核任务

- 用户确认当前及以后收到Work方案+结果包后直接按流程实现、同步并交付完整新版，不重复确认；正式本地运行继续使用GPU。所有材料位于本版目录，旧版不覆盖。
- 输入方案CODEX_TIMING_RECHECK_20261015.md；原结果ZIP SHA256=51c5dc11ef2e4c85af7a0bf1d34496fd623bedb8d84f367a4c86a300949b7bfc，187项CRC/路径检查通过；原QQ阻止和成功/漂移记录保留。
- 独立重算原成功1080条计时及AP/TP/FP，与Work说明一致。原B2三轮31.75185/24.54046/8.26163ms，不能认定汇总速度差为架构收益。未收到单独EVAL_RESULT_REVIEW文档，已有方案及原始包足以继续。
- 实现2.0平衡区组ABBA+BAAB、固定共同/区组预热、1440条测量、快照/窗口/跨组/位置检查、缓冲写盘与partial、状态分离且不自动重试；HostDetector原边界不变。
- 实现clean旧JPEG字节一致副本/原selection顺序与新PNG的实际官方loader像素/输入shape/dtype/hash/有效参数/AP/全类/预测匹配记录；有条件最多rect单因素，不预归因；不训练、不扫扰动、不重复固定P/R、不跑test。
- model/engine/data字节一致，1.0源码和配置不放松。新增Windows18完整/19仅测速/20仅诊断/21打包，默认GPU device0/CUDA12.1/RTX4060 Laptop/FP32，校验真实权重SHA，无CPU回退。
- 实际7项工程单元、CPU随机checkpoint/6合成JPEG的八区组48条集成、旧新入口观测/原源权重只读、rect入口、异常partial、缺配置失败及ZIP排除检查通过。源代码解析、解包后清单、help/pack-only验证通过。CPU不支持真实GPU速度结论。
- 真实Windows/GPU、两份真实权重重载、30秒/200次正式预热、1440样本和真实clean差异原因仍pending；不声称已消除漂移或B6整体收益。
- 本轮开始时v0.2.4/onboarding若干旧安装文件/日志已在工作区缺失；未修改或提交这些既存删除。
- 完整发布包SHA256：7683640dae441f6aa2eb0e977fd901e46765b839578367fad583f6d7d446ce3b
- 用户下一步解压到D:\ModelImprovement\DetailPair_v0.2.6，既有CUDA环境下运行windows\18_recheck_all.cmd，把完整DetailPair_recheck_results ZIP交Work。主体同步及远端核验另记。

## 同步结果

- 主体提交 `0fcdca105e36d7126e44022cecc3028d41f41984` 已推送main，git ls-remote独立核验远端一致。代码、完整发布包、原输入/失败/漂移记录和实际工程检查结果同步成功。
- 仅提交本轮v0.2.6和根介绍/隐藏忽略规则；未提交旧onboarding已存在的删除。真实GPU运行仍由用户本地执行。本同步记录另提交并再次核验远端。
