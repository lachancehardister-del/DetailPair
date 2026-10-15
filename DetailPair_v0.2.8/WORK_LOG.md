# v0.2.8 工作日志

2026-10-15 UTC。依据用户持续授权：收到Work方案/结果即检查、实现、验证、创建独立完整版本包并自动提交推送。根目录仅介绍和隐藏Git配置，旧版本保留。用户现有v0.2.4/onboarding九项未暂存删除未修改、恢复或提交。

输入原件及解包位于work_inputs/20261015；SHA256/大小见verification/input_provenance.json。新任务要求有限复用原40轮B5/B6，不授权训练。方案引用的MECHANISM_RESULT_REVIEW_20261015.md不在附件；按明确任务边界和原始机制记录开展，未冒充已读。科研工具箱未提供。云端setup技能用于核验复用已配置环境，不更改依赖，不新增环境草稿或服务。

从v0.2.7复制完整工程到新v0.2.8，排除旧运行/导出/临时设置/旧输入。新增独立matched_checkpoints库存模块、matched_checkpoint_eval评价入口/配置/协议及Windows24/25/26入口；run.py pack接入新打包器。原model/engine/data/evaluation逐字节保留。实际安装版save_model+save_metrics源码证明checkpoint epoch零起算，CSV为epoch+1。仅明确本地可信文件，元数据/有限性/原run/variant/训练预算和策略/选样核验，取最新共同完成轮<=38，不凭文件名或AP。无配对打包并退出2，无训练恢复。两条件一致EMA/model规则，独立B5训练权重，禁止同B6推理改系数冒充。

配对后仅clean和Gaussian.03共享val，GPU0 FP32实际320/batch1，官方AP+固定P/R/FPPI+逐图匹配/输入身份、统一B5−B6；源/权重前后哈希，失败留档/自动pack。默认不导出原图/标签文件/派生全图/权重/大张量。云端仅随机检查点合成工程集成，不读取用户D盘或声称真实权重存在。

上传92项机制包CRC通过；由B2/B6各32批原记录重算全部summary一致，64唯一train图、配对输入/扰动相同、原total差0及记录的状态不变。B6 gates D/A负cos21/32，A/D平均2.030257%；不证明删A提升AP。69项上传源码清单匹配，23项research/tools未打包，已存在项无差异。旧38轮CSV B5 .26453/B6 .26884仅转述任务文档。

检查：11项单元+完整四组官方AP和固定评价CLI CPU合成集成通过，真实缺配置exit1/默认缺配对exit2及打包排除范围通过；pip check和AST解析通过。初次命令目录错误、初次测试断言错误均保留日志，修正后通过。具体run/证据见CHECKS.md、verification/matched_cpu_final.log和当前验证清单。Windows/CUDA/真实旧检查点和NEU四组评价待本地。无科研性能提升主张。

发布ZIP包含完整独立源码、依赖/配置/Windows入口、协议/状态/CHANGELOG/CHECKS/实际CPU验证；原上传材料另在仓库本版本work_inputs保留。发布ZIP不含runs/exports/work_inputs/原图/权重。

Git同步结果在验证远端后追加。

发布包逐文件SHA256/CRC及解压入口检查通过；包SHA256：0b3c43642560ae15bbba02034f5a29d7f1659933f429e898442ffb9d7ead34d4。包体积：257238 bytes。

提交前补充实际tensor设备/FP32记录与守卫，以及官方validator有效阈值、rect/half/batch/imgsz和增强/类别NMS配置核验；最终复检见verification/matched_cpu_release.log。

最终设备/官方设置增强后，11项+四组CPU合成CLI复检通过；最终发布包SHA256：20e627b17a19625cc2578714aa6972f8654fc45bddf08afa75a35a4eb5f3210d，259514 bytes。先前包校验值仅是构建中间件，交付以此值及releases/*.sha256为准。

## GitHub 同步验证

2026-10-15T06:45:57.533501+00:00：实现与完整代码ZIP已推送origin/main；远端实查commit `1fab332f56d88f640b0c0c8e8047387f0eaa1ea2` 与本地HEAD一致。上传材料/原始结果及工程日志均在v0.2.8目录内，无真实原图/标签文件/权重上传。最后仅追加本同步记录并同步，不改代码包；v0.2.4用户已有九项删除仍未暂存。
