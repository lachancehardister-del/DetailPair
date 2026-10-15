# DetailPair v0.2.8：旧 B5/B6 同轮检查点有限对照

本轮只复用 `pilot__20261011_112235_644745` 的旧40轮实验。先盘点可信本地检查点，用元数据、原CSV和模型身份选择不超过第38轮的**最新共同完成轮**；有配对才评价 clean 与 Gaussian std=.03。没有配对就返回库存与 `blocked_missing_matched_checkpoint`，不训练、不恢复、不微调。完整B6正式扩展仍停止。

## Windows 本地 GPU 运行

完整解压到 `D:\ModelImprovement\DetailPair_v0.2.8`，入口应为该目录下的 `matched_checkpoint_eval.py`。原版本、失败和结果全部保留。既有环境为 `D:\ModelImprovement\envs\detailpair_cuda_v021`；入口直接使用其 python.exe，不升级或重装。PowerShell：

```powershell
cd D:\ModelImprovement\DetailPair_v0.2.8
.\windows\24_matched_checkpoint_eval.cmd
```

程序先列库存，配对成功后实查 Torch2.5.1+cu121、torchvision0.20.1+cu121、Ultralytics8.3.221 和 GPU0；FP32、320×320、batch1，无CPU回退。没有测速流程，不会因 QQ 被公平测速守卫拦截。

默认位置在 `configs/matched_checkpoint_eval.yaml`：

- B5/B6：`D:\ModelImprovement\DetailPair_v0.2.2\runs\pilot__20261011_112235_644745\B5_s0\training\weights` 与对应 B6 目录。
- 各自 `training\results.csv` 和同原run的 `subset_s0\selection.json`。
- 数据：`D:\datasets\NEU-DET_DetailPair_s0_20261011_104211_157196\data.yaml`。

如果旧目录迁移了，只修改配置的 `weight_dirs`、`csv`、`selection` 和必要的数据路径，保留检查点内部的原run身份。`weight_dirs` 可明确增加已有归档的解压后权重目录；仅检查这些目录的直接文件，不扫整盘，不自动解压/下载权重。不要求上传权重。复制配置后可传入路径（支持空格）：

```powershell
.\windows\24_matched_checkpoint_eval.cmd "D:\my configs\matched.yaml"
```

只盘点：`.\windows\25_inventory_only.cmd`，无需GPU，配对成功也不会评价。CPU仅供明确 `--synthetic-check` 的工程检查，不能用于真实结果。

## 结果与状态

每次新建 `runs/matched_时间戳`，输出库存、拒绝原因、安装版保存源码、轮次映射、配对理由、权重SHA256、训练配置/选样哈希、环境、四组原始指标和逐图预测/匹配。每次官方AP与固定P/R均记录实际输入shape/FP32张量哈希，两模型逐项核对。表格差值统一 **B5−B6**。原CSV只作为轮次和历史参考，不能视为本轮实际320输入下的AP。

结束或失败都会打印 `exports/DetailPair_matched_results_*.zip`，把这个完整结果ZIP交给Work。默认排除原图、标签文件、派生全图、权重和完整张量；保留指标、GT框/匹配、哈希与失败。手工重打包：

```powershell
.\windows\26_pack_matched.cmd
```

退出码0：完成评价，或显式仅库存模式完成（以status区分）；2：缺失可确认的同轮配对，已打包库存；1：其他失败，已尽力打包。`best.pt` 如果被剥离为 epoch=-1，不能凭文件名或AP猜轮次。

## 当前证据与边界

已重算上传的64图机制原始记录：B2/B6各32批，汇总一致；B6门控处D与A梯度在21/32批反向，A/D范数比平均2.03%。这不证明删A改善AP。方案引用的 `MECHANISM_RESULT_REVIEW_20261015.md` 未随附件提供，当前审阅是Codex依据原始记录重算，未冒充Work文档。原始文件及审计保留在仓库版本目录的 `work_inputs` 与 `verification`。

旧第38轮CSV：B5 .26453、B6 .26884（方案报告值，未收到该旧原始结果包，不作为新评价）。本轮只支持单种子短预算有限对照；历史初始化张量哈希不可证，不宣称训练轨迹完全相同。关闭训练幅值损失不会减少相同架构的Haar门控推理开销。运行结果只回传Work，不自动追求正结果、补训或调参。

协议见 [MATCHED_CHECKPOINT_PROTOCOL.md](MATCHED_CHECKPOINT_PROTOCOL.md)，实测与待测见 [CHECKS.md](CHECKS.md)。源码保留旧入口供追溯；当前应使用24入口，勿运行旧训练脚本。模型、损失、训练核心与v0.2.7完全相同。
