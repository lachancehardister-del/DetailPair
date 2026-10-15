# DetailPair v0.2.7：冻结终点权重的局部机制诊断

Work已决定停止当前完整B6的正式扩展。本版只检查现有幅值目标/门控与检测梯度的局部关系；不训练新模型、不更新参数或BN统计、不跑test、不追加种子/B5/YOLO11s/消融，也不继续测速。

## Windows 本地运行（GPU）

完整解压，使入口位于 `D:\ModelImprovement\DetailPair_v0.2.7\mechanism_diagnose.py`。保留旧代码、权重、数据、失败和结果；复用既有CUDA环境，不安装/升级依赖。

```bat
conda activate D:\ModelImprovement\envs\detailpair_cuda_v021
cd /d D:\ModelImprovement\DetailPair_v0.2.7
windows\22_mechanism_diagnosis.cmd
```

正式入口固定GPU device0、CUDA12.1、FP32、320×320、batch2，无CPU回退。配置为 `configs/mechanism_diagnosis.yaml`，仍绑定v0.2.4的B2/B6 best.pt、run、seed0和两份SHA256。可显式修改迁移后的路径，不能替换权重身份。带空格的配置可作为.cmd第1参数传入。

## 本轮具体做什么

- **只读train**：从规范排序后的1438train路径，用独立 `random.Random(178)` 打乱取前64张；两模型同样顺序、32批。核验选中图/标签与两个旧selection哈希，保存类别框计数；缺类照实报告，不重新抽样。val/test图像和标签不读取、不用于选样。
- **原始损失模式、冻结状态**：读取未融合终点副本；使用原paired loss所需训练损失路径，但所有BN和随机层设eval，只开启layer1参数梯度。strip后的requires_grad=False会显式恢复目标层梯度；完整loss系数从checkpoint train_args恢复，缺失则失败，不猜默认值。
- **同一扰动**：noise_std=.03、brightness±.05，独立torch.Generator种子100003+批次索引。记录噪声/亮度/扰动输入哈希，并核对B2/B6每批完全相同。这是新诊断政策，不是复现历史训练批次。
- **精确分解与验证**：D=(clean_loss.sum()+noisy_loss.sum())/2；C=batch×.05×原region_loss；B6 A=batch×.02×原amplitude_loss。每批与原model.loss在同副本/输入/BN模式/step下比较total和整个目标层梯度；rtol1e-4/atol1e-5，失败保留，不自动放松。还核对前景/背景重组成原C/A及A的梯度。
- **局部梯度**：B6分别记录gates、low/detail投影、整个替换层；B2记录原layer1。输出各损失norm、对D的norm比、cos/内积，包含A前景/背景分量；不跨模型拼接不同维数梯度。zero_norm/disconnected/not_applicable分别标注，未定义cos为null。
- **门控/幅值**：B6三子带分别输出框并集内/外的门控mean/median/分位数/near1/near_floor，raw/gated/clean-reference幅值和按原共享归一化分母计算的误差贡献。框内不是分割真值，框外可含漏标缺陷及有padding时的边界。
- **只做autograd.grad**：没有optimizer、参数更新、训练epoch循环或.pt导出；逐批释放图，不累积.grad。每批及整个条件前后核验参数与所有BN buffers，磁盘权重/源图/标签哈希末尾再核验。

原model.py/engine.py/data.py保持不变。全局确定性设置、关闭TF32以及冻结BN属于本轮诊断模式，记录在execution_policy.json，不代表历史训练设置或B5消融。

## 输出与回传

`runs/mechanism_时间戳` 保存选样/环境/配置/身份、loss模式、每条件32批 `batches.jsonl`、`summary.json`、原损失一致性及状态哈希、两条件输入/噪声对齐、失败记录。汇总保留全部批次，报告有效/失败/未执行、空区域、零/未连通梯度、负cos比例；32个相关批次不是32次独立训练。

某条件失败后不换参数或选样重试，剩余批次标not_run，另一条件仍尝试。状态校验失败或未知时明确报告，不冒充只读检查通过。原始参数恢复修正前的工程失败记录也保留。

结束或失败时自动打印 `exports/DetailPair_mechanism_results_*.zip`。把完整ZIP交给Work。默认不含原图/标签文件、权重或完整大张量；只包含必要身份、哈希、统计、日志、配置与源码。手工重打包：

```bat
windows\23_pack_mechanism.cmd
```

原GPU测速和clean结论来自本次Work审阅：320实际输入B2/B6 AP50:95为35.1279%/34.2651%，延迟6.4685/7.1978ms；旧独立val实际352。**本轮未收到审阅引用的214项原始结果ZIP，因此这些数字是Work的复核结论，不是本轮独立重算。**

云端完成范围见 CHECKS.md。CPU随机权重/合成图检查不能称为NEU/RTX4060诊断。实图梯度即使冲突，也只说明这些终点/样本/冻结BN模式下有局部迹象，不能归因60个额外FP、证明去掉A会提升AP或宣布创新成立。后续是否设计单一监督修改/受控训练，由Work决定，本版不自动进入。
