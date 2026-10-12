# DetailPair v0.2.4：B2/B6，200轮上限＋50轮早停

按新指令收窄为两个条件。每条件最多200轮，patience50；不是250轮。两者从随机初始化开始，不接着旧40轮训练；单种子长预算比较，不保证改进效果。此前CUDA错误未定位，本版不宣称修复。详细冻结协议见research/LONG_COMPARISON_PROTOCOL_v0.2.4.md。

## Windows开始

完整解压，使run.py位于D:\ModelImprovement\DetailPair_v0.2.4；保留所有旧目录、数据、权重和失败记录。在新的Anaconda Prompt执行：

```bat
conda activate D:\ModelImprovement\envs\detailpair_cuda_v021
cd /d D:\ModelImprovement\DetailPair_v0.2.4
windows\14_compare_200_patience50.cmd
```

复用GPU环境，无需安装。不要同时跑其他训练或再启动同一脚本。默认数据D:\datasets\NEU-DET_DetailPair_s0_20261011_104211_157196\data.yaml；路径有变时作为cmd第一参数。全部1438train/180val，seed0，320，batch2，epochs200/patience50。原固定划分/模型/系数/优化器保留；test只参与审计，不做检测评价。

两个条件在独立进程按保存的随机顺序依次训练；控制台实时输出，也写B2/B6_console.log。原数据审计与哈希复查；已人工复核9个近重复候选的探索覆盖显式开启，硬错误仍拒绝。失败条件退出后仍尝试另一个条件的新进程；不自动改参数或续训。独立进程不是CUDA故障修复保证。

## 结果与回传

父目录runs/long_comparison_时间戳保存protocol/order/children、实际epoch数量、安装环境和实际EarlyStopping源码。每条件另外一个runs/pilot_时间戳保存完整audit/selection、results.csv、auxiliary、独立best重载metrics/status/failure及本地weights。早停和条件差异可能让实际轮数、耗时不同，必须报告。

结束或失败时自动pack，控制台打印exports中的ZIP。回传全部结果ZIP，不挑最好条件；权重不入ZIP，仍保留本地。人工终止后用已有入口：

```bat
D:\ModelImprovement\envs\detailpair_v02\python.exe run.py pack
```

本次无需先跑13_checkpoint_diagnose，但该诊断包仍保留；新指令不等于已有CUDA错误已消除。失败则暂停正式结论，上传日志。下一步比较B2/B6完整val与训练趋势，再判断是否值得多种子及正式消融。

当前网页仅验证代码语法、配置与顺序、拒绝改动逻辑、源码一致性及ZIP；Torch不在网页环境，未跑新版Windows、早停或真实200轮训练。模型公式/所有对照和原创新性局限见research/README_v0.2_archive.md及NOVELTY_AUDIT.md。当前核心源码没有改变，不因新版本号声称新创新。
