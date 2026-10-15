# DetailPair v0.2 完整模型与创新性诊断实验包

基线YOLO11n，改进浅层Haar细节门控，配对检测＋特征一致性＋细节幅值监督。这是完整可训练原型，不含训练好的真实缺陷权重；创新性未确立。核验结论和近期区别见research/NOVELTY_AUDIT.md。本版本保持B6机制，补足13个对照、门控日志及独立扰动种子，未把代码更新当作新的科研贡献。

## 1. D盘Anaconda环境与首次检查

完整解压到 **D:\ModelImprovement\DetailPair_v0.2**，能看到run.py。旧v0.1目录保留。

打开Anaconda Prompt；Anaconda本身已在D盘。项目代码和Python环境分开，新项目环境显式放在D盘，不依赖conda默认envs_dirs：

```bat
cd /d D:\ModelImprovement\DetailPair_v0.2
windows\00_create_env_d.cmd
conda activate D:\ModelImprovement\envs\detailpair_v02
windows\01_setup_cpu.cmd
windows\02_check.cmd
windows\99_pack.cmd
```

若已有专门用于本项目的兼容环境，可激活它并检查Python3.12与本包依赖，跳过00_create_env_d；具体旧环境名称/路径尚未知，不能假定它已经兼容。首次推荐独立v02环境，避免改动其他项目。00脚本若发现目标python.exe已存在会停在提示，不覆写；激活后的python命令应来自目标环境。

01安装CPU Torch2.5.1/torchvision0.20.1及requirements；安装需联网下载公开软件，不需付费API/外部账号。核心版本固定，完整跨平台传递依赖锁定尚未完成；实际安装环境记录在结果包。训练同步/在线日志关闭，不上传图像或权重。NOTICE说明第三方许可证及科研技能来源。

02执行环境记录、13条件单批前向/反向/EMA模型复制、Haar/梯度/初值检查、等价性及幅值反例检查，以及B6合成矩形图像一轮训练/验证/保存/重载/预测。**合成结果只能检查流程。** CPU是首次检查选择，不是对你的显卡能力的推断。

99打印exports\DetailPair_results_时间戳.zip，回传这个文件用于判断能否进入真实数据先导。任一步失败保留install_logs、verification或runs日志，仍可运行99。数据及.pt权重默认不进结果包。新包已交付后才安排上述本地命令。

## 2. 复用现有YOLO数据

使用自己的现成data.yaml，不要求重下或重传。支持YOLO每行class_id cx cy width height，坐标归一化0–1，图像/标注对应images/labels；支持目录、列表及txt清单。只有VOC XML时本版不能直接读取，转换器仍需单独补充。configs/data.example.yaml仅示例，类别顺序必须与真实标签核对。

```bat
python run.py audit --data "D:\你的实际数据目录\data.yaml"
windows\03_pilot.cmd "D:\你的实际数据目录\data.yaml" cpu
python run.py collect
windows\99_pack.cmd
```

audit检查字段/缺文件/类别/边界/重复框、跨划分字节和解码重复、相同dHash候选。跨划分精确重复阻止训练；dHash告警先人工复核，只有探索先导可显式加--accept-near-duplicates-for-pilot。审计不证明许可、近重复全覆盖或同一工件独立划分。原始来源、许可、分组与划分核验尚缺，不自动沿用其他项目结果当本项目证据。

默认pilot.yaml：B1/B2/B5/B6、每条件3轮、单种子、64训练/32验证、320输入、batch2、从头初始化。不跑测试集；按种子固定同样样本并随机运行顺序，逐条件复查原始哈希；选样复制到私有工作目录，缓存也写副本旁。这个规模只能筛选和检查流程。

通过四条件真实先导后，扩展创新性诊断入口：

```bat
python run.py pilot --data "D:\你的实际数据目录\data.yaml" --device cpu --config configs\novelty_pilot.yaml
python run.py collect
windows\99_pack.cmd
```

扩展为12条件（默认不含单视图B0）。正式候选配置包含13条件、3种子、150轮、全train/val，只是未冻结候选，**现在先运行02_check，勿直接开始正式长训练**。模型/目标系数在model.py固定，pilot记录implemented_objectives.json，改变设计需新版本；配置中的旧consistency_weight/energy_weight不是动态调参入口。

## 3. 模型和消融

| 条件 | 模型/训练 |
|---|---|
| B0 | 原YOLO11n，单视图检测 |
| B1 | 原模型，双视图检测 |
| B2 | B1＋浅层输出一致性 |
| B3 | 无门控Haar投影，双视图检测；算子对照 |
| B4 | Haar门控，双视图检测 |
| B5 | B4＋输出一致性 |
| B6 | B5＋框内/外归一化子带幅值目标；完整候选 |
| B7 | 普通卷积瓶颈，近似层参数匹配＋B2训练；不匹配实际延迟 |
| SPD | space-to-depth＋3×3卷积，双视图；算子适配非原方法完整复现 |
| E_UNIFORM | B6幅值目标改全图；区域及归一化同时变化，需报告尺度 |
| E_SHIFT | B6参考幅值循环错位，检验空间对应是否必要 |
| E_SIGNED | B6幅值项改带符号子带L1 |
| E_FFT | B5＋浅层输出log FFT幅值对齐；PCW启发的目标适配非原方法复现 |

原始/扰动视图共享检测器与几何变换，扰动只Gaussian σ.03和亮度±.05。参考分支只在辅助目标stop-gradient，仍接收检测梯度；无冻结教师/EMA教师。推理只一张图，门控仍在内部。λ=.05、μ=.02，框内/外各自按像素和参考幅度归一化，背景.1。准确术语是平均绝对幅值，旧配置energy键仅保留名称兼容。各辅助项数值单位不同，同一系数不等于相同梯度预算。

无门控Haar线性投影可等价2×2 stride2卷积；幅值不能分辨符号；无扰动时幅值项可能偏向全通过，均已给出解析条件和数值检查。auxiliary.jsonl记录门控均值、>0.99比例和幅值保留比，不能只凭数值趋稳宣称噪声识别成功。

## 4. 单条件、评价及效率

```bat
python run.py train --data "D:\实际数据\data.yaml" --manifest "runs\实际audit目录\audit.json" --variant B6 --epochs 3 --batch 2 --imgsz 320 --seed 0 --device cpu
```

默认仍64/32选样；--train-limit 0 --val-limit 0使用全部对应划分。可选--weights是可信本地.pt，不自动下载预训练权重。正式比较统一初始化与公共层映射，不把不兼容加载当公平迁移。没有自动恢复旧run入口；每次新建run目录并保存失败。

```bat
python run.py evaluate --weights "runs\实际条件\training\weights\best.pt" --data "runs\实际pilot\subset_s0\data.yaml" --device cpu
python run.py robustness --weights "runs\实际条件\training\weights\best.pt" --data "runs\实际pilot\subset_s0\data.yaml" --device cpu
python run.py benchmark --weights "runs\实际条件\training\weights\best.pt" --device cpu
```

评价默认val，测试集只有方案冻结后显式--split test使用。直接给原始yaml评价可能产生Ultralytics缓存，优先使用副本yaml。robustness复制图像，测三档Gaussian及椒盐，保留种子/实际SNR；不等于真实生产噪声。benchmark为FP32未融合网络前向、batch1、20预热100样本、同步GPU，保存原始时间及p50/p95；不含预处理/NMS，不等于端到端延迟。

环境回传确认GPU后才决定CUDA版本/真实先导规模；包内CUDA12.1安装脚本仅候选，未在Windows或GPU执行，不应在现有环境中盲目覆盖CPU安装。ONNX入口需要额外onnx，未执行引擎验证；本轮不安排安装。缺弱/小目标分组AP、固定阈值每图误检、bootstrap、完整WAMF/PCW/SF原方法复现及端到端/边缘部署；这些仍是正式论文证据缺口。

## 5. 保存与实际检查范围

runs保存实际环境、配置、样本映射、原始CSV/辅助JSONL、评价/预测、状态及失败堆栈。collect按单个pilot/条件分组报告均值/样本标准差，不跨划分混合结果；单次std为null。pack包含源码、配置、诊断报告和日志，排除权重/输入副本；原始、失败、旧版本保留。

网页侧Linux CPU已执行范围见verification/VERIFICATION.md；没有Windows、本机GPU、真实缺陷提升或正式三种子结果。本轮新检索给出可核验差异，不能保证创新性或发表。先回传02_check结果，再进入真实先导。
