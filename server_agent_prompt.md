# Server Agent 首轮执行 Prompt：CSIG2026 金睛杯赛道一

你现在位于一个用于参加 **CSIG2026“金睛杯”光学时敏弱小目标探测挑战赛——赛道一：红外视频卫星空中动目标检测** 的服务器仓库根目录。请在当前仓库内完成首轮调研、硬件与数据盘点、可运行工程骨架和提交链路离线验证。

本轮目标是得到一个 **CPU 可运行、无需模型权重、无需真实数据也能端到端自测** 的工程基础，并为下一轮模型开发留下清晰接口。不要追求模型精度。

请自主推进并完成所有不受外部认证阻塞的工作；不要只给建议或计划。若官方数据下载受百度网盘登录、验证码、限速或权限阻塞，记录阻塞和官方链接后继续完成其余任务，禁止改用非官方镜像。

## 0. 最高优先级边界

本轮严格禁止：

- 训练或微调任何模型；
- 提供、调用或暗中触发任何训练入口；
- 对完整训练集、验证集或测试集运行全量推理；
- 启动长时间 benchmark、超参搜索或 GPU 压力任务；
- 向 Codabench 上传或提交任何结果；
- 构建、拉取或运行评分 Docker 容器；
- 删除远程资源、已有数据、已有结果或用户文件；
- 生成 checkpoint、训练日志或实验跟踪记录；
- 从非官方镜像下载比赛数据；
- 将没有明确许可证的 DeepPro 源码直接复制进本项目模块。

`codalab/codalab-legacy:py312` 仅作为已知评分环境信息记录。此次只生成和校验文本结果 zip，不构建、不拉取、不运行该镜像。

若任何后续要求与本节冲突，以本节为准。

## 1. 事实来源与优先级

先读取并记录访问日期。研究时使用以下资料：

1. 当前 V3 官方资料，优先级最高：
   - 官网主页：<https://jinsight-cup.github.io/JinSight-ChallengeV3/>
   - 赛事说明：<https://jinsight-cup.github.io/JinSight-ChallengeV3/intro/index.html>
   - 赛事日程：<https://jinsight-cup.github.io/JinSight-ChallengeV3/schedule/index.html>
   - 仓库根目录 `prompt.md`
   - 仓库根目录 `参赛须知与提交结果说明(3).pdf`
   - 仓库根目录 `Codabench 比赛操作流程(3).pdf`
2. 官方工具包 DeepPro：
   - 仓库：<https://github.com/TinaLRJ/DeepPro>
   - 本轮固定提交：`8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28`
   - 只能针对该提交调研和适配，不要悄悄跟随 `main`。
3. 数据集论文：
   - 《红外视频卫星空中动目标检测数据集及其评估》
   - <https://www.cjig.cn/rc-pub/front/front-article/download/147851829/lowqualitypdf/%E7%BA%A2%E5%A4%96%E8%A7%86%E9%A2%91%E5%8D%AB%E6%98%9F%E7%A9%BA%E4%B8%AD%E5%8A%A8%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B%E6%95%B0%E6%8D%AE%E9%9B%86%E5%8F%8A%E5%85%B6%E8%AF%84%E4%BC%B0.pdf>

冲突处理规则：

- 当前 V3 官网和仓库内两份当前比赛 PDF 优先于旧挑战赛材料、旧 README、论文中的历史比赛描述；
- 不要把不同版本的数据规模拼接成一个数字；
- 当前 V3 官网称赛道一共有 **1400 段序列，训练/验证/测试为 1000/200/200**，并列出 IRAir、IRSatVideo-LEO、IRSatVideo-GEO、实测卫星热红外视频四类数据；
- 固定提交中的旧 DeepPro README 称其 SatVideoIRSDT 划分为 **1001/202/200，共 1403 段**。必须把这项差异以及数据组成/命名差异写入调研文档，标注其来源版本，不得用旧数字覆盖 V3 当前规则；
- 对无法从官方资料确认的细节明确标为“待官方确认”，不要自行猜测；
- 所有关键事实附来源链接或本地文件名；历史成绩与获胜方案必须区分“论文报告”“旧届比赛”和“当前 V3 规则”。

特别核查坐标风险：固定提交中的
`tools_forSatVideoIRSTD/seg2centroid_txt.py` 虽然先计算 OpenCV 的 `(cx, cy)`，但写出时使用了 `(row, col)`，也就是数值顺序疑似为 `(y, x)`；而当前提交 PDF 的文字描述是“横坐标、纵坐标”。不要擅自断言线上评分器究竟采用哪一种。工程内部固定使用图像坐标 `(x=列, y=行)`，默认提交 `xy`，同时提供兼容开关并在文档中显著提示此风险。

## 2. 第一动作：服务器与仓库盘点

在下载或安装大体积内容之前，先做只读/低风险检查并记录：

- 操作系统、架构、Python 与 pip 版本；
- CPU 型号、逻辑/物理核心数；
- 总内存和可用内存；
- NVIDIA GPU 型号、数量、显存、驱动版本；
- `nvidia-smi`、CUDA runtime/toolkit 和 PyTorch CUDA 状态；缺失时如实记录，不要为了盘点强装 GPU 栈；
- 仓库所在文件系统、总空间、可用空间、磁盘类型（能可靠判断时才写 NVMe/SSD/HDD）；
- 到赛事官网、GitHub 和官方数据链接域名的基本连通性；不要输出代理口令、cookie、token 或其他秘密；
- 当前仓库文件和 Git 状态，保留所有已有用户文件和改动。

生成：

- `docs/machine_inventory.json`：机器可读的实际盘点结果，至少包含
  `collected_at`、`os`、`python`、`cpu`、`memory`、`gpus`、`nvidia_driver`、
  `cuda`、`torch`、`storage`、`network_checks`、`data_inventory` 和
  `measurement_notes`；
- `docs/hardware_estimate.md`：硬件结论、测量命令、已知限制和后续资源建议。

JSON 中未知值使用 `null` 并解释原因，不要伪造值。

### 下载硬闸门

只有在仓库目标磁盘 **实际可用空间不少于 250 GB** 时，才允许按需下载官方训练集、验证集或 DeepPro。可用空间低于 250 GB 时：

- 不得开始数据下载；
- 在报告中记录实际可用空间和“因空间闸门跳过”；
- 继续完成全部代码、文档和合成测试。

即使空间足够，也不要无目的地下载全部数据。优先完成工程骨架；只在能改善格式核验时下载官方资料或少量官方数据。下载前记录 URL、目标路径和预计大小；下载后记录实际压缩大小、解压大小、文件数和校验信息（若官方提供）。

官方数据入口以 V3 官网当时列出的“赛道一训练集和验证集”链接为准。若百度网盘认证阻塞，记录后继续；禁止使用第三方转存或非官方镜像。

如需获取 DeepPro：

- 必须检出并验证提交 `8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28`；
- 放在与本项目源码分离且被 Git 忽略的位置，例如 `.external/DeepPro/`；
- 记录远程 URL、提交 hash、获取时间和许可证检查结果；
- 保持上游目录只读，不修改它；
- 若该提交没有明确许可证，允许阅读、运行适配和通过路径动态加载，但禁止把其源码复制进 `src/jinsight_track1` 或提交到本项目；
- 本项目只实现薄适配层，并要求用户显式传入 DeepPro 源码根目录和权重路径；
- 不运行 DeepPro 的 `train.py`。

## 3. 竞赛调研交付物

生成 `docs/competition_research.md`，内容至少包括：

1. 任务定义：质心检测为主任务，跟踪为附加任务；
2. 当前 V3 完整赛程：
   - 报名和截止时间；
   - 训练/验证阶段；
   - 最终评测阶段；
   - 代码、模型、算法报告审查阶段；
   - 成绩复核/公布阶段；
3. 当前 V3 赛道一数据规模 1400 和 1000/200/200 划分；
4. 四类数据的来源、目标特点和官方分辨率描述：
   - IRAir；
   - IRSatVideo-LEO；
   - IRSatVideo-GEO；
   - 实测卫星热红外视频；
5. 训练/验证标签发布情况和测试标签不公开规则；
6. Recall、Precision、F1、轨迹完整度、轨迹准确度及最终计分方式；
7. 当前 PDF 规定的提交格式、检测模式 ID 为 0、跟踪模式 ID 区分目标、每序列一个同名 txt、zip 根目录不得有子目录；
8. 两个评测阶段的提交次数限制，以当前 PDF 为准；
9. DeepPro/DeepPro-Plus 的方法概要、输入时序窗口、固定提交中的历史基线结果和运行依赖；
10. 数据集论文中的历史基线、参赛/获奖方案总结。只总结有证据的方案，不杜撰方法细节；
11. 一张“版本差异与裁决”表，至少对比当前 V3、固定 DeepPro README、数据集论文在序列数、划分、数据组成/命名、比赛版本和用途上的差异；
12. 坐标顺序风险专节：
    - 内部 `(x=列, y=行)`；
    - 当前 PDF 的“横坐标、纵坐标”文字；
    - 旧 `seg2centroid_txt.py` 实际写出 `row,col` 的代码证据；
    - 默认 `xy` 和 `--coordinate-order xy|yx` 的使用方法；
    - 上线提交前必须用官方小样或评分反馈确认，不允许仅凭旧脚本推断；
13. 仍待官方确认的问题清单。

历史数值应原样标明单位/百分比口径并附来源，不把代理指标等同于官方评分。

## 4. 硬件估算要求

在 `docs/hardware_estimate.md` 中明确区分“规划建议”“理论估算”和“本机实测”：

- 本轮无训练最低建议：8 核 CPU、32 GB RAM、250 GB 可用 NVMe，GPU 可选；
- 推荐后续开发：16 核 CPU、64 GB RAM、1×24 GB NVIDIA GPU、500 GB NVMe；
- 2×24 GB GPU 只作为并行实验扩展，不是基线必需配置；
- 当前没有实际数据时，可以基于官方序列数与分辨率给范围估算，但必须写出假设、公式和不确定性；
- 一旦下载数据，必须按实际文件数、实际分辨率分布、压缩大小、解压大小重新计算；
- 不得把理论估算写成实测值；
- 给出训练、缓存、预测结果和临时 zip 的空间余量建议，但本轮不训练。

## 5. Python 工程骨架

建立标准 `src` 布局，至少包含：

```text
pyproject.toml
README.md
.gitignore
src/jinsight_track1/
  __init__.py
  types.py
  data.py
  detector.py
  deeppro_adapter.py
  windowing.py
  postprocess.py
  tracking.py
  evaluation.py
  submission.py
  hardware.py
  cli.py
tests/
```

可按职责增加小模块，但不要做成单个巨型脚本。基础安装必须在 CPU、无 PyTorch、无 DeepPro 的环境中可运行。建议基础依赖保持为 NumPy、SciPy、Pillow/imageio 等必要小集合；PyTorch、OpenCV 和 DeepPro 集成放在可选依赖中。不要因为一个可选包缺失而导致 `import jinsight_track1` 失败。

在 `.gitignore` 中忽略至少：

- 官方数据和本地数据目录；
- `.external/` 或其他上游仓库目录；
- 模型权重与 checkpoint；
- 缓存、虚拟环境和构建产物；
- 推理结果、提交 zip、临时文件和实验日志。

不要忽略源码、测试、文档或必要的小型合成 fixture。

### 5.1 固定公共契约

下列名称和语义必须稳定，并从顶层包导出：

```python
@dataclass(frozen=True, slots=True)
class Detection:
    x: float
    y: float
    score: float


@dataclass(frozen=True, slots=True)
class TrackPoint:
    frame_id: int
    track_id: int
    x: float
    y: float


@dataclass(slots=True)
class SequencePrediction:
    sequence_name: str
    frames: dict[int, list[TrackPoint]]
```

可以增加校验方法或便利方法，但不要改变字段含义：

- `frame_id` 为提交中使用的一基帧号；
- 内部坐标始终是 `(x=列, y=行)`；
- `track_id=0` 表示仅检测模式；
- 跟踪模式的序列内 ID 从 1 开始。

定义运行时可检查或可静态检查的 `Detector` Protocol。其核心职责是接收单个时间窗口的图像数组并返回与每帧对齐的分割/置信度图；清楚记录输入输出 shape、dtype、数值范围和边界窗口行为。序列级窗口拼接不应耦合到具体模型。

### 5.2 数据发现与校验

实现：

- 递归/按约定发现序列，不把标签目录误当输入序列；
- 按自然顺序读取帧；
- 检查空序列、重复帧号、缺帧、不可读图像、序列内尺寸不一致；
- 支持多种分辨率，不写死 256×256；
- 输出数据摘要：序列数、帧数、扩展名、dtype、分辨率分布、每序列帧数范围、磁盘大小；
- 默认只读，不重命名、不移动、不转换官方数据；
- `inspect-data` 允许限制最大序列数和最大帧数，方便少量真实数据检查。

真实目录命名尚不确定时，发现策略应可配置，并以清楚的错误消息失败。不要为迎合合成测试而假定真实数据只有一种布局。

### 5.3 DeepPro 延迟加载适配器

实现薄适配层：

- 顶层导入时不导入 torch、cv2 或 DeepPro；
- 只有显式选择 DeepPro detector 时才检查可选依赖、上游源码路径、提交信息和权重；
- 缺失时给出可操作错误，不自动开始下载或训练；
- 对输入归一化、维度排列、设备、时序窗口长度、输出 shape 做显式校验；
- 适配器只负责推理，不暴露训练函数；
- 使用用户提供的外部源码路径，不复制上游实现；
- 提供可用 mock/fake detector，使整个测试套件不依赖 DeepPro 或权重。

### 5.4 序列窗口推理

实现通用序列推理接口：

- 支持可配置窗口长度和 overlap；
- 保持输出帧与输入帧一一对应，不重复、不漏帧；
- 明确定义首尾窗口 padding/cropping 与重叠区域融合策略；
- 控制内存，不把整个大序列无条件堆进 RAM/GPU；
- 支持批量大小 1 的 CPU 流程；
- 合成 smoke 使用简单、确定性的 detector，不读取模型权重。

### 5.5 质心后处理

从二值/概率掩码提取 8 连通域质心：

- 阈值可配置；
- 可配置最小连通域面积；
- 质心以浮点 `(x=列, y=行)` 返回；
- score 定义为连通域内概率的可解释聚合值；
- 结果有确定性排序；
- 空掩码返回空列表；
- 无 OpenCV 的基础安装也能运行，可用 SciPy/NumPy 实现；
- 专门测试非对称位置，确保不会把 x/y 写反。

### 5.6 跟踪器

实现一个简单、CPU 可运行的多目标跟踪器：

- 每条轨迹使用二维位置和速度的常速度 Kalman 状态；
- 使用匈牙利算法做预测中心与检测中心的距离匹配；
- 使用运动门控拒绝不合理匹配；
- 默认距离门限为当前图像对角线的 `2%`；
- 默认 `max_age=5`；
- 默认 `min_hits=2`；
- 最多对 `3` 帧短缺失做插值；
- 每个序列重置 ID，ID 从 1 开始；
- 参数可由 CLI 显式覆盖；
- 相同输入产生确定性输出。

测试至少覆盖：

- 单目标连续运动；
- 短暂丢失后保持 ID；
- 丢失超过限制后新建 ID；
- 两目标交叉运动时尽量依靠速度预测保持 ID；
- 多分辨率下 2% 对角线门限的计算。

不要声称这个简单跟踪器能达到比赛附加分；它只是可运行基线。

### 5.7 代理评估

实现仅用于本地回归的代理评估：

- 能读取合成点标注/预测；
- 给定明确可配置的点匹配半径，用匈牙利匹配计算 TP/FP/FN、Recall、Precision、F1；
- 跟踪代理指标如实现，必须标为 proxy；
- 零分母行为有定义且有测试；
- 文档显著说明它不是官方评分器，未知的官方匹配细节不得自行假设。

## 6. 提交文本与 zip

严格实现可写出、可重新解析、可校验的提交格式。每帧一行：

```text
五位帧号 目标数 [目标ID 坐标1 坐标2]*
```

要求：

- 帧号一基、至少五位零填充，例如 `00001`；
- 空帧也要写行，例如 `00001 0`；
- 声明的目标数必须与后续三元组数量一致；
- 检测模式为默认模式，所有 ID 固定为 `0`；
- 跟踪模式必须由用户显式启用，ID 使用跟踪结果；
- 每个序列生成一个与序列名对应的 `.txt`；
- zip 根目录直接包含全部 txt，禁止任何子目录；
- zip 内 txt 数量必须与待提交序列数量一致；
- 文件名、帧顺序和点顺序确定性；
- 拒绝路径穿越、重复序列名、重复 zip entry 和意外额外文件；
- 打包后重新打开 zip，逐个解析并校验，而不是只相信写入成功。

坐标规则：

- 内部固定 `(x, y)`；
- CLI 和序列化器提供 `--coordinate-order xy|yx`；
- 默认 `xy`，即输出 `ID x y`；
- `yx` 仅在序列化边界交换为 `ID y x`；
- 解析器也必须显式接收坐标顺序，解析后恢复内部 `(x, y)`；
- README 和调研文档给出两种模式的同一个非对称示例。

## 7. CLI

提供一个统一入口，例如：

```text
jinsight-track1 inspect-data
jinsight-track1 smoke
jinsight-track1 infer
jinsight-track1 package
jinsight-track1 estimate-hardware
```

也应支持 `python -m jinsight_track1.cli ...`。

必须满足：

- 五个子命令各自的 `--help` 可在基础 CPU 环境执行；
- 不提供 `train` 子命令；
- `inspect-data`：只读盘点数据并可输出 JSON；
- `smoke`：生成临时合成多分辨率序列，完成发现、窗口推理、后处理、可选跟踪、文本生成、zip 打包和回读校验，不读权重；
- `infer`：默认只处理显式给出的输入和输出路径，有序列/帧数量限制选项，支持 fake/simple detector 和可选 DeepPro adapter；
- `package`：只打包并严格验证已有预测 txt；
- `estimate-hardware`：输出/更新硬件与数据规模估算，清楚区分实测与估算；
- `infer` 和 `package` 支持 `--coordinate-order xy|yx`；
- 跟踪模式必须用显式参数开启，默认检测模式；
- 破坏性覆盖默认关闭；若输出已存在，应安全失败或要求明确 `--overwrite`；
- 日志简洁，长循环显示可关闭的进度，不泄露敏感信息。

README 至少给出安装、五个 CLI 示例、坐标约定、检测/跟踪模式、DeepPro 可选配置、数据下载闸门和本轮禁训说明。

## 8. 测试与验证

使用临时目录和程序生成的小图像建立测试，不提交真实比赛数据或权重。

必须覆盖：

1. 多分辨率合成序列的端到端 smoke；
2. 空帧、单目标、多目标；
3. 非对称目标位置的质心 `x/y` 顺序；
4. `xy` 与 `yx` 序列化/反序列化；
5. 五位帧号；
6. 检测模式 ID 全为 0；
7. 跟踪短暂丢失、超龄重建和交叉运动；
8. zip 根目录无子目录；
9. txt 数量与序列数量一致；
10. zip 中所有 txt 可重新解析；
11. 缺失 torch/DeepPro 时基础导入和 CLI help 仍可用；
12. 不一致目标数、非法行、重复序列和路径穿越被拒绝；
13. 窗口首尾与 overlap 不丢帧、不重复帧；
14. 代理指标零分母和简单已知样例。

实际运行并记录：

```bash
python -m pip install -e .
pytest -q
python -m jinsight_track1.cli --help
python -m jinsight_track1.cli inspect-data --help
python -m jinsight_track1.cli smoke --help
python -m jinsight_track1.cli infer --help
python -m jinsight_track1.cli package --help
python -m jinsight_track1.cli estimate-hardware --help
python -m jinsight_track1.cli smoke
```

如果环境使用其他 Python 可执行文件，记录实际命令。不要为了让结果好看而隐藏跳过或失败。

若已有或成功获得少量官方数据，再执行一个有严格序列/帧上限的 **只读** `inspect-data`；没有数据或认证阻塞不算工程失败。不得因此触发全量推理。

验证本轮没有生成 checkpoint、训练日志，也没有启动训练进程。

## 9. 最终报告

生成仓库根目录 `SERVER_AGENT_REPORT.md`，至少包含：

- 执行时间和环境；
- 已完成项与对应文件；
- 未完成项及明确原因；
- 实际 CPU、内存、GPU/CUDA、磁盘和网络盘点摘要；
- 是否满足 250 GB 下载闸门；
- 实际数据状态：未下载/部分下载/已有数据，以及实测序列、帧、文件、分辨率、压缩/解压大小；
- DeepPro 是否获取、固定提交是否验证、许可证检查结论；
- 实际执行的安装、测试、CLI 和少量数据检查命令；
- `pytest` 通过/失败/跳过数量；
- smoke 产生的序列/txt 数量及 zip 结构验证结果；
- 坐标顺序风险和上线前确认动作；
- 官方规则、代理评估、DeepPro 兼容性、数据访问等未解决风险；
- 下一轮建议，重点放在小规模真实数据格式核验、官方坐标确认、DeepPro 权重推理适配和后续训练资源准备；
- 明确声明本轮没有训练、没有全量推理、没有 Codabench 提交、没有运行评分容器。

## 10. 完成定义

只有同时满足以下条件才算完成：

- `docs/competition_research.md`、`docs/hardware_estimate.md`、
  `docs/machine_inventory.json` 和 `SERVER_AGENT_REPORT.md` 已生成；
- 模块化 `src/jinsight_track1` 工程和测试已生成；
- 基础安装不依赖 torch、DeepPro 或 GPU；
- 五个 CLI 均存在且没有训练入口；
- 合成 smoke 端到端通过；
- `pytest -q` 已实际执行，所有失败已修复或在报告中给出无法修复的外部原因；
- 提交 txt 和 zip 已回读严格验证；
- 默认检测 ID 为 0，跟踪须显式启用；
- 内部 `x/y` 约定、`xy|yx` 开关和旧脚本风险均有代码、测试和文档；
- 250 GB 下载闸门被实际执行；
- 未发生训练、全量推理、Codabench 提交或评分容器运行；
- 最终回复简洁列出完成结果、测试结果、关键风险和文件路径。

开始执行。先做硬件/磁盘/仓库盘点，再根据闸门决定是否下载；不要先启动大下载或模型任务。
