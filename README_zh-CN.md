<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="resources/banner-dark.png">
  <img src="resources/banner-light.png" alt="GraspPanda：面向杂乱场景 6-DoF 抓取检测的工具箱" width="820">
</picture>

[English](README.md) | **简体中文**

[![tests](https://github.com/Daeda1used/GraspPanda/actions/workflows/test.yml/badge.svg)](https://github.com/Daeda1used/GraspPanda/actions/workflows/test.yml)
[![docs](https://img.shields.io/badge/docs-mkdocs--material-2da44e)](https://daeda1used.github.io/GraspPanda)
[![PyPI](https://img.shields.io/badge/PyPI-coming%20soon-lightgrey)](#-安装与快速开始)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776ab)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A5%202.1-ee4c2c)](https://pytorch.org)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![status](https://img.shields.io/badge/status-alpha-orange)](#-路线图)

[文档](https://daeda1used.github.io/GraspPanda) ·
[快速开始](#-安装与快速开始) ·
[Model zoo](#-model-zoo) ·
[路线图](#-路线图) ·
[设计文档](docs/design/ARCHITECTURE.md) ·
[参与贡献](CONTRIBUTING.md)

</div>

**GraspPanda 之于杂乱场景 6-DoF 抓取检测，就像 MMDetection 之于目标检测：**
在同一套代码里，流程的每个阶段都能在 YAML 里替换，已发表的方法逐阶段对照参考实现移植并验证，
所有数字都出自同一套固定的评测。

🐼 *Panda 指的是熊猫这种动物，不是那款机械臂。GraspPanda 不绑定任何机器人。*

> [!NOTE]
> **Alpha 阶段。** 现在已经可用：配置系统、两个基准的数据读取、两个评测器（与官方工具逐阶段一致）、
> 拆成可替换组件的六个方法（GSNet、EconomicGrasp、FineGrasp、HGGD、GraspNet-baseline、Scale-Balanced-Grasp，
> 用转换后的权重各自与参考实现逐阶段一致）、十一个可互换的场景编码器，
> 以及 `train` / `test` / `infer` / `demo` / `analyze` / `sweep` 命令。Model zoo 仍然是空的，这是有意为之：
> 方法必须先在本工具箱内训练*并*评测才能进入，第一个从零复现正在训练中。

<div align="center">
<img src="resources/terminal-demo.png" alt="grasppanda print-config 校验配置，发现拼错的键，并拒绝不兼容的组件替换" width="820">

*真实输出。拼错的键和不兼容的替换，在加载任何数据之前就被拦下。*

<img src="resources/demo.webp" alt="GraspPanda 在 GraspNet-1Billion 杂乱测试场景上渲染的抓取预测" width="720">

*GSNet 在 GraspNet-1Billion 测试帧上的预测（碰撞过滤后得分最高的 40 个），由 GraspPanda 的实现加载参考代码训练的权重算出。场景图像来自 GraspNet-1Billion，CC BY-NC-SA 4.0。*
</div>

## 📰 News

- **2026-10** — 重建后的第一个公开版本。六个方法拆成可互换组件，各自与参考实现逐阶段一致；十一个场景编码器；`grasppanda components`、`demo`、`analyze`；GraspClutter6D 端到端可用。v0.56 及以前的版本是更早的、无关的代码库，仍可通过原有 tag 和 `legacy` 分支获取。
- **进行中** — 为每个槽位（seed 提议、朝向、局部分组、head 与损失、后处理、小型构件）加入更多可互换模块，把 HGGD 拆成槽位，以及一条命令拼出组合、一条命令在真实帧上试跑组合。

## 🤔 为什么做 GraspPanda

杂乱场景抓取检测已经有很强的方法和很好的基准，但要公平地比较它们，比想象中难。

- **加和不加碰撞过滤的数字被混在一起比。** GSNet 在 GraspNet-1Billion（RealSense，seen）上不加过滤是 65.70 AP，加了是 67.12。拿其中一个去比另一个，差距就超过了不少论文声称的提升。
- **同一个方法，数字不一样。** HGGD 自己的 README 把 59.36 AP（RealSense，seen）列为论文结果，把 64.45 列为仓库结果；后来两篇复测它的论文给出的是 58.35 和 63.37。
- **每个方法各住各的仓库**，各有各的数据流程、标签文件、编译算子，往往还带一份自己的评测器。

GraspPanda 的回答是：一条流程、一个评测器，外加一条规则——**只宣传验证过的能力**。

## ✨ 特性

| | |
|---|---|
| 🎯 **认证过的复现** ⏳ | 方法必须在本工具箱内训练并评测，AP 与发表数字相差不超过 1，或者把差距解释清楚，才能进入 model zoo。每个数字都标明出处：`reported`、`author-ckpt` 或 `reproduced`。 |
| 🧩 **可替换的组件** | backbone、种子提议、接近方向、局部 grouping、head bundle、loss、后处理都是配置里的槽位。一次消融就是一个基础配置加一行覆盖。 |
| 🛡️ **类型化契约，严格配置** | 未知的键、错误的类型、不兼容的替换都会报错并给出修改建议，而且发生在构建任何东西之前。不会悄悄插值、投影或压缩维度。 |
| 📏 **一套固定的评测** | 对 GraspNet-1Billion 和 GraspClutter6D 官方协议做 clean-room 重写，与官方工具逐阶段一致（1,592 个记录帧，零差异），速度约 40 倍，内存约九分之一。加和不加碰撞过滤的结果总是同时报告。 |
| 🔍 **诚实的输入** | `predict()` 永远拿不到真值。由真值派生的输入（比如多数流程在测试时使用的真值工作区裁剪）必须写在配置里，并标注在结果上。 |
| 📦 **精简的核心** | 只依赖 `torch`、`numpy`、`pyyaml`。稀疏卷积等重型依赖是可选项；算子都有纯 PyTorch 的参考实现。 |
| 🧾 **可复现的运行** | 每次运行都会保存展开后的配置、命令行覆盖、代码、git 状态和环境信息。 |

⏳ 表示仍在进行中的部分（第一个从零复现正在训练）；其余现在就能用。

### 一个配置就是一次完整实验

```yaml
_base_:
  - ../../../configs/_base_/default_runtime.yaml
  - ../../../configs/_base_/schedules/gsnet_10e.yaml

custom_imports: [../toy_components.py]      # 仓库外的代码，不需要 fork

model:
  type: ToyDetector
  backbone: {type: ToyBackbone, out_channels: 64}
  grouper:  {type: ToyGrouper, radius: 0.05, nsample: 16}
  head:     {type: ToyHead, num_angle: 12, num_depth: 4}
```

```bash
# 替换组件：旧类型的参数会被丢弃，而不是悄悄合并进来
grasppanda print-config <config> --opts optimizer.type=SGD --check

# 查看一次运行实际用到的全部设置，包括默认值
grasppanda print-config <config> --defaults
```

### 一条流程，类型化的契约

<div align="center">
<img src="resources/pipeline.png" alt="流程槽位：backbone、proposal、view、grouper 和 head bundle，以及每个槽位输出的数据契约" width="820">
</div>

## 🗺️ 总览

✅ 已可用 · 🚧 进行中 · 📅 计划中

| 模块 | 状态 | 内容 |
|---|---|---|
| **核心** | ✅ | 支持继承和三层严格校验的 YAML 配置 · 自动延迟导入的注册表 · 构建时注入 · 运行快照 |
| **数据契约** | ✅ | `Observation` / `Supervision` · `PointSet` / `PointFeatures` · `SeedSet` · `ApproachSet` · `GroupFeatures` · `GraspGroup` |
| **算子** | ✅ | 最远点采样 · ball / cylinder query · kNN · 插值 · grasp NMS · 体素化 · 空间填充曲线 · view 模板（纯 PyTorch 参考实现，可在 CPU 上测试）· 四个热点算子有可选的 CPU / CUDA 编译内核，在实际运行的设备上与参考实现自检一致（[详情](docs/advanced_guides/operators.md)） |
| **命令行** | ✅ `train`、`test`、`infer`、`demo`、`analyze`、`components`、`sweep`、`prepare`、`print-config`、`env` · 📅 `profile` | |
| **数据集** | ✅ GraspNet-1Billion 和 GraspClutter6D 的读取、split 清单、带内容哈希的标签资产（在记录帧上与参考预处理一致） | |
| **评测** | ✅ GraspNet-1Billion 和 GraspClutter6D 评测器，带 coverage 和出处的结果记录 | |
| **方法** | ✅ GSNet、EconomicGrasp、FineGrasp、GraspNet-baseline、Scale-Balanced-Grasp（共用一个 seed-based 骨架）和 HGGD（自成一族）：用转换后的权重各自与参考实现逐阶段一致 · 🚧 全量测试集评测和从零复现 · 📅 Contact-GraspNet | |
| **场景编码器** | ✅ 稀疏 U-Net（spconv）、SpUNet（MinkUNet-34C 结构）、OA-CNNs、PointNet++（两种形式）、PointNeXt、PTv3、带 Concerto 和 Utonia 权重的 PTv3、LitePT，以及稠密与下采样编码器之间的两个适配器；凡是有官方权重的都能加载，输出与官方代码一致（[详情](docs/model_zoo_backbones.md)）· 编码器与方法的每一种组合，要么能训练，要么在配置阶段被拒绝并给出改法（[互换矩阵](docs/advanced_guides/swapping.md)） | |

## 🏆 Model zoo

目前还没有任何方法通过认证，所以这张表列的是**目标**：原作者在 GraspNet-1Billion
上报告的数字（RealSense，seen / similar / novel 的 AP）。最后一列会在方法通过准入门槛后填上。

| 方法 | 发表 | 碰撞过滤 | 作者报告的 AP（seen / similar / novel） | 本工具箱复现 |
|---|---|:---:|---|:---:|
| GraspNet-baseline | CVPR 2020 | 开 | 47.47 / 42.27 / 16.61（其 README 的 "In repo" 行） | — |
| GSNet | ICCV 2021 | 关 | 65.70 / 53.75 / 23.98 | — |
| GSNet | ICCV 2021 | 开 | 67.12 / 54.81 / 24.31 | — |
| EconomicGrasp | ECCV 2024 | 关 | 68.21 / 61.19 / 25.48 | — |
| HGGD | RA-L 2023 | 见注 | 59.36 / 51.20 / 22.17（其 README 的 "In paper" 行）· 64.45 / 53.59 / 24.59（"In repo" 行，arXiv 版论文同此） | — |
| Scale-Balanced-Grasp | CoRL 2022 | 关 | 58.95 / 52.97 / 22.63 | — |
| Scale-Balanced-Grasp | CoRL 2022 | 开 | 63.83 / 58.46 / 24.63 | — |
| FineGrasp | arXiv 2025 | 关 | 71.67 / 62.83 / 27.40 | — |
| FineGrasp | arXiv 2025 | 开 | 73.71 / 64.56 / 28.14 | — |

注：HGGD 公开了两组数字，且没有说明是否做碰撞过滤；它发布的测试脚本在保存结果前会过滤碰撞。这两组数字之间差在哪些设置上，正是这个工具箱要查清楚的事情之一。

准入规则：配置 + 权重（sha256）+ 训练日志 + 固定评测器产出的结果记录，
并且每个 split 上 `|复现 − 报告| ≤ 1 AP`，否则要写明差距和原因。

## 🚀 安装与快速开始

先按你的 CUDA 版本装好 PyTorch，然后：

```bash
git clone https://github.com/Daeda1used/GraspPanda.git
cd GraspPanda
pip install -e .
```

三条命令看到框架在工作，不需要任何数据集：

```bash
grasppanda env                                                              # 环境报告
grasppanda train projects/example_project/configs/toy_train.yaml            # 在 CPU 上几秒钟训练一个玩具检测器
grasppanda print-config projects/example_project/configs/toy_detector.yaml \
    --opts model.grouper.type=ToyDepthGrouper                               # 被拒绝，并说明原因
```

### 五分钟得到第一批抓取

第一张图不需要任何数据集。Model zoo 目前还没有权重，原作者的权重各有其许可证，因此本工具不会替你下载任何东西。

**1. 检查安装**（无需下载，CPU，几秒钟）：现场训练一个玩具检测器。

```bash
grasppanda train projects/example_project/configs/toy_train.yaml --opts runtime.run_id=toy
grasppanda demo  projects/example_project/configs/toy_train.yaml \
    --weights work_dirs/toy_train/toy/checkpoints/best.pth --example
```

`--example` 是随软件包提供的合成桌面场景。玩具模型的抓取没有实际意义，这条命令只说明整个流程可以跑通。

**2. 真实方法。** 自行下载原作者的权重（链接与许可证见 `configs/<method>/README.md`），转换一次，然后运行：

```bash
python -m tools.convert_checkpoint checkpoint-rs.tar baseline_realsense.pth \
    --source-format graspnet-baseline --camera realsense
grasppanda demo graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e \
    --weights baseline_realsense.pth --example --collision-filter
```

命令会打印得分最高的抓取，并写出 `grasps.npy`（GraspNet 17 列格式，相机坐标系，单位米）、`grasps.png` 和 `grasps.gif`。

**3. 你自己的相机。** 点云方法只需要深度图和相机内参；HGGD 还需要与深度图配准的彩色图：

```bash
grasppanda demo <config> --weights <file> --depth depth.png --intrinsics fx,fy,cx,cy \
    --workspace -0.25 -0.2 0.35 0.25 0.2 0.8
```

这些模型在 GraspNet-1Billion 上训练（RealSense D435 与 Azure Kinect，桌面场景），在其他相机上的效果尚未测量。同一套流程在 Python 里是 `GraspInferencer`；`grasppanda analyze` 说明一次评测的 AP 来自哪里：见[推理与可视化](docs/user_guides/inference.md)。

### 基准数据集

磁盘上有 GraspNet-1Billion 时（`export GRASPPANDA_GRASPNET1B_ROOT=...`，
并安装稀疏卷积后端：`pip install -e ".[spconv-cu118]"`）：

```bash
grasppanda prepare graspnet1b --check                        # 检查数据集目录
grasppanda prepare graspnet1b --labels                       # 内存映射的抓取标签
grasppanda prepare graspnet1b --graspness --camera realsense # 逐点 graspness 标签
grasppanda train gsnet/gsnet_sunet14_g1b-realsense_10e
grasppanda test  gsnet/gsnet_sunet14_g1b-realsense_10e --weights <checkpoint>
```

在 GraspNet-1Billion 上训练的模型，一条命令就能在 GraspClutter6D 上评测（见[跨数据集评测](docs/user_guides/cross_dataset.md)）：

```bash
grasppanda test gsnet/gsnet_sunet14_g1b-realsense-to-gc6d-d435_10e --weights <checkpoint>
```

`grasppanda components` 列出每个槽位可以放什么。换场景编码器只要一行覆盖；不兼容的替换会被拒绝并说明原因：

```bash
grasppanda print-config gsnet/gsnet_sunet14_g1b-realsense_10e --check \
    --opts "model.backbone={type: PTv3, in_channels: 3, grid_size: 0.005}"
```

从配置构建一个检测器并运行：

```python
import torch
from grasppanda.config import Config
from grasppanda.registry import DETECTORS
from grasppanda.structures import Observation, PointSet, SampleKey

config = Config.load("projects/example_project/configs/toy_detector.yaml")
model = DETECTORS.build(config["model"]).eval()

observation = Observation(
    keys=[SampleKey("demo", 0, "none", 0)],
    points=PointSet.from_dense(torch.rand(1, 2048, 3)),
)
grasps = model.predict(observation)[0]     # GraspGroup：translation、rotation、width、depth、score
array = grasps.to_graspnet_array()         # (64, 17)，GraspNet 的交换格式
```

示例检测器只是一个接线模板：它没有训练过，输出的抓取是随意的。
`pip install grasppanda` 会随第一个正式版本提供。

## 🧭 路线图

- [x] **M0 — 核心。** 配置系统、注册表、数据契约、参考算子、命令行、CI、文档骨架、品牌素材。
- [ ] **M1 — GraspNet-1Billion、评测器、GSNet。** ✅ 评测器与官方 API 逐值一致。✅ GSNet 用转换后的权重与参考实现逐阶段一致。🚧 从零训练。
- [ ] **M2 — EconomicGrasp 和 HGGD。** 一个同族的第二方法，和一个刻意选来检验抽象的异类。✅ 两者用作者权重都与参考实现逐阶段一致。🚧 全量评测和训练。
- [ ] **M3 — Scale-Balanced-Grasp、GraspNet-baseline、FineGrasp。** 第一张受控对比表：相同 backbone、相同预算、相同输入、相同评测器。✅ 三者用作者权重都与参考实现逐阶段一致。📅 对比表。
- [ ] **M4 — GraspClutter6D 和 Contact-GraspNet。** ✅ 数据读取、评测器（在 254 个记录帧上与官方工具一致）、标签资产、五个方法的迁移配置（在 312 帧子集上检查过）。📅 全量迁移结果表、在 GraspClutter6D 上训练、Contact-GraspNet。
- [ ] **M5 — 发布。** PyPI、model zoo、文档站。

## 📚 教程

- [配置指南](docs/user_guides/config.md)：继承、合并规则、命令行覆盖、校验。
- [新增组件](docs/advanced_guides/add_component.md)：不 fork 就能注册 backbone 或 head bundle。
- [数据契约](docs/advanced_guides/data_flow.md)：每个阶段接收什么、返回什么。
- [架构设计](docs/design/ARCHITECTURE.md)：完整设计，以及每个决定背后的依据。

## 📝 引用

如果 GraspPanda 对你的研究有帮助，请引用：

```bibtex
@software{grasppanda,
  title  = {GraspPanda: A Toolbox for 6-DoF Grasp Detection in Clutter},
  author = {{GraspPanda Contributors}},
  year   = {2026},
  url    = {https://github.com/Daeda1used/GraspPanda}
}
```

也请引用你用到的方法和数据集；每个方法的页面都列出了对应的文献。

## 🙏 致谢

GraspPanda 借鉴了 [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)、
[MMEngine](https://github.com/open-mmlab/mmengine)、
[Detectron2](https://github.com/facebookresearch/detectron2) 和
[Pointcept](https://github.com/Pointcept/Pointcept) 的设计，建立在
[GraspNet-1Billion](https://graspnet.net) 与
[GraspClutter6D](https://huggingface.co/datasets/GraspClutter6D/GraspClutter6D) 两个基准，
以及所复现方法的作者们的工作之上。方法实现参照论文和对参考代码行为的研究独立编写，
没有拷贝任何第三方源文件。

## 📄 License

GraspPanda 采用 [Apache License 2.0](LICENSE)。数据集、上游仓库和已发布的权重
各自保留原有许可，并按方法逐一记录。
