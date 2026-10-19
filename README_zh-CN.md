<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="resources/banner-dark.png">
  <img src="resources/banner-light.png" alt="GraspPanda：面向杂乱场景 6-DoF 抓取检测的工具箱" width="820">
</picture>

[English](README.md) | **简体中文**

[![tests](https://github.com/Daeda1used/GraspPanda/actions/workflows/test.yml/badge.svg)](https://github.com/Daeda1used/GraspPanda/actions/workflows/test.yml)
[![docs](https://img.shields.io/badge/docs-mkdocs--material-2da44e)](https://daeda1used.github.io/GraspPanda)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776ab)](pyproject.toml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![status](https://img.shields.io/badge/status-alpha-orange)](#-路线图)

[文档](https://daeda1used.github.io/GraspPanda) ·
[快速上手](https://daeda1used.github.io/GraspPanda/get_started/) ·
[组合浏览器](https://daeda1used.github.io/GraspPanda/explorer/) ·
[模型库](#-模型库) ·
[设计](https://daeda1used.github.io/GraspPanda/design/ARCHITECTURE/) ·
[参与贡献](CONTRIBUTING.md)

<a href="https://daeda1used.github.io/GraspPanda/explorer/"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="resources/explorer-dark.webp">
  <img src="resources/explorer-light.webp" alt="组合浏览器：按两个家族排列的六个方法；GraspNet-baseline 换上另一篇论文的 PTv3 场景编码器，可以运行；再换上 EconomicGrasp 的检测头，因设计上不兼容被拒绝，并给出原因；HGGD 换上 DINOv2 图像编码器；Compare 视图先列出后处理在全部六个方法下的结果，再列出场景编码器在五个基于种子点的方法下的结果" width="820">
</picture></a>

</div>

**GraspPanda 是一个开源工具箱，用于杂乱场景中的 6-DoF 平行夹爪抓取检测。** 它从 GraspNet-1Billion 出发，也能读取 GraspClutter6D。六个已发表的方法已经移植进来，并与各自的参考代码逐阶段一致。基于种子点的骨架的各个阶段、HGGD 的图像编码器、后处理和训练方案都是槽位，一行命令就能替换。每个选项都在它适用的每个方法下组合过，被接受的组合都用真实帧试跑过；少数没有记录的选项会说明原因。

<details>
<summary>为什么做 GraspPanda</summary>

<br>

杂乱场景抓取检测已经有很强的方法和很好的基准，但要公平地比较它们，比想象中难。

- **加和不加碰撞过滤的数字被混在一起比。** GSNet 在 GraspNet-1Billion（RealSense，seen）上不加过滤是 65.70 AP，加了是 67.12。拿其中一个去比另一个，差距就超过了不少论文声称的提升。
- **同一个方法，数字不一样。** HGGD 自己的 README 把 59.36 AP（RealSense，seen）列为论文结果，把 64.45 列为代码仓库的结果；后来复测它的两篇论文 [RNGNet](https://arxiv.org/abs/2406.01767) 和 [GraRe](https://arxiv.org/abs/2608.00946) 给出的是 58.35 和 63.37。
- **每个方法各住各的仓库**，各有各的数据流程、标签文件和编译算子，往往还带一份自己的评测器。

GraspPanda 的回答是：一条流程，每个数据集一个锁定版本的评测器，外加一条规则——只宣传验证过的能力。

</details>

## ✨ 特性

| | |
|---|---|
| 🧩 **任何阶段都能换**<br>编码器、种子点提议、视角、分组、检测头、损失、后处理和训练方案都是槽位。 | 🔬 **实测，而非假设**<br>每个选项都标明它在几个方法下能跑、能学，以及为什么不适用于其余方法或没有记录。 |
| 🎯 **忠实移植**<br>两个家族共六个方法，用转换后的权重与各自的参考实现逐阶段一致。 | 📏 **锁定版本的评测器**<br>GraspNet-1Billion 和 GraspClutter6D，与官方工具一致；碰撞过滤开、关都报告。GraspNet-1Billion 的评测器在 1,592 个记录帧上与 `graspnetAPI` 逐值一致（[测试](tests/integration/test_eval_parity.py)）。 |
| 🛡️ **错误的替换尽早拦下**<br>大多数错误的组合在构建任何东西之前就被拒绝，并指出冲突和改法；剩下的由一次短暂的 probe 发现。 | 📦 **核心很小**<br>只依赖 `torch`、`numpy`、`scipy`、`pyyaml`、`pillow` 和 `tqdm`；稀疏卷积和模型库都是可选扩展。 |

## 🚀 快速上手

先按你的 CUDA 版本装好 PyTorch，再装 GraspPanda 和下面示例要用的可选扩展，用一个极简示例检测器检查安装：

```bash
git clone https://github.com/Daeda1used/GraspPanda.git && cd GraspPanda
pip install -e ".[spconv-cu118,timm]"   # spconv 要与你的 CUDA 版本匹配
grasppanda env                                                     # 查看安装了什么
grasppanda train projects/example_project/configs/toy_train.yaml   # 极简示例检测器，CPU 上一分钟内训完
```

一行命令替换一个阶段：给 GraspNet-baseline 换上另一篇论文的场景编码器，给 HGGD 换上预训练的图像编码器。

```bash
grasppanda compose graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e \
    --set backbone=ptv3 -o baseline_ptv3.yaml
grasppanda compose hggd/hggd_r34w8_g1b-realsense_15e \
    --set image_encoder=dinov2-small-reg-depth-stem -o hggd_dinov2.yaml
# 准备好 GraspNet-1Billion 时，每次 probe 都在真实帧上短暂训练；否则只做合成场景检查，结果为 `wired`
grasppanda probe baseline_ptv3.yaml
grasppanda probe hggd_dinov2.yaml
```

`compose` 检查各阶段之间的契约并写出配置，需要时自动插入适配器。PTv3 在全部 5 个基于种子点的方法下都能跑，其中 4 个方法下训练得和原装组件不相上下（FineGrasp 下较差）。DINOv2 编码器在 HGGD 下能跑，训练得和 HGGD 的原装组件不相上下。`probe` 只说明一个组合能训练，不说明它有多好；替换后的模型在你训练之前没有权重。

**下一步：** [运行一个方法](https://daeda1used.github.io/GraspPanda/get_started/run_a_method/)（用已发布的权重，几分钟就能拿到第一批抓取）· [替换一个组件](https://daeda1used.github.io/GraspPanda/get_started/swap/) · [打开组合浏览器](https://daeda1used.github.io/GraspPanda/explorer/)

## 🧩 内容一览

| 方法 | 发表 | 家族 | 输入 |
|---|---|---|---|
| [GraspNet-baseline](https://daeda1used.github.io/GraspPanda/methods/graspnet_baseline/) | CVPR 2020 | 基于种子点的骨架 | 深度 |
| [GSNet](https://daeda1used.github.io/GraspPanda/methods/gsnet/) | ICCV 2021 | 基于种子点的骨架 | 深度 |
| [Scale-Balanced-Grasp](https://daeda1used.github.io/GraspPanda/methods/scale_balanced_grasp/) | CoRL 2022 | 基于种子点的骨架 | 深度 |
| [EconomicGrasp](https://daeda1used.github.io/GraspPanda/methods/economicgrasp/) | ECCV 2024 | 基于种子点的骨架 | 深度 |
| [FineGrasp](https://daeda1used.github.io/GraspPanda/methods/finegrasp/) | arXiv 2025 | 基于种子点的骨架 | 深度 |
| [HGGD](https://daeda1used.github.io/GraspPanda/methods/hggd/) | RA-L 2023 | 自成一族 | RGB-D |

**数据集：** GraspNet-1Billion 和 GraspClutter6D，各有读取器、标签资产和与官方一致的评测器。**槽位**及其适用的方法：

| 槽位 | 适用于 |
|---|---|
| 后处理、训练方案 | 全部 6 个方法 |
| 场景编码器、种子点提议、视角、分配器、分组、检测头、损失 | 基于种子点 · 5 个方法 |
| 图像编码器 | 仅 HGGD |

<details>
<summary>全部槽位，以及一个检测器是怎样组装起来的</summary>

<br>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="resources/pipeline-dark.png">
  <img src="resources/pipeline.png" alt="GraspNet-1Billion 和 GraspClutter6D 为两个检测器家族提供数据。基于种子点的骨架有场景编码器、种子点提议、视角、分配器、分组、检测头和损失这些槽位；HGGD 自成一族，有一个图像编码器槽位，热力图阶段和区域阶段是固定的。后处理和训练方案为所有方法共用。" width="820">
</picture>

| 槽位 | `--set` 键 | 可以放什么 |
|---|---|---|
| [图像编码器](https://daeda1used.github.io/GraspPanda/slots/image_encoders/) | `image_encoder` | HGGD 的原装组件（内置编码器），以及接到 timm、transformers、torchvision 的桥接 |
| [场景编码器](https://daeda1used.github.io/GraspPanda/slots/scene_encoders/) | `backbone` | 基于点、稀疏体素和 Transformer 的编码器，合适的地方带有已发布的权重 |
| [种子点提议](https://daeda1used.github.io/GraspPanda/slots/proposal/) | `proposal` | 打分或不打分的种子点，及其采样器 |
| [视角](https://daeda1used.github.io/GraspPanda/slots/view/) | `view` | 任意大小的视角网格、回归方向、完整旋转 |
| [分配器](https://daeda1used.github.io/GraspPanda/slots/head_loss/) | `assigner` | 标签如何与预测的抓取对应 |
| [分组](https://daeda1used.github.io/GraspPanda/slots/grouper/) | `grouper` | 在种子点之间或在场景中裁剪：形状、编码器、池化 |
| [检测头](https://daeda1used.github.io/GraspPanda/slots/head_loss/) | `head` | 各移植方法的检测头和一个解耦网格检测头 |
| [损失](https://daeda1used.github.io/GraspPanda/slots/losses/) | `<stage>.<term>_loss`，例如 `head.score_loss` | focal、quality、鲁棒和集合损失，按损失项设置 |
| [后处理](https://daeda1used.github.io/GraspPanda/slots/post/) | `post` | 碰撞、桌面和工作空间过滤、NMS、筛选 |
| [训练方案](https://daeda1used.github.io/GraspPanda/slots/recipe/) | `optimizer`、`schedule`、`hooks`、`augment`、`recipe` | 优化器、学习率调度、钩子、数据增强；调整场景编码器的预设只适用于基于种子点的方法 |

每个选项在哪个方法下能跑，以及为什么不适用于其余方法或没有记录：见[组合浏览器](https://daeda1used.github.io/GraspPanda/explorer/)和[证据页面](https://daeda1used.github.io/GraspPanda/evidence/)。

</details>

## 🏆 模型库

目标：原作者在 GraspNet-1Billion 上报告的数字（RealSense，seen / similar / novel 的 AP）；最后一列是在本工具箱内训练出的模型，如果有的话。

| 方法 | 发表 | 碰撞过滤 | 作者报告的 AP（seen / similar / novel） | 本工具箱复现 |
|---|---|:---:|---|:---:|
| GraspNet-baseline | CVPR 2020 | 未说明 | 27.56 / 26.11 / 10.55（论文） | — |
| GraspNet-baseline | CVPR 2020 | 开 | 47.47 / 42.27 / 16.61（代码仓库） | — |
| GSNet | ICCV 2021 | 关 | 65.70 / 53.75 / 23.98 | 68.83 / 60.35 / 25.20 |
| GSNet | ICCV 2021 | 开 | 67.12 / 54.81 / 24.31 | 71.23 / 62.29 / 26.42 |
| Scale-Balanced-Grasp | CoRL 2022 | 关 | 58.95 / 52.97 / 22.63 | — |
| Scale-Balanced-Grasp | CoRL 2022 | 开 | 63.83 / 58.46 / 24.63 | — |
| EconomicGrasp | ECCV 2024 | 关 | 68.21 / 61.19 / 25.48 | — |
| FineGrasp | arXiv 2025 | 关 | 71.67 / 62.83 / 27.40 | — |
| FineGrasp | arXiv 2025 | 开 | 73.71 / 64.56 / 28.14 | — |
| HGGD | RA-L 2023 | 未说明 | 59.36 / 51.20 / 22.17（论文）· 64.45 / 53.59 / 24.59（代码仓库） | — |

方法要在本工具箱内训练、成绩与报告数字相差不超过 1 AP（或者把差距解释清楚），才能进入模型库。目前还没有方法通过，所以 GraspPanda 暂未发布自己训练的权重；各方法用已发布的权重运行（[运行一个方法](https://daeda1used.github.io/GraspPanda/get_started/run_a_method/)）。

<details>
<summary>关于这些数字</summary>

<br>

- **GSNet** 按其配置中的训练方案在本工具箱内从零训练：只训练了一次，一个随机种子。它比论文数字高出的幅度超过了准入规则的允许范围，原因尚不清楚。已知的是：移植版与参考实现逐阶段一致；用参考代码训练的模型（标签集有改动）同样高于论文；作者的原始代码没有公开（[详情](https://daeda1used.github.io/GraspPanda/model_zoo/#trained-and-evaluated-here)）。它的权重暂未发布。
- **其他方法作者发布的权重**在本工具箱里跑完了整个测试集，碰撞过滤开、关各一次。这些 `author-ckpt` 数字列在[模型库页面](https://daeda1used.github.io/GraspPanda/model_zoo/)，每个都有提交在仓库里的结果记录可查。它们不是复现。
- **HGGD** 公开了两组数字，且没有说明是否做碰撞过滤；它发布的测试脚本在保存抓取前会过滤碰撞。
- **准入规则：** 配置、带 sha256 的权重、训练日志和锁定版本的评测器产出的结果记录，每个 split 上相差不超过 1 AP，否则写明差距。

</details>

## 📰 新闻

- **2026-10** — 会训练的替换还和原装组件对照着训练，看它是否[训练得和原装组件一样](https://daeda1used.github.io/GraspPanda/advanced_guides/learns_like_reference/)。
- **2026-10** — 文档按任务重新组织：运行一个方法、替换一个组件、查看证据。

<details>
<summary>更早的新闻</summary>

<br>

- **2026-10** — 拒绝都会说明原因：每次拒绝标明类别，适配器或某个设置能解决的，`compose` 会自动处理。
- **2026-10** — [组合浏览器](https://daeda1used.github.io/GraspPanda/explorer/)：选一个方法、换一个槽位，查看实测记录和拒绝的原因。`grasppanda explore` 可以在本地运行它。
- **2026-10** — 每个槽位都有了可选组件，取自已发表的代码和模型库，每个都在它适用的方法下试跑过。
- **2026-10** — `grasppanda compose` 用方法配置和预设拼出一个组合；`grasppanda probe` 在真实帧上短暂训练它。
- **2026-10** — 重建后的第一个公开版本。v0.56 及更早的版本属于另一套无关的代码，仍保留在各自的标签和 `legacy` 分支上。

</details>

## 🧭 路线图

<details>
<summary>里程碑</summary>

<br>

- [x] **M0 — 核心。** 配置系统、注册表、数据契约、参考算子、命令行、CI、文档、品牌。
- [ ] **M1 — GraspNet-1Billion、它的评测器、第一个方法。** ✅ 评测器与官方一致。✅ GSNet 与参考实现逐阶段一致，已在本工具箱内训练并在整个测试集上评测。🚧 与论文的差距尚未解释；权重暂未发布。
- [ ] **M2 — EconomicGrasp，以及作为第二个家族的 HGGD。** ✅ EconomicGrasp 和 HGGD 用作者的权重与参考实现逐阶段一致。🚧 在本工具箱内训练。
- [ ] **M3 — 其余基于种子点的方法。** ✅ GraspNet-baseline、Scale-Balanced-Grasp 和 FineGrasp 与参考实现一致。📅 受控对比：相同的编码器、训练预算、输入和评测器。
- [ ] **M4 — GraspClutter6D。** ✅ 读取器、与官方一致的评测器、标签资产、迁移配置。📅 在 GraspClutter6D 上训练；Contact-GraspNet。
- [ ] **M5 — 发布。** PyPI、模型库、文档网站。

</details>

## 📝 引用

<details>
<summary>BibTeX</summary>

```bibtex
@software{grasppanda,
  title  = {GraspPanda: A Toolbox for 6-DoF Grasp Detection in Clutter},
  author = {{GraspPanda Contributors}},
  year   = {2026},
  url    = {https://github.com/Daeda1used/GraspPanda}
}
```

也请引用你用到的方法和数据集；每个方法页面都列出了它的参考文献。

</details>

## 🙏 致谢

GraspPanda 借鉴了 [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)、[MMEngine](https://github.com/open-mmlab/mmengine)、[Detectron2](https://github.com/facebookresearch/detectron2) 和 [Pointcept](https://github.com/Pointcept/Pointcept) 的思路，建立在 [GraspNet-1Billion](https://graspnet.net) 和 [GraspClutter6D](https://huggingface.co/datasets/GraspClutter6D/GraspClutter6D) 基准以及所移植方法的作者的工作之上。方法和评测器依据论文和参考代码的行为编写，没有复制那些仓库的任何文件。改编自 MIT、BSD 或 Apache-2.0 项目的代码列在 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 中；[NOTICE](NOTICE) 说明数据集、上游代码和已发布权重各自的许可证。

🐼 *Panda 指的是熊猫这种动物，不是那款机械臂：GraspPanda 不绑定任何机器人。*

## 📄 许可证

[Apache License 2.0](LICENSE)。数据集、上游仓库和已发布的权重保留各自的许可证，按方法分别注明。
