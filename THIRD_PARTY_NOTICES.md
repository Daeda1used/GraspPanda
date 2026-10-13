# Third-party notices

GraspPanda is licensed under the Apache License, Version 2.0 (see `LICENSE` and `NOTICE`).
The files listed below are adapted from third-party code under permissive licences. Each
file keeps the upstream copyright notice at its top; this page gathers them.

How to add an entry: one section per upstream repository, with the repository URL, the
commit the code was taken from, the licence (SPDX identifier and the copyright line of
the upstream licence file), and the files in this repository that contain adapted code.

Components whose status is `bridge` call a library that the user installs as an optional
extra. They contain none of the library's code and are not listed here; the licence of
each library is its own.

<!-- Sections below are appended by contributions; keep them in the form shown above. -->

## Pointcept (Point Transformer V1 and V2)

- Repository: https://github.com/Pointcept/Pointcept
- Commit: `1342eda`
- Licence: MIT. `Copyright (c) 2023 Pointcept`
- Files: `grasppanda/models/backbones/point_transformer.py` (from
  `pointcept/models/point_transformer/point_transformer_seg.py` and
  `pointcept/models/point_transformer_v2/point_transformer_v2m2_base.py`)

## Point-MAE (point-patch tokeniser and transformer encoder)

- Repository: https://github.com/Pang-Yatian/Point-MAE
- Commit: `7445a68`
- Licence: MIT. `Copyright (c) 2022 PANG-Yatian, YUAN-Li`
- Files: `grasppanda/models/backbones/point_tokens.py` (from `models/Point_MAE.py`)

## PointMamba (token order, order scaling and block layout)

- Repository: https://github.com/LMD0311/PointMamba
- Commit: `80858e0`
- Licence: Apache-2.0 (the licence file carries no copyright line; `models/block_scan.py`
  carries `Copyright (c) 2023, Tri Dao, Albert Gu`)
- Files: `grasppanda/models/backbones/point_tokens.py` (class `PointMamba`, from
  `models/point_mamba_scan.py`; changes: packed batches of clouds, a per-centre read-out
  in place of the pooled classification feature, operators of `grasppanda.ops`)

## Uni3D (layout of the point encoder)

- Repository: https://github.com/baaivision/Uni3D
- Commit: `64e03c3`
- Licence: MIT. `Copyright (c) 2023 BAAI-Vision`
- Files: `grasppanda/models/backbones/point_tokens.py` (class `Uni3D`, from
  `models/point_encoder.py`; the transformer blocks are built by `timm`)

## Volt

- Repository: https://github.com/YilmazKadir/Volt
- Commit: `8dcd608`
- Licence: MIT. `Copyright (c) 2026 Kadir Yilmaz`
- Files: `grasppanda/models/backbones/volt.py` (from `pointcept/models/volt/volt_base.py`
  and `pointcept/models/volt/decoder.py`)

## OctFormer

- Repository: https://github.com/octree-nn/octformer
- Commit: `c2975ab`
- Licence: MIT. `Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>`
- Files: `grasppanda/models/backbones/octformer.py` (from `models/octformer.py` and
  `models/octformerseg.py`)
## facebookresearch/votenet

- Repository: https://github.com/facebookresearch/votenet
- Commit: `2f6d6d36ff98d96901182e935afe48ccee82d566`
- Licence: MIT. `Copyright (c) Facebook, Inc. and its affiliates.`
- Its `pointnet2/pointnet2_modules.py` is "modified based on" https://github.com/erikwijmans/Pointnet2_PyTorch
  (commit `b5ceb6d9ca0467ea34beb81023f96ee82228f626`, Unlicense), whose multi-scale segmentation layout is the
  default of `PointNet2MSGDense`.
- Files: `grasppanda/models/backbones/pointnet2_msg.py`

## linhaojia13/PointMetaBase

- Repository: https://github.com/linhaojia13/PointMetaBase
- Commit: `c364a671ee255453c0a03b2474af555664604db9`
- Licence: MIT. `Copyright (c) Guocheng Qian.`
- Files: `grasppanda/models/backbones/pointnext_family.py` (class `PointMetaBase`)

## guochengqian/openpoints

- Repository: https://github.com/guochengqian/openpoints
- Commit: `bc778ec20c51fa9a72861c0ffe32718815367b6a`
- Licence: MIT. `Copyright (c) Guocheng Qian.`
- Files: `grasppanda/models/backbones/pointnext_family.py` (class `PointVector`, from
  `models/backbone/pointvector.py`, "Official implementation of PointVector" by Xin Deng, WenYu Zhang,
  Qing Ding and XinMing Zhang)
  and `grasppanda/models/backbones/assanet.py` (class `ASSANet`, from `models/layers/local_aggregation.py`
  and `models/backbone/pointnetv2.py`; layout of `cfgs/s3dis/assanet.yaml` in
  https://github.com/guochengqian/PointNeXt, commit `b7ec10fd8b9acdb1c3b18e557f0d3465eb919164`, MIT,
  `Copyright (c) Guocheng Qian.`)

## antao97/dgcnn.pytorch

- Repository: https://github.com/antao97/dgcnn.pytorch
- Commit: `07d534c2702905010ec9991619f552d8cacae45b`
- Licence: MIT. `Copyright (c) 2020 An Tao`
- Files: `grasppanda/models/backbones/dgcnn.py`

## isl-org/Open3D-ML

- Repository: https://github.com/isl-org/Open3D-ML
- Commit: `a3d0f7a51d4eef0b94b9000a7d74af05e2d54c4f`
- Licence: MIT. `Copyright (c) 2020 www.open3d.org`
- Files: `grasppanda/models/backbones/randlanet.py`

## ma-xu/pointMLP-pytorch

- Repository: https://github.com/ma-xu/pointMLP-pytorch
- Commit: `1ab7f04e88d2aeb8fd16921a2b2fc4fba950b6df`
- Licence: Apache-2.0. The licence file is the unmodified licence text and names no copyright holder; the
  code is by Xu Ma and the contributors of that repository.
- Changes (Apache-2.0, section 4b): packed batches of clouds of any size, anchors by ratio, GraspPanda's
  operators in place of `pointnet2_ops`, no object category token, no classifier, no batch norm on the
  per-cloud context vector.
- Files: `grasppanda/models/backbones/pointmlp.py`

## Matrix-ASC/DeLA

- Repository: https://github.com/Matrix-ASC/DeLA
- Commit: `0f427c36c5a991a28397ffdaaf8c205b80d6b30d`
- Licence: MIT. `Copyright (c) 2023 Matrix-ASC`
- Files: `grasppanda/models/backbones/dela.py`
## MinkowskiEngine

- Repository: https://github.com/NVIDIA/MinkowskiEngine
- Commit: `02fc608bea4c0549b0a7b00ca1bf15dee4a0b228`
- Licence: MIT. `Copyright (c) 2020 NVIDIA CORPORATION.` `Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu)`
- Files: `grasppanda/models/backbones/minkunet.py` (the layer plan of `examples/minkunet.py`, the
  blocks of `MinkowskiEngine/modules/resnet_block.py` and the initialisation of `examples/resnet.py`,
  rewritten on `grasppanda.ops.sparse`; no MinkowskiEngine code is called).
## RGB Matters

- Repository: https://github.com/GouMinghao/rgb_matters
- Commit: 2e176728f6
- Licence: MIT. Copyright (c) 2022 Gou Minghao
- Files: `grasppanda/models/views/angular_cost.py` (the angular class loss, adapted from
  `rgbd_graspnet/net/loss.py`)

## open-mmlab/mmdetection

- Repository: https://github.com/open-mmlab/mmdetection
- Commit: `cfd5d3a985b0249de009b67d04f37263e11cdf3d`
- Licence: Apache-2.0. `Copyright 2018-2023 OpenMMLab.`
- Changes (Apache-2.0, section 4b): the loss functions of `mmdet/models/losses/gfocal_loss.py`,
  `varifocal_loss.py`, `ghm_loss.py` and `balanced_l1_loss.py` as GraspPanda loss primitives (a penalty
  per element, a mask, weights, `weight x mean`), without the registry, the reductions by an average
  factor and the label conventions of the detection heads; the last bin is a valid target of the
  distribution focal loss; GHM counts only the supervised elements of a call.
- Files: `grasppanda/models/losses/detection.py`

## OpenPCDet (RoI-grid pooling, RoI-aware pooling, voxel set abstraction)

- Repository: https://github.com/open-mmlab/OpenPCDet
- Commit: `233f849`
- Licence: Apache-2.0. The licence file names no copyright holder; the pooling kernels are
  signed "Written by Shaoshuai Shi".
- Files: `grasppanda/models/groupers/roi_pool.py` (classes `RoIGridGroup`, `RoIAwareGroup`
  and `VoxelSetAbstractionGroup`; from `pcdet/models/roi_heads/pvrcnn_head.py`,
  `pcdet/models/roi_heads/partA2_head.py`, `pcdet/ops/roiaware_pool3d` and
  `pcdet/models/backbones_3d/pfe/voxel_set_abstraction.py`). Changes are listed at the top
  of the file: the region is a gripper-aligned box, the CUDA extensions are replaced by
  operators of this package, and a linear layer per cell stands in for the sparse
  convolutions of Part-A².
## Pointcept

- Repository: https://github.com/Pointcept/Pointcept
- Commit: `1342eda30e96cbb5fadf5374ff8eb18f6de15c71`
- Licence: MIT. `Copyright (c) 2023 Pointcept`
- Files: `grasppanda/transforms/pointcept.py` (`RandomJitter`, `RandomDropout`,
  `RandomShift`, adapted from `pointcept/datasets/transform.py`)
