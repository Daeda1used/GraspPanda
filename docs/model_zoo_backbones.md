# Encoders checked against their authors' code

Each scene encoder below was checked to one of three depths. Nothing here is a
grasp-detection result: the claims are about architecture fidelity, not accuracy.

- **Weights verified** (5): `PointNet2`, `PTv3`, `PTv3Sonata`, `PTv3Utonia`, `LitePT`.
  Released weights load strictly, and the outputs match the official implementation.
- **Compared on synthetic weights** (3): `PointNeXt`, `OACNNs`, `SpUNet`. The outputs match
  the official code when both are given the same analytic or hashed weights; no released
  checkpoint was loaded.
- **Unit-tested only** (4): `PointNet2Dense`, `FeaturePropagation`,
  `FurthestPointSubsample`, `MinkUNet`. Unit tests only: not compared with the authors'
  code.

Every registered encoder, checked or not, is in
[Registered classes](components.md#scene-encoders); how each fares under every method is
on [Does it run?](advanced_guides/measured_combinations.md#scene-encoders).

## What the status means

| Status | Meaning |
|---|---|
| `tested` | Unit tests on CPU: output shape and channels, binding to the right point set, gradients reaching every parameter, deterministic evaluation, batch independence, variable-size batches, rejection of misuse. |
| `tested` + parity | Additionally: the outputs match the official implementation when both sides are given the same analytic or hashed weights. No released checkpoint was loaded. The measured difference is stated. |
| `weights-verified` | Additionally: a checkpoint released by the authors loads with no missing and no unexpected key, and the outputs match the official implementation run on the same input. The measured difference is stated. |

"Matches" is always a number. All comparisons are in strict float32 on one fixed
20 000-point tabletop scene, on CPU and on an RTX A6000; the fixtures and the script that
recorded them are in the repository (see [Reproducing the evidence](#reproducing-the-evidence)).

## The encoders and how each was checked

| Backbone | Paper | `output` | Status | Needs |
|---|---|---|---|---|
| `PointNet2` | Qi et al., NeurIPS 2017 | `subsampled` | `weights-verified` | none |
| `PointNet2Dense` | same | `dense` | `tested` | none |
| `FeaturePropagation` | Qi et al., NeurIPS 2017, §3.4 | `dense` | `tested` | none |
| `FurthestPointSubsample` | Qi et al., NeurIPS 2017, §3.2 (furthest point sampling) | `subsampled` | `tested` | none |
| `PTv3` | Wu et al., CVPR 2024 | `dense` | `weights-verified` | spconv |
| `PTv3Sonata` | Wu et al., CVPR 2025 (Sonata); Zhang et al., NeurIPS 2025 (Concerto) | `dense` | `weights-verified` | spconv |
| `PTv3Utonia` | Zhang et al., ICML 2026 | `dense` | `weights-verified` | spconv |
| `LitePT` | Yue et al., CVPR 2026 | `dense` | `weights-verified` | spconv |
| `PointNeXt` | Qian et al., NeurIPS 2022 | `dense` | `tested` + parity | none |
| `OACNNs` | Peng et al., CVPR 2024 | `dense` | `tested` + parity | spconv |
| `MinkUNet` | Choy et al., CVPR 2019 | `dense` | `tested` | spconv |
| `SpUNet` | Choy et al., CVPR 2019 (the residual sparse U-Net and its 34C sizes) | `dense` | `tested` + parity | spconv |

`FeaturePropagation` and `FurthestPointSubsample` wrap another encoder: they are adapters
compose inserts, not options of the slot.

??? note "Official code, default configuration and parameter counts"

    | Backbone | Paper | Official code, licence | `output` | Default configuration | Parameters: ours / reference | Status | Optional dependency |
    |---|---|---|---|---|---|---|---|
    | `PointNet2` | Qi et al., NeurIPS 2017 | [charlesq34/pointnet2](https://github.com/charlesq34/pointnet2) (MIT); PyTorch lineage [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch) (Unlicense), [VoteNet](https://github.com/facebookresearch/votenet) (MIT) | `subsampled` | GraspNet-baseline backbone: SA 2048/1024/512/256 points, radii 0.04/0.1/0.2/0.3 m, 64/32/16/16 neighbours, 2 FP levels, 1024 seeds × 256 ch, xyz only | 641 856 / 641 856 (official model, counted when recording the fixture) | `weights-verified` | none |
    | `PointNet2Dense` | same | same | `dense` | Semantic-segmentation network of the paper (`pointnet2_sem_seg`): SA 1024/256/64/16 points, radii 0.1/0.2/0.4/0.8 m, 32 neighbours, 4 FP levels, 128 ch | 945 824 / 945 824 derived from the published layer list (949 536 with the redundant conv biases of the TensorFlow code) | `tested` | none |
    | `FeaturePropagation` | Qi et al., NeurIPS 2017, §3.4 | same | `dense` (wraps a `subsampled` backbone) | One propagation level as in GraspNet-baseline: 3 neighbours, inverse distance, MLP 256-256 | 132 096 on top of `PointNet2` / no reference (a wrapper) | `tested` | none |
    | `FurthestPointSubsample` | Qi et al., NeurIPS 2017, §3.2 (furthest point sampling) | same | `subsampled` (wraps a `dense` backbone) | The 1024 seeds of GraspNet-baseline, sampled over the scene coordinates with the rules of `PointNet2`; the wrapped backbone's features at those points | 0 on top of the wrapped backbone / no reference (a wrapper) | `tested`; keeps exactly the points `PointNet2` ends on, index for index (unit test on four rule combinations, and at full size against the GraspNet-baseline backbone) | none |
    | `PTv3` | Wu et al., CVPR 2024 | [Pointcept](https://github.com/Pointcept/Pointcept), [PointTransformerV3](https://github.com/Pointcept/PointTransformerV3) (MIT) | `dense` | ScanNet `semseg-pt-v3m1-0-base`: enc 2-2-2-6-2 blocks, 32–512 ch, dec 2-2-2-2, patch 1024, four curves, grid 0.02 m, 6 in, 64 out | 46 166 272 / 46 166 272 (official model); paper: 46.2 M | `weights-verified` | spconv |
    | `PTv3Sonata` | Wu et al., CVPR 2025 (Sonata); Zhang et al., NeurIPS 2025 (Concerto) | [sonata](https://github.com/facebookresearch/sonata), [Concerto](https://github.com/Pointcept/Concerto) (code Apache-2.0) | `dense` | Released base encoder (`sonata`, `concerto_base`): encoder only, 3-3-3-12-3 blocks, 48–512 ch, grid 0.02 m, 9 in, all five stages concatenated: 1232 out | 108 461 376 / 108 461 376 (tensor elements of `concerto_base.pth`) | `weights-verified` (with the four default `concerto_*.pth` files, which their model card calls pre-release Utonia models; see the note under the table of weight files). `sonata.pth`, `sonata_small.pth` and `concerto_base_origin.pth` load strictly (`tests/integration/test_backbone_released_weights.py`), without an activation comparison | spconv |
    | `PTv3Utonia` | Zhang et al., ICML 2026 | [Utonia](https://github.com/Pointcept/Utonia) (code Apache-2.0) | `dense` | Released `utonia` encoder: as above with 54–576 ch and rotary embedding (base 10), grid 0.01 m, 1386 out | 137 253 744 / 137 253 744 (official model built from `utonia.pth`) | `weights-verified` | spconv |
    | `LitePT` | Yue et al., CVPR 2026 | [LitePT](https://github.com/prs-eth/LitePT) (MIT); also in Pointcept | `dense` | ScanNet `semseg-litept-small-v1m1` (LitePT-S): enc 2-2-2-6-2 blocks, 36–504 ch, sparse convolution in stages 0–2, attention with PointROPE in stages 3–4, decoder of unpooling steps only, patch 1024, four curves, grid 0.02 m, 6 in, 72 out | 12 712 932 / 12 712 932 (official model); paper: 12.7 M. Also checked: LitePT-S* 15 988 716 (16.0 M), LitePT-B 45 112 788 (45.1 M), LitePT-L 85 880 448 (85.9 M) | `weights-verified` | spconv |
    | `PointNeXt` | Qian et al., NeurIPS 2022 | [PointNeXt](https://github.com/guochengqian/PointNeXt), [openpoints](https://github.com/guochengqian/openpoints) (MIT) | `dense` | PointNeXt-S for S3DIS: width 32, stride 4 × 4 stages, radius 0.1 m × 2 per stage, 32 neighbours, 4 in, 32 out | 790 720 / 790 720 (official `BaseSeg` without head). With the official head: S 0.792 M, B 3.828 M, L 7.125 M, XL 41.576 M, equal to the published numbers | `tested`, plus parity with the official code. Official S3DIS checkpoints of PointNeXt-S, -B, -L and -XL are public in the Google Drive folders of the PointNeXt [model zoo](https://github.com/guochengqian/PointNeXt/blob/master/docs/modelzoo.md) (the folder of PointNeXt-S was opened on 2026-10-13 and holds a `checkpoint` sub-folder); they have not been loaded here | none |
    | `OACNNs` | Peng et al., CVPR 2024 | [Pointcept](https://github.com/Pointcept/Pointcept) `oacnns-v1m1` (MIT) | `dense` | ScanNet `semseg-oacnns-v1m1-0-base`, the paper's OA-CNN (L): stem 64 ch, enc 3-3-9-8 blocks with 64-64-128-256 ch, dec 256 ch, four pooling grids per stage (8-12-16-16, 6-9-12-12, 4-6-8-8, 3-4-6-6 cells), grid 0.02 m, 9 in, 20 out | 51 494 484 / 51 494 484 (official model built from the published config file) | `tested`, plus parity with the official code (no checkpoint has been released) | spconv |
    | `MinkUNet` | Choy et al., CVPR 2019 | [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) `examples/minkunet.py` @ `02fc608` (MIT), adapted onto `grasppanda.ops.sparse` | `dense` | MinkUNet34C / Res16UNet34C: 3³ stem, enc 2-3-4-6 blocks with 32-64-128-256 ch, dec 2-2-2-2 with 256-128-96-96 ch, grid 0.02 m, 3 in, 96 out without the final layer. `layers`, `planes` and `block` give MinkUNet 14A–D, 18A/B/D, 34A–C, 50 and 101 | 37 847 808 with a 32-channel final layer, the tensor elements of the released PointContrast Res16UNet34C (batch-norm counters aside) | `tested`. The PointContrast and Mask3D backbones load key for key (`tests/integration/test_backbone_released_weights.py`); no activation comparison with MinkowskiEngine was recorded for this class | spconv |
    | `SpUNet` | Choy et al., CVPR 2019 (the residual sparse U-Net and its 34C sizes) | [Pointcept](https://github.com/Pointcept/Pointcept) `SpUNet-v1m1` (MIT) | `dense` | ScanNet `semseg-spunet-v1m1-0-base`: 5³ stem 32 ch, enc 2-3-4-6 blocks with 32-64-128-256 ch, dec 2-2-2-2 blocks with 256-128-96-96 ch, grid 0.02 m, 6 in, 20 out (`out_channels: null` returns the 96 decoder channels) | 39 157 844 / 39 157 844 (official model built from the published config file); 39 155 904 without the final layer | `tested`, plus parity with the official code (its authors have released no checkpoint) | spconv |

Since the review at the end of this page, DGCNN, OctFormer, Point Transformer V1 and V2,
PointMamba, PointMLP and the multi-scale PointNet++ (`PointNet2MSG`) have been registered as
well, as `tested` or `bridge` classes: none of them was compared with its authors' code.

### Evidence

The largest difference between our features and the official implementation's, per
encoder and weight file, on CPU and CUDA, with the fixtures that pin it.

??? note "The differences, row by row"

    Largest absolute difference between our features and the official implementation's, over
    the stored rows, CPU / CUDA. `|f|` is the largest reference feature, for scale. Per-channel
    mean and standard deviation over all 20 000 points agree to better than 2e-5 in every row.

    | Backbone | Oracle | Weights | Strict load | Max. difference (CPU / CUDA) | `|f|` |
    |---|---|---|---|---|---|
    | `PointNet2` | GraspNet-baseline `Pointnet2Backbone` with its CUDA extension | `checkpoint-rs.tar` (released) | yes | 5.2e-6 / 9.5e-7; all 1024 seed indices and coordinates identical | 4.1 |
    | `PTv3` | `PointTransformerV3/model.py` @ `3229e9b` | Pointcept `scannet-semseg-pt-v3m1-0-base` (released, MIT) | yes | 2.1e-5 / 1.3e-5 | 3.0 |
    | `PTv3` | same | analytic, with relative position bias, post-norm and patch sizes 64/64/48/16 | same keys and shapes | 4.8e-7 / 4.8e-7 | 1.5 |
    | `PTv3Sonata` | `concerto` package @ `5b41117` | `concerto_tiny.pth` | yes | 2.5e-6 / 2.9e-6 | 3.6 |
    | `PTv3Sonata` | same | `concerto_small.pth` | yes | 7.6e-6 / 1.4e-6 | 5.5 |
    | `PTv3Sonata` | same | `concerto_base.pth` | yes | 6.4e-6 / 2.9e-6 | 8.1 |
    | `PTv3Sonata` | same | `concerto_large.pth` | yes | 9.1e-6 / 3.8e-6 | 8.7 |
    | `PTv3Sonata` | `sonata` @ `18c09ff` and `concerto` packages | analytic, with decoder and layer scale | same keys and shapes | 1.1e-5 / 1.2e-5 | 4.7 |
    | `PTv3Utonia` | `utonia` package @ `da776a0` | `utonia.pth` | yes | 9.5e-6 / 2.5e-6 | 7.7 |
    | `PTv3Utonia` | same | analytic, with decoder and layer scale | same keys and shapes | 1.1e-5 / 5.5e-6 | 4.7 |
    | `LitePT` | `LitePT/litept/model.py` @ `436d048`, attention in float32 (see *LitePT: two reference outputs* below) | `scannet-semseg-litept-small-v1m1` (released, MIT): LitePT-S | yes | 1.0e-5 / 1.7e-6 | 4.4 |
    | `LitePT` | same | `scannet-insseg-litept-small-v1m2`: LitePT-S*, with decoder blocks | yes | 1.7e-5 / 4.1e-6 | 5.1 |
    | `LitePT` | same | `structured3d-semseg-litept-base-v1m1`: LitePT-B | yes | 2.6e-5 / 1.7e-6 | 4.3 |
    | `LitePT` | same | `structured3d-semseg-litept-large-v1m1`: LitePT-L | yes | 1.2e-5 / 2.6e-6 | 5.0 |
    | `LitePT` | same | analytic: a stage that convolves and attends, decoder blocks of both kinds, a stride of 3, PointROPE bases from 10 to 100 | same keys and shapes | 5.6e-6 / 1.9e-6 | 6.2 |
    | `LitePT` | same | analytic: attention from the input level on, two curves, post-norm, no query/key/value bias, fixed attention scale | same keys and shapes | 2.2e-5 / 1.4e-5 | 2.4 |
    | `PointNeXt` (S and B) | openpoints `BaseSeg` @ `bc778ec` with its CUDA extension, built from the published YAML | analytic | same keys and shapes | 1.3e-7 / 1.2e-7 | 0.35 |
    | serialisation codes | `PointTransformerV3/serialization` | — | — | equal, all four orders, depths 1/3/7/10/16 | — |
    | `OACNNs` | Pointcept `oacnns_v1m1_base.py` @ `1342eda`, built from the published config file | hashed, batch norm calibrated (see below) | same keys and shapes | 3.2e-5 / 4.2e-5 (CUDA: largest of 60 runs, smallest 2.5e-5; the scatter-add of the pooling is unordered there) | 10.6 |
    | `OACNNs` | same file; three stages, a width that changes at every level, one to three pooling grids per stage | analytic | same keys and shapes | 1.5e-7 / 1.5e-7 | 0.21 |
    | `OACNNs` | same model as the row above, on a crafted cloud of 2 583 voxels whose second level does not start at cell 0 | hashed, batch norm calibrated | same keys and shapes | 3.9e-5 / 3.5e-5 (CUDA: largest of 60 runs) | 7.4 |
    | `SpUNet` | Pointcept `spconv_unet_v1m1_base.py` @ `1342eda`, built from the published config file | hashed, batch norm calibrated | same keys; same shapes but for the 1×1 kernels, which load strictly through `convert_pointcept_state` | 3.2e-5 / 3.7e-5 (CUDA: largest of 20 runs) | 13.9 |
    | `SpUNet` | same file; three stages, no two widths alike, one to three blocks per stage, no final layer | analytic | as above | 3.0e-7 / 0 (bit-identical) | 0.90 |

    "Analytic" weights: both sides fill every tensor of their state dict with the same
    closed-form function of the tensor's name and shape. It needs no file, exercises options
    no released checkpoint uses, and fails if the two state dicts differ in a single key or
    shape. The fixtures are sensitive to the details that matter: for example, fusing before
    the decoder's positional convolution moves the `PTv3` output by 1e-2, another curve order
    by 4e-2, and dropping the radius normalisation moves the `PointNeXt` output by 3e-3, all
    orders of magnitude above the agreement in the table.

    "Hashed, batch norm calibrated" is the same idea for deep batch-norm networks, which the
    analytic weights do not survive: a sine of the flat index makes every weight matrix
    nearly rank two, and batch norm with made-up running statistics normalises nothing.
    Through the 23 residual blocks of the published OA-CNNs the features then reach 3e2 and
    rounding alone separates the two implementations by 1.3 on the GPU; with calibrated
    statistics but sine weights, one and the same model still disagreed with itself between
    CPU and GPU by 3 % of its features. For these fixtures every value is a 32-bit integer
    hash of the tensor's name and flat index (exact on every platform, full-rank matrices),
    and each side then sets the batch-norm statistics with one training-mode pass of its own
    code over the input, so the comparison covers batch norm in training mode as well.
    Sensitivity, measured the same way: keeping the voxels the official `OACNNs` loses at an
    odd axis (`crop_odd_extent: false`) moves the published configuration by 1.6; anchoring
    its pooling grids at cell 0 instead of the lowest occupied cell moves the crafted-cloud
    fixture by 1.1 and the tabletop scene not at all, which is why that fixture exists;
    dropping the `1e-6` of the voting softmax fails the published configuration. For `SpUNet`,
    reading a 1×1 kernel the way its shape suggests moves the two fixtures by 15.7 and 0.49,
    and counting cells from one cell further by 9.1 and 0.31.

    Additional checks that need no oracle and run in the unit suite: `PointNet2` and
    `PointNet2Dense` equal the paper's definition written out point by point in NumPy
    (float64, 1e-10) for five configurations; the Hilbert code equals Skilling's algorithm
    and visits every cell of a cube once through face neighbours.

    `OACNNs` has its own: the first block of a small network equals the adaptive aggregation
    written out cube by cube in plain loops, on a batch of two clouds one of which does not
    start at cell 0 of its level; an isolated voxel in the lost cell of an odd axis leaves
    the features of the rest of its cloud untouched; clouds with different bounding boxes do
    not interact. Together they pin what a single-cloud oracle cannot see: the per-cloud
    bounding box and the per-cloud softmax shift. For `SpUNet`, a converted 1×1 kernel is
    compared with the 1×1 convolution of spconv itself.

    One caveat for anyone repeating this on a GPU: PyTorch lets cuDNN compute convolutions in
    TF32 by default. On the A6000 that alone moves the `PointNet2` features by 4.6e-3, in the
    official implementation too. The comparisons switch it off; normal use does not.

??? note "LitePT: two reference outputs"

    The official LitePT attention has no float32 path. It casts queries, keys and values to
    float16 and calls flash-attention, so "the official output" exists in two forms, and the
    fixtures hold both:

    - **Attention in float32**: the official forward pass with that cast switched off
      (`Tensor.half` returns its tensor while recording, and a plain softmax attention stands
      in for flash-attention, which accepts float16 only). This is the computation the port
      performs, and the one the table above compares with.
    - **As released**: the official forward pass untouched, with flash-attention 2.6.3 and the
      authors' CUDA PointROPE extension.

    The port differs from the model as released by what float16 attention costs the official
    model itself:

    | Weights | Official model, float16 against float32 attention (all points) | `LitePT` against the official model as released (stored rows, CPU / CUDA) |
    |---|---|---|
    | LitePT-S, ScanNet | 7.5e-4 | 5.1e-4 / 5.1e-4 |
    | LitePT-S*, ScanNet instances | 2.0e-3 | 9.2e-4 / 9.2e-4 |
    | LitePT-B, Structured3D | 7.4e-4 | 5.4e-4 / 5.3e-4 |
    | LitePT-L, Structured3D | 6.4e-4 | 4.7e-4 / 4.7e-4 |

    With analytic weights the logits, and with them the effect of the cast, are arbitrary
    (6.1e-5 for the first analytic model, 6.3e-2 for the second), so only the float32 form is
    compared there.

    The released-weight fixtures use 1.25 mm voxels. The published models attend only after
    three poolings; at that grid the two attending stages hold 4 265 and 1 215 voxels of the
    scene, several patches of 1024 with an incomplete last one, where 5 mm would leave them
    308 and 80 voxels, one patch each. The fixtures are sensitive to what matters: a
    PointROPE base of 90 instead of 100 in one stage moves the LitePT-S output by 5.5e-2, a
    patch size of 512 in one stage by 0.23, two curves swapped by 0.42.

    All six references were also recorded without flash-attention and without the compiled
    extension: the official PyTorch fallback of PointROPE, the plain attention on float16 and
    on float32 inputs, PyTorch 2.7.1. The port agrees with those to 2.7e-5 or better too.
    Where both attention backends ran on the same float16 inputs they differ by 4.8e-4 to
    9.2e-4 with the released weights: the order of the float16 effect itself, and far below
    what a difference in meaning causes (5e-2 and more, as above).

### Speed

Forward time and memory of each encoder on one GPU, measured on a shared machine: an order
of magnitude.

??? note "Times and memory"

    One RTX A6000, PyTorch 2.7.1 + cu118, spconv 2.3.8, float32, evaluation mode, default
    CUDA settings, the 20 000-point fixture scene (batch 4: the scene and three mirrored or
    shifted copies). Forward time is the minimum and the median of 12 runs after warm-up;
    memory is the peak allocated by PyTorch. **The machine was shared while measuring** (load
    average about 37 on 16 cores, a second job on the GPU), and repeated passes varied by up to
    ±40 %, so read the times as an order of magnitude; memory is stable.

    | Backbone, configuration | Batch 1: ms (min / median) | Batch 1: GiB | Batch 4: ms (min / median) | Batch 4: GiB |
    |---|---|---|---|---|
    | `PointNet2`, default | 306 / 427 | 0.16 | 270 / 294 | 0.64 |
    | `PointNet2Dense`, default | 111 / 181 | 0.19 | 125 / 144 | 0.73 |
    | `PointNet2Dense`, GraspNet-baseline points and radii | 209 / 280 | 0.36 | 251 / 357 | 1.43 |
    | `FeaturePropagation` around `PointNet2` | 262 / 328 | 0.20 | 292 / 347 | 0.77 |
    | `PTv3`, grid 0.02 m (1 227 voxels) | 29 / 40 | 0.19 | 33 / 34 | 0.23 |
    | `PTv3`, grid 0.005 m (10 993 voxels) | 35 / 35 | 0.24 | 69 / 74 | 0.41 |
    | `PTv3Sonata` base, grid 0.02 m | 87 / 103 | 0.73 | 56 / 75 | 1.29 |
    | `PTv3Sonata` base, grid 0.005 m | 62 / 128 | 0.82 | 107 / 134 | 1.52 |
    | `PTv3Sonata` tiny, grid 0.005 m | 44 / 44 | 0.56 | 51 / 68 | 0.85 |
    | `PTv3Utonia`, grid 0.01 m (4 266 voxels) | 63 / 95 | 1.21 | 90 / 126 | 1.95 |
    | `PTv3Utonia`, grid 0.005 m | 79 / 108 | 1.40 | 162 / 227 | 2.67 |
    | `LitePT` S, grid 0.02 m † | 23 / 34 | 0.19 | 27 / 33 | 0.22 |
    | `LitePT` S, grid 0.005 m † | 28 / 37 | 0.21 | 36 / 43 | 0.31 |
    | `LitePT` S* (decoder blocks), grid 0.005 m † | 32 / 43 | 0.23 | 45 / 48 | 0.33 |
    | `LitePT` B, grid 0.005 m † | 37 / 49 | 0.37 | 52 / 56 | 0.48 |
    | `LitePT` L, grid 0.005 m † | 39 / 53 | 0.53 | 63 / 95 | 0.66 |
    | † a later session, same protocol and load; `PTv3`, grid 0.005 m, measured again in it | 36 / 49 | 0.24 | 88 / 120 | 0.41 |
    | `PointNeXt` S | 540 / 617 | 1.70 | 677 / 724 | 4.36 |
    | `PointNeXt` B | 536 / 678 | 1.71 | 794 / 978 | 4.37 |
    | `OACNNs`, grid 0.005 m, 3 in, 512 out | 82 / 113 | 0.30 | 88 / 123 | 0.60 |
    | `SpUNet`, grid 0.005 m, 3 in, 512 out | 20 / 24 | 0.27 | 31 / 40 | 0.61 |
    | `SpUNet`, grid 0.02 m, 3 in, 512 out | 19 / 19 | 0.24 | 22 / 23 | 0.48 |

    The PointNet++ family is slow for its size because it runs on the pure-PyTorch reference
    operators: furthest point sampling is a Python loop of one step per sample, so most of the
    time is interpreter overhead, which is also why batch 4 costs about the same as batch 1.
    A compiled sampling backend, held to the reference by the operator tests, is where that
    time would go.

    The `OACNNs` and `SpUNet` rows come from a later session with the same protocol (load
    average about 30; the batch of four is the scene, two mirrored copies and a shifted one).
    `OACNNs` spends its time in about two thousand small operations, the voting and pooling
    of four grids in each of 23 blocks, not in convolutions. It has no row at 0.02 m: the
    scene is 15 voxels deep there, and with the official bounding-box rule the network needs
    16 along every axis (`crop_odd_extent`, below).

## Released weights

The checkpoints of the self-supervised PTv3 family, of LitePT and of the released
`SpUNet` backbones: what each file is, its licence and what it loads into.

??? note "Sonata, Concerto and Utonia"

    Files under `components/` of the local weight store, and what they are.

    | File | What it is | Released by | Licence | Loads into | Configuration stored in the file |
    |---|---|---|---|---|---|
    | `concerto_tiny.pth` (SHA-256 `2d5ac74f…`) | Concerto: PTv3 encoder pre-trained by joint 2-D/3-D self-supervised learning | Pointcept group, [Hugging Face](https://huggingface.co/Pointcept/Concerto) | code Apache-2.0; weights CC-BY-NC 4.0 per the repository README (the model card's metadata says Apache-2.0) | `PTv3Sonata` | enc 1-1-1-3-1, 16–256 ch, heads 1-2-4-8-16; 4.86 M parameters |
    | `concerto_small.pth` (`251441d9…`) | same | same | same | `PTv3Sonata` | enc 2-2-2-6-2, 32–512 ch, heads 2-4-8-16-32; 38.65 M |
    | `concerto_base.pth` (`845ec7de…`) | same | same | same | `PTv3Sonata` (the defaults) | enc 3-3-3-12-3, 48–512 ch, heads 3-6-12-24-32; 108.46 M |
    | `concerto_large.pth` (`04de4954…`) | same | same | same | `PTv3Sonata` | enc 3-3-3-12-3, 64–768 ch, heads 4-8-16-32-48; 207.67 M |
    | `utonia.pth` (`a0a2e5e2…`) | Utonia: one PTv3 encoder pre-trained across domains (the authors show indoor, outdoor and object data), with rotary position embedding | Pointcept group, [Hugging Face](https://huggingface.co/Pointcept/Utonia) | as Concerto | `PTv3Utonia` (the defaults) | enc 3-3-3-12-3, 54–576 ch, heads 3-6-12-24-32, `rope_base` 10; 137.25 M |
    | `sonata.pth`, `sonata_small.pth` (under `$GRASPPANDA_CACHE/weights_components`, not in the local weight store) | Sonata: self-supervised PTv3 encoder, same architecture and size as `concerto_base` (`sonata_small`: as `concerto_small`) | Meta, [Hugging Face](https://huggingface.co/facebook/sonata) | code Apache-2.0; weights CC-BY-NC 4.0 (repository README; the model card states none) | `PTv3Sonata`: every tensor loads with equal shape (`pretrained: sonata`, `sonata-small`); activations were not compared with the `sonata` package | as `concerto_base` / `concerto_small` |
    | `concerto_base_origin.pth` (same place) | The Concerto model of the paper | Pointcept group, [Hugging Face](https://huggingface.co/Pointcept/Concerto) | weights CC-BY-NC 4.0 (model card) | `PTv3Sonata` (the defaults): every tensor loads with equal shape (`pretrained: concerto-base-origin`); activations were not compared | as `concerto_base` |

    **Which file is "Concerto".** The model card of `Pointcept/Concerto` says: "The default models(concerto_large/base/small/tiny) are the pre-release version of our next work, Utonia, which can deal with input without color and normal. If colors and normals are not available, please set them to zeros. [...] The original Concerto model is `concerto_base_origin.pth`." The four `concerto_*.pth` rows above are therefore the pre-release models, not the networks evaluated in the Concerto paper; the paper's are `concerto_base_origin.pth` and `concerto_large_origin.pth`. The activation fixtures of this page were recorded with the default files.

    All of them:

    - are **encoder-only** (`enc_mode: true`): the state dict has `embedding.*` and `enc.*`
      and no decoder. Dense features are the encoder stages concatenated through the pooling
      maps (`concat_stages`), the authors' linear-probing protocol. Setting `enc_mode: false`
      adds a decoder with random weights;
    - expect **9 input channels**: coordinates, colour, normal. The authors advise zeros for
      a modality that is missing. The coordinate channels go through their data pipeline
      first (centring, and for Utonia a scale factor); reproducing that is the caller's job;
    - contain a `mask_token` that only pre-training uses; keep `mask_token: true` to load
      strictly;
    - store `enable_flash: true`. That selects an attention backend, not an architecture;
      flash-attention is not used here and is not needed;
    - are **non-commercial** by the stricter of the two statements their authors make. Using
      them makes the resulting model non-commercial. GraspPanda does not ship or download
      them.

    ```yaml
    model:
      backbone:
        type: PTv3Sonata              # concerto_small
        grid_size: 0.005
        enc_depths: [2, 2, 2, 6, 2]
        enc_channels: [32, 64, 128, 256, 512]
        enc_num_head: [2, 4, 8, 16, 32]
    ```

??? note "LitePT, PTv3, OA-CNNs and SpUNet"

    | File | What it is | Loads into | Backbone parameters |
    |---|---|---|---|
    | `scannet-semseg-litept-small-v1m1.pth` (SHA-256 `4fc1c778…`) | LitePT-S trained for ScanNet semantic segmentation | `LitePT` (the defaults) | 12.71 M |
    | `scannet-insseg-litept-small-v1m2.pth` (`86408c63…`) | LitePT-S* trained for ScanNet instance segmentation (PointGroup) | `LitePT` with `dec_depths: [2, 2, 2, 2]`, `dec_conv: [true, true, true, false]`, `dec_attn: [false, false, false, true]` | 15.99 M |
    | `structured3d-semseg-litept-base-v1m1.pth` (`53b18ba6…`) | LitePT-B trained for Structured3D semantic segmentation | `LitePT` with enc 3-3-3-12-3, 54–576 ch, heads 3-6-12-24-32, `dec_channels: [72, 108, 216, 432]`, `dec_num_head: [4, 6, 12, 24]` | 45.11 M |
    | `structured3d-semseg-litept-large-v1m1.pth` (`e50bd216…`) | LitePT-L, same task | `LitePT` with enc 3-3-3-12-3, 72–864 ch, heads 4-8-16-32-48, `dec_channels: [72, 144, 288, 576]`, `dec_num_head: [4, 8, 16, 32]` | 85.88 M |

    Released by the Photogrammetry and Remote Sensing Lab of ETH Zurich on
    [Hugging Face](https://huggingface.co/prs-eth/LitePT) (revision `a8e76e9`; each file is
    `<name>/model/model_best.pth` there). Code and weights are **MIT**: the repository's
    `LICENSE`, and `license: mit` on the model card, both read on 2026-10-12. The files are
    Pointcept training checkpoints: the backbone is in `state_dict` under the prefix
    `module.backbone.`, next to a task head and the optimiser state. All four

    - expect **6 input channels**, colour and normal as the authors' data pipeline prepares
      them (reproducing it is the caller's job). A backbone built with another `in_channels`
      has a stem convolution these weights do not fit;
    - were trained on rooms at 0.02 m voxels, with float16 attention (see
      *LitePT: two reference outputs* under [Evidence](#evidence));
    - attend only in the two coarsest stages, on voxels of 8 and 16 grid cells. At a 5 mm
      grid those are 4 cm and 8 cm, and the fixture scene is 308 and 80 voxels there: fewer
      than one patch of 1024, so the attention is global.

    The repository also offers LitePT-S for nuScenes and Waymo (4 input channels, 0.05 m
    voxels), LitePT-L for nuScenes, a ScanNet200 instance model and a Waymo detection model.
    None of those was downloaded or checked.

    ```yaml
    model:
      backbone:
        type: LitePT                  # LitePT-S, untrained, on the 5 mm grid
        in_channels: 3
        grid_size: 0.005
    ```

    Also used for verification: the ScanNet segmentation checkpoint of PTv3
    (`scannet-semseg-pt-v3m1-0-base/model/model_best.pth` on
    [Hugging Face](https://huggingface.co/Pointcept/PointTransformerV3), MIT, SHA-256
    `40206376…`, backbone keys under `module.backbone.`), and the GraspNet-baseline RealSense
    checkpoint (`checkpoint-rs.tar`, non-commercial research licence, backbone keys under
    `view_estimator.backbone.`).

    No checkpoint of OA-CNNs has been released by its authors: on 2026-10-12 the Pointcept
    organisation on Hugging Face held PTv3, Sonata, Concerto and Utonia only, and the Pointcept
    model zoo lists a training command for it. `OACNNs` is therefore `tested` with parity
    against the official code, not `weights-verified`.

    Checkpoints of `SpUNet-v1m1` **are** released, by works that use it as their backbone, and
    they load into `SpUNet` key for key after `convert_pointcept_state`
    (`tests/integration/test_backbone_released_weights.py`; files loaded on 2026-10-13):

    | Release | Files loaded here | Licence | `SpUNet` options |
    |---|---|---|---|
    | [RayYoh/GaussianCross](https://huggingface.co/RayYoh/GaussianCross) (GaussianCross, ACM MM 2025) | `scannet/semseg-spunet-base-lin` (157 MB), `scannet/semseg-spunet-base` (470 MB) | MIT (model card) | `out_channels: null` for the linear-probing file, defaults for the fine-tuned one |
    | [KadirYilmaz/Volt](https://huggingface.co/KadirYilmaz/Volt) `teacher_weights/` (the distillation teachers of Volt, ECCV 2026) | `scannet_unet_teacher.pth` (470 MB) | MIT (model card) | defaults |

    The MSC archives the Pointcept README links (Masked Scene Contrast, CVPR 2023, on the
    author's SharePoint) were not downloaded here. `SpUNet` stays `tested`: the loads prove
    names and shapes, and no activation comparison with the official model was recorded for
    these files. A ScanNet SpUNet converted by a third party into another library's layout
    also exists (`torch-pointcloud/spunet-v1m1.scannet20.pointcept`, MIT according to its
    card, SHA-256 `11658b4d…`); its card does not say which checkpoint it was converted from,
    and it was not used here.

    How to start a grasp model from any of these files, and the full table of released
    weights that load into `SpUNet`, `PTv3`, `PTv3Sonata`, `LitePT` and `MinkUNet`:
    [Scene encoders: sparse-voxel networks and released weights](slots/scene_encoders_sparse.md).

## Where the implementations deliberately differ from upstream

Each difference exists for the scene-encoder contract (batch independence, deterministic
evaluation, no compiled extra) and is exact for a single cloud.

??? note "Every difference, and the upstream behaviours kept as options"

    | Topic | Upstream | Here | Why |
    |---|---|---|---|
    | PTv3 attention backend | flash-attention in half precision; without it, the patch size shrinks to the smallest cloud of the batch | `scaled_dot_product_attention` in the input precision with the fixed-patch layout of the flash path | No extra dependency; a cloud's result does not depend on its batch neighbours |
    | PTv3 serialisation depth | one depth for the batch, from its largest cloud; the Hilbert curve depends on it | one depth per cloud | Batch independence |
    | PTv3 order shuffling | every forward pass, in evaluation mode too | training mode only | Deterministic evaluation |
    | PTv3 voxel representative | a random point of each voxel (training transform) | the first point of each voxel | Deterministic; features go back to exactly the points that produced the voxel |
    | PTv3 multi-dataset norms (`pdnorm_*`, PPT) | one set of normalisation layers per training dataset, chosen per sample by a `condition` | one set, chosen when the weights are loaded (`pretrained_condition`); the model is a plain PTv3 | The `Backbone` contract carries no per-sample dataset condition. The released `ppt` checkpoints load with the statistics of the chosen dataset; the other sets are dropped |
    | PTv3 `cls_mode` (encoder without decoder) | returns the coarsest level | `enc_mode: true` returns the deepest `concat_stages` encoder stages concatenated on every input voxel | A scene encoder returns one feature per point; encoder-only checkpoints load strictly |
    | LitePT attention backend | flash-attention on float16 queries, keys and values; there is no other path | `scaled_dot_product_attention` in the input precision, on the same patches | No extra dependency. Measured: with the released weights the official output itself moves by 6e-4 to 2e-3 between the two precisions |
    | LitePT PointROPE | a CUDA extension, or a PyTorch fallback that looks the rotations up in a table | computed in PyTorch from the grid coordinates | No compiled code; equal to both (the evidence was recorded with each) |
    | LitePT serialisation depth | one depth for the batch, from its largest cloud | one depth per cloud | Batch independence |
    | LitePT order shuffling | at every pooling, on every forward pass, in evaluation mode too; the model's `shuffle_orders` only reaches the input level | training mode only, at every level, switched by `shuffle_orders` | Deterministic evaluation |
    | LitePT patch size of a decoder stage that attends | ignored: the level keeps the layout its encoder stage computed | has to equal the encoder's; anything else is an error | An argument that silently does nothing is a trap |
    | LitePT voxel representative | a random point of each voxel (training transform) | the first point of each voxel | As for PTv3 |
    | LitePT `enc_mode` (encoder without decoder) | supported | not implemented | No released checkpoint uses it, and the model then returns its coarsest level instead of dense features |
    | PointNet++ sampling below the first level | searches again | takes the first rows of the previous sample | Identical result (furthest point samples are nested), about half the sampling time |
    | OA-CNNs sparse grid | the bounding box of the batch, so which voxels a cloud loses at an odd axis (`crop_odd_extent`) depends on its batch neighbours | the bounding box of each cloud | Batch independence |
    | OA-CNNs voting weights | `exp(logit - m) / (sum + 1e-6)` with `m` the largest logit of the batch; because of the constant, `m` changes the result | `m` is the largest logit of the cloud | Batch independence |
    | OA-CNNs pooling grids | anchored at the lowest occupied cell of the batch at each level | anchored at the lowest occupied cell of the cloud | Batch independence |
    | OA-CNNs voxel representative | a random point of each voxel (training transform) | the first point of each voxel | As for PTv3 |
    | OA-CNNs arguments without effect (`groups`, `enc_num_ref`, `dec_depth`, `down_ratio`) | accepted; none enters the computation, the decoder has no blocks | not parameters | A config key that changes nothing is a trap |
    | SpUNet 1×1 convolutions (the final layer, the projected shortcuts) | spconv convolutions of kernel size 1, stored as `(out, 1, 1, 1, in)` and applied as that memory viewed as `(in, out)` | facade `Linear` layers with an `(out, in)` weight | The facade defines a 1×1 convolution as a linear layer; `convert_pointcept_state` re-lays an official state dict, changing no value |
    | SpUNet `enc_mode` (no decoder, one pooled vector per cloud) | supported | not implemented | Not a scene encoder |
    | SpUNet voxel representative | a random point of each voxel (training transform) | the first point of each voxel | As for PTv3 |

    Two upstream behaviours are kept as named constructor arguments because checkpoints depend
    on them: `PointNet2.fps_skip_near_origin` (the PyTorch sampling kernel never selects a
    point within 3.2 cm of the frame origin) and `PTv3.fuse_before_decoder_cpe` (in PT-v3m1
    the first positional convolution of each decoder stage reads the skip features only).
    `PTv3` follows the serialisation-depth rule of the original release, which its weights
    were trained with; Pointcept ≥ 1.6 adds one to the largest coordinate first, which differs
    only when that coordinate plus one is a power of two.

    `OACNNs.crop_odd_extent` keeps a third: the way the official model treats the border of
    its grid. It declares the sparse grid as the exact bounding box of the voxels, and a
    stride-2 convolution without padding cannot place the last cell of an axis with an odd
    number of cells. At every level the voxels in that cell are left out of the down-sampling
    and get zeros back from the up-sampling, and a cloud thinner than 16 voxels along any
    axis runs out of cells (the official code then stops inside spconv with "out spatial
    shape reach zero"; here the error names the cloud and says what to change).
    `crop_odd_extent: false` keeps every voxel and has no such limit; the two agree whenever
    every extent is a multiple of 16. A tabletop cloud is easily thinner than 16 voxels at the
    published 0.02 m grid (the fixture scene is 15 deep), so a grasp detector needs a finer
    grid or `crop_odd_extent: false`. The batch dependence that the per-cloud evaluation
    removes is not small: batching the fixture scene with its own left half moves the
    *official* features of the two clouds by up to 1.1 on features of magnitude 10.6
    (published configuration), where a batch here reproduces the clouds run alone to 2e-5.

    At a 5 mm grid, what has to be re-tuned in `OACNNs` is `point_grid_size`: the pooling
    grids are counted in cells, so the published 8 to 16 cells of the first stage shrink from
    0.32–0.64 m to 0.08–0.16 m. `SpUNet` has nothing counted in metres or cells besides the
    grid itself; its receptive field shrinks with the grid, as that of `SparseUNet14` does.
    The official `SpUNet` pads its sparse grid, loses no voxel and does not depend on the
    batch (measured as for `OACNNs`: 3.4e-5 on features of magnitude 13.9, which is rounding).

## Reproducing the evidence

The unit suite needs no files; the oracle comparisons need the checkpoints named below.

??? note "Commands and environments"

    ```bash
    # unit suite: contracts, written-out oracles, published sizes (CPU, no files needed)
    pytest tests/unit -k "backbone or ops"

    # oracle comparisons; the analytic ones run anywhere, the others need the checkpoints
    GRASPPANDA_WEIGHTS=/path/to/weights pytest tests/integration/test_backbone_oracle.py -rs
    ```

    `$GRASPPANDA_WEIGHTS` holds `graspnet_baseline/checkpoint-rs.tar`,
    `components/concerto_{tiny,small,base,large}.pth`, `components/utonia.pth` and
    `components/scannet-semseg-pt-v3m1-0-base.pth`. A test is skipped when its file is absent
    and fails when the file's SHA-256 is not the recorded one.

    The fixtures themselves are re-recorded with `tools/make_backbone_fixtures.py`, which runs
    the official code and is documented in its docstring. It never imports `grasppanda`; the
    packages an oracle needs (torch_scatter, timm, addict, the compiled extensions of
    graspnet-baseline and openpoints) belong in a separate environment.

    LitePT has its own test file and four more checkpoints, all MIT:

    ```bash
    # components/{scannet-semseg-litept-small-v1m1,scannet-insseg-litept-small-v1m2}.pth,
    # components/structured3d-semseg-litept-{base,large}-v1m1.pth
    GRASPPANDA_WEIGHTS=/path/to/weights pytest tests/integration/test_backbone_litept_oracle.py -rs
    ```

    Its fixtures were recorded in an environment built for the purpose: PyTorch 2.4.1 + cu118
    (the newest release flash-attention publishes a cu118 wheel for), flash-attention 2.6.3,
    spconv 2.3.8, torch_scatter, timm, addict, colorhash, NumPy 1.26, and the official
    PointROPE extension compiled out of tree for the GPU at hand, as its README instructs.
    Without flash-attention and the extension `make_backbone_fixtures.py litept` still runs,
    on the official PyTorch PointROPE and a plain softmax attention, and writes into the
    fixture which backend it used.

    The sparse CNNs of Pointcept have their own test file, which needs no checkpoint:

    ```bash
    pytest tests/integration/test_backbone_sparse_cnn_oracle.py -rs
    ```

    Their fixtures were recorded from Pointcept at commit `1342eda` (MIT), unpacked from the
    GitHub archive, with spconv 2.3.8, timm, torch_geometric 2.6.1 and torch_cluster 1.6.3
    (torch_geometric 2.8 moved the grid clustering OA-CNNs uses into pyg-lib). The recorder
    imports the official model file without `pointcept/models/__init__.py`, which would pull
    in every model of the zoo and its compiled extensions; no upstream code is stubbed or
    modified.

## Reviewed, not checked against their authors' code

Licences were read from the repositories on 2026-10-12 (GitHub licence metadata, and the
licence file where GitHub could not classify it). "Pointcept" means the model is
available inside [Pointcept](https://github.com/Pointcept/Pointcept) under MIT; the
dependencies listed for those rows are the imports of the Pointcept implementation.
"Not checked" means exactly that.

??? note "Point-cloud and voxel encoders"

    | Encoder | Paper | Code, licence | Extra dependencies | Why not yet, and what it would take |
    |---|---|---|---|---|
    | DGCNN (segmentation form) | Wang et al., TOG 2019 | [WangYueFt/dgcnn](https://github.com/WangYueFt/dgcnn), [dgcnn.pytorch](https://github.com/antao97/dgcnn.pytorch) (both MIT) | none | On the list for this round, dropped for time. Needs a k-NN operator in feature space (the reference `knn` takes 3-D coordinates only), chunked: at 20 000 points every EdgeConv layer compares all pairs of points. The PyTorch port needs no compiled code, so it can serve as an oracle directly. |
    | SPVCNN | Tang et al., ECCV 2020 | [spvnas](https://github.com/mit-han-lab/spvnas) (MIT, archived), [torchsparse](https://github.com/mit-han-lab/torchsparse) (MIT) | torchsparse | A third sparse backend for one model. Port the point-voxel branch onto the facade instead. |
    | KPConv | Thomas et al., ICCV 2019 | [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch) (MIT) | its compiled C++ wrappers (`cpp_subsampling`, `cpp_neighbors`) | Neighbourhoods are precomputed on the CPU in the data loader. That pipeline has to become a transform, or be redone on the GPU with the reference operators. |
    | KPConvX | Thomas et al., CVPR 2024 | [ml-kpconvx](https://github.com/apple/ml-kpconvx) (MIT) | not checked | Successor of KPConv; review together with it. |
    | OctFormer | Wang, SIGGRAPH 2023 | [octformer](https://github.com/octree-nn/octformer) (MIT); Pointcept | ocnn (MIT), [dwconv](https://github.com/octree-nn/dwconv) (no licence file) | Octree attention. PTv3 covers the same niche with fewer dependencies; a port needs an octree structure and a depth-wise octree convolution. |
    | Swin3D | Yang et al., arXiv 2304.06906 | [Swin3D](https://github.com/microsoft/Swin3D) (MIT); Pointcept | MinkowskiEngine, the `Swin3D` package, timm | Tied to MinkowskiEngine. Not worth a port before the sparse facade can express its attention. |
    | Point Transformer V1 | Zhao et al., ICCV 2021 | Pointcept. The widely copied POSTECH-CVLab reproduction carries no licence and cannot be a source | pointops (Pointcept's compiled library), einops | k-NN vector attention is feasible on the reference operators but slow without a compiled k-NN. |
    | Point Transformer V2 | Wu et al., NeurIPS 2022 | Pointcept | pointops, torch_geometric, torch_scatter, timm, einops | Grid pooling already exists here (`PTv3Sonata`); grouped vector attention needs k-NN grouping. Superseded by PTv3. |
    | Stratified Transformer | Lai et al., CVPR 2022 | [Stratified-Transformer](https://github.com/dvlab-research/Stratified-Transformer) (MIT); Pointcept | pointops2 (compiled), torch_geometric, torch_scatter, timm | Same dependency problem as PTv2. |
    | PointMamba | Liang et al., NeurIPS 2024 | [PointMamba](https://github.com/LMD0311/PointMamba) (Apache-2.0) | mamba-ssm (Apache-2.0, compiled CUDA) | An object-level model. The serialisation operators here are the ones it needs; blocked on a compiled dependency and on scene-scale evidence. |
    | Point Cloud Mamba | Zhang et al., 2024 | [PCM](https://github.com/zhang-tao-whu/PCM) (no licence) | not checked | Unlicensed code cannot be a source; implement from the paper only. |
    | PointMLP | Ma et al., ICLR 2022 | [pointMLP-pytorch](https://github.com/ma-xu/pointMLP-pytorch) (Apache-2.0) | not checked | Published for classification and part segmentation; no scene-scale configuration to reproduce. |
    | PointNet++ with multi-scale grouping, as in Contact-GraspNet | Sundermeyer et al., ICRA 2021 | [contact_graspnet](https://github.com/NVlabs/contact_graspnet) (an NVIDIA licence shipped as `License.pdf`; read it before use) | none | A small extension of `PointNet2`: several radii per level, features concatenated. Take the configuration from the paper, not from the code. |

??? note "Image encoders used by RGB-D grasp methods"

    These are image encoders, not scene encoders. HGGD's image-encoder slot takes several of
    them (DINOv2, DINOv3, EfficientViT, FastViT, the Hiera encoder of SAM 2, ResNets:
    [Image encoders](slots/image_encoders.md)), and a seed-based method reads one through point
    painting. None of them was compared with its authors' code here.

    | Encoder | Paper | Code, licence | Extra dependencies | Note |
    |---|---|---|---|---|
    | ResNet encoder-decoders | He et al., CVPR 2016 | [torchvision](https://github.com/pytorch/vision) (BSD-3-Clause); the encoder inside [HGGD](https://github.com/THU-VCLab/HGGD) (MIT) | torchvision | The obvious first image encoder: permissive, small, the one HGGD-style methods are built on. |
    | DINOv2 | Oquab et al., TMLR 2024 | [dinov2](https://github.com/facebookresearch/dinov2) (Apache-2.0) | not checked | Strong frozen RGB features. |
    | DINOv3 | Siméoni et al., 2025 | [dinov3](https://github.com/facebookresearch/dinov3) (the custom "DINOv3 License") | not checked | The licence has to be read before anything depends on it. |
    | EfficientViT | Cai et al., ICCV 2023 | [efficientvit](https://github.com/mit-han-lab/efficientvit) (Apache-2.0) | not checked | A fast encoder for real-time use. |
    | FastViT | Vasu et al., ICCV 2023 | [ml-fastvit](https://github.com/apple/ml-fastvit) (a custom Apple licence) | not checked | Custom licence; read before use. |
    | MambaVision | Hatamizadeh and Kautz, CVPR 2025 | [MambaVision](https://github.com/NVlabs/MambaVision) (NVIDIA Source Code License-NC) | not checked | Non-commercial code: cannot be a source of code here. |
    | SAM 2 image encoder | Ravi et al., 2024 | [sam2](https://github.com/facebookresearch/sam2) (Apache-2.0) | not checked | More useful as a provider of instance masks than as a scene encoder. |
