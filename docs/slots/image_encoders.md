# Image encoders

The slot `model.image_encoder`: the network that turns the colour (or RGB-D) image into
feature maps. Registry `IMAGE_ENCODERS`, base class
`grasppanda.models.image_encoders.ImageEncoder`.

**No AP is claimed for anything on this page.** Every option below was built, trained for
20 iterations on eight real frames and asked to predict (`grasppanda probe`). That shows it
*works*; how good it is takes a full training run.

## 1. The task of the slot

An image encoder is the 2-D counterpart of a scene encoder: image in, a short list of feature
maps out. It knows nothing about grasps. Two stages consume it:

- **HGGD** (`model.image_encoder` of the detector). HGGD's image network is *encoder → decoder
  → five map heads*: the grasp-centre heat-map at full resolution, and at the proposal stride
  (8) the in-plane angle classes with three offsets per class. The decoder map at the proposal
  stride is also the feature map the region stage attaches to its points. Until now the
  encoder was fixed (ResNet-34 block layout at an eighth of the width, 1.6 M parameters,
  from scratch); it is now any registered image encoder.
- **The seed-based family** (GSNet, EconomicGrasp, FineGrasp, GraspNet-baseline,
  Scale-Balanced-Grasp) reads points only. The adapters `ImagePainted` and
  `ImagePaintedSubsampled` put an image encoder next to their scene encoder: every point
  reads the image features at its own pixel (point painting, section 5).

## 2. Input and output

| | |
|---|---|
| input | `image`: `(B, in_channels, H, W)`, float, **upright** (rows are image rows) |
| `input_layout` | what the channels are: `rgb` = colour in `[0, 1]`, not normalised (the encoder normalises with the statistics its weights expect); `rgbd` = the same followed by one depth channel in metres; `drgb` = HGGD's own order `[depth, R, G, B]`, the reference encoder only |
| output | a list of `(B, out_channels[i], h_i, w_i)` maps, finest first |
| `out_strides`, `out_channels` | nominal stride (`H / h_i`, rounded) and width of every map; the bridges **measure** them with one forward pass at construction |

What the neighbours require:

- **HGGD** hands over its network image, `(B, 4, U, V)` = `[depth, R, G, B]` with `u` along the
  *width* (640 × 360 by default), depth in metres with the mean removed and clipped to ±1.
  For any encoder but the reference one, `ImageNet` turns the image upright
  (`(B, C, 360, 640)`), re-orders it to the encoder's layout (colour first; depth last for
  `rgbd`, dropped for `rgb`), and turns the maps back. The maps may have **any strides and
  widths**: a pyramid decoder projects every map to `model.encoder.decoder_channels`, merges
  them top-down, resamples to exactly `image / proposal.stride` (80 × 45) and goes to full
  resolution with transposed convolutions. So a four-level pyramid, a five-level pyramid and a
  single-stride ViT all fit; nothing is refused for its strides.
- **The reference encoder** (`HGGDResidualEncoder`) keeps the reference decoder (transposed
  convolutions with additive skips, laid out for 640 × 360). With it the model is the one of
  the shipped config: same parameter names (converted checkpoints load), same seeded
  initialisation (`tests/unit/test_image_encoders.py`).
- **Point painting** needs `input_layout: rgb` (the points already carry the depth), the colour
  image, the camera matrix and points in the camera frame.

What the config checker does: `image_encoder` is a typed slot (`Cfg[ImageEncoder]`), so a
node is checked against the constructor of its class before anything is built (unknown
options, wrong value types, `depth: rgbd` instead of one of the three modes). Contract
problems that need the built model are raised at build with the way out, for example
`encoder.base_width` set next to a foreign encoder.

### RGB-D, honestly

No library ships weights that were pretrained on RGB-D, and timm's `in_chans=4` only repeats
the colour filters. So depth is an explicit option of every bridge, `depth:`

| `depth` | What the encoder sees | Pretrained weights |
|---|---|---|
| `rgb_only` (default of the presets) | colour; `in_channels` 3 | used exactly as released. HGGD's heat-map then has no depth cue; depth still reaches the region stage as points |
| `fourth_channel` | colour + depth; the first convolution (or patch embedding; all parallel stem convolutions of FastViT / RepViT) gets a fourth input | colour filters as released; the depth filters start at **zero** (`depth_init: zero`: the network starts as the colour model) or at the mean colour filter (`depth_init: mean`) and are always trained, also with `frozen: true` |
| `depth_stem` | colour + a 3-channel residual computed from depth by a small stem of its own (two convolutions, the last starts at zero) | untouched; with `frozen: true` only the stem and the consumer's decoder learn |

## 3. Options at a glance

Columns: the name to write after `--set image_encoder=`; what it is; the paper; the class that
runs it; the weights and their licence; the extra to install
(`pip install "grasppanda[timm]"`, `[transformers]`, `[torchvision]`); and what the probe
measured under HGGD on this machine (parameters of the whole detector and peak GPU memory of
the probe at batch size 4; no timings, the GPU was shared with other jobs).
Measured here: 38 `runs`. The records are the `family-slot/HGGD/image_encoder/*` cells of
[the measured file](../advanced_guides/measured_combinations.md).

Sources of the columns: licences and arXiv numbers were read in this session from timm's
`pretrained_cfg`, from the model cards on the Hugging Face hub and from torchvision's weight
enums; venues and groups are the ones recorded in the project's research report where it
marks them as checked, otherwise "venue not checked". **NC** = non-commercial weights:
supported, never shipped; you download them under their own terms.

| `--set image_encoder=` | What it is | Paper, venue, group | Class | Weights and their licence | Extra | HGGD, measured |
|---|---|---|---|---|---|---|
| `hggd-residual` | HGGD's own image encoder, written out: the reference network as a slot value | HGGD (Chen et al., RA-L 2023), THU-VCLab | `HGGDResidualEncoder` | none released for the encoder alone; the authors' HGGD checkpoints convert with `tools/convert_checkpoint.py` and load into this model. | — | runs (3.4 M, 8.0 GiB) |
| `resnet18` | ResNet-18 from timm with ImageNet-1k weights, colour only | ResNet (He et al., CVPR 2016); weights of *ResNet strikes back*, arXiv 2110.00476 | `TimmImageEncoder` | `resnet18.a1_in1k`, Apache-2.0 (timm). | `timm` | runs (13.5 M, 8.7 GiB) |
| `resnet34` | ResNet-34 from timm with ImageNet-1k weights, colour only | ResNet (He et al., CVPR 2016); weights arXiv 2110.00476 | `TimmImageEncoder` | `resnet34.a1_in1k`, Apache-2.0 (timm). | `timm` | runs (23.6 M, 9.1 GiB) |
| `resnet34-rgbd` | ResNet-34 from timm with ImageNet-1k weights and a fourth input channel for depth | as `resnet34`; the fourth channel is this bridge's | `TimmImageEncoder` | `resnet34.a1_in1k`, Apache-2.0 (timm). | `timm` | runs (23.6 M, 9.0 GiB) |
| `resnet50` | ResNet-50 from timm with ImageNet-1k weights, colour only | ResNet (He et al., CVPR 2016); weights arXiv 2110.00476 | `TimmImageEncoder` | `resnet50.a1_in1k`, Apache-2.0 (timm). | `timm` | runs (26.0 M, 10.1 GiB) |
| `convnextv2-tiny` | ConvNeXt V2-Tiny from timm (FCMAE, fine-tuned on ImageNet-22k and 1k), colour only | ConvNeXt V2 (Woo et al., CVPR 2023, arXiv 2301.00808), facebookresearch | `TimmImageEncoder` | `convnextv2_tiny.fcmae_ft_in22k_in1k`, CC BY-NC 4.0: NON-COMMERCIAL weights. **NC** | `timm` | runs (30.2 M, 10.4 GiB) |
| `convnextv2-atto` | ConvNeXt V2-Atto from timm (FCMAE, fine-tuned on ImageNet-1k), colour only | ConvNeXt V2 (Woo et al., CVPR 2023, arXiv 2301.00808), facebookresearch | `TimmImageEncoder` | `convnextv2_atto.fcmae_ft_in1k`, CC BY-NC 4.0: NON-COMMERCIAL weights. **NC** | `timm` | runs (5.7 M, 8.9 GiB) |
| `efficientnetv2-s` | EfficientNetV2-S from timm (ImageNet-21k, fine-tuned on 1k), colour only | EfficientNetV2 (Tan et al., ICML 2021, arXiv 2104.00298), Google | `TimmImageEncoder` | `tf_efficientnetv2_s.in21k_ft_in1k`, Apache-2.0 (timm). | `timm` | runs (22.2 M, 10.0 GiB) |
| `mobilenetv4-conv-small` | MobileNetV4-Conv-Small from timm with ImageNet-1k weights, colour only | MobileNetV4 (Qin et al., arXiv 2404.10518; venue not checked) | `TimmImageEncoder` | `mobilenetv4_conv_small.e2400_r224_in1k`, Apache-2.0 (timm). | `timm` | runs (3.6 M, 8.5 GiB) |
| `mobilenetv4-hybrid-medium` | MobileNetV4-Hybrid-Medium from timm with ImageNet-1k weights, colour only | MobileNetV4 (Qin et al., arXiv 2404.10518; venue not checked) | `TimmImageEncoder` | `mobilenetv4_hybrid_medium.e500_r224_in1k`, Apache-2.0 (timm). | `timm` | runs (10.9 M, 9.5 GiB) |
| `repvit-m1` | RepViT-M1 from timm with distilled ImageNet-1k weights, colour only | RepViT (Wang et al., CVPR 2024, arXiv 2307.09283), THU-MIG | `TimmImageEncoder` | `repvit_m1.dist_in1k`, Apache-2.0 (timm). | `timm` | runs (7.0 M, 9.2 GiB) |
| `fastvit-sa12` | FastViT-SA12 from timm with Apple's ImageNet-1k weights, colour only | FastViT (Vasu et al., ICCV 2023, arXiv 2303.14189), Apple | `TimmImageEncoder` | `fastvit_sa12.apple_in1k`, Apple's FastViT licence (timm: `fastvit-license`; read it before any use beyond research). | `timm` | runs (12.7 M, 9.6 GiB) |
| `fastvit-t8` | FastViT-T8 from timm with Apple's ImageNet-1k weights, colour only | FastViT (Vasu et al., ICCV 2023, arXiv 2303.14189), Apple | `TimmImageEncoder` | `fastvit_t8.apple_in1k`, Apple's FastViT licence (timm: `fastvit-license`). | `timm` | runs (5.5 M, 9.0 GiB) |
| `efficientvit-b1` | EfficientViT-B1 (MIT Han lab) from timm with ImageNet-1k weights, colour only | EfficientViT (Cai et al., ICCV 2023, arXiv 2205.14756), MIT Han lab | `TimmImageEncoder` | `efficientvit_b1.r224_in1k`, Apache-2.0 (timm). | `timm` | runs (6.9 M, 9.3 GiB) |
| `swinv2-tiny` | Swin Transformer V2-Tiny from timm with ImageNet-1k weights, colour only | Swin Transformer V2 (Liu et al., CVPR 2022, arXiv 2111.09883), Microsoft | `TimmImageEncoder` | `swinv2_tiny_window8_256.ms_in1k`, MIT (timm). | `timm` | runs (29.9 M, 10.7 GiB) |
| `maxvit-tiny` | MaxViT-Tiny from timm with ImageNet-1k weights, colour only | MaxViT (arXiv 2204.01697; venue not checked), Google | `TimmImageEncoder` | `maxvit_tiny_tf_224.in1k`, Apache-2.0 (timm). | `timm` | runs (32.6 M, 13.7 GiB) |
| `hiera-small-abswin` | Hiera-Small (absolute window positions) from timm, ImageNet-12k then 1k, colour only | Hiera (Ryali et al., ICML 2023, arXiv 2306.00989) with window-aware absolute positions (arXiv 2311.05613) | `TimmImageEncoder` | `hiera_small_abswin_256.sbb2_e200_in12k_ft_in1k`, Apache-2.0 (timm). | `timm` | runs (35.9 M, 10.7 GiB) |
| `sam2-hiera-tiny` | The image encoder of SAM 2.1 (Hiera-Tiny) from timm, colour only | SAM 2 (Ravi et al., arXiv 2408.00714; venue not checked), Meta FAIR | `TimmImageEncoder` | `sam2_hiera_tiny.fb_r896_2pt1`, Apache-2.0 (timm). | `timm` | runs (29.2 M, 10.4 GiB) |
| `dinov2-small-reg` | DINOv2 ViT-S/14 with registers from timm (LVD-142M), colour only | DINOv2 (Oquab et al., arXiv 2304.07193) with registers (Darcet et al., arXiv 2309.16588); venues not checked; Meta FAIR | `TimmImageEncoder` | `vit_small_patch14_reg4_dinov2.lvd142m`, Apache-2.0 (timm). | `timm` | runs (24.3 M, 9.2 GiB) |
| `dinov2-small-reg-depth-stem` | Frozen DINOv2 ViT-S/14 with registers (timm) and a trained depth stem in front | as `dinov2-small-reg`; the depth stem is this bridge's | `TimmImageEncoder` | `vit_small_patch14_reg4_dinov2.lvd142m`, Apache-2.0 (timm). | `timm` | runs (24.3 M, 9.0 GiB) |
| `dinov3-vits16` | DINOv3 ViT-S/16 from timm (LVD-1689M), colour only | DINOv3 (Siméoni et al., arXiv 2508.10104), Meta FAIR | `TimmImageEncoder` | `vit_small_patch16_dinov3.lvd1689m`, DINOv3 License (timm: `dinov3-license`; Meta's own hub repositories are gated, timm's mirror is not). On this machine also `checkpoint: _weights/components/dinov3_small.safetensors`. | `timm` | runs (23.9 M, 9.4 GiB) |
| `dinov3-convnext-tiny` | ConvNeXt-Tiny distilled from DINOv3 (LVD-1689M) from timm, colour only | DINOv3 (Siméoni et al., arXiv 2508.10104), ConvNeXt student, Meta FAIR | `TimmImageEncoder` | `convnext_tiny.dinov3_lvd1689m`, DINOv3 License (timm: `dinov3-license`). | `timm` | runs (30.2 M, 10.3 GiB) |
| `siglip2-base` | The image tower of SigLIP 2 Base/16 from timm (WebLI), colour only | SigLIP 2 (Tschannen et al., arXiv 2502.14786; venue not checked), Google | `TimmImageEncoder` | `vit_base_patch16_siglip_256.v2_webli`, Apache-2.0 (timm). | `timm` | runs (88.7 M, 11.4 GiB) |
| `eva02-tiny` | EVA-02 Tiny/14 from timm (masked image modelling on ImageNet-22k), colour only | EVA-02 (arXiv 2303.11331; venue not checked) | `TimmImageEncoder` | `eva02_tiny_patch14_224.mim_in22k`, MIT (timm). | `timm` | runs (7.9 M, 8.8 GiB) |
| `pe-spatial-tiny` | Perception Encoder Spatial Tiny/16 from timm, colour only | Perception Encoder (Bolya et al., arXiv 2504.13181; venue not checked), Meta FAIR | `TimmImageEncoder` | `vit_pe_spatial_tiny_patch16_512.fb`, Apache-2.0 (timm). | `timm` | runs (7.9 M, 8.4 GiB) |
| `hf-dinov2-small` | DINOv2 ViT-S/14 through transformers `AutoBackbone`, four layers, colour only | DINOv2 (Oquab et al., arXiv 2304.07193), Meta FAIR | `HFImageEncoder` | `facebook/dinov2-small`, Apache-2.0 (hub card). | `transformers` | runs (24.4 M, 9.9 GiB) |
| `hf-dinov2-small-reg` | DINOv2 ViT-S/14 with registers through transformers `AutoBackbone`, colour only | DINOv2 with registers (arXiv 2309.16588), Meta FAIR | `HFImageEncoder` | `facebook/dinov2-with-registers-small`, Apache-2.0 (hub card). | `transformers` | runs (24.2 M, 9.6 GiB) |
| `hf-hiera-tiny` | Hiera-Tiny through transformers `AutoBackbone`, colour only, at its fixed 224 x 224 | Hiera (Ryali et al., ICML 2023, arXiv 2306.00989), Meta FAIR | `HFImageEncoder` | `facebook/hiera-tiny-224-hf`, CC BY-NC 4.0: NON-COMMERCIAL weights. **NC** | `transformers` | runs (29.5 M, 7.4 GiB) |
| `hf-depth-anything-v2-small` | The encoder of Depth Anything V2 Small (a DINOv2 ViT-S/14) through transformers | Depth Anything V2 (Yang et al., arXiv 2406.09414; venue not checked) | `HFImageEncoder` | `depth-anything/Depth-Anything-V2-Small-hf`, Apache-2.0 (hub card). The Base and Large checkpoints are CC BY-NC 4.0. | `transformers` | runs (24.4 M, 9.9 GiB) |
| `hf-siglip2-base` | The image tower of SigLIP 2 Base/16 through transformers `AutoModel`, colour only | SigLIP 2 (Tschannen et al., arXiv 2502.14786), Google | `HFImageEncoder` | `google/siglip2-base-patch16-256`, Apache-2.0 (hub card). | `transformers` | runs (95.1 M, 11.1 GiB) |
| `hf-convnextv2-tiny` | ConvNeXt V2-Tiny through transformers `AutoBackbone` (ImageNet-22k), colour only | ConvNeXt V2 (Woo et al., CVPR 2023), facebookresearch | `HFImageEncoder` | `facebook/convnextv2-tiny-22k-224`, Apache-2.0 according to its hub card (timm lists the same family as CC BY-NC 4.0; check before commercial use). | `transformers` | runs (30.2 M, 11.4 GiB) |
| `hf-swinv2-tiny` | Swin Transformer V2-Tiny through transformers `AutoBackbone`, colour only | Swin Transformer V2 (Liu et al., CVPR 2022), Microsoft | `HFImageEncoder` | `microsoft/swinv2-tiny-patch4-window8-256`, Apache-2.0 (hub card). | `transformers` | runs (29.9 M, 10.2 GiB) |
| `hf-radio-v3-base` | C-RADIOv3-B (NVIDIA's agglomerative ViT-B/16) through transformers and its hub code, colour only | RADIO (Ranzinger et al., CVPR 2024, arXiv 2312.06709; RADIOv2.5, arXiv 2412.07679), NVIDIA | `HFImageEncoder` | `nvidia/C-RADIOv3-B`, NVIDIA Open Model License (hub card: `nvidia-open-model-license`; read it before any use beyond research). | `transformers` + `open_clip_torch` | runs (100.4 M, 11.1 GiB) |
| `tv-resnet34` | ResNet-34 from torchvision with its ImageNet-1k weights, colour only | ResNet (He et al., CVPR 2016) | `TorchvisionImageEncoder` | `ResNet34_Weights.IMAGENET1K_V1` from download.pytorch.org (torchvision, BSD-3-Clause code; the weights carry no separate licence statement). | `torchvision` | runs (23.6 M, 8.4 GiB) |
| `tv-convnext-tiny` | ConvNeXt-Tiny from torchvision with its ImageNet-1k weights, colour only | ConvNeXt (Liu et al., CVPR 2022, arXiv 2201.03545) | `TorchvisionImageEncoder` | `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` from download.pytorch.org (torchvision). | `torchvision` | runs (30.2 M, 10.0 GiB) |
| `tv-efficientnetv2-s` | EfficientNetV2-S from torchvision with its ImageNet-1k weights, colour only | EfficientNetV2 (Tan et al., ICML 2021) | `TorchvisionImageEncoder` | `EfficientNet_V2_S_Weights.IMAGENET1K_V1` from download.pytorch.org (torchvision). | `torchvision` | runs (22.6 M, 11.4 GiB) |
| `tv-mobilenetv3-large` | MobileNetV3-Large from torchvision with its ImageNet-1k weights, colour only | MobileNetV3 (venue not checked) | `TorchvisionImageEncoder` | `MobileNet_V3_Large_Weights.IMAGENET1K_V2` from download.pytorch.org (torchvision). | `torchvision` | runs (5.3 M, 9.1 GiB) |
| `tv-regnet-y-800mf` | RegNetY-800MF from torchvision with its ImageNet-1k weights, colour only | RegNet (venue not checked) | `TorchvisionImageEncoder` | `RegNet_Y_800MF_Weights.IMAGENET1K_V2` from download.pytorch.org (torchvision). | `torchvision` | runs (8.0 M, 9.1 GiB) |

All of them are **universal for HGGD** in the sense that matters here: the pyramid decoder
takes any strides, so the only method-specific option is `hggd-residual`, whose decoder is
laid out for HGGD's image size. None of them applies to a seed-based method directly; those
use an image encoder through point painting (section 5), which takes the same nodes with
`depth: rgb_only`.

The three bridge classes expose far more than the presets:
`TimmImageEncoder(name=<any timm model with features_only>)`,
`HFImageEncoder(name=<hub id or local directory>)`,
`TorchvisionImageEncoder(name=<resnet*, regnet*, or a model with .features>)`.

## 4. How to use an option (HGGD)

One command writes a config, a second one tries it on real frames:

```console
$ grasppanda compose hggd/hggd_r34w8_g1b-realsense_15e --set image_encoder=resnet34 -o hggd_resnet34.yaml
grasppanda compose hggd/hggd_r34w8_g1b-realsense_15e --set image_encoder=resnet34

what compose did
  • image_encoder: preset `resnet34` (TimmImageEncoder). ResNet-34 from timm with ImageNet-1k
    weights, colour only.

  - contracts: HGGD is not built on the seed-based skeleton, so there are no stage contracts to
    compare; the config check below is the verdict.

config check
  ✓ valid  (hash 9a0c0dd2272e)

✓ written hggd_resnet34.yaml
  next: grasppanda probe hggd_resnet34.yaml

$ grasppanda probe hggd_resnet34.yaml --iters 20 --frames 8
  model       HGGD · 23,639,190 parameters
  data        8 frames of the config's dataset · 4 per iteration
  device      cuda
  trained     20 of 20 iterations · 0.42 s per iteration (median, without data; data 0.27 s), measured on this machine now
  memory      peak 9.1 GiB on the GPU, this process
  losses      first → last
    loss/heatmap                696 → 2.925
    loss/angle                  9583 → 759.6
    loss/offsets                0.7543 → 0.4479
    loss/anchor                 744.3 → 2.415
    loss/anchor_offset          1.612 → 1.04
    loss                        1.103e+04 → 766.5
  predicted   2 frame(s) · 389, 385 grasps

✓ runs: trained 20 iterations on 8 frames of the dataset and predicted on 2 frame(s), with finite losses.
```

The same thing as a YAML fragment, in a config that inherits the HGGD config:

```yaml
model:
  encoder:
    decoder_channels: 64            # width of the pyramid decoder and of the region features
  image_encoder:
    type: TimmImageEncoder
    name: resnet34.a1_in1k          # any timm name: python -c "import timm; print(timm.list_models('resnet*'))"
    pretrained: true                # timm downloads; or  checkpoint: /path/to/file.safetensors
    depth: rgb_only                 # rgb_only | fourth_channel | depth_stem
    frozen: false                   # true: fixed encoder, only the depth handling and the decoder learn
```

Change one option of a preset on the command line:

```console
$ grasppanda compose hggd/hggd_r34w8_g1b-realsense_15e --set image_encoder=dinov2-small-reg \
    --opts model.image_encoder.depth=depth_stem model.image_encoder.frozen=true
```

The settings that matter, and how to choose them:

| Setting | Where | How to choose |
|---|---|---|
| `depth` | every bridge | `rgb_only` to use pretrained features as they are; `fourth_channel` for convolutional pyramids you fine-tune; `depth_stem` when the encoder is frozen |
| `frozen` | every bridge | `true` for foundation ViTs (DINOv2/v3, SigLIP 2, RADIO) and small data; batch-norm statistics stay fixed too |
| `out_indices` | timm, transformers | which levels come out. Pyramids: all (default). ViTs: timm returns the last three blocks, `HFImageEncoder` the last one; `[3, 6, 9, 12]` gives the decoder four depths to merge |
| `input_size` | every bridge | models with one fixed size (MaxViT, Hiera, EVA-02, timm's DINOv2 and SigLIP): the image is resampled to it and the model is built for it with `model_args: {img_size: [H, W]}`. Choose multiples of 32 (pyramids) or of the patch size near 360 × 640 |
| `size_multiple` | every bridge | models that take any multiple of their patch or window (DINOv3: 16; HF DINOv2: 14; SAM 2 Hiera and HF Swin V2: 32): the image is resampled up to the next multiple |
| `model_args` | every bridge | free-form arguments of the library (`timm.create_model`, `from_pretrained`, `torchvision.models.get_model`) as one mapping: `drop_path_rate`, `window_size`, `trust_remote_code` … |
| `model.encoder.decoder_channels` | HGGD | 64 by default (the reference network has 32 at the proposal stride). Wider helps a wide encoder, costs memory in the region stage |
| `model.encoder.detach_features` | HGGD | `true` (reference recipe): the region loss does not train the image network |

Pretrained weights:

- **timm**: `pretrained: true` downloads the tag's weights from the hub
  (`HF_HOME` chooses the cache). A local file: `checkpoint: /path/file.safetensors` (timm's key
  names; on this machine for example `_weights/components/dinov3_small.safetensors` for
  `vit_small_patch16_dinov3.lvd1689m`, tried here). Random weights: `pretrained: false`.
- **transformers**: `name` is a hub id or a local directory with `config.json` and weights;
  `pretrained: false` builds the architecture from its configuration alone; `config: {model_type: dinov2, …}`
  builds from a configuration you write, without any download.
- **torchvision**: `weights: DEFAULT` (or the name of a weight enum entry), `weights: null` for
  random weights, `checkpoint:` for a local state dict.
- Every probe on this page ran with the **pretrained weights named in the table**, downloaded
  in this session. Random weights were used only by the unit tests.

## 5. Point painting: image features for the seed-based methods

`ImagePainted` (wraps a `dense` scene encoder) and `ImagePaintedSubsampled` (wraps a
`subsampled` one) encode the colour image once, project every point into it with the camera
matrix, sample the feature maps there bilinearly and hand the result to the 3-D network:

- `fusion: input`: as extra input channels. The wrapped encoder's `in_channels` is what the
  detector feeds **plus** `paint_channels` (GSNet: 3 constant ones + 32 = 35).
- `fusion: output`: as a zero-initialised residual on the encoder's output features (sampled at
  the output points, so it also works after sub-sampling). `in_channels` stays as it was.

What the data pipeline delivers, and what it does not (checked in
`grasppanda/datasets/scene.py`, `grasppanda/transforms/frame.py`, `grasppanda/structures/observation.py`):

- `images: [rgb]` on a scene dataset puts the colour image into `observation.rgb`
  (`(B, 3, H, W)` in `[0, 1]`); `observation.intrinsics` is always there; points are in the
  camera frame. The pixel of a point is therefore its projection, and sub-sampling does not
  matter. This is all the adapter uses, and it needed one additive change of the skeleton:
  a scene encoder that declares `uses_observation = True` is called as
  `backbone(points, feats, observation)`.
- **Missing: painting under point-space augmentation.** A dataset that serves images refuses
  `RandomFlip` and `RandomRotation` (a moved cloud no longer projects onto its image). The
  per-point pixel index exists inside the pipeline (`PointSample.pixels`, it survives
  sampling) but is not part of `Observation`. Carrying it into the batch (one field, one line
  in `to_batch` and in `collate`) and letting the rigid transforms pass when only pixel-indexed
  image data is present would lift the restriction. Not done here: it changes the data
  contract of every method.
- **Not built: compose support.** `--set backbone=` takes encoders, not wrappers, and a
  painted config also has to change the dataset node of every split. So painting is a YAML
  fragment you write, not a compose choice; the presets `image-painted` and
  `image-painted-subsampled` under `grasppanda/presets/backbone/` hold the nodes.

The complete fragment for GSNet (a file next to the shipped configs, here
`configs/experimental/my_gsnet_painted.yaml`):

```yaml
_base_:
  - ../gsnet/gsnet_sunet14_g1b-realsense_10e.yaml

model:
  backbone:
    _delete_: true                  # replace the base config's encoder node
    type: ImagePainted              # ImagePaintedSubsampled around PointNet2 / PointNet2Transformer
    fusion: input                   # input | output
    paint_channels: 32
    image_scale: 0.5                # encode 640 x 360 instead of the 1280 x 720 sensor image
    image_encoder:                  # any image-encoder node with depth: rgb_only
      type: TimmImageEncoder
      name: resnet18.a1_in1k
      pretrained: true
      out_indices: [1, 2]           # strides 4 and 8
      depth: rgb_only
    backbone:
      type: SparseUNet14
      in_channels: 35               # 3 constant ones + 32 painted channels
      out_channels: 512
      voxel_size: 0.005

data:
  train:
    dataset:
      images: [rgb]
      transforms:                   # the base list without RandomFlip and RandomRotation
        - {type: RandomSample, num_points: 15000}
        - {type: FilterObjects, min_visible_points: 50}
        - {type: SampleGraspLabels, point_fraction: 0.25, min_points: 300, collision: zero_scores, grids: deferred}
        - {type: ConstantFeatures, channels: 3}
  val: {dataset: {images: [rgb]}}
  test: {dataset: {images: [rgb]}}
```

Measured: one such config per method (the method's own encoder wrapped, ResNet-18 with
ImageNet weights, written by a script from the shipped configs), `grasppanda probe <config>
--iters 20 --frames 8` on real GraspNet-1Billion frames:

| Method | Adapter, wrapped encoder | Verdict | Parameters | Loss first → last | Grasps on 2 frames | Peak GPU |
|---|---|---|---|---|---|---|
| EconomicGrasp | `ImagePainted(SparseUNet14)`, `fusion: input` | runs | 16,567,486 | 18.55 → 4.287 | 1024, 1024 | 2.5 GiB |
| FineGrasp | `ImagePainted(SparseUNet14)`, `fusion: input` | runs | 19,962,689 | 18.44 → 5.859 | 1024, 1024 | 12.3 GiB |
| GraspNet-baseline | `ImagePaintedSubsampled(PointNet2)`, `fusion: input` | runs | 1,717,260 | 3.489 → 1.623 | 318, 57 | 4.1 GiB |
| GraspNet-baseline | `ImagePaintedSubsampled(PointNet2)`, `fusion: output` | runs | 1,758,444 | 3.367 → 1.132 | 542, 539 | 4.1 GiB |
| GSNet | `ImagePainted(SparseUNet14)`, `fusion: input` | runs | 16,171,023 | 15.03 → 1.301 | 1024, 1024 | 2.2 GiB |
| GSNet | `ImagePainted(SparseUNet14)`, `fusion: output` | runs | 16,135,663 | 15.55 → 1.795 | 1024, 1024 | 2.2 GiB |
| Scale-Balanced-Grasp | `ImagePaintedSubsampled(PointNet2Transformer)`, `fusion: input` | runs | 2,268,492 | 2.958 → 1.727 | 8, 0 | 9.7 GiB |

An untrained Scale-Balanced-Grasp returns few grasps after 20 iterations with its own encoder
too (11 and 1 in the measured file); the column shows that prediction ran, not how well.

These probes are not cells of the measured grid (the grid composes its cells, and compose
does not write painted configs); the reports are kept under
`$GRASPPANDA_CACHE/probe_grid/wave5-enc2d-painted/`.

## 6. Pitfalls actually met

- **HGGD's image is transposed.** Its tensors are `(B, C, U, V)` with `u` along the width. A
  pretrained network fed that tensor sees a mirrored, rotated picture. `ImageNet` therefore
  turns the image upright for every encoder but its own; an encoder written for this slot
  always gets `(B, C, H, W)`.
- **`in_chans=4` of timm is not RGB-D pretraining**; it repeats the colour filters. The bridges
  do not use it (section 2).
- **Fixed-size transformers.** timm's DINOv2, EVA-02, SigLIP, MaxViT and Hiera assert the image
  size they were built for; 360 is not a multiple of 14, 16 or 32. `input_size` /
  `size_multiple` resample the image; the presets carry values that ran.
- **Swin V2 in timm** needs every stage to be a multiple of its window: 384 × 640 with windows
  of 6 × 10 cells ran (`swinv2-tiny`); the transformers implementation pads by itself
  (`hf-swinv2-tiny`).
- **Hiera**: the MAE checkpoints (`hiera_*_224.mae*` in timm, `facebook/hiera-*-224-hf`) have
  position embeddings for 224 × 224 only. `hf-hiera-tiny` runs by squeezing the image to that
  square; `hiera-small-abswin` and `sam2-hiera-tiny` take the image at its size.
- **Channels-last maps**: timm's Swin returns `(B, h, w, C)`; with 96 channels and a 96-row map
  the two cannot be told apart by shape. The bridge reads timm's declared output format.
- **Re-parameterisable stems** (FastViT, RepViT) read the image with several parallel
  convolutions; `fourth_channel` widens all of them. Convolutions are widened in place, so
  TensorFlow-style "same" padding (MaxViT, EfficientNetV2) keeps working.
- **Depth Anything V2** is a depth-estimation model on the hub, not a backbone:
  `api: depth_backbone` takes the DINOv2 encoder out of it; its backbone returns tokens, which
  the bridge reshapes.
- **Gated repositories**: Meta's own DINOv3 repositories on the hub answer 401 without an
  accepted licence and a token. timm's mirrors of the same weights are not gated and are what
  the presets use.
- **The config hash of the HGGD configs changed** although the model did not: the hash covers
  the resolved config including defaults, and the detector has one more option.
- **Shared GPU**: several probes first failed with CUDA out-of-memory errors while other jobs
  filled the card; they were probed again one at a time. The peak memory in the table is this
  process only.
- **`images` is a list that a child config replaces.** FineGrasp's dataset already serves
  `images: [depth]` for its normals; a painted FineGrasp config needs `images: [depth, rgb]`.
  With `[rgb]` alone the first batch stops with "DepthNormals needs the depth image of …;
  configure the dataset with `images: [depth]`".
- **A nested encoder node gets its defaults filled in by the config loader**, so a wrapper
  cannot inject `in_channels` into the encoder it wraps; the painted adapters read it from the
  wrapped node instead (section 5).

## 7. Not wrapped

- **MambaVision** ([nvidia/MambaVision-T-1K](https://huggingface.co/nvidia/MambaVision-T-1K),
  weights also on this machine). Its hub code stops with "requires the following packages
  that were not found in your environment: mamba_ssm", a compiled CUDA extension; the card's
  licence is `nvclv1` (non-commercial). Tried through `HFImageEncoder(api: model,
  trust_remote_code)`; not built here.
- **Meta's gated DINOv3 repositories**
  ([facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)):
  401 without a token. `HFImageEncoder` loads them once you have access; the presets use
  timm's mirrors instead.
- **timm's MAE Hiera** (`hiera_tiny_224.mae_in1k_ft_in1k`): cannot be built for another image
  size (position-embedding size mismatch on load). Use `hiera-small-abswin`.
- **RGB-D fusion encoders** (two streams with fusion at every stage; the research report
  of this project (local notes, section 5.D) lists CMX, CMNeXt, TokenFusion,
  GeminiFusion, DFormer and others; none was opened in this session): they are not a library
  model behind one `forward(image)`, and each needs its own adapted source and licence check.
  The slot contract (`input_layout: rgbd`) is ready for one.
- **torchvision's transformers** (Swin, ViT, MaxViT): `TorchvisionImageEncoder` cuts chains of
  stages only and says so; timm has the same architectures.
- **Decoder injection for point painting** (image features added at every level of the 3-D
  decoder; DITR in the same report, section 5.G, which records its code as AGPL-3.0; not
  opened in this session): it needs the wrapped encoder to expose its levels. Input and
  output fusion are what is offered.
