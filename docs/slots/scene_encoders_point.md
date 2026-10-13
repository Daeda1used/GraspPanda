# Scene encoders: point-based networks

Options for the slot `model.backbone` of the seed-based methods that work on the raw
point cloud: set abstraction (PointNet++ and its descendants), residual MLPs, graph
convolutions, random-sampling networks and point-wise networks on a grid pyramid.
Sparse-voxel CNNs and serialised transformers have pages of their own.

Everything on this page comes from existing code: a library bridge, source adapted from
a permissively licensed repository, or released weights for a class that already
existed. **No grasp AP is claimed for any option here.** What was checked is stated per
option: the contract on tiny clouds in the unit suite, and `grasppanda compose` plus
`grasppanda probe` on real GraspNet-1Billion frames under the five seed-based methods.

## 1. The task of the slot

The scene encoder is the first learned stage of a seed-based detector (GSNet,
EconomicGrasp, FineGrasp, GraspNet-baseline, Scale-Balanced-Grasp). It turns one
depth-camera view of a cluttered table, 15 000 to 20 000 points, into a feature vector
per point. Everything after it reads those features: the proposal picks seeds from
them, the view stage chooses an approach direction per seed, the grouper crops the
scene around each seed and adds the seed's feature, and the head predicts the grasp.
The encoder decides what a seed "knows" about its surroundings before any crop is
taken, so its receptive field and its resolution are what matter for this task.

```text
depth frame -> points (N, 3) -> [ backbone ] -> proposal -> view -> grouper -> head -> grasps
                                  this slot
```

HGGD is a detector family of its own (image encoder plus region network) and has no
scene point encoder to replace.

## 2. Input and output

```python
class Backbone(nn.Module):                # grasppanda/models/base.py
    output: "dense" | "subsampled"        # where the returned features live
    in_channels: int
    out_channels: int
    def forward(self, points: PointSet, feats: Tensor) -> PointFeatures: ...
```

| | |
|---|---|
| `points` | A `PointSet`: `coords` of shape `(N, 3)`, **metres, camera frame**, all clouds of the batch concatenated, with batch offsets (`points.sizes`, `points.batch_index`). Clouds of a batch may differ in size. Point spacing on a table top is about 5 mm. |
| `feats` | `(N, in_channels)`. GSNet and EconomicGrasp feed constant ones (`point_features: ones`, 3 channels); FineGrasp feeds coordinates and normals (6 channels); GraspNet-baseline and Scale-Balanced-Grasp feed nothing (`in_channels: 0`). With `point_features: ones` the detector builds `ones(N, backbone.in_channels)`, so any `in_channels` works. |
| result, `output = "dense"` | `PointFeatures(feats, points)`: one row per input point, bound to the *same* `PointSet`. |
| result, `output = "subsampled"` | `PointFeatures(feats, points.subset(index))`: features on a child set that records which input rows it holds; every cloud of the batch must yield the same number of rows (the seeds). |
| `out_channels` | Read by the detector after the encoder is built and injected into proposal, view stage and grouper as their `in_channels`. |

What the neighbours require, and what the config checker does with it:

- `GraspnessFPS` (GSNet, EconomicGrasp, FineGrasp) scores every scene point, so it
  requires `dense`. `SubsampledSeeds` (GraspNet-baseline, Scale-Balanced-Grasp) takes
  the encoder's own subsample as seeds, so it requires `subsampled`.
- The checker compares `backbone.output` with `proposal.requires` and refuses a
  mismatch. `grasppanda compose` then inserts the adapter that converts:
  `FeaturePropagation` (subsampled to dense, one interpolation level) or
  `FurthestPointSubsample` (dense to subsampled, keeps `npoint` furthest-point samples),
  and says so in its output.
- When an encoder has an `out_channels` option, compose sets it to the width of the
  encoder it replaces (512 for the sparse U-Net methods, 256 for the PointNet++ ones),
  so the other stages keep their width. When the width is fixed by the architecture,
  the other stages are built that wide.
- One thing the checker cannot see: the crop of Scale-Balanced-Grasp adds the seed
  feature to a 256-channel crop feature (`seed_residual: gated`), so the encoder must
  return 256 channels. An encoder of another width passes the check and fails when the
  model is built, with the fix in the message (rows marked "fails at build" below).
- No contract checks the unit of length. Every published configuration of these
  networks is for rooms (first radius 0.1 m, grids from 4 cm). The presets below
  restate radii and grids for a table top and say what they were derived from; all of
  them are choices by analogy, not tuned values.

## 3. Options at a glance

"Name" is what you write after `--set backbone=`. Route: **L** library bridge, **A**
adapted source, **W** released weights for an existing class. All start from random
weights unless section 5 says otherwise.

| Name | Class | What it is | Paper | Code and licence | Extra | Output, channels |
|---|---|---|---|---|---|---|
| `pointnet2-msg` | `PointNet2MSG` | PointNet++ with two balls per level (multi-scale grouping), GraspNet-baseline level sizes | PointNet++, Qi et al., NeurIPS 2017 | A: [facebookresearch/votenet](https://github.com/facebookresearch/votenet)@2f6d6d3, MIT | none | subsampled, 1024 seeds, 256 |
| `pointnet2-msg-dense` | `PointNet2MSGDense` | the same with four propagation levels back to every point | as above | as above; layout of [erikwijmans/Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)@b5ceb6d, Unlicense | none | dense, 256 |
| `pointnext`, `pointnext-b`, `pointnext-l`, `pointnext-xl` | `PointNeXt` (existing class) | the four published sizes S / B / L / XL | PointNeXt, Qian et al., NeurIPS 2022 | class already in the toolbox | none | dense, 32 (XL: 64) |
| `pointmetabase`, `pointmetabase-xl` | `PointMetaBase` | PointNeXt skeleton; offsets embedded once per stage and added, expansion-1 blocks | Meta Architecture for Point Cloud Analysis, Lin et al., CVPR 2023 | A: [linhaojia13/PointMetaBase](https://github.com/linhaojia13/PointMetaBase)@c364a67, MIT | none | dense, 32 (XL: 64) |
| `pointvector`, `pointvector-xl` | `PointVector` | PointNeXt skeleton; blocks aggregate neighbours as 3-vectors | PointVector, Deng et al., CVPR 2023 (University of Science and Technology of China, per the file header) | A: [guochengqian/openpoints](https://github.com/guochengqian/openpoints)@bc778ec, MIT ("official implementation of PointVector") | none | dense, 32 (XL: 64) |
| `assanet` | `ASSANet` | PointNet++ with separable, anisotropic set abstraction | ASSANet, Qian et al., NeurIPS 2021 spotlight | A: openpoints@bc778ec, MIT | none | dense, 256 (published: 64) |
| `pointmlp` | `PointMLP` | residual MLP hierarchy with geometric affine, part-segmentation form | PointMLP, Ma et al., ICLR 2022 | A: [ma-xu/pointMLP-pytorch](https://github.com/ma-xu/pointMLP-pytorch)@1ab7f04, Apache-2.0 | none | dense, `out_channels` |
| `dela` | `DeLA` | offsets encoded once per stage, then MLPs and edge max-pooling on a grid pyramid | DeLA, Chen et al., arXiv 2308.16532 (2023; no venue on the arXiv record) | A: [Matrix-ASC/DeLA](https://github.com/Matrix-ASC/DeLA)@0f427c3, MIT | none | dense, 256 |
| `dgcnn`, `dgcnn-static` | `DGCNN` | three EdgeConv layers at full resolution; graph rebuilt in feature space (`dgcnn`) or kept spatial (`dgcnn-static`) | DGCNN, Wang et al., ACM Transactions on Graphics 2019 | A: [antao97/dgcnn.pytorch](https://github.com/antao97/dgcnn.pytorch)@07d534c, MIT | none | dense, `out_channels` |
| `randlanet` | `RandLANet` | random subsampling, local spatial encoding, attentive pooling | RandLA-Net, Hu et al., CVPR 2020 oral | A: [isl-org/Open3D-ML](https://github.com/isl-org/Open3D-ML)@a3d0f7a, MIT | none | dense, `out_channels` |
| `pyg-pointnet`, `pyg-edgeconv`, `pyg-point-transformer`, `pyg-pointgnn` | `PyGPointEncoder` | a stack of one PyTorch Geometric point convolution at full resolution | PointNet++ (NeurIPS 2017), DGCNN (TOG 2019), Point Transformer (ICCV 2021), Point-GNN (CVPR 2020) layers; library: Fey and Lenssen, ICLR 2019 workshop | L: `torch_geometric` (MIT) | `torch_geometric` | dense, `out_channels` |
| weights | `PointNeXt`, `PointNet2`, `PointNet2Dense` | released checkpoints that fit key for key, section 5 | PointNeXt; VoteNet (Qi et al., ICCV 2019); Contrastive Scene Contexts (Hou et al., CVPR 2021) | W: `tools/convert_backbone_weights.py` | none | as the class |

Where each fact was read, on 2026-10-13. Venues: the comment field of the arXiv records
(PointNeXt, PointVector, ASSANet, PointMLP, RandLA-Net, VoteNet, Contrastive Scene
Contexts), the proceedings pages (PointNet++ at NeurIPS 2017, Point Transformer at ICCV
2021, Point-GNN at CVPR 2020), the PointMetaBase README ("CVPR 2023") and the citation
entry of the DGCNN README. Licences and commits: the licence file and head of each
clone. Research groups are not listed because no affiliation page was opened; the
repository owner in the code column is the nearest verified fact.

### Measured fit per method

From the measured records (`docs/advanced_guides/measured_combinations.md`, where the
commands and numbers of every cell are): `runs` = 20 training iterations on 8 real
frames and a prediction on 2, with finite losses and real seeds; "in adapter" = compose
wrapped the encoder in the named adapter first.

<!-- fit-table:start -->
| `--set backbone=` | GSNet | EconomicGrasp | FineGrasp | GraspNet-baseline | Scale-Balanced-Grasp | peak GPU memory of the probes that ran, GiB |
|---|---|---|---|---|---|---|
| `pointnet2-msg` | runs in `FeaturePropagation` | runs in `FeaturePropagation` | runs in `FeaturePropagation` | runs | runs | 4 to 16 |
| `pointnet2-msg-dense` | runs | runs | runs | runs in `FurthestPointSubsample` | runs in `FurthestPointSubsample` | 3 to 14 |
| `pointnext` | runs | runs | runs | runs in `FurthestPointSubsample` | fails at build: encoder width is not 256 | 2 to 12 |
| `pointnext-b` | runs | runs | runs | runs in `FurthestPointSubsample` | fails at build: encoder width is not 256 | 3 to 13 |
| `pointnext-l` | runs | runs | runs | runs in `FurthestPointSubsample` | fails at build: encoder width is not 256 | 4 to 16 |
| `pointnext-xl` | runs | runs | runs | runs in `FurthestPointSubsample` | fails at build: encoder width is not 256 | 8 to 28 |
| `pointmetabase` | runs | runs | runs | runs in `FurthestPointSubsample` | fails at build: encoder width is not 256 | 3 to 14 |
| `pointmetabase-xl` | runs | runs | runs | runs in `FurthestPointSubsample` | fails at build: encoder width is not 256 | 6 to 21 |
| `pointvector` | runs | runs | runs | runs in `FurthestPointSubsample` | fails at build: encoder width is not 256 | 4 to 16 |
| `pointvector-xl` | runs | runs | runs | runs in `FurthestPointSubsample` | fails at build: encoder width is not 256 | 8 to 28 |
| `assanet` | runs | runs | runs | runs in `FurthestPointSubsample` | runs in `FurthestPointSubsample` | 2 to 11 |
| `pointmlp` | runs | runs | runs | runs in `FurthestPointSubsample` | runs in `FurthestPointSubsample` | 7 to 28 |
| `dela` | runs | runs | runs | runs in `FurthestPointSubsample` | runs in `FurthestPointSubsample` | 7 to 25 |
| `dgcnn` | runs | runs | runs | runs in `FurthestPointSubsample` | runs in `FurthestPointSubsample` | 8 to 30 |
| `dgcnn-static` | runs | runs | runs | runs in `FurthestPointSubsample` | runs in `FurthestPointSubsample` | 9 to 36 |
| `randlanet` | runs | runs | runs | runs in `FurthestPointSubsample` | runs in `FurthestPointSubsample` | 3 to 15 |
| `pyg-edgeconv` | runs | runs | runs | runs in `FurthestPointSubsample` | runs in `FurthestPointSubsample` | 5 to 20 |
| `pyg-point-transformer` | runs | runs | runs | runs in `FurthestPointSubsample` | runs in `FurthestPointSubsample` | 6 to 21 |
| `pyg-pointgnn` | runs | runs | runs | runs in `FurthestPointSubsample` | runs in `FurthestPointSubsample` | 9 to 34 |
| `pyg-pointnet` | runs | runs | runs | runs in `FurthestPointSubsample` | runs in `FurthestPointSubsample` | 4 to 18 |
<!-- fit-table:end -->

100 cells: 92 `runs`, 8 `fails` at build. All options are **universal** in the sense of
the contract: each runs under every seed-based method, directly or in the adapter
compose inserts, except where the row says "fails at build" (a fixed output width other
than 256 under Scale-Balanced-Grasp, see section 2). None is specific to one method. Which of them is *good* for grasping is not
known. According to the research notes of this project (not re-opened for this page),
the point-based encoders with published scene-level grasp results are PointNet++
(GraspNet-baseline; multi-scale grouping in Contact-GraspNet and GPR), PointNeXt
(vMF-Contact, on its own simulated data) and EdgeConv graphs (GraNet).

## 4. How to use each option

The pattern is the same for all of them:

```bash
grasppanda compose --list backbone                        # what exists
grasppanda compose --show backbone=pointmetabase          # one preset with its notes
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e \
    --set backbone=pointmetabase -o gsnet_pointmetabase.yaml
grasppanda probe gsnet_pointmetabase.yaml                 # trains 20 iterations on 8 frames
grasppanda train gsnet_pointmetabase.yaml
```

Real output of the compose step (2026-10-13):

```text
slot      component             from
────────  ────────────────────  ──────────────────────
backbone  PointMetaBase         preset `pointmetabase`
proposal  GraspnessFPS          base config
view      TemplateViewStage     base config
assigner  NearestLabelAssigner  base config
grouper   CylinderGroup         base config
head      SWADHead              base config

contracts
  ✓ backbone · proposal
  ...
config check
  ✓ valid  (hash 7c25e7980f79)
```

Under a method whose proposal needs the other kind of output, compose adds the adapter
and says what it did (`grasppanda compose
graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e --set backbone=randlanet`):

```text
backbone  RandLANet in FurthestPointSubsample  preset `randlanet`, in the adapter FurthestPointSubsample

  • wrapped RandLANet in the adapter FurthestPointSubsample (preset `furthest-point-subsample`):
    SubsampledSeeds requires 'subsampled' scene features, but RandLANet returns 'dense' ones (one
    feature per scene point). An adapter changes the model; its notes below say how.
  • FurthestPointSubsample.npoint: 1024, the number of seeds of the proposal
    (`model.proposal.num_seeds`): the points the adapter keeps are the seeds.
```

Options are overridden with `--opts model.backbone.<option>=<value>`; the path addresses
the encoder wherever an adapter put it. In a config written by hand the fragment is the
`model.backbone` node of the preset, for example:

```yaml
model:
  backbone:
    type: PointMetaBase
    in_channels: 3        # constant ones
    radius: 0.04
```

### PointNet++ with multi-scale grouping: `pointnet2-msg`, `pointnet2-msg-dense`

```bash
grasppanda compose graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e \
    --set backbone=pointnet2-msg -o baseline_msg.yaml
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e \
    --set backbone=pointnet2-msg-dense -o gsnet_msg.yaml
```

```yaml
model:
  backbone:
    type: PointNet2MSG
    in_channels: 0
    npoints: [2048, 1024, 512, 256]
    radii: [[0.02, 0.04], [0.05, 0.1], [0.1, 0.2], [0.2, 0.3]]    # one list per level
    nsamples: [[32, 64], [16, 32], [16, 16], [16, 16]]
```

- `PointNet2MSG` is a drop-in for the GraspNet-baseline encoder: the same level sizes,
  the same 1024 seeds and 256 channels, two balls per level. Use it where the method
  takes seeds from the encoder; `PointNet2MSGDense` where the proposal scores every
  point.
- `radii`, `nsamples` and `sa_mlps` take one list per level with one entry per scale; a
  level's width is the sum over its scales. `fp_mlps` (coarsest first) decides where the
  output lives: `k` entries stop `k` levels above the coarsest.
- Choosing radii: the larger ball of a level is what a single-scale network would use;
  the smaller one helps where points are dense (near the camera). Cost grows with the
  sum of `nsamples` over the scales.

### PointNeXt sizes: `pointnext`, `pointnext-b`, `pointnext-l`, `pointnext-xl`

```bash
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set backbone=pointnext-b -o gsnet_pointnext-b.yaml
```

| preset | `blocks` | `width` | `sa_layers` / `sa_use_res` | parameters with the official segmentation head |
|---|---|---|---|---|
| `pointnext` (S) | 1, 1, 1, 1, 1 | 32 | 2 / true | 0.792 M |
| `pointnext-b` | 1, 2, 3, 2, 2 | 32 | 1 / false | 3.828 M |
| `pointnext-l` | 1, 3, 5, 3, 3 | 32 | 1 / false | 7.125 M |
| `pointnext-xl` | 1, 4, 7, 4, 4 | 64 | 1 / false | 41.576 M |

The presets set `radius: 0.04` (room value: 0.1). Depth costs ball queries: every
inverted residual block runs one on the reference operators. Released weights: section 5.

### PointMetaBase and PointVector: `pointmetabase[-xl]`, `pointvector[-xl]`

```bash
grasppanda compose economicgrasp/economicgrasp_sunet14_g1b-realsense_10e \
    --set backbone=pointmetabase-xl -o economicgrasp_pointmetabase-xl.yaml
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e \
    --set backbone=pointvector -o gsnet_pointvector.yaml
```

```yaml
model:
  backbone:
    type: PointVector          # or PointMetaBase
    in_channels: 3
    radius: 0.04
    blocks: [1, 4, 7, 4, 4]    # XL; the default 1-3-5-3-3 is L
    width: 64
```

- Both take the arguments of `PointNeXt` (`blocks`, `width`, `strides`, `radius`,
  `radius_scaling`, `nsample`). PointMetaBase also has `expansion` (1 as published);
  PointVector has `block_nsample` (8, fixed in the official code).
- PointMetaBase is the cheap one: its blocks share one ball query and one offset
  embedding per stage. The unit suite pins its size to the published ones (2.738 M for
  L, 15.284 M for XL, 19.672 M for XXL = `blocks: [1, 5, 9, 5, 5], width: 64`, with the
  segmentation head).
- PointVector here is the scene network of the official code (`flag: 1`: sum over
  neighbours, ReLU). The object-classification variants of that code are not built.
- Output width is `width` (32 or 64): under Scale-Balanced-Grasp see section 6.

### ASSANet: `assanet`

```bash
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set backbone=assanet -o gsnet_assanet.yaml
```

- `radii`, `nsamples`, `mlps`: per level and scale, as for multi-scale PointNet++, but
  the two scales of a level run one after the other. `reduction`: `mean` (published),
  `max` or `sum`. `fp_mlps` lists the propagation widths finest first; the first entry
  is the output width (the preset uses 256, the published network 64).
- The unit suite pins the default network to the published size (2.425 M with the
  segmentation head).

### PointMLP: `pointmlp`

```bash
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set backbone=pointmlp -o gsnet_pointmlp.yaml
```

- Neighbourhoods are by count (`k_neighbors`), anchors by ratio (`reducers`): nothing to
  convert from rooms to a table top.
- `gmp_dim` appends a max-pooled vector of the whole frame to every point (0 turns it
  off). `embed_dim: 32` with `res_expansion: 0.25` is the authors' small setting.
- It reads the coordinates itself as its first three input channels; with FineGrasp it
  also receives coordinates and normals as features (the official model reads
  coordinates and normals).

### DeLA: `dela`

```bash
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set backbone=dela -o gsnet_dela.yaml
```

```yaml
model:
  backbone:
    type: DeLA
    in_channels: 3
    grid_size: [0.005, 0.01, 0.02, 0.04]   # first entry: the point spacing of the input
```

- `grid_size[0]` is the spacing of the input cloud and sets the unit of the offset
  encoding; the other entries are the cells the cloud is subsampled to. Keep the factor
  of 2 between levels of the official pyramid unless you also change `k`.
- Level sizes depend on the frame, so DeLA is `dense` only; under GraspNet-baseline and
  Scale-Balanced-Grasp compose wraps it in `FurthestPointSubsample`.
- Not built: the auxiliary relative-coordinate loss and the stochastic depth of the
  official recipe (a backbone of this slot returns features only).

### DGCNN: `dgcnn`, `dgcnn-static`

```bash
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set backbone=dgcnn -o gsnet_dgcnn.yaml
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set backbone=dgcnn-static \
    --opts model.backbone.k=48 -o gsnet_dgcnn-static_k48.yaml
```

- `k`: neighbours per point. `dynamic: true` rebuilds the graph in feature space for
  layers 2 and 3 (the paper); `false` keeps the spatial graph. `emb_dims`: width of the
  frame-level max-pooled vector, 0 to leave it out. `knn_chunk`: queries per block of
  the neighbour search (memory against speed).
- `coords_as_features: true` (default) puts the coordinates in front of the input
  features. Leave it on with `point_features: ones`, otherwise the network sees
  constants only.
- No subsampling: memory is proportional to `N x k`. Under FineGrasp the probes peaked
  at 30 GiB (`dgcnn`) and 36 GiB (`dgcnn-static`, `k: 32`); under GSNet at 8 to 11 GiB.

### RandLA-Net: `randlanet`

```bash
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set backbone=randlanet -o gsnet_randlanet.yaml
```

- `sub_sampling_ratio` and `dim_output` take one entry per level; `num_neighbors` is by
  count. Among the cheapest encoders on this page (3 to 4 GiB in the probes under GSNet,
  EconomicGrasp and GraspNet-baseline).
- Sampling is random in training and a fixed permutation in evaluation (`seed`), so
  inference is repeatable and a frame gives the same result alone or in a batch.
- A cloud needs at least `num_neighbors x prod(ratios[:-1])` points (1024 at the
  defaults).

### PyTorch Geometric layers: `pyg-pointnet`, `pyg-edgeconv`, `pyg-point-transformer`, `pyg-pointgnn`

```bash
pip install "grasppanda[torch_geometric]"
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e \
    --set backbone=pyg-point-transformer -o gsnet_pyg-pt.yaml
```

```yaml
model:
  backbone:
    type: PyGPointEncoder
    in_channels: 3
    conv: point_transformer     # pointnet | edge | point_transformer | pointgnn
    channels: [64, 64, 128]     # one layer per entry
    k: 16
    conv_args: {}               # further arguments of the library layer
```

| preset | `conv` | library layer |
|---|---|---|
| `pyg-pointnet` | `pointnet` | `torch_geometric.nn.PointNetConv` |
| `pyg-edgeconv` | `edge` | `torch_geometric.nn.EdgeConv` |
| `pyg-point-transformer` | `point_transformer` | `torch_geometric.nn.PointTransformerConv` |
| `pyg-pointgnn` | `pointgnn` | `torch_geometric.nn.PointGNNConv` (all `channels` equal) |

Only the layers are the library's. The stack around them is not a published network: it
is the quickest way to try a message-passing layer in a grasp detector. The graph is
built by GraspPanda, so `torch_geometric` alone is enough (no `pyg-lib`,
`torch_cluster`). Without the library the class says how to install it:

```text
The backbone PyGPointEncoder needs 'torch_geometric.nn', which is not installed. Install it with: pip install "grasppanda[torch_geometric]"
```

## 5. Released weights that fit existing classes

`tools/convert_backbone_weights.py` cuts the encoder out of a released checkpoint,
checks it **strictly** against the class (every tensor present, right shapes; anything
left over must be a part the format is known to carry) and writes a file whose keys
start with `backbone.`. Nothing is downloaded by the tool or by a test.

| `--format` | Source file | Fits | Tensors | Trained on | Code licence, weight licence | Size |
|---|---|---|---|---|---|---|
| `pointnext-s` (`-b`, `-l`, `-xl`) | `..._ckpt_best.pth` in the S3DIS folders linked from [PointNeXt `docs/modelzoo.md`](https://github.com/guochengqian/PointNeXt/blob/master/docs/modelzoo.md) (Google Drive) | `PointNeXt(in_channels=4, ...)` of that size | S: 106 | S3DIS area 5, rooms at 4 cm, colour + height | MIT; weights: not stated | S: 9.6 MB |
| `votenet` | `pretrained_votenet_on_sunrgbd.tar`, `pretrained_votenet_on_scannet.tar` in `votenet_pretrained_models.zip` linked from the [VoteNet README](https://github.com/facebookresearch/votenet) (Google Drive) | `PointNet2(in_channels=1, radii=[0.2, 0.4, 0.8, 1.2])` | 96 | SUN RGB-D single-view scans; ScanNet | MIT; weights: not stated | 11.5 MB each |
| `csc-pointnet2` | `partition8_4096_15k_pointnet.pth`, the PointNet++ pre-training linked from the [Contrastive Scene Contexts README](https://github.com/facebookresearch/ContrastiveSceneContexts) | `PointNet2(radii=[0.2, 0.4, 0.8, 1.2])`, i.e. the GraspNet-baseline layout, coordinates only | 96 (the file's `fp3`, `fp4` are left behind) | ScanNet frame pairs, self-supervised | MIT; weights: not stated | 6.4 MB |
| `csc-pointnet2-dense` | the same file | `PointNet2Dense` with the GraspNet-baseline level sizes and `fp_mlps` 256-256 / 256-256 / 256-128 / 128-32 | 120 | as above | as above | as above |

The dataset terms of S3DIS, SUN RGB-D and ScanNet apply to weights trained on them; none
of the files states a licence of its own, so treat them as research-only until you have
checked.

What was verified on 2026-10-13 with the real files: `pointnext-s`, both `votenet` files
and both `csc` formats convert and load with `strict=True`, and a converted file loads
into a full detector through the engine (`matched 96, unexpected 0, mismatched 0`, the
66 missing tensors are the stages after the encoder). The B / L / XL formats use the
same code path with the options the unit suite pins to the published sizes; their files
(46 to 500 MB) were not downloaded. **Activations were not compared with the upstream
implementations**, so the classes keep their status (`PointNeXt`: `tested`).

```bash
python -m tools.convert_backbone_weights --list
python -m tools.convert_backbone_weights partition8_4096_15k_pointnet.pth csc_pointnet2.pth \
    --format csc-pointnet2
```

```text
wrote 96 tensors to csc_pointnet2.pth
source sha256: a6812612720295ff2625537aa1c5c88c0ba9748de2e4b4f501c84e86730c482e
model.backbone: {'type': 'PointNet2', 'radii': [0.2, 0.4, 0.8, 1.2]}
```

Then start a training from it; the other stages start from random weights and the key
report lists them as missing:

```bash
grasppanda train graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e \
    --opts runtime.load_from=csc_pointnet2.pth runtime.allow_weight_mismatch=true
```

Read before you rely on such an initialisation:

- **Radii are not stored in a checkpoint.** The VoteNet and CSC networks were trained
  with room radii (0.2 to 1.2 m). They load under the GraspNet-baseline radii (0.04 to
  0.3 m) without complaint; because offsets are divided by the radius, the network then
  sees a table top enlarged about five times. Whether that helps is untested. Printing
  the room radii, as the tool does, is the honest default; choosing table-top radii is
  an experiment.
- **Input channels are part of the weights.** PointNeXt checkpoints read colour and a
  height above the floor (4 channels); VoteNet reads one height channel. A camera-frame
  table-top cloud has no floor height: define what you feed (zero, or height above the
  table plane) and keep `in_channels` as printed.
- The integration test `tests/integration/test_backbone_released_weights.py` repeats
  the strict loading when the files lie in
  `$GRASPPANDA_WEIGHTS/components/released_backbones/`, and is skipped otherwise.
- Some research checkpoints pickle their experiment config next to the weights (the CSC
  file holds an `omegaconf` object). The tool reads tensors with PyTorch's restricted
  loader and replaces every other class the file names by an inert stand-in; nothing of
  the file is executed.

## 6. Pitfalls met while building this page

- **Scale-Balanced-Grasp needs 256 channels from the encoder.** `pointnext*`,
  `pointmetabase*` and `pointvector*` end on their stem width (32 or 64); the composed
  config passes the check and fails at build:
  `DepthCylinderGroup: the gated seed residual adds 32-channel seed features to
  256-channel crop features; the two must be equal ... End mlp_channels on 32, or use an
  encoder that returns 256 channels, or set seed_residual: none.` The presets of the
  encoders whose width is free (`assanet`, `pointnet2-msg-dense`, and every class with
  `out_channels`) are 256 wide for this reason.
- **Probing Scale-Balanced-Grasp needs its width histogram.** Without
  `--opts data.train.dataset.scale_statistic.source=legacy_import
  data.train.dataset.scale_statistic.reference_sha256=<sha>` the probe falls back to a
  synthetic scene and reports `wired`, not `runs`
  (`docs/user_guides/combinations.md`, "The width histogram").
- **Constant input features starve encoders that read geometry through features.**
  With `point_features: ones`, an EdgeConv or a point-wise MLP sees constants. `DGCNN`,
  `PyGPointEncoder`, `RandLANet`, `PointMLP` therefore read the coordinates themselves;
  the set-abstraction encoders get geometry through grouped offsets and do not need it.
- **Absolute coordinates.** RandLA-Net's published neighbourhood encoding contains the
  absolute positions of both points, and DGCNN / PointMLP / the bridge read camera-frame
  coordinates as features: their features are not translation invariant.
- **Frame-level vectors.** `dgcnn` (`emb_dims`) and `pointmlp` (`gmp_dim`) append a
  max-pooled vector of the whole frame to every point, so features depend on the
  workspace crop. Both can be switched off.
- **Padded balls under mean or sum.** A ball query pads a sparse ball by repeating its
  first hit. Under max-pooling that is invisible; `assanet` (mean) and `pointvector`
  (sum) count the repeated point several times, as their upstream code does.
- **`torch_geometric` 2.8 without `pyg-lib`.** Its `knn_graph` raises
  `'knn_graph' requires 'pyg-lib>=0.6.0'`; the bridge builds the graph itself.
- **Memory.** The last column of the table in section 3 is the peak of the probes. The
  upper end is always FineGrasp, whose own stages are heavy; there the full-resolution
  encoders (`dgcnn*`, `pyg-pointgnn`) reach 30 to 36 GiB. On a GPU shared with other
  jobs these probes ran out of memory several times before they went through alone.

## 7. Not wrapped

| Candidate | Opened | Why not |
|---|---|---|
| KPConv (ICCV 2019), rigid and deformable | [HuguesTHOMAS/KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch) (MIT per the GitHub licence record; model files read), [Open3D-ML](https://github.com/isl-org/Open3D-ML) (MIT) | Not a thin adaptation: it needs kernel-point dispositions (an optimisation whose results the upstream caches as files), radius neighbourhoods with a shadow point, a grid pyramid with barycentre subsampling, all computed in the upstream data pipeline with compiled code; the deformable form also owns loss terms, for which the slot has no channel. A follow-up of its own. |
| KPConvX / KPConvD (CVPR 2024) | licence file of [apple/ml-kpconvx](https://github.com/apple/ml-kpconvx) (MIT, Copyright 2024 Apple Inc.); the code was not read | Same prerequisites as KPConv according to the research notes. Not attempted. |
| RandLA-Net, authors' repository | licence file of [QingyongHu/RandLA-Net](https://github.com/QingyongHu/RandLA-Net) | CC BY-NC-SA 4.0 (and TensorFlow); the MIT implementation of Open3D-ML was adapted instead. |
| Contact-GraspNet's PointNet++ MSG layout and weights | hyper-parameters as reported in the research notes | NVIDIA Source Code License (non-commercial): no code taken. Its three-scale layout can be written as a `PointNet2MSG` config by hand; its checkpoints were not converted (the TensorFlow original and the PyTorch port are two different networks). |
| `torch_geometric.nn.XConv` (PointCNN), `GravNetConv` | `torch_geometric` 2.8.1 | They build their neighbour graph inside the layer through `knn` / `knn_graph`, which needs `pyg-lib` in this version of the library. |
| PointNeXt checkpoints of the Hugging Face organisation `torch-pointcloud` | research report only | Third-party conversions with their own key layout; the official files fit directly. |
| DeepLA-Net (CVPR 2025), LinNet (NeurIPS 2024), PointMixer (ECCV 2022) | research report only | No licence file in the repositories according to the research notes; not opened further. |
| SPoTr, X-3D, HPENet V2, PAConv, RepSurf, PVCNN, DeepGCN, BAAF-Net | research report only | Not reached in this wave. All are MIT or Apache-2.0 according to the research notes; SPoTr, X-3D and HPENet V2 are forks of the same code base as `PointMetaBase` and would follow its pattern. |
