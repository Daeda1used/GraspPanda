# Third-party notices

SPGrasp and its SAM2-derived source retain the author repository's Apache-2.0 `LICENSE` and `LICENSE-sam2.txt` in the local compatibility overlay. The adapter retains the native model and rasterizer, with explicit continuous-target, prompt-sampling and width-decoding corrections described in the [planar sequence guide](MODULES.md#prompted-planar-sequences). SAM2 initialization weights follow the [SAM2 terms](https://github.com/facebookresearch/sam2#license).

The root MIT license covers GraspPanda-authored code only. Downloaded repositories retain their own licenses. Origins and commits are recorded in `grasppanda/resources/upstreams.lock.json`; per-method notes are in [Methods & papers](METHODS.md).

GraspNet baseline/Graspness and derivatives may carry academic-use terms. Public source does not imply commercial permission. CGAL/scikit-geometry and Ultralytics/FastSAM include GPL/LGPL/AGPL terms; review the exact components before redistributing a combined image or wheel bundle. SDKs, checkpoints and GraspNet-1B data have separate terms.

`upstream/`, environments, checkpoints, data and local experiment outputs are excluded from Git distribution. Source-download instructions and patches do not relicense their underlying projects. No blanket commercial or binary redistribution permission is claimed.

PointNeXt/OpenPoints and FineGrasp components are fetched from pinned author sources. FineGrasp uses its native point operators with the configuration utilities from RoboOrchardCore 0.7 and scoped imports of model dependencies. The broader RoboOrchard robotics application is not installed. Consult each source's license before redistribution.

PointMLP uses the author's point hierarchy and feature-propagation blocks, plus the bundled PointNet2 operators. The installer builds these operators locally with the selected CUDA architecture. Original source and license files remain in the pinned repository; GraspPanda does not redistribute a pretrained PointMLP model.

PointVector uses its native segmentation encoder, vector aggregation blocks and feature propagation from the pinned OpenPoints repository. The adapter provides XYZ inputs, feature projection, grasp seed mapping and configurable query sizes. Original source notices remain with OpenPoints; no PointVector pretrained weights are downloaded or redistributed.

Grouped seed interaction loads `GraspGNN`, `MultiHeadAttn` and `AttentionModule` from the pinned [GCF-GraphGrasp repository](https://github.com/qzsrh/GCF-GraphGrasp). Its root MIT license carries the iSEE lab copyright notice and remains with the downloaded source. GraspPanda adds scene/depth isolation, component configuration and checkpoint mapping around those classes. This component does not redistribute the fork's bundled binaries or a pretrained model, and does not identify the fork as a separate verified publication.

ConvNeXt V2, RepViT, MobileNetV4, Lion and Muon use the locked timm implementations. Author papers and original implementation links are listed in [Modules](MODULES.md). These convolutional image encoder replacements initialize without external pretrained weights.

Sonata PTv3 is imported from the pinned Apache-2.0 author source. GraspPanda supplies input/output mappings and native attention-cache handling. The adapter does not download or redistribute the separately licensed pretrained Sonata weights.

VMamba is imported from its pinned MIT-licensed author repository, including its native selective-scan and cross-scan/merge implementations. The installer builds the CUDA operator locally; original notices remain with the downloaded source. GraspPanda adds RGB-D feature projections and scoped checkpoint/driver compatibility handling. No pretrained VMamba weights are redistributed or downloaded.

ASL single-label and Poly-1 are implemented from their documented mathematical objectives, with source references in [Training controls](MODULES.md#loss-formulations). ASL is checked against the locked timm implementation; its saturated-probability handling uses stable complements and bounded power bases. No classification datasets or pretrained loss-specific models are downloaded.

HGGD/RNG loss adapters invoke the locally downloaded, pinned native target generators with scoped loss primitives. Their sigmoid ASL and Poly-1 variants retain method-specific balancing and normalization, as described in [RGB-D training controls](MODULES.md#rgb-d-training-controls). RGB-D observation augmentation uses torchvision transforms and the native camera/input conventions. Original method source and its terms remain separate from the toolbox.

Compatibility patches are stored in `grasppanda/resources/patches/` and applied only to local build copies or overlays. The scikit-geometry patch uses its exact-construction kernel consistently in skeleton bindings; this can change numerical behavior relative to the original mixed-kernel binding. Patches retain the underlying project licenses.

DINOv2 and DINOv3 encoders use locked timm code and revision-pinned timm weight conversions, downloaded separately. DINOv2 weights retain Apache-2.0 terms; DINOv3 weights and derived checkpoints retain the [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md), including its redistribution and research-attribution requirements. GraspPanda's MIT license does not replace these terms. Sources and papers are linked in [Modules](MODULES.md#pretrained-dino-image-features).

DeepLA ResLFE blocks and CUDA operators are fetched from the [pinned author repository](https://github.com/zeng-ziyin/DeepLA-Net/tree/7f572899de7db26d2c5eac538395d9932faafb89). That revision has no project-level license declaration; its vendored timm notice does not license the whole project. GraspPanda does not redistribute the DeepLA source, compiled operators or pretrained weights, and its MIT license grants no rights to them. The local build adds current-stream/device handling. Consult the authors for use or redistribution rights beyond those explicitly granted by the source.

PointMetaBase is fetched from its pinned MIT-licensed author repository, including its OpenPoints Python fork. Its CUDA source files match the shared OpenPoints operator build; the installer verifies this equality. GraspPanda changes the Python import namespace and provides XYZ input, feature projection and original-point seed mapping. Original source notices remain with the download; no PointMetaBase pretrained weights are included.

GtG2 uses its pinned MIT-licensed model and geometry source. GPG is downloaded under its BSD-2-Clause license and built locally with an original GraspPanda Python binding; no pyGPG binding source is redistributed. GraphSAGE and GATv2 use the locked PyTorch Geometric implementation. Source notices remain in their downloaded repositories; trained ensembles and candidate data are generated locally.

PointMamba and its vendored Mamba implementation are downloaded under Apache-2.0; causal-conv1d is pinned separately under BSD-3-Clause. The installer compiles native operators into isolated namespaces. GraspPanda adds camera-coordinate grouping, sequence inverse mapping and grasp feature interpolation; original source and license notices stay in the downloads. No PointMamba pretrained weights are redistributed.

Point Cloud Mamba (PCM) is fetched from [SkyworkAI's pinned source](https://github.com/SkyworkAI/PointCloudMamba/tree/57fdb29d3cf1a977b4ed971919b6d83b5134d1dd), which has no project-level license declaration. Its vendored Mamba Apache-2.0 and causal-convolution BSD-3-Clause notices do not license the whole project. GraspPanda does not redistribute PCM source, compiled operators or pretrained weights. Its MIT license grants no rights to them; consult the authors for rights beyond those expressly granted. The installer verifies that its point operators match shared OpenPoints and builds its distinct scan/convolution ABI locally. Adapter correspondence fixes are documented in [Modules](MODULES.md#point-cloud-mamba-hierarchy).

OctFormer and its CUDA octree depthwise convolution are downloaded from pinned MIT-licensed author repositories. O-CNN is installed through the dependency lock. The installer compiles depthwise operators locally; original notices remain in the downloaded sources. GraspPanda adds metric-feature/query mapping, scene-local window padding and BatchNorm-safe checkpoint recomputation, and wires native stage controls into the segmentation constructor. No author weights, source checkout or compiled binaries are redistributed.

Point Transformer V2 uses the original mode 1 model and pointops from the pinned [MIT-licensed Pointcept source](https://github.com/Pointcept/Pointcept/blob/9f37497e4f3005c90bbbe7221b86439c29d60611/LICENSE). Original notices stay with the download. GraspPanda supplies grasp feature/row mappings, scoped grouped-linear and interpolation repairs, BatchNorm-safe checkpoint recomputation, and isolated CUDA packaging. No Pointcept source, compiled binaries or pretrained weights are redistributed.

LitePT loads the pinned [standalone author source](https://github.com/prs-eth/LitePT) under its MIT license, copyright 2025 Photogrammetry and Remote Sensing Lab. The original license remains with the downloaded checkout. Its PointROPE build uses an isolated module name and adds current-stream/device handling and a shared-memory synchronization barrier; the adapter uses non-mutating autograd outputs. The optional PyTorch rotation is the author implementation. FlashAttention is installed from its [official release](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.3) with its upstream license and a locked checksum.

OA-CNNs loads hash-checked [Pointcept source](https://github.com/Pointcept/Pointcept) under its MIT license, copyright 2023 Pointcept. Source and license remain in the downloaded checkout. The adapter isolates imports, exposes actual stem/decoder layer counts and retains sparse row correspondence; configurable normalization and lattice policies are described in [Modules](MODULES.md#oa-cnns-adaptive-sparse-hierarchy). It does not redistribute segmentation weights.

KPConvX uses the pinned [Apple author repository](https://github.com/apple/ml-kpconvx) under its MIT license, copyright 2024 Apple Inc.; its LICENSE and ACKNOWLEDGEMENTS remain with the download. GraspPanda isolates the Python imports, supplies the shared point operators, repairs kernel-cache paths and exposes documented configuration variants. Original source, trained weights and compiled third-party binaries are not redistributed.

The KPConvX cylindrical component uses the same pinned author blocks and source terms. It adds local graph packing, per-cylinder normalization, memory chunking and grasp-specific query/feature mappings. It is a local adaptation, not a separate author-released grasping method.


<details>
<summary>PointSP sampling adaptation and notices</summary>

The observation sampler implements the weighted-random and local/global downsampling rules from [PointSP](https://github.com/tangsankou/PointSP/blob/8206043f27e8b849fecde48841ff4b2513e88439/PCT_Pytorch/sampling.py). GraspPanda uses an exact CPU KD-tree, NumPy sampling, explicit point-label index maps and fixed-size padding. It does not copy the classification trainer or reproduce its full inference protocol. The repository root and nested PCT source carry the following notices.

```text
BSD 3-Clause License

Copyright (c) 2021, University of Michigan
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

```text
MIT License

Copyright (c) 2021 Strawberry-Eat-Mango

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

</details>

Network sampling uses PointSP density/filtering rules with shared Pointcept KNN and a GraspPanda CUDA masked-FPS implementation. The operator checks its inputs and device, uses one eligible start per scene, preserves row indices and specifies deterministic tie handling. It does not redistribute or execute the original time-seeded PointSP FPS kernel.

Utonia and Concerto are loaded from pinned author repositories under Apache-2.0. Their separately downloaded pretrained weights are CC-BY-NC-4.0, as declared by [Utonia](https://github.com/Pointcept/Utonia#license) and [Concerto](https://github.com/Pointcept/Concerto#license). GraspPanda preserves the source notices and does not redistribute those sources or weights in its archive. The adapter supplies missing-modality inputs, independent scene transforms/encoding, hierarchical feature lifting and a trainable grasp projection; the root MIT license does not replace the pretrained-weight terms.

Flash3D uses the original [Apache-2.0 source](https://github.com/cruise-automation/Flash3D), its pinned MIT-licensed ThunderKittens headers, Apache-2.0 Transformer Engine and BSD-3-Clause glog. Source pins are in `component_sources.lock.json`; its scoped patches and compiler manifest are grouped under `resources/flash3d/`. Patches correct native gradients, RNG initialization and CUDA compatibility without replacing the native hierarchy. The installer retains source notices and the NVIDIA redistribution archive licenses with locally built artifacts. NVIDIA compiler/runtime components remain subject to their own terms. No compiled Flash3D binaries, author model weights or compiler archives are included in the GraspPanda source download.

PointTPA adaptation uses the pinned [MIT-licensed author source](https://github.com/H-EmbodVis/PointTPA), fetched locally with its original notices. GraspPanda retains its dynamic projection, serialization grouping, zero initialization and parallel residual placement. Concerto/Utonia initialization weights retain their separate CC-BY-NC-4.0 terms when those backbones are selected.
