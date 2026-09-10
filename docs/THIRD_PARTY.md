# Third-party notices

SPGrasp and its SAM2-derived source retain the author repository's Apache-2.0 `LICENSE` and `LICENSE-sam2.txt` in the local compatibility overlay. The adapter retains the native model and rasterizer, with explicit continuous-target, prompt-sampling and width-decoding corrections described in the [planar sequence guide](REFERENCE.md#prompted-planar-sequences). SAM2 initialization weights follow the [SAM2 terms](https://github.com/facebookresearch/sam2#license).

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

ASL single-label and Poly-1 are implemented from their documented mathematical objectives, with source references in [Training controls](REFERENCE.md#loss-formulations). ASL is checked against the locked timm implementation; its saturated-probability handling uses stable complements and bounded power bases. No classification datasets or pretrained loss-specific models are downloaded.

HGGD/RNG loss adapters invoke the locally downloaded, pinned native target generators with scoped loss primitives. Their sigmoid ASL and Poly-1 variants retain method-specific balancing and normalization, as described in [RGB-D training controls](REFERENCE.md#rgb-d-training-controls). RGB-D observation augmentation uses torchvision transforms and the native camera/input conventions. Original method source and its terms remain separate from the toolbox.

Compatibility patches are stored in `grasppanda/resources/patches/` and applied only to local build copies or overlays. The scikit-geometry patch uses its exact-construction kernel consistently in skeleton bindings; this can change numerical behavior relative to the original mixed-kernel binding. Patches retain the underlying project licenses.

DINOv2 and DINOv3 encoders use locked timm code and revision-pinned timm weight conversions, downloaded separately. DINOv2 weights retain Apache-2.0 terms; DINOv3 weights and derived checkpoints retain the [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md), including its redistribution and research-attribution requirements. GraspPanda's MIT license does not replace these terms. Sources and papers are linked in [Modules](REFERENCE.md#pretrained-dino-image-features).

DeepLA ResLFE blocks and CUDA operators are fetched from the [pinned author repository](https://github.com/zeng-ziyin/DeepLA-Net/tree/7f572899de7db26d2c5eac538395d9932faafb89). That revision has no project-level license declaration; its vendored timm notice does not license the whole project. GraspPanda does not redistribute the DeepLA source, compiled operators or pretrained weights, and its MIT license grants no rights to them. The local build adds current-stream/device handling. Consult the authors for use or redistribution rights beyond those explicitly granted by the source.

PointMetaBase is fetched from its pinned MIT-licensed author repository, including its OpenPoints Python fork. Its CUDA source files match the shared OpenPoints operator build; the installer verifies this equality. GraspPanda changes the Python import namespace and provides XYZ input, feature projection and original-point seed mapping. Original source notices remain with the download; no PointMetaBase pretrained weights are included.

GtG2 uses its pinned MIT-licensed model and geometry source. GPG is downloaded under its BSD-2-Clause license and built locally with an original GraspPanda Python binding; no pyGPG binding source is redistributed. GraphSAGE and GATv2 use the locked PyTorch Geometric implementation. Source notices remain in their downloaded repositories; trained ensembles and candidate data are generated locally.

PointMamba and its vendored Mamba implementation are downloaded under Apache-2.0; causal-conv1d is pinned separately under BSD-3-Clause. The installer compiles native operators into isolated namespaces. GraspPanda adds camera-coordinate grouping, sequence inverse mapping and grasp feature interpolation; original source and license notices stay in the downloads. No PointMamba pretrained weights are redistributed.

Point Cloud Mamba (PCM) is fetched from [SkyworkAI's pinned source](https://github.com/SkyworkAI/PointCloudMamba/tree/57fdb29d3cf1a977b4ed971919b6d83b5134d1dd), which has no project-level license declaration. Its vendored Mamba Apache-2.0 and causal-convolution BSD-3-Clause notices do not license the whole project. GraspPanda does not redistribute PCM source, compiled operators or pretrained weights. Its MIT license grants no rights to them; consult the authors for rights beyond those expressly granted. The installer verifies that its point operators match shared OpenPoints and builds its distinct scan/convolution ABI locally. Adapter correspondence fixes are documented in [Modules](REFERENCE.md#point-cloud-mamba-hierarchy).

OctFormer and its CUDA octree depthwise convolution are downloaded from pinned MIT-licensed author repositories. O-CNN is installed through the dependency lock. The installer compiles depthwise operators locally; original notices remain in the downloaded sources. GraspPanda adds metric-feature/query mapping, scene-local window padding and BatchNorm-safe checkpoint recomputation, and wires native stage controls into the segmentation constructor. No author weights, source checkout or compiled binaries are redistributed.

Point Transformer V2 uses the original mode 1 model and pointops from the pinned [MIT-licensed Pointcept source](https://github.com/Pointcept/Pointcept/blob/9f37497e4f3005c90bbbe7221b86439c29d60611/LICENSE). Original notices stay with the download. GraspPanda supplies grasp feature/row mappings, scoped grouped-linear and interpolation repairs, BatchNorm-safe checkpoint recomputation, and isolated CUDA packaging. No Pointcept source, compiled binaries or pretrained weights are redistributed.

LitePT loads the pinned [standalone author source](https://github.com/prs-eth/LitePT) under its MIT license, copyright 2025 Photogrammetry and Remote Sensing Lab. The original license remains with the downloaded checkout. Its PointROPE build uses an isolated module name and adds current-stream/device handling and a shared-memory synchronization barrier; the adapter uses non-mutating autograd outputs. The optional PyTorch rotation is the author implementation. FlashAttention is installed from its [official release](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.3) with its upstream license and a locked checksum.

OA-CNNs loads hash-checked [Pointcept source](https://github.com/Pointcept/Pointcept) under its MIT license, copyright 2023 Pointcept. Source and license remain in the downloaded checkout. The adapter isolates imports, exposes actual stem/decoder layer counts and retains sparse row correspondence; configurable normalization and lattice policies are described in [Modules](REFERENCE.md#oa-cnns-adaptive-sparse-hierarchy). It does not redistribute segmentation weights.

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

PointCNN++ uses the [Pointelligence author implementation](https://github.com/robbyant-research/pointelligence), under Apache-2.0, and its pinned NVIDIA CUTLASS dependency under the BSD 3-Clause license. The installer retains their license notices and Pointelligence's legal notice in the local artifact. Integration changes namespace Python imports, map Triton registration to the native Torch 2.5 API, remove the external Pointcept registry requirement, expose residual geometry constants as configuration, and rebuild decoder residual neighborhoods instead of reusing cross-resolution upsampling indices. Author convolution kernels remain intact.

RALA loads the pinned author's `segmentation/mmseg/models/backbones/RALA.py`. The downloaded segmentation tree retains its [Apache-2.0 license and MMSegmentation notice](https://github.com/qhfan/RALA/blob/a86314f62103c21a1c7e6d826fe4cb0f5586e7b0/segmentation/LICENSE). This notice concerns that subtree; no repository-wide license is inferred for the separate classification/detection directories. GraspPanda removes framework registration and checkpoint-loader imports at load time, retaining the native numerical implementation. RGB-D projections, configuration, gradient checkpointing and normalization-statistic controls are toolbox adaptations. No author weights are bundled.

PointHR is fetched from the author's pinned source repository; no explicit license file was found in that revision. The GraspPanda MIT license covers the adapter only and does not grant rights over the downloaded PointHR source. No author source or weights are bundled in the distribution. The adapter retains native multi-resolution attention and fusion, uses matching shared Pointcept CUDA operators, and adds effective decoder-width controls, mean fusion, safe interpolation and checkpoint handling as described in [Modules](REFERENCE.md#pointhr-multi-resolution-point-features).

PointRWKV is fetched from the [author repository](https://github.com/hithqd/PointRWKV) at its pinned revision, which has no repository-level license file. Author source and weights are not bundled; the adapter license does not grant rights over that source. The adapter uses the released model's recurrence, local graph and three feature-propagation stages, adds attribute-aware patch encoding and configurable widths, and replaces the shape-category head with grasp features. Its optional chunk recurrence preserves the released state updates. Source-versus-paper differences are documented in [Modules](REFERENCE.md#pointrwkv-released-code-hierarchy).

Swin3D is fetched from [Microsoft's pinned author repository](https://github.com/microsoft/Swin3D) under its MIT license. The native artifact retains that license. The adapter uses the released sparse attention, convolution, pooling and interpolation operators, with runtime, indexing, tie-selection and gradient-checkpointing compatibility changes described in [Modules](REFERENCE.md#swin3d-sparse-window-hierarchy). Author pretrained RGB segmentation checkpoints are not bundled or automatically loaded.


SP2T is fetched from the [pinned author repository](https://github.com/WallelWan/SP2T) under its MIT license, which remains in the prepared runtime. GraspPanda retains the native Point representation, sparse proxy map/reduce operations, local attention and hierarchy. The [component reference](REFERENCE.md#sp2t-sparse-proxy-hierarchy) describes runtime corrections and grasp input adaptations. Segmentation weights are linked upstream and are not bundled as grasp weights.

<details>
<summary>Classification objective sources and MbLS notice</summary>

LogitNorm and LogitClip are implemented as local PyTorch formulations of their published equations. The original trainers are used for numerical comparison, not downloaded by the installer or redistributed. Their reviewed repositories have no project-level license declaration; GraspPanda's MIT license does not grant rights to those repositories. Source revisions, parameters and the LogitClip threshold/scale distinction are linked in the [loss reference](REFERENCE.md#logit-normalization-margin-penalties-and-clipping).

MbLS follows the author's constant-alpha objective, adapted to native grasp masks and per-term reductions. Its MIT notice is retained below.

```text
MIT License

Copyright (c) 2022 Bingyuan Liu

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

Quality BCE, Varifocal and MAL use local PyTorch formulas with explicit grasp-label mappings and reductions. Their detached focusing weights and powered MAL targets are checked against the [Apache-2.0 DEIM source](https://github.com/Intellindust-AI-Lab/DEIM/blob/09d35d53d39ee3145a1e61e3a989b28b9468d1dd/LICENSE), copyright INTELLINDUST INFORMATION TECHNOLOGY (SHENZHEN) CO., LTD. and affiliates. The DEIM detector, matcher and source checkout are not redistributed or installed; [quality-head documentation](REFERENCE.md#quality-score-heads) identifies the adaptation and original papers. Native grasp modules retain their existing source terms.


MambaVision uses the pinned [NVIDIA author source](https://github.com/NVlabs/MambaVision/tree/7860a506b2eb844eaaae676f08461ce8c3c26f43) and author Hugging Face weights under the [NVIDIA Source Code License-NC](https://github.com/NVlabs/MambaVision/blob/7860a506b2eb844eaaae676f08461ce8c3c26f43/LICENSE), restricted to non-commercial research/evaluation. The toolbox's MIT license does not replace those terms. Source notices remain with the installer download; model code and weights are not redistributed in the GraspPanda archive. The adapter binds the native blocks to RGB-D grasp projections, maps HF layer-scale names, and makes SDPA evaluation respect dropout mode. The shared selective-scan operator and autograd interface use the already pinned PointMamba/Mamba source and its notices.

PRIME photometric augmentation loads the pinned [Apache-2.0 author implementation](https://github.com/amodas/PRIME-augmentations) (ECCV 2022). GraspPanda supplies full-resolution PIL/CPU integration and a chunked adaptation of the smooth-color calculation. The loaded random-filter identity impulse is centered, and explicitly fixed mixture depths remain fixed. Spatial diffeomorphisms and the classification JSD objective are not applied to grasp training. Sources and their notices are fetched locally; no classifier weights are needed. The following license also applies to the adapted color calculation in `grasppanda/training/prime.py`.

<details>
<summary>PRIME adaptation: Apache License 2.0</summary>

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

```

</details>

<details>
<summary>Scale-Balanced-Grasp sampling and clean-scene preparation</summary>

The NcM loader and DSN network are loaded from the pinned [author implementation](https://github.com/mahaoxiang822/Scale-Balanced-Grasp). The toolbox adapts its object-balanced allocation, clustering, per-object mixing and CAD preparation with explicit camera, cache and degenerate-input handling. The source notice is retained below.

```text
MIT License

Copyright (c) [2022] [Haoxiang Ma]

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

Contact-score refinement imports the pinned Generalizing-Grasp networks and selected geometry definitions at runtime; its auxiliary weight downloads remain with the author source terms. That repository has no project-level license declaration. GraspPanda does not redistribute those networks or weights, and its MIT license grants no additional rights to them. The predicted-instance, coordinate and optimization adaptations are described in the [refinement reference](REFERENCE.md#contact-score-refinement).
