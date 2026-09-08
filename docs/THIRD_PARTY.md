# Third-party notices

The root MIT license covers GraspPanda-authored code only. Downloaded repositories retain their own licenses. Origins and commits are recorded in `grasppanda/resources/upstreams.lock.json`; per-method notes are in [Methods & papers](METHODS.md).

GraspNet baseline/Graspness and derivatives may carry academic-use terms. Public source does not imply commercial permission. CGAL/scikit-geometry and Ultralytics/FastSAM include GPL/LGPL/AGPL terms; review the exact components before redistributing a combined image or wheel bundle. SDKs, checkpoints and GraspNet-1B data have separate terms.

`upstream/`, environments, checkpoints, data and local experiment outputs are excluded from Git distribution. Source-download instructions and patches do not relicense their underlying projects. No blanket commercial or binary redistribution permission is claimed.

PointNeXt/OpenPoints and FineGrasp components are fetched from pinned author sources. FineGrasp uses its native point operators with the configuration utilities from RoboOrchardCore 0.7 and scoped imports of model dependencies. The broader RoboOrchard robotics application is not installed. Consult each source's license before redistribution.

PointMLP uses the author's point hierarchy and feature-propagation blocks, plus the bundled PointNet2 operators. The installer builds these operators locally with the selected CUDA architecture. Original source and license files remain in the pinned repository; GraspPanda does not redistribute a pretrained PointMLP model.

ConvNeXt V2, RepViT, MobileNetV4, Lion and Muon use the locked timm implementations. Author papers and original implementation links are listed in [Modules](MODULES.md). Image encoder replacements initialize without external pretrained weights.

Sonata PTv3 is imported from the pinned Apache-2.0 author source. GraspPanda supplies input/output mappings and native attention-cache handling. The adapter does not download or redistribute the separately licensed pretrained Sonata weights.

ASL single-label and Poly-1 are implemented from their documented mathematical objectives, with source references in [Training controls](MODULES.md#loss-formulations). ASL is checked against the locked timm implementation; its saturated-probability handling uses stable complements and bounded power bases. No classification datasets or pretrained loss-specific models are downloaded.

Compatibility patches are stored in `grasppanda/resources/patches/` and applied only to local build copies or overlays. The scikit-geometry patch uses its exact-construction kernel consistently in skeleton bindings; this can change numerical behavior relative to the original mixed-kernel binding. Patches retain the underlying project licenses.
