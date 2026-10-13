# GSNet

> [Graspness Discovery in Clutters for Fast and Accurate Grasp Detection](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.html) (ICCV 2021)

GSNet predicts a point-wise *graspness* score, samples seeds only where it is high,
classifies one of 300 approach views per seed, crops a cylinder along that view and
regresses grasp score and width over a grid of in-plane angles and depths.

## Components

| Slot | Component | Notes |
|---|---|---|
| `backbone` | `SparseUNet14` | 14-layer sparse U-Net on spconv, 5 mm voxels, per-point 512-d features |
| `proposal` | `GraspnessFPS` | objectness and graspness per point, then furthest point sampling to 1024 seeds |
| `view` | `TemplateViewStage` | 300 template views; a sampled view in training, the arg-max in inference |
| `assigner` | `NearestLabelAssigner` | nearest label point per seed, labels at the chosen view |
| `grouper` | `CylinderGroup` | one cylinder per seed among the seeds, radius 0.05 m, 16 neighbours |
| `head` | `SWADHead` | score and width grids over 12 angles and 4 depths |

The full default model is [`configs/_base_/models/gsnet.yaml`](../_base_/models/gsnet.yaml).
Options marked "reference quirk" there reproduce behaviour of the reference
implementation that changes numbers.

## Equivalence with the reference implementation

Checked with `tools/make_gsnet_fixtures.py` and `tests/integration/test_gsnet_parity.py`
against [graspnet/graspness_unofficial](https://github.com/graspnet/graspness_unofficial)
at `b5abf5a`, using weights converted with `tools/convert_checkpoint.py`:

| What | Result |
|---|---|
| Parameters | 15,353,775, equal to the reference |
| Seeds and chosen views | identical on 72 of 72 recorded frames |
| Backbone features, view scores, head outputs | within 5e-6 relative |
| Decoded grasps | score within 1.1e-6, width within 9e-8 m, rotation within 1.2e-7 |
| Training batch: every loss term | within 2e-7 relative |

This shows that the same weights give the same predictions. It is not an accuracy
claim: see the results below.

The same holds on the whole test split (90 scenes, 23,040 frames, RealSense). One
checkpoint was run through the reference code (`tools/oracle/gsnet_reference.py`) and
through `grasppanda test`; both sets of predictions were scored with GraspPanda's
evaluator, which equals the official one frame by frame
(`tests/integration/test_eval_parity.py`):

| Pipeline | Collision filter | Seen | Similar | Novel |
|---|:---:|---|---|---|
| reference code | off | 69.79 | 60.83 | 25.52 |
| GraspPanda | off | 69.78 | 60.88 | 25.52 |
| reference code | on | 72.04 | 62.85 | 26.60 |
| GraspPanda | on | 72.09 | 62.93 | 26.62 |

The two differ by at most 0.08 AP; they draw their 15,000 input points with different
random generators. The checkpoint is not the authors': it was trained with the reference
code on a modified graspness label set, which is why these numbers are evidence about the
pipeline and do not appear in the results table.

## Results on GraspNet-1Billion

AP on seen / similar / novel. Inputs follow the reference: the workspace crop is derived
from the ground-truth segmentation at test time (`gt_assisted: workspace`).

| Config | Camera | Collision filter | Provenance | Seen | Similar | Novel |
|---|---|:---:|---|---|---|---|
| [gsnet_sunet14_g1b-realsense_10e](gsnet_sunet14_g1b-realsense_10e.yaml) | RealSense | off | reported (paper) | 65.70 | 53.75 | 23.98 |
| | RealSense | on | reported (paper) | 67.12 | 54.81 | 24.31 |
| | RealSense | off / on | reproduced | pending | pending | pending |

The model trained here from scratch is not finished; this row is filled in, with its
weights, log and result records, only when it is.

## Usage

```bash
grasppanda prepare graspnet1b --labels
grasppanda prepare graspnet1b --graspness --camera realsense
grasppanda train gsnet/gsnet_sunet14_g1b-realsense_10e
grasppanda test  gsnet/gsnet_sunet14_g1b-realsense_10e --weights <checkpoint>
```

## Notes

- The reference implementation is third-party; no official training code exists.
- The reference test script enables its collision filter by default, while its
  published numbers are without it. Both variants are always evaluated here.
- Point-wise graspness labels must come from `grasppanda prepare`. Label files produced
  by other generators are row-aligned to a preprocessing they do not record.

## Citation

```bibtex
@inproceedings{wang2021graspness,
  title     = {Graspness Discovery in Clutters for Fast and Accurate Grasp Detection},
  author    = {Wang, Chenxi and Fang, Hao-Shu and Gou, Minghao and Fang, Hongjie and Gao, Jin and Lu, Cewu},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2021}
}
```
