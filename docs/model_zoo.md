# Model zoo

No method has been trained and evaluated here to the admission rule yet, so nothing is
certified and the zoo holds no weights of its own. What has been measured so far is how
the weights **released by the authors** score in this toolbox: see
[Authors' checkpoints scored by this toolbox](#authors-checkpoints-scored-by-this-toolbox)
and [GraspClutter6D transfers](#graspclutter6d-transfers). Those numbers have the
provenance `author-ckpt`; they are not reproductions.

## Admission gate

A method enters only with all of: config, weights with sha256, training log, a result
record produced by the pinned evaluator, and the run snapshot.

Every number carries its provenance and they are never mixed in one column:

| Provenance | Meaning |
|---|---|
| `reported` | Taken from the paper or the authors' repository. |
| `author-ckpt` | The authors' released weights, evaluated in this toolbox. |
| `reproduced` | Trained and evaluated in this toolbox. |

Gate: `|reproduced − reported| ≤ 1.0 AP` on each of seen / similar / novel. If the gate
fails the entry may still be published, with the gap, the `author-ckpt` number and the
diagnosed cause stated. Results with and without collision filtering are always separate
rows.

A measured number is shown only when a result record backs it. The record of every
`author-ckpt` and `reproduced` row is committed under `configs/<method>/records/` and
named by the row of `configs/<method>/metafile.yaml`; it holds the per-scene values, the
coverage, the post-processing chain, the inputs derived from ground truth, the config
hash, the code commit and the sha256 of the weights. A test recomputes every measured
number of the tables below from those per-scene values.

### Recording a result

```bash
grasppanda test <config> --weights <checkpoint> --opts runtime.work_dir=<work directory>
python -m tools.record_result <work directory> --method <directory below configs/>
```

The second command refuses a run that did not score every frame of the whole test split,
that ran on uncommitted code, or whose weights cannot be tied to the file the metafile
lists (for a converted checkpoint: the file it was converted from, which the converter
notes in the converted file). Otherwise it copies the result
records to `configs/<method>/records/<model>_<provenance>_<split>_<variant>.json`, with
the directories of the machine replaced by the name of their environment variable, and
prints the rows to paste into the metafile, the numbers computed from the per-scene
values. Run again, it changes nothing and says whether the metafile states the rows.
The table lines of this page and of the method's README are then written by hand; the
tests name the lines that are missing.

## Targets

Author-reported AP on GraspNet-1Billion, RealSense, seen / similar / novel.

| Method | Collision filter | Reported | Reproduced |
|---|:---:|---|:---:|
| GSNet (ICCV 2021) | off | 65.70 / 53.75 / 23.98 | — |
| GSNet (ICCV 2021) | on | 67.12 / 54.81 / 24.31 | — |
| EconomicGrasp (ECCV 2024) | off | 68.21 / 61.19 / 25.48 | — |
| HGGD (RA-L 2023), "In paper" row of its README | unstated | 59.36 / 51.20 / 22.17 | — |
| HGGD (RA-L 2023), "In repo" row and arXiv version | unstated | 64.45 / 53.59 / 24.59 | — |
| Scale-Balanced-Grasp (CoRL 2022) | off | 58.95 / 52.97 / 22.63 | — |
| Scale-Balanced-Grasp (CoRL 2022) | on | 63.83 / 58.46 / 24.63 | — |
| FineGrasp (arXiv 2025) | off | 71.67 / 62.83 / 27.40 | — |
| FineGrasp (arXiv 2025) | on | 73.71 / 64.56 / 28.14 | — |

## Authors' checkpoints scored by this toolbox

GraspNet-1Billion, protocol `graspnet1b-sv-v1`: the whole test split, 90 scenes and
23,040 frames per camera, all scored. AP in percent on seen / similar / novel.

`author-ckpt` is the checkpoint the authors released, converted with
`tools/convert_checkpoint.py` and run through `grasppanda test`: `off` is the prediction
as emitted (variant `raw`), `on` the model-free collision filter of the method's
reference (variant `cd`). `reported` stands beside a measured number only where its
source states the same collision filter; a reported number whose source does not state
the filter has a line of its own and is paired with nothing. These are **not**
reproductions: nothing in this table was trained here.

| Method | Camera | Collision filter | `author-ckpt` | `reported` | `author-ckpt` − `reported` |
|---|---|:---:|---|---|---|
| GraspNet-baseline | RealSense | off | 32.64 / 27.48 / 10.42 | — | — |
| GraspNet-baseline | RealSense | on | 47.90 / 42.70 / 16.98 | 47.47 / 42.27 / 16.61 | +0.43 / +0.43 / +0.37 |
| GraspNet-baseline | RealSense | not stated | — | 27.56 / 26.11 / 10.55 | — |
| Scale-Balanced-Grasp | RealSense | off | 57.78 / 51.66 / 22.12 | 58.95 / 52.97 / 22.63 | −1.17 / −1.31 / −0.51 |
| Scale-Balanced-Grasp | RealSense | on | 62.29 / 56.80 / 24.00 | 63.83 / 58.46 / 24.63 | −1.54 / −1.66 / −0.63 |
| HGGD | RealSense | off | 58.33 / 47.96 / 22.03 | — | — |
| HGGD | RealSense | on | 63.70 / 52.90 / 24.18 | — | — |
| HGGD | RealSense | not stated | — | 59.36 / 51.20 / 22.17 | — |
| HGGD | RealSense | not stated | — | 64.45 / 53.59 / 24.59 | — |
| HGGD | Kinect | off | 56.98 / 43.97 / 18.63 | — | — |
| HGGD | Kinect | on | 60.34 / 46.83 / 19.27 | — | — |
| HGGD | Kinect | not stated | — | 60.26 / 48.59 / 18.43 | — |
| HGGD | Kinect | not stated | — | 61.17 / 47.02 / 19.37 | — |
| EconomicGrasp | RealSense | off | 69.28 / 61.50 / 25.17 | 68.21 / 61.19 / 25.48 | +1.07 / +0.31 / −0.31 |
| EconomicGrasp | RealSense | on | 70.83 / 62.56 / 25.80 | — | — |
| EconomicGrasp | Kinect | off | 63.71 / 52.28 / 19.73 | 62.59 / 51.73 / 19.54 | +1.12 / +0.55 / +0.19 |
| EconomicGrasp | Kinect | on | 65.47 / 53.64 / 20.36 | — | — |
| FineGrasp | RealSense | off | 71.56 / 62.80 / 27.42 | 71.67 / 62.83 / 27.40 | −0.11 / −0.03 / +0.02 |
| FineGrasp | RealSense | on | 73.69 / 64.62 / 28.18 | 73.71 / 64.56 / 28.14 | −0.02 / +0.06 / +0.04 |

What to know when reading it:

- **Inputs derived from ground truth.** GraspNet-baseline, Scale-Balanced-Grasp,
  EconomicGrasp and FineGrasp crop the workspace with the ground-truth segmentation at
  test time, as their references do; their records are stamped `gt_assisted: workspace`.
  HGGD reads the whole image and nothing in its input is derived from ground truth
  (`gt_assisted` is empty). The two kinds of row are not like-for-like.
- **The filter of each reference.** `on` is not one setting: every config keeps the
  collision filter of its method's reference (voxels of 0.005 m for EconomicGrasp and
  0.01 m for the others), and HGGD's `on` row is what its released test script dumps,
  its own filter followed by NMS. The chain with all its parameters is in each record.
- **Unpaired reported numbers.** The HGGD lines "not stated" are, for each camera, the
  "In paper" and "In repo" rows of its README, in that order. The GraspNet-baseline line
  is the paper's number as its repository README quotes it ("In paper"); the README says
  of the "In repo" row only that it adds collision detection. EconomicGrasp's reported
  numbers are read as filter `off`: neither its README nor the paper says, and the
  shipped test command leaves the filter off.
- **One run each.** Every number is one run with seed 0. The spread over point-sampling
  seeds, and for HGGD over its random draws at inference, was not measured on whole
  splits.
- **The differences are stated, not explained, here.** Each method's README
  (`configs/<method>/README.md`, section "Results") says for every difference what was
  investigated and what was not.
- **Not scored.** GSNet, whose reference is a third-party implementation and for which
  no checkpoint of the authors has been scored: its first row will be the model trained
  here. GraspNet-baseline, Scale-Balanced-Grasp and FineGrasp have a row for RealSense
  only.

## GraspClutter6D transfers

The same converted checkpoints, trained by their authors on GraspNet-1Billion RealSense,
scored on GraspClutter6D: protocol `graspclutter6d-sv-v1`, the whole test split, 235
scenes and 3,055 frames of the RealSense D435, all scored. AP / AP0.8 / AP0.4 in percent.
`reported` is Table IX of the GraspClutter6D paper
([arXiv 2504.06866](https://arxiv.org/abs/2504.06866) v3), which states neither the
collision filter nor the checkpoints that were scored, so it is paired with no line; it
has no row for HGGD.

| Method | Camera | Collision filter | `author-ckpt` | `reported` | `author-ckpt` − `reported` |
|---|---|:---:|---|---|---|
| GraspNet-baseline | RealSense D435 | off | 8.69 / 10.35 / 6.96 | — | — |
| GraspNet-baseline | RealSense D435 | on | 17.14 / 20.60 / 12.86 | — | — |
| GraspNet-baseline | RealSense D435 | not stated | — | 17.95 / 21.54 / 13.53 | — |
| Scale-Balanced-Grasp | RealSense D435 | off | 21.77 / 26.49 / 16.46 | — | — |
| Scale-Balanced-Grasp | RealSense D435 | on | 22.71 / 27.61 / 17.01 | — | — |
| Scale-Balanced-Grasp | RealSense D435 | not stated | — | 22.69 / 27.57 / 16.95 | — |
| HGGD | RealSense D435 | off | 14.25 / 17.41 / 9.23 | — | — |
| HGGD | RealSense D435 | on | 14.56 / 17.79 / 9.33 | — | — |
| EconomicGrasp | RealSense D435 | off | 21.69 / 26.64 / 15.79 | — | — |
| EconomicGrasp | RealSense D435 | on | 22.25 / 27.30 / 16.06 | — | — |
| EconomicGrasp | RealSense D435 | not stated | — | 21.67 / 26.62 / 15.90 | — |

- **Inputs derived from ground truth.** GraspNet-baseline, Scale-Balanced-Grasp and
  EconomicGrasp crop the workspace to the image rectangle around the ground-truth
  segmentation, the crop of the official GraspClutter6D tool (`gt_assisted: workspace`).
  It is not the crop the weights were trained with; what that costs was measured on a
  subset only, see [Cross-dataset evaluation](user_guides/cross_dataset.md). HGGD reads
  the whole image (`gt_assisted` is empty) and its `on` lines add NMS to the filter, as
  on GraspNet-1Billion: its lines are not like-for-like with the others.
- **Which measured line the reported number is close to differs by method**: `on` for
  GraspNet-baseline (0.81 AP below the reported value; `off` is 9.26 below) and for
  Scale-Balanced-Grasp (0.02 above; `off` is 0.92 below), `off` for EconomicGrasp (0.02
  above; `on` is 0.58 above). Whether the paper applied a collision filter for some
  methods and not for others was not investigated and cannot be read from the paper.
- **One run each**, with seed 0, as above.
- **No line yet**: GSNet, for which no checkpoint of the authors has been scored; the
  transfer of the model trained here enters with that model. Everything else on that
  page of the user guide (other cameras, other crops, the Kinect weights) was measured
  on the 24-scene subset and is not a benchmark number.
