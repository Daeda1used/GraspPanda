---
hide: [toc]
---

# Model zoo

Every number on this page comes from a result record of a whole test split, scored by the
evaluator of this toolbox and recomputed by a test from its per-scene values. AP is in
percent; *Filter* says whether each method's own collision filter was off or on.

- **Authors' weights scored here** (`author-ckpt`): GraspNet-baseline, Scale-Balanced-Grasp,
  EconomicGrasp, FineGrasp, HGGD.
  Those numbers have the provenance `author-ckpt`; they are not reproductions.
- **Trained here** (`reproduced`): GSNet, EconomicGrasp and FineGrasp, from scratch with the
  recipe of their configs. EconomicGrasp lands on its authors' checkpoint, and with it just
  outside the [admission gate](#admission-gate) on seen; GSNet is above its paper by more
  than 1 AP on every split, FineGrasp below it. Their weights are not published yet, so the
  zoo cannot download them.

## GraspNet-1Billion

Protocol `graspnet1b-sv-v1`: the whole test split, 90 scenes and 23,040 frames per camera;
AP on seen / similar / novel. `author-ckpt` is the released checkpoint, converted with
`tools/convert_checkpoint.py`; `reproduced` is a model trained from scratch here with the
recipe of its config; both ran through `grasppanda test`. A `reported` number stands on a
measured line only where its source states the same collision filter; one that does not
has a line *not stated*.

<div class="gp-figures" markdown>

| Method | Camera | Filter | `reported` | `author-ckpt` | `reproduced` |
|---|---|:---:|---|---|---|
| GraspNet-baseline | RealSense | off | — | 32.63 / 27.48 / 10.42 | — |
| GraspNet-baseline | RealSense | on | 47.47 / 42.27 / 16.61 | 47.90 / 42.70 / 16.97 | — |
| GraspNet-baseline | RealSense | not stated | 27.56 / 26.11 / 10.55 | — | — |
| GSNet | RealSense | off | 65.70 / 53.75 / 23.98 | — | 68.83 / 60.35 / 25.20 |
| GSNet | RealSense | on | 67.12 / 54.81 / 24.31 | — | 71.23 / 62.29 / 26.42 |
| Scale-Balanced-Grasp | RealSense | off | 58.95 / 52.97 / 22.63 | 57.78 / 51.67 / 22.12 | — |
| Scale-Balanced-Grasp | RealSense | on | 63.83 / 58.46 / 24.63 | 62.29 / 56.81 / 23.99 | — |
| EconomicGrasp | RealSense | off | 68.21 / 61.19 / 25.48 | 69.28 / 61.50 / 25.17 | 69.34 / 61.41 / 26.21 |
| EconomicGrasp | RealSense | on | — | 70.83 / 62.56 / 25.80 | 71.05 / 62.54 / 26.81 |
| EconomicGrasp | Kinect | off | 62.59 / 51.73 / 19.54 | 63.71 / 52.28 / 19.73 | — |
| EconomicGrasp | Kinect | on | — | 65.47 / 53.64 / 20.36 | — |
| FineGrasp | RealSense | off | 71.67 / 62.83 / 27.40 | 71.56 / 62.80 / 27.42 | 69.77 / 59.72 / 25.77 |
| FineGrasp | RealSense | on | 73.71 / 64.56 / 28.14 | 73.69 / 64.62 / 28.18 | 72.43 / 61.87 / 26.94 |
| HGGD | RealSense | off | — | 58.36 / 47.95 / 22.08 | — |
| HGGD | RealSense | on | — | 63.72 / 52.91 / 24.16 | — |
| HGGD | RealSense | not stated | 59.36 / 51.20 / 22.17 | — | — |
| HGGD | RealSense | not stated | 64.45 / 53.59 / 24.59 | — | — |
| HGGD | Kinect | off | — | 56.98 / 43.97 / 18.56 | — |
| HGGD | Kinect | on | — | 60.30 / 46.85 / 19.22 | — |
| HGGD | Kinect | not stated | 60.26 / 48.59 / 18.43 | — | — |
| HGGD | Kinect | not stated | 61.17 / 47.02 / 19.37 | — | — |

</div>

??? note "Measured minus reported"

    Every line whose reported number states its collision filter.

    | Method | Camera | Filter | `author-ckpt` | `reproduced` |
    |---|---|:---:|---|---|
    | GraspNet-baseline | RealSense | on | +0.43 / +0.43 / +0.36 | — |
    | GSNet | RealSense | off | — | +3.13 / +6.60 / +1.22 |
    | GSNet | RealSense | on | — | +4.11 / +7.48 / +2.11 |
    | Scale-Balanced-Grasp | RealSense | off | −1.17 / −1.30 / −0.51 | — |
    | Scale-Balanced-Grasp | RealSense | on | −1.54 / −1.65 / −0.64 | — |
    | EconomicGrasp | RealSense | off | +1.07 / +0.31 / −0.31 | +1.13 / +0.22 / +0.73 |
    | EconomicGrasp | Kinect | off | +1.12 / +0.55 / +0.19 | — |
    | FineGrasp | RealSense | off | −0.11 / −0.03 / +0.02 | −1.90 / −3.11 / −1.63 |
    | FineGrasp | RealSense | on | −0.02 / +0.06 / +0.04 | −1.28 / −2.69 / −1.20 |

    The differences are stated here, not explained: each method's README
    (`configs/<method>/README.md`, section "Results") says for every one what was
    investigated and what was not.

??? note "What to know when reading it"

    - **Inputs derived from ground truth.** GraspNet-baseline, GSNet, Scale-Balanced-Grasp,
      EconomicGrasp and FineGrasp crop the workspace with the ground-truth segmentation at
      test time, as their references do; their records are stamped `gt_assisted: workspace`.
      HGGD reads the whole image and nothing in its input is derived from ground truth
      (`gt_assisted` is empty). The two kinds of line are not like-for-like.
    - **The filter of each reference.** `off` is the prediction as emitted (variant `raw`),
      `on` the model-free collision filter of each method's reference (variant `cd`), which
      is not one setting: voxels of 0.005 m for EconomicGrasp and 0.01 m for the others, and
      for HGGD what its released test script dumps, its own filter followed by NMS. The
      chain with all its parameters is in each record.
    - **Reported numbers without a filter.** The HGGD lines *not stated* are, for each
      camera, the "In paper" and "In repo" rows of its README, in that order. The
      GraspNet-baseline line is the paper's number as its repository README quotes it ("In
      paper"); the README says of the "In repo" row only that it adds collision detection.
      EconomicGrasp's reported numbers are read as filter `off`: neither its README nor the
      paper says, and the shipped test command leaves the filter off.
    - **One run each.** Every number is one run with seed 0. The spread over point-sampling
      seeds, and for HGGD over its random draws at inference, was not measured on whole
      splits.
    - **Empty cells.** GSNet has no `author-ckpt` number: the file released with its
      reference, a third-party implementation, is no reference for the benchmark
      (`configs/gsnet/README.md`). GraspNet-baseline, Scale-Balanced-Grasp and FineGrasp
      have lines for RealSense only.

??? note "The models trained here: the admission gate, and what is known about each gap"

    - **Against the admission gate.** The gate asks for at most 1.0 AP between `reproduced`
      and `reported` on each split. EconomicGrasp meets it on similar and novel and is 0.13 AP
      outside it on seen. GSNet and FineGrasp are outside it on every split, and neither gap
      is explained.
    - **EconomicGrasp lands on its authors' checkpoint**: +0.06 / −0.09 / +1.04 AP from it
      without the filter, +0.22 / −0.02 / +1.01 with it. That checkpoint is itself 1.07 AP
      above the reported number on seen, so what separates the training here from the
      published table is what separates the authors' own file from it. Why the table is
      lower is not known (`configs/economicgrasp/README.md`).
    - **GSNet is above the paper** on every split, most on similar. What is known: this
      implementation equals the public reference implementation stage by stage, including
      every loss term of a recorded training batch, so its training step is the reference's;
      a model trained independently with the reference code, on a modified graspness label
      set, also scores above the paper; and the paper's numbers come from the authors'
      original code, which is not public (`configs/gsnet/README.md`).
    - **FineGrasp is below the paper** on every split, most on similar. What is known: the
      authors' weights score the paper's numbers through the same test path, and a recorded
      training batch gives the reference's losses and gradients. The authors' normal images
      and graspness labels are not published, so a training here generates both from the
      paper's equations, and it ran as four processes with accumulated gradients in place of
      the eight of the published recipe (`configs/finegrasp/README.md`).
    - **One seed.** GSNet was trained three times with it; its line is the first training,
      and the three lie within 1.12 AP of one another on every split. EconomicGrasp and
      FineGrasp were trained once. Other seeds were not trained, nor was a training with
      unmodified reference code run. Each training run is described in its method's README.
    - **The weights are not published yet.** They were trained on GraspNet-1Billion (CC
      BY-NC-SA 4.0), and whether and where they are published has not been decided. The
      metafile lists them with their sha256 and no url, so the zoo, `GraspInferencer` and
      `grasppanda demo` say that they are not published instead of downloading anything; the
      training log and the run snapshot are not published either.

## GraspClutter6D transfers

The same weights, trained on GraspNet-1Billion RealSense, scored on GraspClutter6D: protocol
`graspclutter6d-sv-v1`, the whole test split, 235 scenes and 3,055 frames of the RealSense
D435; AP / AP0.8 / AP0.4. `reported` is Table IX of the GraspClutter6D paper
([arXiv 2504.06866](https://arxiv.org/abs/2504.06866) v3), which states neither the
collision filter nor the checkpoints, so it stands beside no measured number.

<div class="gp-figures" markdown>

| Method | Filter | `reported` | `author-ckpt` | `reproduced` |
|---|:---:|---|---|---|
| GraspNet-baseline | off | — | 8.70 / 10.35 / 6.97 | — |
| GraspNet-baseline | on | — | 17.14 / 20.60 / 12.87 | — |
| GraspNet-baseline | not stated | 17.95 / 21.54 / 13.53 | — | — |
| GSNet | off | — | — | 25.58 / 31.38 / 19.42 |
| GSNet | on | — | — | 26.87 / 33.00 / 20.04 |
| Scale-Balanced-Grasp | off | — | 21.76 / 26.48 / 16.45 | — |
| Scale-Balanced-Grasp | on | — | 22.69 / 27.59 / 17.00 | — |
| Scale-Balanced-Grasp | not stated | 22.69 / 27.57 / 16.95 | — | — |
| EconomicGrasp | off | — | 21.69 / 26.64 / 15.79 | 23.24 / 28.58 / 17.04 |
| EconomicGrasp | on | — | 22.25 / 27.30 / 16.06 | 23.72 / 29.16 / 17.20 |
| EconomicGrasp | not stated | 21.67 / 26.62 / 15.90 | — | — |
| FineGrasp | off | — | 12.56 / 15.47 / 9.01 | 16.12 / 19.78 / 11.53 |
| FineGrasp | on | — | 13.14 / 16.20 / 9.19 | 17.00 / 20.85 / 11.86 |
| HGGD | off | — | 14.24 / 17.40 / 9.24 | — |
| HGGD | on | — | 14.56 / 17.75 / 9.38 | — |

</div>

FineGrasp's lines are the method as published: its weights know the camera distance of
their training frames, and one setting moves the authors' from 12.56 to 24.39 AP (see the
note).

??? note "What to know when reading it"

    - **Inputs derived from ground truth.** The five seed-based methods crop the workspace
      to the image rectangle around the ground-truth segmentation, the crop of the official
      GraspClutter6D tool (`gt_assisted: workspace`).
      It is not the crop the weights were trained with; what that costs was measured on a
      subset only, see [Cross-dataset evaluation](user_guides/cross_dataset.md). HGGD reads
      the whole image (`gt_assisted` is empty) and its `on` lines add NMS to the filter, as
      on GraspNet-1Billion: its lines are not like-for-like with the others.
    - **Which `author-ckpt` line the reported number is close to differs by method**: `on` for
      GraspNet-baseline (0.81 AP below the reported value; `off` is 9.25 below) and for
      Scale-Balanced-Grasp (0.00 apart; `off` is 0.93 below), `off` for EconomicGrasp (0.02
      above; `on` is 0.58 above). Whether the paper applied a collision filter for some
      methods and not for others was not investigated and cannot be read from the paper.
    - **FineGrasp knows its camera distance.** Its encoder reads the coordinates of every
      point, and this split lies farther from the camera than any training cloud (0.75 m
      in the median against 0.42 m). With the depth coordinate of the features moved to
      the training distance, and nothing else changed, the same weights score 24.39 AP
      without the filter and 25.02 with it, and the model trained here does alike; the
      lines above are the method as published
      ([its README](https://github.com/Daeda1used/GraspPanda/tree/main/configs/finegrasp#results-on-graspclutter6d)).
    - **One run each**, with seed 0, and one training with one seed behind every
      `reproduced` number, as above. The paper has no row for GSNet, FineGrasp or HGGD.
    - **EconomicGrasp trained here transfers better than its authors' checkpoint**, by
      1.55 AP without the filter and 1.47 with it; on GraspNet-1Billion the two differ on
      novel scenes only, by about 1 AP in the same direction.
    - **The subset numbers of the user guide** (other cameras, other crops, the Kinect
      weights) were measured on 24 scenes and are not benchmark numbers.

## Admission gate

A method enters only with all of: config, weights with sha256, training log, a result record
produced by the pinned evaluator, and the run snapshot.

Gate: `|reproduced − reported| ≤ 1.0 AP` on each of seen / similar / novel. If the gate fails
the entry may still be published, with the gap, the `author-ckpt` number and the diagnosed cause
stated. Results with and without collision filtering are always separate rows.

??? note "Provenance, and where the records are"

    Every number carries its provenance, and provenances are never mixed in one column:

    | Provenance | Meaning |
    |---|---|
    | `reported` | Taken from the paper or the authors' repository. |
    | `author-ckpt` | The authors' released weights, evaluated in this toolbox. |
    | `reproduced` | Trained and evaluated in this toolbox. |

    A measured number is shown only when a result record backs it. The record of every
    `author-ckpt` and `reproduced` row is committed under `configs/<method>/records/` and
    named by the row of `configs/<method>/metafile.yaml`; it holds the per-scene values, the
    coverage, the post-processing chain, the inputs derived from ground truth, the config
    hash, the code commit and the sha256 of the weights. A test recomputes every measured
    number of the tables above from those per-scene values.

    Two directories beside it hold whole-split records that no line of this page states:
    `repeats/`, the config of a zoo row trained again (GSNet), and `variants/`, the weights
    of a zoo row with one option changed (FineGrasp's features on GraspClutter6D). The
    method's README states their numbers, and a test holds the two together.

### Recording a result

Two commands turn a finished test run into a row of this page.

??? note "The commands, and what the second one checks"

    ```bash
    grasppanda test <config> --weights <checkpoint> --opts runtime.work_dir=<work directory>
    python -m tools.record_result <work directory> --method <directory below configs/>
    ```

    The second command refuses a run that did not score every frame of the whole test split,
    that ran on uncommitted code, or whose weights cannot be tied to the file the metafile
    lists (for a converted checkpoint: the file it was converted from, which the converter
    notes in the converted file). Otherwise it copies the result records to
    `configs/<method>/records/<model>_<provenance>_<split>_<variant>.json`, with the
    directories of the machine replaced by the name of their environment variable, and
    prints the rows to paste into the metafile, the numbers computed from the per-scene
    values. Run again, it changes nothing and says whether the metafile states the rows.
    The table lines of this page and of the method's README are then written by hand; the
    tests name the lines that are missing.

    While the metafile has no weights entry for the model, say what the weights are with
    `--provenance author-ckpt` or `--provenance reproduced`; the tool then prints the entry
    to complete. Weights trained here may be listed before they are published: the entry
    then has `url: null`, `license: null` and a short `note` saying that they are not
    published yet, beside their sha256.
