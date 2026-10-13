# Experimental swap configs

**Support level: experimental. No AP is claimed for any config in this folder.**

Each file replaces one stage of a reference method, the scene encoder, and leaves the
rest of that method's reference config untouched. They exist to show what a swap looks
like when it is written down, and to be a starting point that is known to run. None has
been trained to convergence, none has been evaluated on a benchmark split, and no paper
reports a GraspNet-1Billion number for any of these combinations.

What "known to run" means here:

1. `grasppanda print-config experimental/<name> --check` passes (also checked for every
   shipped config by `tests/unit/test_repository.py`);
2. the same encoder under the same method builds, back-propagates and predicts on
   synthetic data in `tests/unit/test_swap_matrix.py`;
3. the full-size model builds and predicts on a GPU
   (`tests/integration/test_encoder_swaps.py`);
4. the config trained for a few iterations on real GraspNet-1Billion frames with real
   labels, and then predicted and was scored on validation frames (table below).

| Config | Method | Encoder | Why it is here |
|---|---|---|---|
| `gsnet_ptv3_g1b-realsense_10e` | GSNet | `PTv3` at a 5 mm grid | A dense encoder for a dense encoder: the plain one-line swap |
| `gsnet_litept_g1b-realsense_10e` | GSNet | `LitePT` (LitePT-S) at a 5 mm grid | A light encoder that convolves at fine scale and attends at coarse scale |
| `gsnet_oacnns_g1b-realsense_10e` | GSNet | `OACNNs` at a 5 mm grid | A sparse CNN for a sparse CNN, with adaptive pooling grids that are counted in cells |
| `gsnet_spunet_g1b-realsense_10e` | GSNet | `SpUNet` at a 5 mm grid | The smallest step from the reference: the same kind of network, three times as deep |
| `gsnet_pointnet2-fp_g1b-realsense_10e` | GSNet | `FeaturePropagation` around `PointNet2` | A sub-sampling encoder under a proposal that needs dense features, through the adapter the checker names |
| `gsnet_pointnext_g1b-realsense_10e` | GSNet | `PointNeXt` with a tabletop radius | A room-scale encoder whose unit of length has to be changed |
| `economicgrasp_ptv3_g1b-realsense_10e` | EconomicGrasp | `PTv3` at a 5 mm grid | The same swap under compact labels and the 5 mm matching rule |
| `graspnet-baseline_sunet14-fps_g1b-realsense_18e` | GraspNet-baseline | `FurthestPointSubsample` around `SparseUNet14` | A dense encoder under a method written for a subsample, with the reference seeds kept |

Every file starts with a comment that lists what must be re-tuned (grid or radius,
optimizer and schedule, batch size, widths) and why. The short version, for all of them:
the recipe is the reference method's, tuned by its authors for the reference encoder;
the unit of length (`grid_size`, `voxel_size`, `radius`) was set by analogy with the
reference, not searched; and the head sees other seeds and other label statistics than
under the reference encoder. See
[Swapping components](../../docs/advanced_guides/swapping.md).

## The training check

Run on 2026-10-12, one RTX A6000 shared with other jobs, PyTorch 2.7.1 + cu118,
spconv 2.3.8. Each config was trained on the `sample` split (8 frames of training
scene 0, with their real grasp, graspness, compact and tolerance labels), for 5 epochs,
then predicted on 20 frames of `val_dev` and scored by the GraspNet-1Billion evaluator:

```bash
grasppanda train experimental/<name> --opts \
    data.train.dataset.split=sample train.epochs=5 evaluation.interval=5 \
    data.val.dataset.frame_step=128 runtime.log.interval=1 \
    data.train.num_workers=2 data.val.num_workers=2 evaluation.num_workers=4
```

| Config | Parameters | Iterations | Total loss, first → last | Supervised seeds, last iteration | Wall time |
|---|---|---|---|---|---|
| `gsnet_ptv3_g1b-realsense_10e` | 46 352 303 | 10 (batch 4) | 17.37 → 2.44 | 1.00 | 171 s |
| `gsnet_litept_g1b-realsense_10e` | 12 903 031 | 10 (batch 4) | 9.68 → 2.46 | 1.00 | 86 s |
| `gsnet_oacnns_g1b-realsense_10e` | 52 317 519 | 10 (batch 4) | 5.63 → 3.01, not monotonic (see below) | 1.00 | 105 s |
| `gsnet_spunet_g1b-realsense_10e` | 39 900 527 | 10 (batch 4) | 6.01 → 1.65 (see below) | 1.00 | 86 s |
| `gsnet_pointnet2-fp_g1b-realsense_10e` | 1 140 943 | 10 (batch 4) | 5.78 → 2.17 | 1.00 | 80 s |
| `gsnet_pointnext_g1b-realsense_10e` | 967 727 | 10 (batch 4) | 8.31 → 2.73 | 1.00 | 75 s |
| `economicgrasp_ptv3_g1b-realsense_10e` | 46 748 766 | 10 (batch 4) | 20.98 → 5.45 | 0.10 | 78 s |
| `graspnet-baseline_sunet14-fps_g1b-realsense_18e` | 14 981 516 | 20 (batch 2) | 3.18 → 1.58 | 0.49 | 107 s |

What this shows: the config, the data pipeline, the label assets and the model fit
together; losses are finite and fall on eight frames; a checkpoint is written;
prediction, both post-processing variants and the evaluator run on the result. What it
does not show: anything about accuracy. The AP printed after ten iterations on eight
frames was 0.00 to 0.02 and is not a result.

Two runs show what "the recipe is not tuned" looks like. Under `OACNNs` the total loss
went 5.63, 15.78, 3.21 and then, for iterations 4 to 6, no scene point passed the
graspness threshold: the frames had no real seed (`metric/real_seeds` 0), so the view
and grasp losses were zero and only the proposal was trained. Seeds came back at
iteration 7 (0.36 of the slots) and were complete from iteration 8. Under `SpUNet` the
fraction of real seeds fell to 0.01 in iteration 3 and to 0.54 in iteration 8 and was
0.89 or more otherwise. GSNet's learning rate of 1e-3 without warm-up is too much for
the first steps of these two deep sparse CNNs (52 M and 40 M parameters). In the other
runs at least 86 % of the seed slots held a real seed at every iteration (the minimum,
0.857, is one iteration of `gsnet_ptv3`).

"Supervised seeds" is `metric/supervised_seeds`, the fraction of seeds that take part
in the grasp losses. It is 1 for GSNet by construction (no distance limit), about a
tenth for EconomicGrasp at this stage (a seed counts only within 5 mm of a label point,
and an untrained proposal puts seeds anywhere), and about half for the baseline (seeds
on an object). It is the first number to compare with a run of the reference encoder.

## Full runs

Not run here: each is a multi-hour job. The commands are the ones in the header of each
file; the assets are those of the reference method
(`grasppanda prepare graspnet1b --labels`, `--graspness --camera realsense`,
`--compact-labels`, `--tolerance`; see the method READMEs). A number from such a run has
no paper to be compared with: report it as trained here, over three seeds
(`docs/design/ARCHITECTURE.md` §11.5), next to a run of the reference encoder under the
same recipe, and label it experimental.
