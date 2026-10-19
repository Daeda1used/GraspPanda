# Experimental swap configs

**Support level: experimental. No AP is claimed for any config in this folder.**

Each file is one method's shipped config with one stage swapped and nothing else changed. It
shows what a swap looks like when it is written down, and it starts from a combination that
trained on real frames. None has been trained to convergence or evaluated on a benchmark split,
and no paper reports a number for any of these combinations.

## One per method

| Method | Config | Swap | Probe record |
|---|---|---|---|
| GraspNet-baseline | `graspnet-baseline_sunet14-fps_g1b-realsense_18e` | scene encoder `sparse-unet14`, in the adapter `FurthestPointSubsample` | [`runs`](https://daeda1used.github.io/GraspPanda/explorer/#m=GraspNet-baseline&backbone=sparse-unet14) · `encoder/GraspNet-baseline/SparseUNet14` |
| GSNet | `gsnet_ptv3_g1b-realsense_10e` | scene encoder `ptv3` | [`runs`](https://daeda1used.github.io/GraspPanda/explorer/#m=GSNet&backbone=ptv3) · `encoder/GSNet/PTv3@ptv3` |
| Scale-Balanced-Grasp | `sbg_ptv3_g1b-realsense_18e` | scene encoder `ptv3`, in the adapter `FurthestPointSubsample` | [`runs`](https://daeda1used.github.io/GraspPanda/explorer/#m=Scale-Balanced-Grasp&backbone=ptv3) · `encoder/Scale-Balanced-Grasp/PTv3@ptv3` |
| EconomicGrasp | `economicgrasp_ptv3_g1b-realsense_10e` | scene encoder `ptv3` | [`runs`](https://daeda1used.github.io/GraspPanda/explorer/#m=EconomicGrasp&backbone=ptv3) · `encoder/EconomicGrasp/PTv3@ptv3` |
| FineGrasp | `finegrasp_ptv3_g1b-realsense_10e` | scene encoder `ptv3`, reading FineGrasp's six point features | [`runs`](https://daeda1used.github.io/GraspPanda/explorer/#m=FineGrasp&backbone=ptv3) · `encoder/FineGrasp/PTv3@ptv3` |
| HGGD | `hggd_dinov2-small-reg-depth-stem_g1b-realsense_15e` | image encoder `dinov2-small-reg-depth-stem` | [`runs`](https://daeda1used.github.io/GraspPanda/explorer/#m=HGGD&image_encoder=dinov2-small-reg-depth-stem) · `family-slot/HGGD/image_encoder/dinov2-small-reg-depth-stem` |

A probe record is one short training run: 20 iterations on 8 RealSense frames of one
GraspNet-1Billion scene, then a prediction. `runs` means the combination trained and predicted
without error, not that it is accurate. The config hash of each file is the one its record was
probed with (a unit test checks it). Any other combination under any method is one command away
([Compose and probe](https://daeda1used.github.io/GraspPanda/user_guides/combinations/)), and
the [explorer](https://daeda1used.github.io/GraspPanda/explorer/) shows every record.

<details markdown="1"><summary>Five more scene encoders under GSNet</summary>

These were the first swaps written down, all under GSNet. Each shows another kind of encoder.

| Config | Encoder | Why it is here | Probe record |
|---|---|---|---|
| `gsnet_litept_g1b-realsense_10e` | `LitePT` (LitePT-S) at a 5 mm grid | A light encoder that convolves at fine scale and attends at coarse scale | `runs` · `encoder/GSNet/LitePT@litept` |
| `gsnet_oacnns_g1b-realsense_10e` | `OACNNs` at a 5 mm grid | A sparse CNN for a sparse CNN, with adaptive pooling grids that are counted in cells | `runs-unstable` · `encoder/GSNet/OACNNs` |
| `gsnet_spunet_g1b-realsense_10e` | `SpUNet` at a 5 mm grid | The smallest step from the reference: the same kind of network, three times as deep | `runs` · `encoder/GSNet/SpUNet@spunet` |
| `gsnet_pointnet2-fp_g1b-realsense_10e` | `FeaturePropagation` around `PointNet2` | A sub-sampling encoder under a proposal that needs dense features, through the adapter the checker names | `runs` · `encoder/GSNet/PointNet2` |
| `gsnet_pointnext_g1b-realsense_10e` | `PointNeXt` with a tabletop radius | A room-scale encoder whose unit of length has to be changed | `runs-empty` · `encoder/GSNet/PointNeXt@pointnext` |

`runs-unstable`: it trained and predicted, but a loss or a weight became non-finite or an
iteration had no real seed. `runs-empty`: it trained without error but predicted no grasp on
the probe frames. Both are starting points to tune, not combinations that run as they are.

</details>

<details markdown="1"><summary>What each file has passed</summary>

1. `grasppanda print-config experimental/<name> --check` passes (checked for every shipped
   config by `tests/unit/test_repository.py`);
2. it is what `grasppanda compose` writes today for the command in its header, and its config
   hash is the one of its probe record (`tests/unit/test_combine_compose.py`);
3. its probe record above;
4. the full-size model builds and predicts one synthetic frame on a GPU
   (`tests/integration/test_encoder_swaps.py`);
5. the first eight files (all but Scale-Balanced-Grasp, FineGrasp and HGGD) also trained for 5
   epochs and were scored on validation frames (*The training check* below).

</details>

<details markdown="1"><summary>How these files are made</summary>

None of them is written by hand. Each is the output of `grasppanda compose` for the command in
its own header, for example

```bash
grasppanda compose finegrasp/finegrasp_sunet14_g1b-realsense_10e --set backbone=ptv3 \
    -o configs/experimental/finegrasp_ptv3_g1b-realsense_10e.yaml
```

The numbers (grid size, radii, widths, sampling rules) live in the presets under
`grasppanda/presets/<slot>/`, the adapter and the fitted width come from compose, and the
header repeats the re-tuning notes of the presets. To change a number, edit the preset and
regenerate (`python -m tools.make_combination_docs`); a unit test fails when a file is not what
compose writes today. When the first eight files were moved onto presets their resolved content
did not change: their config hashes are the ones the hand-written files had.

</details>

<details markdown="1"><summary>The training check (2026-10-12)</summary>

Run on one RTX A6000 shared with other jobs, PyTorch 2.7.1 + cu118, spconv 2.3.8. Each config
was trained on the `sample` split (8 frames of training scene 0, with their real grasp,
graspness, compact and tolerance labels), for 5 epochs, then predicted on 20 frames of
`val_dev` and scored by the GraspNet-1Billion evaluator:

```bash
grasppanda train experimental/<name> --opts \
    data.train.dataset.split=sample train.epochs=5 evaluation.interval=5 \
    data.val.dataset.frame_step=128 runtime.log.interval=1 \
    data.train.num_workers=2 data.val.num_workers=2 evaluation.num_workers=4
```

| Config | Parameters | Iterations | Total loss, first → last | Supervised seeds, last iteration | Wall time |
|---|---|---|---|---|---|
| `graspnet-baseline_sunet14-fps_g1b-realsense_18e` | 14 981 516 | 20 (batch 2) | 3.18 → 1.58 | 0.49 | 107 s |
| `gsnet_ptv3_g1b-realsense_10e` | 46 352 303 | 10 (batch 4) | 17.37 → 2.44 | 1.00 | 171 s |
| `economicgrasp_ptv3_g1b-realsense_10e` | 46 748 766 | 10 (batch 4) | 20.98 → 5.45 | 0.10 | 78 s |
| `gsnet_litept_g1b-realsense_10e` | 12 903 031 | 10 (batch 4) | 9.68 → 2.46 | 1.00 | 86 s |
| `gsnet_oacnns_g1b-realsense_10e` | 52 317 519 | 10 (batch 4) | 5.63 → 3.01, not monotonic (see below) | 1.00 | 105 s |
| `gsnet_spunet_g1b-realsense_10e` | 39 900 527 | 10 (batch 4) | 6.01 → 1.65 (see below) | 1.00 | 86 s |
| `gsnet_pointnet2-fp_g1b-realsense_10e` | 1 140 943 | 10 (batch 4) | 5.78 → 2.17 | 1.00 | 80 s |
| `gsnet_pointnext_g1b-realsense_10e` | 967 727 | 10 (batch 4) | 8.31 → 2.73 | 1.00 | 75 s |

What this shows: the config, the data pipeline, the label assets and the model fit together;
losses are finite and fall on eight frames; a checkpoint is written; prediction, both
post-processing variants and the evaluator run on the result. What it does not show: anything
about accuracy. The AP printed after ten iterations on eight frames was 0.00 to 0.02 and is not
a result.

Two runs show what "the recipe is not tuned" looks like. Under `OACNNs` the total loss went
5.63, 15.78, 3.21 and then, for iterations 4 to 6, no scene point passed the graspness
threshold: the frames had no real seed (`metric/real_seeds` 0), so the view and grasp losses
were zero and only the proposal was trained. Seeds came back at iteration 7 (0.36 of the slots)
and were complete from iteration 8. Under `SpUNet` the fraction of real seeds fell to 0.01 in
iteration 3 and to 0.54 in iteration 8 and was 0.89 or more otherwise. GSNet's recipe, a
learning rate of 1e-3 without warm-up, is too much for the first steps of these two deep sparse
CNNs (52 M and 40 M parameters). In the other runs at least 86 % of the seed slots held a real
seed at every iteration (the minimum, 0.857, is one iteration of `gsnet_ptv3`).

"Supervised seeds" is `metric/supervised_seeds`, the fraction of seeds that take part in the
grasp losses. It is 1 for GSNet by construction (no distance limit), about a tenth for
EconomicGrasp at this stage (a seed counts only within 5 mm of a label point, and an untrained
proposal puts seeds anywhere), and about half for the baseline (seeds on an object). It is the
first number to compare with a run of the reference encoder.

</details>

<details markdown="1"><summary>Full runs</summary>

Not run here: each is a multi-hour job. The commands are the ones in the header of each file;
the assets are those of the method (`grasppanda prepare graspnet1b --labels`, plus what the
method's README lists: `--graspness --camera realsense`, `--compact-labels`, `--tolerance`,
`--scale-statistic`, `--view-grasps`). A number from such a run has no paper to be compared
with: report it as trained here, over three seeds (`docs/design/ARCHITECTURE.md` §11.5), next to
a run of the reference encoder under the same recipe, and label it experimental.

</details>
