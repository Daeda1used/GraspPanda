# Read an analysis

**Goal:** see where the AP of an evaluation comes from (which grasps earn it, which lose
it, and why), and whether two results really differ.

**You need:** a run of `grasppanda test` ([Train and test](train_test.md)). `grasppanda
analyze` reads its result records and nothing else: no dataset, no weights, no predictions.

## 1. Score a model

The example on this page scores the authors' GraspNet-baseline checkpoint on every 64th
frame of the seen test scenes, 120 frames:

```bash
grasppanda test graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e \
    --weights baseline_realsense.pth \
    --opts data.test.dataset.split=test_seen data.test.dataset.frame_step=64 \
    runtime.work_dir=work_dirs/baseline runtime.run_id=test_seen_step64 evaluation.num_workers=4
```

Any method and any split work the same way: `grasppanda test` writes one result record per
post-processing variant into `<run>/results/`.

## 2. Analyze it

```bash
grasppanda analyze work_dirs/baseline --variant cd --objects 5 --plot work_dirs/baseline_cd.png
```

The report has a header, then the AP taken apart by friction threshold, by cause, by scene,
by object and through the post-processing chain. `--plot` draws it (Pillow only).

??? note "The whole report of this run"

    ```text
    work_dirs/baseline/test_seen_step64 · cd
      protocol        graspnet1b-sv-v1 · dataset graspnet1b · split test_seen · camera realsense
      coverage        120 of 120 frames scored (100.0 %), 0 missing
      extent          a SUBSET of split 'test_seen': 120 of its 7,680 frames (4 of 256 frames per scene, 30 of 30 scenes). Not a benchmark number.
      inputs          ground-truth-assisted: workspace
      post-processing ModelFreeCollisionFilter
      provenance      weights sha256 f24ba97fcf30 · code de7d468efc · seed 0 · 2026-10-12T16:31:51+00:00
      metrics         AP 0.4867 · AP0.4 0.4178 · AP0.8 0.5764   (fraction in [0, 1]; multiply by 100 for the figures papers print)

    Where the precision comes from
    friction threshold  μ≤0.2   μ≤0.4   μ≤0.6   μ≤0.8   μ≤1     μ≤1.2   mean
    ──────────────────  ──────  ──────  ──────  ──────  ──────  ──────  ──────
    mean precision      0.1788  0.4178  0.5299  0.5764  0.6005  0.6166  0.4867

    Share of every frame's AP by outcome of the rank that owns it (earned = the precision above):
    outcome                 μ≤0.2   μ≤0.4   μ≤0.6   μ≤0.8   μ≤1     μ≤1.2   mean
    ──────────────────────  ──────  ──────  ──────  ──────  ──────  ──────  ──────
    earned                  0.1788  0.4178  0.5299  0.5764  0.6005  0.6166  0.4867
    lost: no force closure  0.5214  0.2824  0.1703  0.1238  0.0997  0.0836  0.2135
    lost: collision         0.2931  0.2931  0.2931  0.2931  0.2931  0.2931  0.2931
    lost: empty             0.0067  0.0067  0.0067  0.0067  0.0067  0.0067  0.0067
    lost: unfilled ranks    0.0001  0.0001  0.0001  0.0001  0.0001  0.0001  0.0001
    sum                     1.0000  1.0000  1.0000  1.0000  1.0000  1.0000  1.0000

    By split
    split      scenes  frames  mean    μ≤0.2   μ≤0.4   μ≤0.6   μ≤0.8   μ≤1     μ≤1.2
    ─────────  ──────  ──────  ──────  ──────  ──────  ──────  ──────  ──────  ──────
    test_seen  30/30   120     0.4867  0.1788  0.4178  0.5299  0.5764  0.6005  0.6166

    By scene (the 5 lowest and the 5 highest of 30; --scenes 0 lists all)
    scene                 frames  mean precision  AP0.8   AP0.4   grasps/frame
    ────────────────────  ──────  ──────────────  ──────  ──────  ────────────
    realsense/scene_0122  4       0.1866          0.2482  0.0967  155.0
    realsense/scene_0124  4       0.3077          0.3535  0.2674  163.2
    realsense/scene_0111  4       0.3347          0.3861  0.2471  147.0
    realsense/scene_0100  4       0.3417          0.4168  0.2924  132.0
    realsense/scene_0101  4       0.3905          0.4546  0.3111  125.5
    …
    realsense/scene_0121  4       0.6018          0.6721  0.5533  195.8
    realsense/scene_0113  4       0.6135          0.7400  0.4962  204.2
    realsense/scene_0116  4       0.6322          0.7470  0.6038  193.5
    realsense/scene_0119  4       0.6798          0.7911  0.6571  173.8
    realsense/scene_0114  4       0.6934          0.8047  0.7022  177.0

    By object (the 5 that lose most of 28; percentage points of AP, mean over the friction thresholds)
    object  frames  ranked/frame  earned  no closure  collision  empty  success
    ──────  ──────  ────────────  ──────  ──────────  ─────────  ─────  ───────
    0       40      8.5           3.07    1.01        4.99       0.13   0.334
    7       40      8.8           3.29    1.95        2.65       0.00   0.417
    58      40      9.8           7.78    1.88        2.53       0.18   0.629
    66      40      9.6           4.51    2.08        2.16       0.00   0.515
    8       40      8.9           2.09    1.69        1.67       0.04   0.381
      earned sums to the AP over all objects; success = earned / (earned + lost), rank-weighted

    Grasps
      stage                           grasps  per frame  removed by this step  left of the predicted
      ──────────────────────────────  ──────  ─────────  ────────────────────  ─────────────────────
      predicted                       47,812  398.4
      after ModelFreeCollisionFilter  19,235  160.3      59.8 %                40.2 %
      frames with an empty prediction: 0

    wrote work_dirs/baseline_cd.png
    ```

## 3. Read the header first

The first block is everything a number needs before it may be quoted: protocol, split and
camera, how many frames were scored, which inputs were derived from ground truth, the
post-processing chain, and the weights and code that produced it. **extent** compares the
frames of the record with the split it names. This run scored every 64th frame, so it is a
subset, and the report says so in capitals: its AP of 0.4867 is a measurement on 120
frames, not the AP of GraspNet-baseline on the seen split.

## 4. See where the precision comes from

AP is the mean of a precision table with one column per friction threshold μ. The first
table gives the mean of each column; their mean is the AP. Here a grasp that needs
μ ≤ 0.2 is rare (0.18), and most of what can be earned is earned by μ ≤ 0.8.

The second table attributes the AP. A frame's AP is a weighted sum over its 50 ranks: rank
1 owns about 9 % of it, rank 50 owns 0.04 %. Every rank either earns its share, or loses it
for exactly one reason: the gripper collides with the scene (checked against the
ground-truth models), nothing is between the fingers, the grasp is free but not
force-closed at this μ, or the rank was never filled. The columns add up to one. In this
run 29 % of the AP is lost to collisions although the model-free collision filter ran.

??? note "By object, by scene, and through the post-processing chain"

    **By object** applies the same split to the object each ranked grasp belongs to. The
    numbers are percentage points of AP: object 0 earns 3.07 points and loses 4.99 to
    collisions, more than any other object. `success` is the share an object's ranks earned
    of what they held. Object ids are the dataset's.

    **By scene** lists the scenes at both ends. Scenes, not frames, are the unit to think
    in: the four frames of a scene show the same objects.

    **Grasps** follows the predictions through the post-processing chain. For a chain with
    two steps each step has its own row. HGGD's `cd` variant on the same frames:

    ```text
    Grasps
      stage                           grasps  per frame  removed by this step  left of the predicted
      ──────────────────────────────  ──────  ─────────  ────────────────────  ─────────────────────
      predicted                       53,917  449.3
      after ModelFreeCollisionFilter  25,908  215.9      51.9 %                48.1 %
      after GraspNMS                  7,954   66.3       69.3 %                14.8 %
      frames with an empty prediction: 0
    ```

    The split by cause and object and the per-step counts are stored in the record by
    `grasppanda test` (fields `breakdown` and `grasp_counts`). Records written before these
    fields existed are read as before; the report then gives the total the chain removed,
    from the grasp counts of the run's records, and says that the rest is not recorded.

## 5. Compare two results

```bash
grasppanda analyze --compare work_dirs/baseline:cd work_dirs/sbg:cd --scenes 5 \
    --plot work_dirs/baseline_vs_sbg.png
```

`A:cd` picks a variant of a run; `--variant` picks one for both. The two records must share
protocol and dataset. Whatever else differs between them (split, cameras,
ground-truth-assisted inputs, post-processing, weights, code) is listed under `differs`, so
that a difference in the numbers is not credited to the wrong cause. The summary pairs
scenes: a bootstrap interval of the mean difference and a sign test, both with a scene as
one observation.

??? note "GraspNet-baseline against Scale-Balanced-Grasp on these 120 frames"

    ```text
    A  work_dirs/baseline/test_seen_step64 · cd
    B  work_dirs/sbg/test_seen_step64 · cd
       protocol graspnet1b-sv-v1 · dataset graspnet1b · 30 paired scenes
       differs: weights: A f24ba97fcf30 · B e6696a21baf5

    Paired summary over scenes (mean precision of a scene; B − A)
      mean A 0.4867 · mean B 0.6284 · mean difference +0.1417 · median +0.1371 · standard deviation 0.0788
      95 % interval of the mean difference +0.1154 to +0.1698 (bootstrap over the 30 scenes, 10 000 resamples, seed 0)
      B higher in 30 scenes, A higher in 0, equal in 0 · two-sided sign test p = 1.86e-09

    Per scene, largest difference first (the 5 largest of 30; --scenes 0 lists all)
    scene                 A       B       B − A
    ────────────────────  ──────  ──────  ───────
    realsense/scene_0122  0.1866  0.5417  +0.3551
    realsense/scene_0124  0.3077  0.6321  +0.3244
    realsense/scene_0118  0.4546  0.6887  +0.2342
    realsense/scene_0101  0.3905  0.6088  +0.2183
    realsense/scene_0111  0.3347  0.5298  +0.1951
    ```

    On this subset Scale-Balanced-Grasp is ahead in all 30 scenes. Both are the authors'
    checkpoints scored on 120 frames; the statement is about these 120 frames.

    `--plot` draws the report: for one record the precision per friction threshold, the
    share earned and lost, and every scene; for a comparison the difference of every scene
    around zero.

## Next

- [Model zoo](../model_zoo.md): whole-split results of every method, with their records.
- [Cross-dataset evaluation](cross_dataset.md): the same weights on GraspClutter6D.
