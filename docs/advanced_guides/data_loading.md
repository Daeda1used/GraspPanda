# Data loading: where the label grids are read

**Applies to the methods that train on dense label grids: GraspNet-baseline, GSNet and
Scale-Balanced-Grasp** (their training recipes have a `SampleGraspLabels` step).
EconomicGrasp and FineGrasp train on compact labels and HGGD on per-view grasps; none of
the three reads these grids during training ([Datasets](../user_guides/datasets.md)).

A dense label holds, for every annotated point of an object, a friction score and a
gripper width for each of 300 views × 12 angles × 4 depths: 14 400 grasps and 57.6 kB
per grid and point. This page explains what a training batch used to carry, what it
carries now, why nothing changed in the numbers, and what the change is worth on a clock.

Nothing here has to be configured: the shipped dataset configs use the fast path. Read
on if you write an assigner, train on your own label store, or want to know where an
iteration spends its time.

## What a batch carried

A training frame of GraspNet-1Billion shows about nine objects. The three recipes keep a
quarter of each object's annotated points, at least 300 (GraspNet-baseline and
Scale-Balanced-Grasp only of the points near the visible surface). With GSNet's recipe that
is about 3 300 label points per frame, each with a score grid and a width grid. The
reference pipelines copy all of them into the sample, and so did `SampleGraspLabels`
(GSNet, batches of 4):

| | per frame | per batch of 4 |
|---|---:|---:|
| label points kept | 3 300 | 13 300 |
| their score and width grids | 350 MiB | 1.4 GiB |
| label points the 1024 seeds of a frame are matched to | 600 | 2 500 |
| what label matching reads of them | 34 MiB of scores, 0.2 MiB of widths | 140 MiB |

The grids were read from the label store in a dataloader worker, pickled to the
training process and copied to the GPU, where label matching looked at a tenth of them.
On the development machine the copy to the GPU alone took 0.3 s per batch, and four
workers delivered 8 frames per second to a model that can train on 16.

## Deferred grids

`SampleGraspLabels` has an option that says where the grids are read:

```yaml
transforms:
  - type: SampleGraspLabels
    point_fraction: 0.25
    min_points: 300
    collision: zero_scores
    grids: deferred        # or: dense
```

| | `grids: dense` | `grids: deferred` |
|---|---|---|
| The worker | draws the label points and copies their grids out of the store | draws the same label points and writes down which rows of which files they are |
| The sample carries | `ObjectGraspLabels.scores`, `.widths` (`.collision`, `.tolerance`) as tensors | `ObjectGraspLabels.deferred`: file names, rows, collision mode; a few kilobytes per object |
| `batch.to(device)` | moves the grids | moves poses and label points |
| The assigner | indexes the tensors at the matched rows | reads the matched rows from the store |

The label points, the object poses, the point labels and every random number drawn are
the same in both. Only the moment at which the grids are read moves: from "before the
forward pass, all of them" to "inside it, the ones that are needed". They cannot be
read earlier, because which label points are needed depends on the seeds, and the seeds
are chosen by the network.

The read is one pass per batch (`DeferredGraspLabels.gather`):

1. the matched (object, label point) pairs of all frames come to the host in one
   transfer and are grouped by object;
2. every *distinct* row is copied once, in file order, from the memory-mapped store
   into one page-locked buffer (`np.take`, which copies straight into the buffer);
3. the buffer goes to the device in one transfer; so do the bit-packed collision rows,
   which are unpacked there and written into the scores as zeros;
4. on the device the rows are expanded to the seeds that share them and the views are
   re-indexed.

Widths and tolerance at the chosen view are 48 numbers per seed and are read cell by
cell. An assigner that needs the widths of all views (GraspNet-baseline,
Scale-Balanced-Grasp) reads them like the scores.

### The numbers do not change

Deferring is a change of storage, and it is held to that: the same label points, poses
and random draws in both storages, identical matched labels on real frames of the three
methods, and bit-identical training steps and trainings for GraspNet-baseline and
Scale-Balanced-Grasp.

??? note "The tests and runs behind it"

    * A deferred reader returns the stored values: zeros where `collision: zero_scores`
      would have written them, the tolerance radius of each stored level, collisions bit
      for bit. `tests/unit/test_data_deferred_labels.py` compares every grid, every
      collision mode, with and without tolerance, repeated and unsorted requests, against
      the arrays the store was built from, on a label grid whose size is not a multiple of
      eight. Seven deliberately broken readers (bit order, row offsets, a forgotten zero)
      each fail that file.
    * The same file builds the synthetic dataset twice, once per storage, and requires
      identical samples and bit-identical `MatchedLabels` for all 32 combinations of the
      options of `NearestLabelAssigner`, for `ScaleBalancedAssigner`, and for the scale
      statistic.
    * `tests/integration/test_deferred_labels.py` repeats this on real training frames for
      the shipped GraspNet-baseline, GSNet and Scale-Balanced-Grasp recipes: three batches
      each, all tensors of `MatchedLabels` identical on the GPU; and on GraspClutter6D
      frames in which one object model occurs twice.
    * A whole training step, losses and gradients of every parameter, is bit-identical
      between the two storages for GraspNet-baseline and Scale-Balanced-Grasp (under
      PyTorch's deterministic kernels, without which their backward pass does not repeat
      itself).
    * A whole training, too. GraspNet-baseline was trained for 100 iterations
      (`runtime.deterministic: true`, two loader workers, shuffling and augmentation on)
      three times: with the code before this change, with the code after it, and with the
      first half run by the old code and the second half resumed by the new. All three end
      with the same weights and the same optimizer state, bit for bit.

**GSNet is a special case, and not because of the labels.** Its sparse convolutions do
not repeat themselves: two training steps on the very same batch, with the same seed,
differ in the last bits of some loss terms and in every gradient, with dense labels and
with PyTorch's deterministic switches on. Two trainings of 25 iterations with the same code
and seed end with weights that differ by 3 · 10⁻³ on average; old code against new
differs by the same amount. A loss curve of GSNet is therefore not reproducible to the
last bit on this stack, whatever the labels. What is shown for GSNet is everything
that *can* be identical: the batches, the random draws, the labels matched inside one
forward pass, and the view and head losses computed from them.

The cause is the sparse-convolution library; GSNet is the one method on dense labels
that uses it.

Two things follow for runs:

* `grids` is not part of a transform's `config()`, so assets derived from a data recipe
  (the scale statistic) are shared by both storages.
* It *is* part of the resolved config, so the config hash of a run changes with it, and
  resuming a run started with dense grids under a config that says `deferred` needs
  `runtime.allow_config_mismatch=true`. The resumed run draws the same batches, in a run
  directory of its own.

## Measurements

With deferred grids an iteration is 3.3 times shorter for GraspNet-baseline, 3.7 times
for GSNet and 1.8 times for Scale-Balanced-Grasp, measured with
`tools/bench_data.py` on the development machine. The model is now the larger part of an
iteration; before, an iteration waited for the loader and for the copy to the device.

??? note "How it was measured, and the three tables"

    `tools/bench_data.py` runs a training config on the real dataset and reports the loader
    alone, an iteration part by part (the device synchronised around each part) and the
    plain training loop. With `--grids dense deferred` both storages run back to back in
    every repeat, on the same frames, with the same sampling and initial weights.

    ```bash
    python -m tools.bench_data gsnet/gsnet_sunet14_g1b-realsense_10e \
        --grids dense deferred --scenes 8 --iterations 24 --warmup 16 --repeats 3
    ```

    On the development machine: an RTX A6000 and 16 cores, both shared with other jobs (load
    average 20 to 30 during the runs), the label store on a network file system and in the
    page cache, four workers, 24 batches after 16 of warm-up, the first eight scenes of the
    training split. Cells are best / median over three repeats. `before` is the code
    before this change (commit `ccb8704`), run right before or after; `dense` and
    `deferred` are the two storages on the current code.

    **GraspNet-baseline**, batch 2

    | best / median | before | dense | deferred |
    |---|---:|---:|---:|
    | loader, frames/s | 6.28 / 6.27 | 7.35 / 7.20 | 39.1 / 29.6 |
    | batch as it leaves a worker, MiB | 805 | 805 | 1.17 |
    | first batch after, s | 6.61 / 6.91 | 3.61 / 3.84 | 1.37 / 1.52 |
    | `batch.to(device)`, ms | 158 / 176 | 155 / 181 | 2.31 / 2.59 |
    | forward, ms | 81.1 / 85.4 | 77.8 / 79.2 | 103 / 109 |
    | of which the assigner, ms | 17.2 / 17.2 | 14.4 / 15.3 | 40.9 / 43.6 |
    | backward, ms | 51.9 / 54.8 | 49.0 / 50.5 | 49.2 / 49.4 |
    | optimizer step, ms | 1.89 / 2.24 | 2.33 / 2.45 | 2.26 / 3.13 |
    | **iteration, s** | 0.54 / 0.56 | 0.51 / 0.52 | **0.16 / 0.17** |
    | of which waiting for the loader, s | 0.29 / 0.30 | 0.26 / 0.27 | 0.02 / 0.02 |

    **GSNet**, batch 4 (`before`: two repeats)

    | best / median | before | dense | deferred |
    |---|---:|---:|---:|
    | loader, frames/s | 7.88 / 7.77 | 7.80 / 7.15 | 35.9 / 34.8 |
    | batch as it leaves a worker, MiB | 1409 | 1409 | 2.77 |
    | first batch after, s | 9.12 / 9.17 | 4.92 / 5.75 | 1.22 / 1.27 |
    | `batch.to(device)`, ms | 286 / 305 | 341 / 353 | 2.94 / 3.00 |
    | forward, ms | 72.4 / 75.9 | 68.1 / 73.2 | 102 / 108 |
    | of which the assigner, ms | 20.5 / 22.5 | 18.9 / 20.0 | 50.8 / 51.9 |
    | backward, ms | 74.4 / 78.7 | 79.5 / 80.6 | 77.2 / 78.1 |
    | optimizer step, ms | 3.00 / 3.01 | 2.78 / 2.84 | 2.73 / 2.98 |
    | **iteration, s** | 0.86 / 0.87 | 0.97 / 0.99 | **0.22 / 0.24** |
    | of which waiting for the loader, s | 0.45 / 0.46 | 0.51 / 0.54 | 0.05 / 0.05 |

    **Scale-Balanced-Grasp**, batch 2 (with a scale histogram of 64 frames instead of the
    shipped 3200, which changes the weights and not the timings)

    | best / median | before | dense | deferred |
    |---|---:|---:|---:|
    | loader, frames/s | 6.96 / 6.57 | 7.37 / 7.03 | 31.7 / 28.4 |
    | batch as it leaves a worker, MiB | 805 | 805 | 1.18 |
    | first batch after, s | 8.41 / 9.33 | 5.59 / 5.83 | 1.82 / 1.94 |
    | `batch.to(device)`, ms | 154 / 171 | 149 / 159 | 3.57 / 4.09 |
    | forward, ms | 141 / 144 | 134 / 139 | 169 / 173 |
    | of which the assigner, ms | 18.5 / 20.3 | 18.5 / 18.7 | 53.1 / 55.7 |
    | backward, ms | 158 / 159 | 132 / 136 | 130 / 135 |
    | optimizer step, ms | 2.46 / 2.62 | 2.90 / 3.15 | 2.25 / 2.80 |
    | **iteration, s** | 0.54 / 0.56 | 0.48 / 0.53 | **0.31 / 0.31** |
    | of which waiting for the loader, s | 0.22 / 0.23 | 0.20 / 0.22 | 0.02 / 0.03 |

    An iteration is 3.3 times shorter for GraspNet-baseline, 3.7 times for GSNet and 1.8
    times for Scale-Balanced-Grasp, whose model is the heaviest of the three. The deferred
    assigner costs 30 to 40 ms more than the dense one, because it reads; the copy to the
    device and the wait for the loader, which it replaces, cost 0.4 to 0.8 s. On the full
    training split the loader keeps the model waiting somewhat longer than on eight scenes
    (60 to 75 ms per iteration instead of 20 to 50) while the workers share their cores
    with the model and with other jobs.

    Ten epochs of GSNet on GraspNet-1Billion are 64 000 iterations: about 15 hours at
    0.87 s, about 4.5 hours at 0.25 s. These totals follow from the iteration times above;
    a full run was not timed.

    The model is now the larger part of an iteration; before, an iteration waited for the
    loader and for the copy to the device.

??? note "On a cold page cache"

    All of the above is with the label store in the page cache, which is where it is after
    the first minutes of a run. Before that, a deferred read is at a disadvantage: the
    dense path reads rows in four workers, the deferred path in the training process, which
    waits for every row that is not cached. On the network store of the development
    machine (a gigabit link) a cold row costs about 4 ms, so the rows of one batch cost
    about 10 s; the same store delivers a file read front to back at 85 to 105 MiB/s, six
    times faster than it delivers scattered rows.

    So the first time a deferred reader opens a label file, the file is also read once,
    front to back, on one of two background threads (two sequential streams fill the
    link). The threads end when their queue is empty. A dataloader worker forked while they
    are still reading starts without them and without the memory maps of the training
    process.

    Measured on a private copy of the label files of four scenes (18 objects, 2.67 GiB) on
    the same network store, dropped from the page cache before every run; 30 batches of four
    frames, seconds from the start of the loader:

    | | loader and label matching | GSNet training loop |
    |---|---:|---:|
    | dense grids | 51 | 73 |
    | deferred, without read-ahead | 81 to 93 | 66 |
    | deferred, with read-ahead | 34 to 36 | 52 |

    With the read-ahead the first batch is slow (22 to 34 s: its rows are fetched while the
    files are read) and the cache is warm from the fifth batch on. Without it, batches are
    still two to three times slower than warm after thirty.

    The files a GSNet training on GraspNet-1Billion reads are 7.5 GiB (scores 2.7, scene
    collisions 2.1, widths 2.7; tolerance levels add 0.7 for GraspNet-baseline). At the
    105 MiB/s of this link that is a good minute, so expect the first one to two minutes
    of a run on a completely cold cache to be slow. This last figure is an estimate from
    the file sizes; a cold start of the whole training set was not measured.

    Other ways of reading cold rows were measured on the same copy (loader and label
    matching, 30 batches) and not adopted: asking the kernel for the rows of a batch before
    copying them (`madvise`, 72 s), copying on eight threads (87 s), both together (61 s),
    and letting the workers ask for the rows they sampled (59 s). They all read scattered
    rows, at the speed of scattered rows.

    `GRASPPANDA_LABEL_READ_AHEAD=0` switches the read-ahead off. Do that on a machine whose
    memory cannot hold the label files next to everything else: there the cache never
    becomes warm, reading files ahead only displaces other pages, and a faster disk for
    `$GRASPPANDA_CACHE` is what helps.

## Reading labels in your own code

An assigner, or any code that consumes `Supervision.objects`, reads grids through one
function and does not care how they are stored:

```python
from grasppanda.structures.observation import gather_grids

# request k reads label point rows[k] of objects[owner[k]]
scores = gather_grids(objects, "scores", owner, rows)             # (K, V, A, D)
widths = gather_grids(objects, "widths", owner, rows, views)      # (K, A, D) at views[k]
mask = gather_grids(objects, "collision", owner, rows)            # (K, V, A, D) or None
```

`objects` can be the objects of every frame of a batch; one call reads them all. For a
single object there are `labels.gather_scores(rows)`, `gather_widths(rows, views)`,
`gather_collision(rows)` and `gather_tolerance(rows, views)`. `labels.dense()` turns
deferred labels into tensors, which reads everything and is meant for inspection.

Indexing `labels.scores` directly still works on dense labels and fails on deferred
ones (`scores` is `None`): an assigner written that way either moves to `gather_grids`
or runs with `grids: dense`.

A label store of your own can be deferred by returning, from the `defer(rows,
collision=...)` method of its label source, an object that implements
`grasppanda.structures.observation.DeferredGrids`: three properties and one class
method, `gather(readers, name, owner, rows, views)`, which reads for a group of readers
at once.

## Decisions, with the measurements behind them

Why the grids are read where they are, and what else was measured and changed or left
alone.

??? note "Why not read in the workers, and how the rows are read"

    **Why not read in the workers?** A worker knows the label points but not the seeds. It
    could ship the rows of every label point that is the nearest one of *some* scene point,
    a superset of whatever the seeds will match. Measured on 24 training frames: 3 300
    label points sampled, 1 390 in that superset, 610 matched by 1024 furthest-point seeds;
    the nearest-neighbour search costs a worker 15 ms per frame. The superset would still
    put 150 MiB of scores and widths per frame through the loader and onto the device (the
    deferred path reads 34 MiB and ships 0.04), and it needs a fallback for a seed whose
    float32 match on the GPU falls outside a float64 superset. Not worth its complexity.

    **Distinct rows, expanded on the device.** Of 4096 seeds about 2 500 distinct label
    points are matched. Reading per seed, or per object with a transfer each, loses to
    reading each row once into one buffer: with one transfer per object the assigner took
    110 to 500 ms on a GPU that other jobs share, because every small transfer waits for
    the device; with one pass per batch it takes about 50 ms.

    **`np.take` into a reused, page-locked buffer.** Copying 146 MiB of rows out of the
    memory maps takes 18 ms with `np.take(..., out=buffer, mode="clip")` and 24 ms with
    fancy indexing into a fresh array; the transfer of 190 MiB to the GPU takes 9 ms from
    page-locked memory and 19 ms from ordinary memory. Reading on three or four threads was
    faster in the best case (7 ms) and slower in the median on a busy machine, and was left
    out.

    **Opening files.** A dataloader worker starts every epoch without open files. NumPy
    resolves a `Path` it is asked to memory-map component by component, a dozen `lstat`
    calls that are round trips on a network file system: 61 of the 84 ms it took to open
    one label file. The store now opens maps by string path, and hands out labels without
    opening their grid files at all, so a worker that defers never opens them. The first
    batch of a run arrives after about 1 s instead of 7 to 9 s.

??? note "The rest of an iteration (GSNet)"

    With deferred grids, a GSNet iteration (batch 4) looks like this on the development
    machine; the device is synchronised around every part, mean over the 30 iterations of
    one run:

    | part | ms |
    |---|---:|
    | wait for the loader | 67 (median 0.7) |
    | `batch.to(device)` | 2.7 |
    | forward: backbone | 25.9 |
    | forward: proposal (seed selection) | 7.2 |
    | forward: view stage | 0.6 |
    | forward: **assigner** | 47.9 |
    | forward: grouper | 3.9 |
    | forward: head | 0.2 |
    | forward: losses and the rest | 5.5 |
    | backward | 52.3 |
    | optimizer step | 2.8 |

    What was changed besides the label storage, each with its measurement:

    | change | before | after | measured on |
    |---|---:|---:|---|
    | Box workspaces work on one table-frame coordinate at a time, in contiguous memory | 54.9 ms | 23.5 ms | `SegmentationBoxWorkspace.inside`, mean of 320 calls on 160 real frames; the 480 masks compared are identical |
    | `align` gathers the valid pixels with `np.take` | 7.9 ms | 3.1 ms | a 1280 × 720 cloud, 60 % valid |
    | One training sample in a worker (both of the above, and no grids) | 280 ms | 92 ms | GSNet recipe, median of 24 frames, files open |
    | Label files opened by string path, and only when a grid is read | 6.7 to 9.2 s | 1.1 to 1.3 s | time to the first batch, GSNet, two pairs of runs |
    | Poses and label points of a batch moved to the device stacked | 22.1 ms | 3.7 ms | mean of 60 batches from the loader, old and new interleaved (medians 4.5 and 1.8 ms) |

    Looked at and left alone:

    * **`pin_memory`, non-blocking transfers.** A deferred batch is 3 MiB and moves in 3 to
      5 ms. There is nothing left to gain.
    * **`persistent_workers`.** Starting the workers of an epoch now costs about a second.
      Workers that persist would carry their random state from one epoch into the next,
      which breaks the rule that the samples of an epoch depend on the seed and the epoch
      only, and exact resume with it.
    * **`prefetch_factor`.** Measured at 2, 4 and 8 on the full training split: the mean
      wait for a batch was 64, 72 and 99 ms. The waits come from how fast the workers are,
      not from how much they may queue.
    * **Graspness maps.** Reading the map of a frame takes 17 to 21 ms (one member of a
      compressed archive, and the check that it labels exactly the pixels of the frame).
      It is the next thing to look at in a worker.
    * **More workers.** A worker now needs about 92 ms per frame of an otherwise idle core;
      four of them deliver 30 to 37 frames per second alone and about as much as GSNet
      consumes (15 to 17) while the model competes for the same cores. On a machine with
      free cores `num_workers: 8` should remove the remaining waits; that was not measured
      here, because the benchmarks were limited to four workers.

    Where the time now is, and what would move it:

    * **The assigner.** Of its 48 ms, about 25 are the copy of 145 MiB of score rows out of
      the page cache and 9 their transfer. The scores take eleven values and 93 to 98 % of
      them are −1: a lossless one-byte code per score, as a derived asset, would quarter
      both, a sparse row format more. Neither is built.
    * **The model.** Backbone, proposal, grouper, head, losses and the backward pass are
      about half of the iteration. Mixed precision, `torch.compile` and channels-last memory change numbers
      and were out of scope. None of them was measured; as expectations only: the sparse
      convolutions and the point operators are C++ and CUDA kernels that `torch.compile`
      does not see, so it can only help the small MLP heads; mixed precision could shorten
      the sparse U-Net's forward and backward, but its sparse convolutions would have to be
      checked for stability in half precision first; channels-last concerns dense 2-D
      convolutions, which among the shipped methods only HGGD has.

## Limits

* **Open files.** A deferred reader keeps one memory map, and with it one file
  descriptor, per label file it has read: 220 for a training on GraspNet-1Billion (two
  per object model of its 40, one per scene of its 100, one more per object with
  tolerance labels). It keeps at most half of the limit on open files (`ulimit -n`),
  4096 at most. A dataset with more label files than that reopens maps, which is slow
  on a network file system: raise the limit.
* **Memory.** The page-locked buffer grows to the largest read, about 300 MiB for 4096
  seeds, and is kept. The label files themselves are not copied: they are file-backed
  pages shared with every other process that maps them.
* **Dense is still there.** `grids: dense` is the reference path and the component's
  default; the identity tests compare against it.
