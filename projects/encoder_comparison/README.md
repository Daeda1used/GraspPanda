# Encoder comparison

**Support level: experimental.** One training run per encoder, one seed.

What does a scene encoder from another paper do under a grasp detection method when
nothing else changes? This project trains GSNet with the encoder it ships and with three
others, with GSNet's recipe, data and evaluator, and scores each on the whole
GraspNet-1Billion test split. It is the first use of the swappable stages beyond a
[probe](https://daeda1used.github.io/GraspPanda/evidence/): a probe says that a combination
trains, this says how good it becomes.

```bash
grasppanda prepare graspnet1b --labels
grasppanda prepare graspnet1b --graspness --camera realsense
grasppanda sweep projects/encoder_comparison/gsnet_encoders.yaml --dry-run        # four runs, each config checked
grasppanda sweep projects/encoder_comparison/gsnet_encoders.yaml --gpus 0,1,2,3 \
    --opts evaluation.num_workers=8                                               # one run per GPU
```

`evaluation.num_workers` gives the evaluator processes of its own: scoring 23,040 frames in
the training process takes hours.

Every run writes into `work_dirs/gsnet_encoders/<encoder>/seed0/`, and the table of the
four into `work_dirs/gsnet_encoders/summary_test.md`. A run that is recorded is not repeated.

| Name | Encoder | Config |
|---|---|---|
| `sunet14` | the 14-layer sparse U-Net GSNet ships | [`gsnet_sunet14_g1b-realsense_10e`](../../configs/gsnet/gsnet_sunet14_g1b-realsense_10e.yaml) |
| `ptv3` | Point Transformer V3 | [`gsnet_ptv3_g1b-realsense_10e`](../../configs/experimental/gsnet_ptv3_g1b-realsense_10e.yaml) |
| `spunet` | SpUNet, the same kind of network three times as deep | [`gsnet_spunet_g1b-realsense_10e`](../../configs/experimental/gsnet_spunet_g1b-realsense_10e.yaml) |
| `litept` | LitePT-S | [`gsnet_litept_g1b-realsense_10e`](../../configs/experimental/gsnet_litept_g1b-realsense_10e.yaml) |

## Results

Not measured yet. When the runs are in, this section holds the AP of each encoder on seen,
similar and novel scenes, with and without the collision filter, from result records of
the whole test split, and what one seed can and cannot say.
