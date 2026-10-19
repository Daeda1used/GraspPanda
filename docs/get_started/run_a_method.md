---
hide: [toc]
---

# Run a method

Every ported method runs from its released weights: download the file, convert it once, and
get grasps on the example scene in minutes. No dataset is needed.

## Which method first?

--8<-- "docs/includes/choose_method.md"

??? note "The example scene"

    ![The example scene: the colour image and the depth image of six primitives on a table](../assets/example-scene.png)

    `--example` is a synthetic table-top scene that ships with GraspPanda: six geometric
    primitives on a table, ray-cast into a 1280 × 720 depth image in millimetres, with a
    colour image, the camera matrix and an object mask.

    - It is computed from formulas, so it needs no download and carries no dataset licence.
    - It is not a camera image: surfaces are perfectly smooth, with no noise, no missing
      depth and no reflection. Every method was trained on real scenes, so the example is
      outside its training distribution.
    - It has no grasp labels, so nothing checks the quality of the predicted grasps. It
      shows that the installation and the weights work end to end.

## Pick a method

Each page walks you through it: the weights and their licence, the conversion, the demo, the
same from Python and in the browser, and a first number on the benchmark.

<div class="grid cards" markdown>

-   **GraspNet-baseline** · CVPR 2020 · seed-based

    The benchmark's own baseline: a PointNet++ encoder, approach views and cylinder crops.

    [Run it →](../methods/graspnet_baseline.md#run-in-five-minutes)

-   **GSNet** · ICCV 2021 · seed-based

    Finds the points worth grasping first, on a sparse U-Net encoder.

    [Run it →](../methods/gsnet.md#run-in-five-minutes)

-   **Scale-Balanced-Grasp** · CoRL 2022 · seed-based

    Balances grasps of every width, so thin objects are not drowned out.

    [Run it →](../methods/scale_balanced_grasp.md#run-in-five-minutes)

-   **EconomicGrasp** · ECCV 2024 · seed-based

    Trains on a compact set of labels with an interaction grouper.

    [Run it →](../methods/economicgrasp.md#run-in-five-minutes)

-   **FineGrasp** · arXiv 2025 · seed-based

    Builds on EconomicGrasp with crops at several scales, for delicate objects.

    [Run it →](../methods/finegrasp.md#run-in-five-minutes)

-   **HGGD** · RA-L 2023 · own family

    A heat-map on the colour and depth images guides where to look for grasps.

    [Run it →](../methods/hggd.md#run-in-five-minutes)

</div>

## On your camera

A depth image and its intrinsics are enough. For the seed-based methods, `--workspace` crops
the cloud to the objects, as they were trained; HGGD reads the whole image and crops no
workspace.

```bash
grasppanda demo <method config> --weights <converted file> \
    --depth depth.png --intrinsics fx,fy,cx,cy \
    --workspace -0.25 -0.2 0.35 0.25 0.2 0.8
```

HGGD also reads the colour image registered to the depth image: add `--rgb rgb.png` and leave
out `--workspace`. The methods were trained on GraspNet-1Billion (RealSense D435 and Azure
Kinect); results on other cameras have not been measured. Units, the depth scale and the
usual mistakes: [Your own camera](../user_guides/inference.md#your-own-camera).

## From Python or in the browser

- [From Python](../user_guides/inference.md#from-python): `GraspInferencer` runs the demo's pipeline and returns the grasps.
- [In the browser](../user_guides/inference.md#run-in-the-browser): `grasppanda explore --weights-dir <folder>` runs a method on the example scene or an uploaded depth image.

## Next

[Swap a component](swap.md) of the method you ran.
