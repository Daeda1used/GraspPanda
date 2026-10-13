"""Record what the pristine HGGD label converter writes for a few views.

HGGD trains on per-view grasp labels that its ``convert_grasp.py`` derives from the
GraspNet-1Billion annotations. This script runs that converter unmodified for a handful
of ``(camera, scene, view)`` units and records, per view, the arrays it saves, every
random choice it made and the intermediate grasp sets, so that another implementation
can be compared with it step by step.

It is an oracle: it is executed with an interpreter that has the upstream dependencies
(open3d, transforms3d, the compiled ``grasp_nms`` extension, NumPy 1.x), never with
GraspPanda's environment, and it imports the upstream modules read-only (run it with
``-B`` so no bytecode is written next to them)::

    /path/to/envs/HGGD/bin/python -B tools/oracle/hggd_convert_reference.py \
        --upstream /path/to/HGGD --dataset-root /path/to/GraspNet-1B \
        --out $GRASPPANDA_CACHE/fixtures/hggd/labels

The script contains no label logic of its own. It observes the converter from outside:

* ``convert_grasp`` builds ``GraspNet('/ssd/graspnet', ...)`` and reads its settings
  from module constants when it is imported. The dataset root is substituted by
  wrapping ``GraspNet.__init__`` for the duration of that import, and the constants
  (``annIds``, ``output_path``, ``skip``, ``camera``, ``graspnet_root``) are set on the
  imported module before ``run([scene])`` is called. No upstream file is edited.
* ``random.sample`` is wrapped to log what it returns, in call order. The converter
  calls it up to three times per view: the scene-model points used for the centre
  refinement, the grasp sub-sample (only when more than 5000 grasps pass the filter) and
  the depth-cloud points of the visibility statistic. ``random`` is seeded with
  ``scene * 1000 + view`` before each view, so a recording can be repeated.
* ``mask_grasp_group``, ``GraspGroup.nms``, ``remove_invisible_grasp_points`` and
  ``GraspNet.loadSceneModel`` are wrapped to capture their arguments and results.
* The compiled extension visits equally scored grasps in the order of an unstable sort.
  That order cannot be derived from its output alone, so the extension is asked twice
  more: once with the same scores on grasps that are too far apart to suppress each
  other (the output is then the visiting order), once with the row number in the
  object-id column (the output then names the surviving rows).

Everything is written to ``--out`` and to a temporary working directory; nothing goes
into the upstream tree or the dataset. Fixtures that already exist are kept, so an
interrupted run can be repeated; delete a file to record it again. The file must stay
valid Python 3.8 and must not import grasppanda.
"""

import argparse
import json
import os
import platform
import random
import shutil
import sys
import tempfile
import time

import numpy as np

FORMAT = 1

#: (camera, scene, view). Scene 0 has few grasps per view (no sub-sampling), scenes 5 and
#: 62 have more than 5000 (sub-sampled), scene 100 is a test scene.
VIEWS = [
    ("realsense", 0, 0),
    ("realsense", 0, 128),
    ("realsense", 5, 17),
    ("realsense", 100, 0),
    ("kinect", 62, 31),
]

#: Thresholds of the converter's NMS call, needed to repeat it on the probe arrays.
NMS_TRANSLATION = 0.02
NMS_ROTATION = 60 * np.pi / 180


def fixture_name(camera, scene, view):
    return "convert_%s_%04d_%04d.npz" % (camera, scene, view)


def seed_for(scene, view):
    return scene * 1000 + view


def upstream_commit(upstream):
    """Commit of the upstream checkout, read from its ``.git`` directory."""
    git = os.path.join(upstream, ".git")
    try:
        with open(os.path.join(git, "HEAD")) as handle:
            head = handle.read().strip()
        if not head.startswith("ref:"):
            return head
        reference = head.split(None, 1)[1]
        loose = os.path.join(git, reference)
        if os.path.isfile(loose):
            with open(loose) as handle:
                return handle.read().strip()
        with open(os.path.join(git, "packed-refs")) as handle:
            for line in handle:
                if line.strip().endswith(reference):
                    return line.split()[0]
    except OSError:
        pass
    return "unknown"


class Spy:
    """Observes one view of the converter through the functions it calls."""

    def __init__(self, module, grasp_module, nms_grasp):
        self.module = module
        self.grasp_module = grasp_module
        self.nms_grasp = nms_grasp
        self.draws = []
        self.captured = {}
        self._sample = random.sample
        self._mask = module.mask_grasp_group
        self._invisible = module.remove_invisible_grasp_points
        self._nms = grasp_module.GraspGroup.nms
        self._scene_model = module.dataset.loadSceneModel

    def __enter__(self):
        spy = self

        def sample(population, k, *args, **kwargs):
            out = spy._sample(population, k, *args, **kwargs)
            spy.draws.append((len(population), np.asarray(out, dtype=np.int64)))
            return out

        def mask_grasp_group(group, threshold):
            masked, mask = spy._mask(group, threshold=threshold)
            spy.captured["total_grasps"] = np.int64(len(group))
            spy.captured["masked"] = np.array(masked.grasp_group_array, dtype=np.float64)
            return masked, mask

        def nms(group, translation_thresh=0.03, rotation_thresh=30.0 / 180.0 * np.pi):
            before = np.array(group.grasp_group_array, dtype=np.float64)
            out = spy._nms(
                group, translation_thresh=translation_thresh, rotation_thresh=rotation_thresh
            )
            spy.captured["refined"] = before
            spy.captured["nms_survivors"] = np.array(out.grasp_group_array, dtype=np.float64)
            spy.captured["nms_thresholds"] = np.array([translation_thresh, rotation_thresh])
            rows = np.arange(len(before), dtype=np.float64)
            # Same scores, nothing close to anything: the output is the visiting order.
            apart = np.zeros_like(before)
            apart[:, 0] = before[:, 0]
            apart[:, 4:13] = np.eye(3).reshape(-1)
            apart[:, 13] = rows
            apart[:, 16] = rows
            order = spy.nms_grasp(apart, translation_thresh, rotation_thresh)[:, 16]
            spy.captured["nms_order"] = np.array(order, dtype=np.int64)
            # The same grasps with the row number as object id: the output names the rows.
            numbered = before.copy()
            numbered[:, 16] = rows
            keep = spy.nms_grasp(numbered, translation_thresh, rotation_thresh)[:, 16]
            spy.captured["nms_keep"] = np.array(keep, dtype=np.int64)
            return out

        def remove_invisible_grasp_points(cloud, grasp_points, th=0.01):
            mask = spy._invisible(cloud, grasp_points, th=th)
            spy.captured["visible_cloud"] = np.array(cloud, dtype=np.float64)
            spy.captured["visible_points"] = np.array(grasp_points, dtype=np.float64)
            spy.captured["visible_mask"] = np.array(mask, dtype=bool)
            spy.captured["visible_distance"] = np.float64(th)
            return mask

        def load_scene_model(*args, **kwargs):
            models = spy._scene_model(*args, **kwargs)
            spy.captured["model_parts"] = [np.array(model.points) for model in models]
            return models

        random.sample = sample
        self.module.mask_grasp_group = mask_grasp_group
        self.module.remove_invisible_grasp_points = remove_invisible_grasp_points
        self.grasp_module.GraspGroup.nms = nms
        self.module.dataset.loadSceneModel = load_scene_model
        return self

    def __exit__(self, *exc_info):
        random.sample = self._sample
        self.module.mask_grasp_group = self._mask
        self.module.remove_invisible_grasp_points = self._invisible
        self.grasp_module.GraspGroup.nms = self._nms
        self.module.dataset.loadSceneModel = self._scene_model


def load_converter(upstream, dataset_root):
    """Import the pristine converter with its dataset pointed at ``dataset_root``."""
    sys.path.insert(0, upstream)
    import customgraspnetAPI

    original = customgraspnetAPI.GraspNet.__init__

    def redirected(self, root, camera="kinect", split="train"):
        original(self, dataset_root, camera=camera, split=split)

    customgraspnetAPI.GraspNet.__init__ = redirected
    try:
        import convert_grasp
    finally:
        customgraspnetAPI.GraspNet.__init__ = original
    convert_grasp.graspnet_root = dataset_root
    convert_grasp.skip = False
    return convert_grasp


def record_view(converter, camera, scene, view, work):
    """Run the converter for one view; return the arrays of its fixture."""
    import customgraspnetAPI.grasp as grasp_module
    from grasp_nms import nms_grasp

    converter.camera = camera
    converter.annIds = [view]
    converter.output_path = os.path.join(work, camera)
    random.seed(seed_for(scene, view))
    tic = time.time()
    with Spy(converter, grasp_module, nms_grasp) as spy:
        converter.run([scene])
    seconds = time.time() - tic

    scene_dir = os.path.join(converter.output_path, "6d_dataset", "scene_%d" % scene)
    arrays = {}
    with np.load(os.path.join(scene_dir, "grasp_labels", "%d_view.npz" % view)) as saved:
        for name in saved.files:
            arrays["saved/" + name] = saved[name]
    with open(os.path.join(scene_dir, "infos", "%d_info.json" % view)) as handle:
        arrays["info"] = np.array(handle.read())

    captured = spy.captured
    parts = captured.pop("model_parts")
    arrays["model_counts"] = np.array([len(part) for part in parts], dtype=np.int64)
    # The converter's first draw picks the scene-model points of the centre refinement.
    arrays["model_points"] = np.concatenate(parts)[spy.draws[0][1]]
    arrays.update(captured)
    arrays["draws"] = np.int64(len(spy.draws))
    for position, (population, indices) in enumerate(spy.draws):
        arrays["draw_%d/population" % position] = np.int64(population)
        arrays["draw_%d/indices" % position] = indices
    arrays["seed"] = np.int64(seed_for(scene, view))
    arrays["seconds"] = np.float64(seconds)
    return arrays


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--upstream", required=True, help="pristine HGGD checkout")
    parser.add_argument("--dataset-root", required=True, help="GraspNet-1Billion")
    parser.add_argument("--out", required=True, help="directory of the fixtures")
    args = parser.parse_args()

    upstream = os.path.abspath(args.upstream)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    pending = [unit for unit in VIEWS if not os.path.isfile(os.path.join(out, fixture_name(*unit)))]
    if pending:
        work = tempfile.mkdtemp(prefix="hggd-convert-")
        os.chdir(work)  # the converter's default output path is relative
        converter = load_converter(upstream, os.path.abspath(args.dataset_root))
        for camera, scene, view in pending:
            arrays = record_view(converter, camera, scene, view, work)
            target = os.path.join(out, fixture_name(camera, scene, view))
            with open(target + ".tmp", "wb") as handle:
                np.savez_compressed(handle, **arrays)
            os.replace(target + ".tmp", target)
            print(
                "%s scene %d view %d: %d grasps saved, %d draws, %.1f s"
                % (
                    camera,
                    scene,
                    view,
                    len(arrays["saved/centers_2d"]),
                    int(arrays["draws"]),
                    float(arrays["seconds"]),
                ),
                flush=True,
            )
        os.chdir(out)
        shutil.rmtree(work, ignore_errors=True)
    index = {
        "format": FORMAT,
        "upstream": {"commit": upstream_commit(upstream), "converter": "convert_grasp.run"},
        "views": [list(unit) for unit in VIEWS],
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    with open(os.path.join(out, "index.json.tmp"), "w") as handle:
        json.dump(index, handle, indent=1)
    os.replace(os.path.join(out, "index.json.tmp"), os.path.join(out, "index.json"))
    print("fixtures in %s" % out)


if __name__ == "__main__":
    main()
