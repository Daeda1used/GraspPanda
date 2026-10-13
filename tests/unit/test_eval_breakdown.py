"""The split of AP by object and cause: exact, and attached to what the evaluator returns."""

from __future__ import annotations

import numpy as np
import pytest

from grasppanda.evaluation.breakdown import Tally, frame_tallies, rank_weights
from grasppanda.evaluation.graspnet1b import GRASPNET1B_SV_V1, GraspNet1BEvaluator
from grasppanda.evaluation.graspnet1b.frame import precision_table

from .eval_synthetic import write_dataset
from .test_eval_evaluator import key as eval_key
from .test_eval_evaluator import toy_prediction
from .test_eval_evaluator import write as write_prediction

PROTOCOL = GRASPNET1B_SV_V1
FRICTIONS = list(PROTOCOL.frictions)


def random_frame(generator: np.random.Generator, grasps: int, objects: int = 3) -> dict:
    """The decisions an evaluator makes for one frame, at random."""
    empty = generator.random(grasps) < 0.1
    collision = empty | (generator.random(grasps) < 0.3)
    friction = generator.choice([*FRICTIONS, -1.0], size=grasps)
    friction[collision] = -1.0
    return {
        "owner": generator.integers(0, objects, size=grasps),
        "collision": collision,
        "empty": empty,
        "friction": friction,
        "ranking": generator.permutation(grasps),
    }


def test_rank_weights_are_the_shares_of_a_precision_table():
    weights = rank_weights(50)
    assert np.isclose(weights.sum(), 1.0) and (np.diff(weights) < 0).all()
    assert np.isclose(weights[0], np.sum(1.0 / np.arange(1, 51)) / 50)  # about 9 % for rank 1
    assert np.isclose(weights[-1], 1.0 / 2500)
    for hit_rank in (0, 7, 49):  # one successful grasp at that rank, nothing else
        ranked = np.full(50, -1.0)
        ranked[hit_rank] = 0.2
        assert np.isclose(precision_table(ranked, PROTOCOL).mean(axis=0)[0], weights[hit_rank])


@pytest.mark.parametrize("grasps", [0, 1, 17, 50, 80])
def test_the_split_adds_up_to_the_precision_table(grasps):
    generator = np.random.default_rng(grasps)
    frame = random_frame(generator, grasps)
    whole, objects = frame_tallies(
        **frame, object_ids=[11, 22, 33], top_k=PROTOCOL.top_k, frictions=FRICTIONS
    )
    table = precision_table(frame["friction"][frame["ranking"]], PROTOCOL)
    assert np.allclose(whole.gain, table.mean(axis=0), atol=1e-12)  # earned = the frame's AP
    total = whole.gain + whole.no_force_closure + whole.collision + whole.empty + whole.unfilled
    assert np.allclose(total, 1.0, atol=1e-12)  # every rank is earned or lost for one reason
    assert whole.frames == 1 and whole.ranked == min(grasps, 50)
    assert np.isclose(whole.unfilled, rank_weights(50)[grasps:].sum())
    assert set(objects) <= {11, 22, 33}
    for name in ("gain", "no_force_closure"):
        parts = sum(getattr(part, name) for part in objects.values()) if objects else np.zeros(6)
        assert np.allclose(parts, getattr(whole, name), atol=1e-12)
    assert np.isclose(sum(part.collision for part in objects.values()), whole.collision)
    assert sum(part.ranked for part in objects.values()) == whole.ranked
    never = Tally().to_dict(6, with_unfilled=False)  # an object that was never ranked
    assert never["gain"] == [0.0] * 6 and "unfilled" not in never


def test_the_evaluator_attaches_the_split_to_its_result(tmp_path):
    """Frame 0: ranks 1 and 3 succeed, 2 and 4 fail; frame 1: an empty prediction."""
    dataset = write_dataset(tmp_path / "data", scenes=(0,), frames=2)
    write_prediction(tmp_path / "dump", eval_key(0), toy_prediction())
    write_prediction(tmp_path / "dump", eval_key(1), np.zeros((0, 17), dtype=np.float32))
    evaluator = GraspNet1BEvaluator(data_root=str(dataset), cache_dir=str(tmp_path / "cache"))
    result = evaluator.evaluate(tmp_path / "dump", [eval_key(0), eval_key(1)])
    split = result.breakdown
    assert split["top_k"] == 50 and split["frictions"] == FRICTIONS
    (scene,) = split["per_scene"].values()
    weights = rank_weights(50)
    assert scene["frames"] == 2 and scene["ranked"] == 4
    assert np.allclose(scene["gain"], weights[0] + weights[2])
    assert np.allclose(np.asarray(scene["gain"]) / 2, [result.metrics["AP"]] * 6)  # the same number
    assert np.isclose(scene["unfilled"], weights[4:].sum() + 1.0)  # frame 1 filled no rank at all
    lost = np.asarray(scene["no_force_closure"]) + scene["collision"] + scene["empty"]
    assert np.allclose(lost, weights[1] + weights[3])
    assert set(split["per_object"]) == {"0", "1"}  # both objects of the scene, by dataset id
    assert sum(entry["ranked"] for entry in split["per_object"].values()) == 4
    assert all(entry["frames"] == 2 for entry in split["per_object"].values())
    gains = np.sum([entry["gain"] for entry in split["per_object"].values()], axis=0)
    assert np.allclose(gains, scene["gain"])
