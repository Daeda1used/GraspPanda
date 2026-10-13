"""Strict validation: schema, component signatures, cross-field rules."""

from __future__ import annotations

import copy

import pytest

from grasppanda.config import Config, ConfigError

from .conftest import MODEL


def issues(data, **kwargs):
    with pytest.raises(ConfigError) as error:
        Config.from_dict(data, **kwargs)
    return {issue.path: issue.message for issue in error.value.issues}


def model(**changes):
    node = copy.deepcopy(MODEL)
    for path, value in changes.items():
        *parents, key = path.split("__")
        target = node
        for parent in parents:
            target = target[parent]
        target[key] = value
    return {"model": node}


def test_valid_config_passes():
    Config.from_dict(model())


def test_unknown_top_level_key_suggests_the_right_one():
    found = issues({"runtim": {"seed": 1}})
    assert "unknown top-level key 'runtim'. Did you mean 'runtime'?" in found["runtim"]


def test_unknown_nested_key():
    assert "Did you mean 'epochs'?" in issues({"train": {"epoch": 3}})["train.epoch"]


def test_unknown_component_parameter_suggests_the_right_one():
    found = issues(model(head__num_angles=6))
    assert (
        "THead has no parameter 'num_angles'. Did you mean 'num_angle'?"
        in found["model.head.num_angles"]
    )


def test_missing_required_parameter():
    data = model()
    del data["model"]["head"]["num_angle"]
    assert issues(data)["model.head.num_angle"] == "THead requires 'num_angle'."


@pytest.mark.parametrize(
    ("change", "path", "fragment"),
    [
        ({"backbone__out_channels": "wide"}, "model.backbone.out_channels", "expected an integer"),
        ({"backbone__out_channels": True}, "model.backbone.out_channels", "expected an integer"),
        ({"backbone__norm": "gn"}, "model.backbone.norm", "expected one of 'bn', 'ln'"),
        ({"backbone__dims": [8, "x"]}, "model.backbone.dims.1", "expected an integer"),
        ({"head__loss_weight": "big"}, "model.head.loss_weight", "expected a number"),
        ({"seeds": {"type": "TBackbone"}}, "model.seeds", "is not a component slot"),
        ({"backbone": "TBackbone"}, "model.backbone", "expected a BACKBONES component"),
        ({"backbone": {"out_channels": 8}}, "model.backbone", "missing 'type'"),
    ],
)
def test_wrong_values(change, path, fragment):
    assert fragment in issues(model(**change))[path]


def test_integers_are_accepted_for_floats_and_null_for_optionals():
    Config.from_dict(model(head__loss_weight=2, backbone__dims=None))


def test_unknown_component_names_the_registry_and_the_right_one():
    message = issues(model(backbone__type="TBackbon"))["model.backbone.type"]
    assert "not registered in BACKBONES" in message and "Did you mean 'TBackbone'" in message


def test_component_from_another_registry_is_explained():
    message = issues(model(head__type="TBackbone"))["model.head.type"]
    assert "registered in BACKBONES, which owns a different slot" in message


def test_component_self_check_runs_only_on_structurally_valid_nodes():
    assert issues(model(seeds=5000))["model"] == "seeds must be at most 4096."
    assert "model" not in issues(model(seeds=5000, head__num_angles=1))


def test_all_problems_are_reported_together():
    data = {**model(head__num_angles=6), "runtim": {}, "train": {"epochs": "ten"}}
    assert set(issues(data)) == {"model.head.num_angles", "runtim", "train.epochs"}


def test_engine_provided_arguments_cannot_be_set():
    found = issues({"optimizer": {"type": "Adam", "params": [], "lrr": 0.1}})
    assert "supplied by the engine" in found["optimizer.params"]
    assert "Did you mean 'lr'?" in found["optimizer.lrr"]


def test_torch_optimizers_and_schedulers_are_available():
    Config.from_dict(
        {
            "optimizer": {"type": "AdamW", "lr": 1e-3},
            "scheduler": {"type": "StepLR", "step_size": 5},
        }
    )


def test_command_requirements():
    found = issues(model(), command="train")
    assert set(found) == {"data.train", "optimizer"}
    assert "`grasppanda train` needs this section" in found["optimizer"]


def test_save_best_must_name_a_declared_variant():
    evaluation = {"variants": {"raw": [], "cd": []}, "save_best": "nms/AP"}
    assert (
        "variant from evaluation.variants (raw, cd)"
        in issues({"evaluation": evaluation})["evaluation.save_best"]
    )
    Config.from_dict({"evaluation": {**evaluation, "save_best": "cd/AP"}})


def test_custom_import_failures_are_config_errors():
    assert (
        "cannot import 'no_such_module_xyz'"
        in issues({"custom_imports": ["no_such_module_xyz"]})["custom_imports.0"]
    )


def test_custom_import_by_path_is_relative_to_the_config(write):
    write(
        "plugin.py",
        "from grasppanda.models import Backbone\nfrom grasppanda.registry import BACKBONES\n\n\n@BACKBONES.register()\nclass PathPluginBackbone(Backbone):\n    output = 'dense'\n\n    def __init__(self, width: int = 4) -> None:\n        super().__init__()\n",
    )
    path = write(
        "sub/exp.yaml",
        "custom_imports: [../plugin.py]\nmodel: {type: TDetector, backbone: {type: PathPluginBackbone, width: 2}, head: {type: THead, num_angle: 2}}\n",
    )
    Config.load(path)
    Config.load(path)  # importing twice must not register twice


def test_defaults_are_written_out_and_the_digest_ignores_run_location():
    config = Config.from_dict(model())
    resolved = config.resolved()
    assert resolved["train"]["epochs"] == 10
    assert resolved["runtime"]["work_dir"] == "work_dirs/config"
    assert resolved["model"]["backbone"] == {
        "type": "TBackbone",
        "out_channels": 16,
        "in_channels": 3,
        "norm": "bn",
        "dims": None,
    }
    assert "in_channels" not in resolved["model"]["head"]  # injected, not a default
    assert config.select("model.head.loss_weight") == 1.0

    moved = Config.from_dict({**model(), "runtime": {"work_dir": "/elsewhere"}})
    changed = Config.from_dict(model(backbone__out_channels=8))
    assert config.digest() == moved.digest()
    assert config.digest() != changed.digest()


def test_the_digest_does_not_depend_on_the_installed_pytorch(monkeypatch):
    """Optimizers and schedulers are PyTorch's classes named by class, and their defaults
    change between versions (2.1 has a ``verbose`` argument of schedulers that 2.7 lacks).
    A config must hash the same on every machine, so those defaults are shown by
    ``resolved`` and left out of the digest; GraspPanda's own defaults stay in it."""
    import torch

    node = {
        **model(),
        "optimizer": {"type": "Adam", "lr": 1e-3},
        "scheduler": {"type": "StepLR", "step_size": 5},
    }
    config = Config.from_dict(node)
    assert config.resolved()["optimizer"]["betas"] == [0.9, 0.999]  # still shown
    before = config.digest()

    original = torch.optim.Adam.__init__

    def another_version(self, params, lr=1e-3, betas=(0.5, 0.5), brand_new_option=7):
        original(self, params, lr=lr, betas=betas)

    monkeypatch.setattr(torch.optim.Adam, "__init__", another_version)
    again = Config.from_dict(node)
    assert again.resolved()["optimizer"]["brand_new_option"] == 7
    assert again.digest() == before
    # What the user wrote for a library class is part of the experiment.
    assert Config.from_dict({**node, "optimizer": {"type": "Adam", "lr": 2e-3}}).digest() != before
    explicit = {**node, "optimizer": {"type": "Adam", "lr": 1e-3, "betas": [0.9, 0.99]}}
    assert Config.from_dict(explicit).digest() != before
    # A scheduler defined by GraspPanda keeps its defaults in the digest.
    cosine = {"type": "WarmupCosineLR", "warmup_steps": 5, "cosine_steps": 50}
    own = Config.from_dict({**node, "scheduler": cosine})
    assert own.resolved()["scheduler"]["min_factor"] == 1e-5  # a default of GraspPanda's own
    spelled = Config.from_dict({**node, "scheduler": {**cosine, "min_factor": 1e-5}})
    other = Config.from_dict({**node, "scheduler": {**cosine, "min_factor": 0.1}})
    assert own.digest() == spelled.digest() != other.digest()


def test_dump_round_trips():
    config = Config.from_dict({**model(), "optimizer": {"type": "Adam", "lr": 1e-5}})
    from grasppanda.config import parse_yaml

    assert parse_yaml(config.dump()) == config.to_dict()
    assert parse_yaml(config.dump(defaults=True)) == config.resolved()
