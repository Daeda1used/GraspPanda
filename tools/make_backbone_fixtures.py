"""Record reference outputs of official backbone implementations.

"Faithful" has to be demonstrated. This script runs an official implementation as an
oracle on a fixed input and stores (a sample of) its output in
``tests/integration/fixtures/backbones/``. The tests in
``tests/integration/test_backbone_*_oracle.py`` then run the GraspPanda backbone on the
same input and compare.

The script never imports ``grasppanda``: each oracle needs its own environment (the
GraspNet-baseline CUDA extension, for example, lives in a Python 3.8 environment), so
it only depends on ``numpy``, ``torch`` and whatever the oracle itself imports. Upstream
checkouts and weights are passed as paths; nothing is downloaded and nothing upstream
is modified.

    # the shared input scene
    python tools/make_backbone_fixtures.py scene

    # codes of the official serialisation (Pointcept/PointTransformerV3, MIT)
    python tools/make_backbone_fixtures.py serialization --upstream <PointTransformerV3>

    # GraspNet-baseline backbone with the authors' weights (needs its pointnet2 extension)
    python tools/make_backbone_fixtures.py pointnet2-graspnet \
        --upstream <graspnet-baseline> --weights <checkpoint-rs.tar>

    # Point Transformer V3, official model (needs spconv, torch_scatter, timm, addict):
    # with analytic weights, and with the released ScanNet weights
    python tools/make_backbone_fixtures.py ptv3 --upstream <PointTransformerV3>
    python tools/make_backbone_fixtures.py ptv3 --upstream <PointTransformerV3> \
        --weights <scannet-semseg-pt-v3m1-0-base.pth> --name ptv3_scannet

    # Sonata-family encoders from the official sonata / concerto / utonia package
    python tools/make_backbone_fixtures.py ptv3-sonata --upstream <Concerto> \
        --package concerto --weights <concerto_small.pth> --name concerto_small
    python tools/make_backbone_fixtures.py ptv3-sonata --upstream <Utonia> \
        --package utonia --name utonia_analytic        # no weights: analytic, with decoder

    # LitePT, official standalone model (needs spconv, torch_scatter, timm, addict,
    # colorhash). The committed fixtures were recorded with flash-attention installed and
    # the official PointROPE extension (built from <LitePT>/libs/pointrope) on PYTHONPATH;
    # without them the official PyTorch PointROPE and a stand-in for flash-attention run,
    # and the fixture says so (see flash_attention()). With analytic weights, and with
    # released weights and their published config:
    python tools/make_backbone_fixtures.py litept --upstream <LitePT> --name litept_analytic
    python tools/make_backbone_fixtures.py litept --upstream <LitePT> --name litept_scannet \
        --config <LitePT>/configs/scannet/semseg-litept-small-v1m1.py \
        --weights <scannet-semseg-litept-small-v1m1.pth>

    # PointNeXt, official openpoints model built from a published YAML, analytic weights
    # (needs the openpoints CUDA extension: cpp/pointnet2_batch, build_ext --inplace)
    python tools/make_backbone_fixtures.py pointnext --upstream <openpoints> \
        --config <PointNeXt>/cfgs/s3dis/pointnext-s.yaml --name pointnext_s

    # OA-CNNs, official Pointcept model (needs spconv, timm, torch_geometric 2.6 and
    # torch_cluster; --commit names the commit of an unpacked archive). Three fixtures,
    # no released weights: the model of the published config file, with hashed weights
    # and calibrated batch norm (see hashed_state, calibrate_norms); a small model that
    # switches on what that file leaves alone, with analytic weights; and that small
    # model on a crafted cloud (see notch_cloud), hashed and calibrated again
    python tools/make_backbone_fixtures.py oacnns --upstream <Pointcept> --commit <sha> \
        --config <Pointcept>/configs/scannet/semseg-oacnns-v1m1-0-base.py --calibrate \
        --name oacnns_base
    python tools/make_backbone_fixtures.py oacnns --upstream <Pointcept> --commit <sha> \
        --name oacnns_analytic
    python tools/make_backbone_fixtures.py oacnns --upstream <Pointcept> --commit <sha> \
        --notch --calibrate --name oacnns_notch

    # SpUNet, official Pointcept model (same packages): the model of the published
    # config file, hashed and calibrated, and a small one without final layer, analytic
    python tools/make_backbone_fixtures.py spunet --upstream <Pointcept> --commit <sha> \
        --config <Pointcept>/configs/scannet/semseg-spunet-v1m1-0-base.py --calibrate \
        --name spunet_base
    python tools/make_backbone_fixtures.py spunet --upstream <Pointcept> --commit <sha> \
        --name spunet_analytic

Weights are not stored. Where an oracle runs with released weights the fixture records
their SHA-256; where it runs without, both sides fill every tensor of the state dict
with :func:`analytic_state`, a closed-form function of the tensor's name and shape, so
the comparison also proves that the two state dicts have the same keys.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import pickle
import runpy
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tests" / "integration" / "fixtures" / "backbones"


# --------------------------------------------------------------------------- inputs


def tabletop_cloud(count: int, seed: int = 0) -> np.ndarray:
    """A synthetic single-view tabletop scene in the camera frame, in metres.

    A tilted table with a dozen spheres and boxes on it, seen from above, with sensor
    noise. Not a benchmark scene: it only has to exercise the operators at a realistic
    scale and density (objects of 3 to 8 cm, about 0.5 m from the camera).
    """
    rng = np.random.RandomState(seed)  # the legacy generator has a frozen stream
    table = int(count * 0.55)
    xy = rng.uniform([-0.30, -0.20], [0.30, 0.20], size=(table, 2))
    points = [np.concatenate([xy, np.zeros((table, 1))], axis=1)]
    remaining = count - table
    objects = 12
    for index in range(objects):
        share = remaining // objects + (index < remaining % objects)
        centre = rng.uniform([-0.25, -0.15], [0.25, 0.15])
        size = rng.uniform(0.03, 0.08)
        if index % 2:  # the upper half of a sphere resting on the table
            direction = rng.normal(size=(share, 3))
            direction[:, 2] = np.abs(direction[:, 2])
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            surface = direction * size / 2 + [0.0, 0.0, size / 2]
        else:  # the top face and two sides of a box
            face = rng.randint(0, 3, size=share)
            uv = rng.uniform(-0.5, 0.5, size=(share, 2)) * size
            surface = np.zeros((share, 3))
            top, front, side = face == 0, face == 1, face == 2
            surface[top] = np.stack([uv[top, 0], uv[top, 1], np.full(top.sum(), size)], axis=1)
            surface[front] = np.stack(
                [uv[front, 0], np.full(front.sum(), -size / 2), uv[front, 1] + size / 2], axis=1
            )
            surface[side] = np.stack(
                [np.full(side.sum(), -size / 2), uv[side, 0], uv[side, 1] + size / 2], axis=1
            )
        points.append(surface + np.array([centre[0], centre[1], 0.0]))
    scene = np.concatenate(points)
    scene += rng.normal(scale=0.001, size=scene.shape)
    # Table frame (z up) to camera frame (z forward), camera 0.55 m away, tilted 35 degrees.
    tilt = np.deg2rad(35.0)
    rotation = np.array(
        [[1, 0, 0], [0, -np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), -np.cos(tilt)]]
    )
    scene = scene @ rotation.T + [0.0, 0.0, 0.55]
    return scene[rng.permutation(len(scene))].astype(np.float32)


def scene() -> np.ndarray:
    """The shared input cloud, stored once so that every fixture refers to the same bits."""
    path = OUT / "tabletop.npz"
    cloud = tabletop_cloud(20000)
    if path.is_file() and not np.array_equal(np.load(path)["cloud"], cloud):
        raise SystemExit(f"{path} does not match tabletop_cloud(); refusing to mix inputs")
    return cloud


def point_features(cloud: np.ndarray, channels: int) -> np.ndarray:
    """Per-point input features: low-order monomials of the coordinates.

    Only float32 subtraction and multiplication are used, so NumPy and PyTorch produce
    the same bits and the features need not be stored.
    """
    x, y, z = (cloud - np.array([0.0, 0.0, 0.5], dtype=np.float32)).T
    monomials = [x, y, z, x * y, y * z, z * x, x * x, y * y, z * z, x * y * z, x * x * y, y * y * z]
    scale = np.float32(4.0)
    return np.stack([monomials[index % 12] * scale for index in range(channels)], axis=1)


def voxels(cloud: np.ndarray, grid_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantise as the official grid sampling does; one voxel per row, sorted.

    Returns:
        Grid coordinates of the voxels, the first point of each voxel, and the voxel
        of every point.
    """
    grid = np.floor(cloud.astype(np.float64) / grid_size).astype(np.int64)
    grid -= grid.min(0)
    return np.unique(grid, axis=0, return_index=True, return_inverse=True)


def analytic_state(model: torch.nn.Module) -> None:
    """Overwrite every parameter and buffer with a closed-form function of its name.

    The values are reproducible on any platform (float64 sines, then cast), bounded,
    and scaled so that activations neither vanish nor explode: weights by
    ``1 / sqrt(fan_in)``, norm scales around one, variances positive.
    """
    state = model.state_dict()
    for name, tensor in state.items():
        leaf = name.rsplit(".", 1)[-1]
        if not tensor.is_floating_point() or leaf == "inv_freq":
            continue  # counters, and the fixed frequencies of rotary embeddings
        phase = zlib.crc32(name.encode()) % 1000
        wave = torch.sin(torch.arange(tensor.numel(), dtype=torch.float64) * 0.37 + phase)
        if leaf == "running_var":
            values = 1.0 + 0.5 * wave.pow(2)
        elif leaf == "running_mean":
            values = 0.1 * wave
        elif tensor.ndim >= 2:
            fan_in = tensor.numel() // tensor.shape[0]
            values = wave * (1.5 / fan_in**0.5)
        elif leaf in ("weight", "gamma"):  # one-dimensional weights are norm scales
            values = 1.0 + 0.1 * wave
        else:
            values = 0.1 * wave
        state[name] = values.reshape(tensor.shape).to(tensor.dtype)
    model.load_state_dict(state)


# -------------------------------------------------------------------------- helpers


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def commit_of(checkout: Path) -> str:
    """Commit a checkout is at, read from its ``.git`` directory (no git process needed)."""
    git = checkout / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[len("ref: ") :]
        if (git / ref).is_file():
            return (git / ref).read_text().strip()
        for line in (git / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return "unknown"


def load_checkpoint(path: Path) -> dict:
    """Unpickle a checkpoint without letting it run code.

    Released inference checkpoints hold tensors only. Training checkpoints also carry
    optimiser and scheduler state, which refers to a few harmless globals; those are
    allowed by name and nothing else is.
    """
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        from torch.optim.lr_scheduler import OneCycleLR

        scalar = (getattr(np, "_core", None) or np.core).multiarray.scalar
        allowed = [
            getattr,
            np.dtype,
            OneCycleLR,
            type(np.dtype("f8")),
            (scalar, "numpy.core.multiarray.scalar"),  # its path before NumPy 2
        ]
        with torch.serialization.safe_globals(allowed):
            return torch.load(str(path), map_location="cpu", weights_only=True)


def state_keys(model: torch.nn.Module) -> str:
    """Digest of the state-dict layout: every key with its shape, sorted."""
    lines = sorted(f"{key} {tuple(value.shape)}" for key, value in model.state_dict().items())
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def save(name: str, **arrays: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.npz"
    arrays["torch_version"] = torch.__version__
    arrays["python_version"] = sys.version.split()[0]
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in arrays.items()})
    print(f"wrote {path.relative_to(REPO)} ({path.stat().st_size / 1024:.0f} KiB)")


def sample_rows(count: int, keep: int) -> np.ndarray:
    """Evenly spaced row indices: enough to compare, small enough to commit."""
    return np.unique(np.linspace(0, count - 1, min(keep, count)).round().astype(np.int64))


def save_dense(name: str, dense: np.ndarray, keep: int, **extra: object) -> None:
    """Store sampled rows of per-point features plus statistics over all rows."""
    rows = sample_rows(len(dense), keep)
    save(
        name,
        feature_rows=rows.astype(np.int32),
        features=dense[rows],
        feature_mean=dense.mean(0),
        feature_std=dense.std(0),
        **extra,
    )


# ---------------------------------------------------------------------------- scene


def record_scene(args: argparse.Namespace) -> None:
    del args
    save("tabletop", cloud=tabletop_cloud(20000))


# -------------------------------------------------------------------- serialization


def record_serialization(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.upstream))
    encode = importlib.import_module("serialization").encode
    rng = np.random.RandomState(0)
    arrays: dict[str, object] = {"upstream_commit": commit_of(args.upstream)}
    for depth in (1, 3, 7, 10, 16):
        grid = rng.randint(0, 1 << depth, size=(256, 3)).astype(np.int64)
        batch = rng.randint(0, 3, size=256).astype(np.int64)
        arrays[f"grid_{depth}"] = grid
        arrays[f"batch_{depth}"] = batch
        for order in ("z", "z-trans", "hilbert", "hilbert-trans"):
            code = encode(torch.from_numpy(grid), torch.from_numpy(batch), depth, order=order)
            arrays[f"code_{depth}_{order}"] = code.numpy()
    save("serialization", **arrays)


# --------------------------------------------------------------- GraspNet-baseline


def record_pointnet2_graspnet(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.upstream / "models"))
    sys.path.insert(0, str(args.upstream / "pointnet2"))
    backbone = importlib.import_module("backbone").Pointnet2Backbone(0).cuda().eval()
    prefix = "view_estimator.backbone."
    checkpoint = load_checkpoint(args.weights)["model_state_dict"]
    state = {key[len(prefix) :]: value for key, value in checkpoint.items() if prefix in key}
    backbone.load_state_dict(state, strict=True)

    cloud = scene()
    with torch.no_grad():
        feats, xyz, end_points = backbone(torch.from_numpy(cloud)[None].cuda())
    feats = feats[0].T.cpu().numpy()  # (C, K) -> (K, C)
    save_dense(
        "pointnet2_graspnet",
        feats,
        keep=128,
        seed_index=end_points["fp2_inds"][0].cpu().numpy().astype(np.int32),
        seed_xyz=xyz[0].cpu().numpy(),
        parameters=sum(p.numel() for p in backbone.parameters()),
        weights_sha256=sha256(args.weights),
        upstream_commit=commit_of(args.upstream),
    )


# ------------------------------------------------------------------------ PTv3 family

#: Voxel edge used for the PTv3 fixtures: fine enough that the scene spans many
#: attention patches, so the completion of the last, incomplete patch is exercised.
PTV3_GRID = 0.005

#: A small PT-v3m1 that switches on what the published configuration leaves off
#: (relative position bias, post-norm, patch sizes that vary by stage).
PTV3_ANALYTIC = {
    "in_channels": 4,
    "order": ("z", "z-trans", "hilbert", "hilbert-trans"),
    "stride": (2, 2, 2),
    "enc_depths": (1, 2, 1, 1),
    "enc_channels": (16, 32, 64, 96),
    "enc_num_head": (2, 4, 8, 8),
    "enc_patch_size": (64, 64, 48, 16),
    "dec_depths": (1, 1, 2),
    "dec_channels": (16, 32, 48),
    "dec_num_head": (2, 4, 8),
    "dec_patch_size": (64, 64, 48),
    "mlp_ratio": 2,
    "enable_rpe": True,
    "pre_norm": False,
}

#: A small Sonata-family model with a decoder and layer scale. Head widths are
#: multiples of 6 so that the same configuration also suits rotary embedding.
SONATA_ANALYTIC = {
    "in_channels": 9,
    "order": ("z", "z-trans", "hilbert", "hilbert-trans"),
    "stride": (2, 2, 2),
    "enc_depths": (1, 2, 1, 1),
    "enc_channels": (24, 48, 96, 96),
    "enc_num_head": (2, 4, 8, 8),
    "enc_patch_size": (64, 64, 48, 16),
    "dec_depths": (1, 1, 2),
    "dec_channels": (24, 48, 96),
    "dec_num_head": (2, 4, 8),
    "dec_patch_size": (64, 64, 48),
    "mlp_ratio": 2,
    "layer_scale": 0.5,
    "mask_token": True,
    "enc_mode": False,
}

#: Keys of a released configuration that select an attention backend, not an architecture.
BACKEND_KEYS = ("enable_flash", "upcast_attention", "upcast_softmax", "traceable")


def freeze_orders(model: torch.nn.Module) -> None:
    """Switch off the random permutation of the serialisation orders.

    The official models shuffle the orders among the blocks on every forward pass, in
    evaluation mode too, through an attribute of the model and of each pooling layer.
    """
    for module in model.modules():
        if hasattr(module, "shuffle_orders"):
            module.shuffle_orders = False


def ptv3_inputs(channels: int, device: torch.device) -> tuple[dict, np.ndarray]:
    """Voxel-level input of an official PTv3-family model and the voxel of every point."""
    cloud = scene()
    grid, first, inverse = voxels(cloud, PTV3_GRID)
    feats = point_features(cloud, channels)
    data = {
        "coord": torch.from_numpy(cloud[first]).to(device),
        "grid_coord": torch.from_numpy(grid).int().to(device),
        "feat": torch.from_numpy(feats[first]).to(device),
        "offset": torch.tensor([len(grid)], device=device),
    }
    return data, inverse.reshape(-1)


def record_ptv3(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.upstream.parent))
    module = importlib.import_module(f"{args.upstream.name}.model")
    device = torch.device("cuda")
    if args.weights is None:
        config = dict(PTV3_ANALYTIC)
        model = module.PointTransformerV3(**config, enable_flash=False)
        analytic_state(model)
        extra = {"weights": "analytic"}
    else:
        config = {}  # the defaults of the official model are the published ScanNet model
        model = module.PointTransformerV3(enable_flash=False)
        state = load_checkpoint(args.weights)["state_dict"]
        prefix = "module.backbone." if any(k.startswith("module.") for k in state) else "backbone."
        model.load_state_dict(
            {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}, strict=True
        )
        extra = {"weights": args.weights.name, "weights_sha256": sha256(args.weights)}
    model = model.to(device).eval()
    freeze_orders(model)
    data, inverse = ptv3_inputs(model.embedding.in_channels, device)
    with torch.no_grad():
        feat = model(data).feat.cpu().numpy()
    save_dense(
        args.name or "ptv3_analytic",
        feat[inverse],
        keep=128,
        grid_size=PTV3_GRID,
        config=json.dumps(config),
        state_keys=state_keys(model),
        voxels=len(feat),
        parameters=sum(p.numel() for p in model.parameters()),
        upstream_commit=commit_of(args.upstream),
        **extra,
    )


def record_ptv3_sonata(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.upstream))
    module = importlib.import_module(f"{args.package}.model")
    device = torch.device("cuda")
    if args.weights is None:
        config = dict(SONATA_ANALYTIC)
        model = module.PointTransformerV3(**config, enable_flash=False)
        analytic_state(model)
        extra = {"weights": "analytic"}
    else:
        model = module.load(str(args.weights), custom_config={"enable_flash": False})
        released = load_checkpoint(args.weights)["config"]
        config = {key: value for key, value in released.items() if key not in BACKEND_KEYS}
        extra = {"weights": args.weights.name, "weights_sha256": sha256(args.weights)}
    model = model.to(device).eval()
    freeze_orders(model)
    data, inverse = ptv3_inputs(model.embedding.in_channels, device)
    with torch.no_grad():
        point = model(data)
        # Encoder-only models return the coarsest level. Concatenate every stage through
        # the pooling maps, the official linear-probing protocol (DefaultSegmentorV2).
        while "pooling_parent" in point:
            parent = point.pop("pooling_parent")
            parent.feat = torch.cat([parent.feat, point.feat[point.pop("pooling_inverse")]], -1)
            point = parent
        feat = point.feat.cpu().numpy()
    save_dense(
        args.name,
        feat[inverse],
        keep=128 if args.weights is None else 16,
        grid_size=PTV3_GRID,
        config=json.dumps(config),
        package=args.package,
        state_keys=state_keys(model),
        voxels=len(feat),
        parameters=sum(p.numel() for p in model.parameters()),
        upstream_commit=commit_of(args.upstream),
        **extra,
    )


# ---------------------------------------------------------------------------- LitePT

#: Voxel edge for the LitePT fixtures with released weights. The published models attend
#: only after three poolings; at 1.25 mm those two stages still hold several patches of
#: 1024 voxels (4 265 and 1 215 voxels of the scene), so the patch layout is exercised.
LITEPT_GRID = 0.00125

#: Small models that switch on what the published configurations leave off. The first has
#: a stage that convolves and attends, decoder blocks of both kinds, a stride of 3 and
#: PointROPE bases that vary by stage; the second attends from the input level on, in two
#: curves, post-norm, without a query/key/value bias and with a fixed attention scale.
LITEPT_ANALYTIC = {
    "litept_analytic": {
        "in_channels": 4,
        "order": ("z", "z-trans", "hilbert", "hilbert-trans"),
        "stride": (2, 3, 2),
        "enc_depths": (1, 2, 5, 2),
        "enc_channels": (16, 32, 48, 96),
        "enc_num_head": (2, 4, 4, 8),
        "enc_patch_size": (64, 64, 48, 16),
        "enc_conv": (True, True, True, False),
        "enc_attn": (False, False, True, True),
        "enc_rope_freq": (100.0, 100.0, 50.0, 10.0),
        "dec_depths": (1, 1, 2),
        "dec_channels": (24, 32, 48),
        "dec_num_head": (2, 4, 4),
        "dec_patch_size": (64, 64, 48),
        "dec_conv": (True, False, False),
        "dec_attn": (False, False, True),
        "dec_rope_freq": (100.0, 100.0, 20.0),
        "mlp_ratio": 2,
    },
    "litept_analytic_dense": {
        "in_channels": 4,
        "order": ("hilbert", "z-trans"),
        "stride": (2, 2),
        "enc_depths": (2, 1, 3),
        "enc_channels": (12, 24, 48),
        "enc_num_head": (1, 2, 4),
        "enc_patch_size": (128, 64, 32),
        "enc_conv": (False, True, False),
        "enc_attn": (True, True, True),
        "enc_rope_freq": (1000.0, 100.0, 10.0),
        "dec_depths": (2, 1),
        "dec_channels": (12, 24),
        "dec_num_head": (2, 2),
        "dec_patch_size": (128, 64),
        "dec_conv": (True, False),
        "dec_attn": (True, True),
        "dec_rope_freq": (100.0, 30.0),
        "mlp_ratio": 3,
        "qkv_bias": False,
        "qk_scale": 0.5,
        "pre_norm": False,
    },
}


def varlen_attention(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """What ``flash_attn.flash_attn_varlen_qkvpacked_func`` computes, in plain PyTorch.

    ``qkv`` is ``(n, 3, heads, width)``; rows ``cu_seqlens[i]:cu_seqlens[i + 1]`` are one
    sequence. The result ``(n, heads, width)`` is ``softmax(q kᵀ · scale) v`` within each
    sequence. Products and the softmax are evaluated in float32 (at least); the result is
    returned in the precision of the input, as the CUDA kernel returns it.
    """
    del max_seqlen  # a buffer size for the CUDA kernel
    if dropout_p:
        raise NotImplementedError("the oracle runs in evaluation mode")
    scale = qkv.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    out = torch.empty_like(qkv[:, 0])
    precise = qkv.to(torch.promote_types(qkv.dtype, torch.float32))
    start = 0
    for stop in cu_seqlens[1:].tolist():
        q, k, v = precise[start:stop].permute(1, 2, 0, 3)  # each (heads, length, width)
        weights = torch.softmax(q @ k.transpose(-1, -2) * scale, dim=-1)
        out[start:stop] = (weights @ v).transpose(0, 1).to(qkv.dtype)
        start = stop
    return out


def flash_attention() -> tuple[object, str]:
    """The ``flash_attn`` module the official LitePT will import, and what it is.

    LitePT has no attention path without flash-attention: it casts queries, keys and
    values to float16 and calls ``flash_attn_varlen_qkvpacked_func``. Where the package is
    not installed, a module holding :func:`varlen_attention` under that name takes its
    place. That replaces a library call with documented semantics, not anything LitePT
    itself computes: the projections, PointROPE, the float16 casts, the patch layout and
    the mapping back all run in the official code. Where the package is installed, the
    recorder measures how far the stand-in is from it (``stand_in_difference``).
    """
    try:
        module = importlib.import_module("flash_attn")
    except ImportError:
        module = type(sys)("flash_attn")
        module.flash_attn_varlen_qkvpacked_func = varlen_attention
        sys.modules["flash_attn"] = module
        return module, "stand-in: softmax attention in PyTorch (tools/make_backbone_fixtures.py)"
    return module, f"flash_attn {module.__version__}"


@contextlib.contextmanager
def replaced(owner: object, name: str, value: object):
    """Swap one attribute for the duration of a block. Nothing upstream is edited."""
    original = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, original)


def load_checkpoint_torch24(path: Path) -> dict:
    """:func:`load_checkpoint` for an environment with real flash-attention.

    The newest PyTorch that flash-attention ships a cu118 wheel for is 2.4, which has the
    allow-list of the safe unpickler but not yet its context manager, and matches entries
    by their own module path: such an environment needs NumPy 1.x, where the scalar
    constructor still lives under the name the checkpoints refer to.
    """
    if hasattr(torch.serialization, "safe_globals"):
        return load_checkpoint(path)
    import _codecs  # its ``encode`` rebuilds the bytes of a NumPy scalar

    from torch.optim.lr_scheduler import OneCycleLR

    allowed = [getattr, np.dtype, OneCycleLR, type(np.dtype("f8")), np.core.multiarray.scalar]
    torch.serialization.add_safe_globals([*allowed, _codecs.encode])
    return torch.load(str(path), map_location="cpu", weights_only=True)


def record_litept(args: argparse.Namespace) -> None:
    flash, backend = flash_attention()
    sys.path.insert(0, str(args.upstream))
    module = importlib.import_module("litept.model")
    device = torch.device("cuda")
    if args.weights is None:
        config = dict(LITEPT_ANALYTIC[args.name])
        grid_size = PTV3_GRID
        model = module.LitePT(**config)
        analytic_state(model)
        extra = {"weights": "analytic"}
    else:
        # A published config is a file of plain assignments; ``model`` holds the backbone.
        published = runpy.run_path(str(args.config))["model"]["backbone"]
        config = {key: value for key, value in published.items() if key != "type"}
        if config.pop("enc_mode", False):
            raise SystemExit("an encoder-only LitePT has no dense output to record")
        grid_size = LITEPT_GRID
        model = module.LitePT(**config)
        state = load_checkpoint_torch24(args.weights)["state_dict"]
        prefix = "module.backbone."
        model.load_state_dict(
            {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}, strict=True
        )
        extra = {"weights": args.weights.name, "weights_sha256": sha256(args.weights)}
    model = model.to(device).eval()
    freeze_orders(model)

    cloud = scene()
    grid, first, inverse = voxels(cloud, grid_size)
    feats = point_features(cloud, model.embedding.in_channels)
    data = {
        "coord": torch.from_numpy(cloud[first]).to(device),
        "grid_coord": torch.from_numpy(grid).to(device),  # int64, as the official transforms
        "feat": torch.from_numpy(feats[first]).to(device),
        "offset": torch.tensor([len(grid)], device=device),
    }

    def run() -> np.ndarray:
        with torch.no_grad():
            return model(dict(data)).feat.float().cpu().numpy()[inverse.reshape(-1)]

    # 1. The official forward pass as released: float16 attention.
    half = run()
    # 2. The same with the stand-in for flash-attention, to measure the stand-in.
    with replaced(flash, "flash_attn_varlen_qkvpacked_func", varlen_attention):
        emulated = run()
        # 3. The same with the attention in float32, which is what the GraspPanda port
        # computes. The only casts to half precision in the official forward pass are
        # the ``.half()`` of the attention module, so ``Tensor.half`` returns its tensor
        # unchanged here. (flash-attention accepts float16 and bfloat16 only, hence the
        # stand-in.)
        with replaced(torch.Tensor, "half", lambda tensor, *args, **kwargs: tensor):
            dense = run()

    ropes = {type(m).__module__ for m in model.modules() if type(m).__name__ == "PointROPE"}
    keep = 64 if args.weights is None else 32
    print(
        f"{args.name}: {len(grid)} voxels, |f| {np.abs(dense).max():.3g}; {backend}; "
        f"PointROPE from {sorted(ropes)}\n  official float16 vs float32 attention: "
        f"{np.abs(half - dense).max():.3g}; backend vs stand-in, both float16: "
        f"{np.abs(half - emulated).max():.3g}"
    )
    if args.weights is not None:
        # With trained weights the output as released is evidence of its own. (Analytic
        # weights make the logits, and with them the effect of the cast, arbitrary.)
        extra["half_features"] = half[sample_rows(len(dense), keep)]
    save_dense(
        args.name,
        dense,
        keep=keep,
        half_difference=np.abs(half - dense).max(),
        stand_in_difference=np.abs(half - emulated).max(),
        attention_backend=backend,
        pointrope=", ".join(sorted(ropes)),
        grid_size=grid_size,
        config=json.dumps(config),
        state_keys=state_keys(model),
        voxels=len(grid),
        parameters=sum(p.numel() for p in model.parameters()),
        upstream_commit=args.commit or commit_of(args.upstream),
        **extra,
    )


# ------------------------------------------------------------------------- PointNeXt


def record_pointnext(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.upstream.parent))
    config = importlib.import_module("openpoints.utils").EasyConfig()
    config.load(str(args.config), recursive=False)
    config.model.cls_args = None  # encoder and decoder only: the backbone
    model = importlib.import_module("openpoints.models").build_model_from_cfg(config.model)
    analytic_state(model)
    model = model.cuda().eval()

    cloud = scene()
    feats = point_features(cloud, config.model.encoder_args.in_channels)
    data = {
        "pos": torch.from_numpy(cloud)[None].cuda(),
        "x": torch.from_numpy(feats).T[None].contiguous().cuda(),
    }
    with torch.no_grad():
        dense = model(data)[0].T.cpu().numpy()  # (C, N) -> (N, C)
    encoder = config.model.encoder_args
    keys = ("in_channels", "width", "blocks", "strides", "sa_layers", "sa_use_res")
    keys += ("expansion", "radius", "nsample")
    save_dense(
        args.name,
        dense,
        keep=128,
        config=json.dumps({key: encoder[key] for key in keys}),
        weights="analytic",
        state_keys=state_keys(model),
        parameters=sum(p.numel() for p in model.parameters()),
        upstream_commit=commit_of(args.upstream),
    )


# --------------------------------------------------------- sparse CNNs of Pointcept

#: A small OA-CNNs that differs from the published model wherever a mix-up would
#: otherwise go unnoticed: three stages, a width that changes at every level, one to
#: three pooling grids per stage, and pooling grids that are not multiples of each other.
#: Keys are the constructor arguments of the GraspPanda backbone.
OACNNS_ANALYTIC = {
    "in_channels": 4,
    "out_channels": 24,
    "embed_channels": 16,
    "enc_channels": (24, 32, 48),
    "enc_depth": (2, 1, 2),
    "dec_channels": (20, 28, 40),
    "point_grid_size": ((5, 9, 16), (3, 7), (4,)),
}

#: A small SpUNet in the same spirit: three stages, no two widths alike, block counts
#: from one to three, and no final layer (the official ``num_classes=0``), which the
#: published segmentation model does not exercise.
SPUNET_ANALYTIC = {
    "in_channels": 4,
    "out_channels": None,
    "base_channels": 12,
    "channels": (16, 24, 40, 32, 20, 28),
    "layers": (1, 2, 3, 2, 1, 2),
}


def pointcept_module(upstream: Path, name: str) -> object:
    """Import one model file of a Pointcept checkout without its model zoo.

    ``pointcept/models/__init__.py`` imports every model of the zoo, and with them
    compiled extensions the sparse CNNs do not use. ``pointcept`` and
    ``pointcept.models`` are therefore entered as bare packages whose path is the
    checkout. Every module imported through them (the model file, ``models.builder``,
    ``models.utils``, ``utils.registry``) is the official file; nothing is stubbed.
    """
    for package in ("pointcept", "pointcept.models"):
        if package not in sys.modules:
            shell = type(sys)(package)
            shell.__path__ = [str(upstream / package.replace(".", "/"))]
            sys.modules[package] = shell
    return importlib.import_module(f"pointcept.models.{name}")


def published_backbone(config: Path) -> dict:
    """The ``model.backbone`` entry of a Pointcept config file, read without running it."""
    import ast

    for statement in ast.parse(config.read_text()).body:
        targets = getattr(statement, "targets", [])
        if any(getattr(target, "id", "") == "model" for target in targets):
            for entry in statement.value.keywords:  # model = dict(type=..., backbone=dict(...))
                if entry.arg == "backbone":
                    return {key.arg: ast.literal_eval(key.value) for key in entry.value.keywords}
    raise SystemExit(f"{config} does not assign `model = dict(backbone=dict(...))`")


def hashed_state(model: torch.nn.Module) -> None:
    """Closed-form state for deep networks: :func:`analytic_state` with a hash for the sine.

    A sine of the flat index makes every weight matrix (nearly) rank two, with a largest
    singular value that grows with the square root of the width. A few blocks tolerate
    that. Through the dozens of residual blocks of a published sparse CNN, rounding
    differences are amplified until the same model disagrees with itself between CPU and
    GPU (measured: 3 % of the output). Here every value comes from a 32-bit integer hash
    of its name and flat index instead: exact on every platform, and the matrices are
    full rank with singular values of order one.
    """
    mask = np.uint64(0xFFFFFFFF)
    state = model.state_dict()
    for name, tensor in state.items():
        if not tensor.is_floating_point():
            continue
        leaf = name.rsplit(".", 1)[-1]
        seed = np.uint64(zlib.crc32(name.encode()))
        bits = (np.arange(tensor.numel(), dtype=np.uint64) + seed) & mask
        for shift, factor in ((16, 0x7FEB352D), (15, 0x31848BAB), (16, 0x2C1B3C6D)):
            bits = ((bits ^ (bits >> np.uint64(shift))) * np.uint64(factor)) & mask
        bits ^= bits >> np.uint64(15)
        wave = torch.from_numpy(bits.astype(np.float64) / 2.0**31 - 1.0)  # uniform in [-1, 1)
        if leaf == "running_var":
            values = 1.0 + 0.5 * wave.pow(2)
        elif leaf == "running_mean":
            values = 0.1 * wave
        elif tensor.ndim >= 2:
            fan_in = tensor.numel() // tensor.shape[0]
            values = wave * (3.0 / fan_in) ** 0.5  # variance 1 / fan_in
        elif leaf == "weight":  # one-dimensional weights are norm scales
            values = 1.0 + 0.1 * wave
        else:
            values = 0.1 * wave
        state[name] = values.reshape(tensor.shape).to(tensor.dtype)
    model.load_state_dict(state)


def calibrate_norms(model: torch.nn.Module, run: object) -> None:
    """Replace the running statistics of every batch norm by those of one input.

    A closed-form state gives batch norm running statistics of order one whatever the
    layer sees, so nothing normalises, and through dozens of residual blocks the
    activations drift by orders of magnitude. ``run`` is therefore called once in
    training mode with a momentum of one, which stores the statistics of that very
    pass. Both sides do this with their own implementation, so the comparison then also
    covers batch norm in training mode.
    """
    norms = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    momenta = [norm.momentum for norm in norms]
    for norm in norms:
        norm.momentum = 1.0
    model.train()
    try:
        run()
    finally:
        model.eval()
        for norm, momentum in zip(norms, momenta):
            norm.momentum = momentum


def notch_cloud() -> np.ndarray:
    """A small crafted cloud whose lowest cells disappear at the first down-sampling.

    One point at the centre of every occupied cell of a 29 x 22 x 17 box, a quarter of
    them occupied. The two lowest layers along the first axis are occupied only in the
    top layer of the third axis, which is the seventeenth, an odd one. A model that
    declares the sparse grid as the bounding box loses that layer when it halves the
    grid, so its next level starts at cell 1 instead of cell 0, and whatever is anchored
    at the lowest occupied cell of a level moves with it. The tabletop scene never gets
    a network into that state.
    """
    rng = np.random.RandomState(0)
    cells = np.argwhere(rng.rand(29, 22, 17) < 0.25)
    cells = cells[(cells[:, 0] >= 2) | (cells[:, 2] == 16)]
    corners = np.array([[0, 0, 16], [28, 21, 0], [5, 0, 0]])  # pin the box whatever was drawn
    cells = np.unique(np.concatenate([cells, corners]), axis=0)
    return ((cells + 0.5) * PTV3_GRID).astype(np.float32)


def sparse_inputs(
    cloud: np.ndarray, channels: int, device: torch.device
) -> tuple[dict, np.ndarray]:
    """Voxel-level input of a Pointcept sparse CNN and the voxel of every point."""
    grid, first, inverse = voxels(cloud, PTV3_GRID)
    data = {
        "grid_coord": torch.from_numpy(grid).int().to(device),
        "feat": torch.from_numpy(point_features(cloud, channels)[first]).to(device),
        "offset": torch.tensor([len(grid)], device=device),
    }
    return data, inverse.reshape(-1)


def left_half(data: dict) -> dict:
    """The voxels in the lower half of the first axis, as a cloud of their own."""
    grid = data["grid_coord"]
    rows = torch.nonzero(grid[:, 0] < int(grid[:, 0].max()) // 2).flatten()
    half = grid[rows] - grid[rows].amin(dim=0)
    count = data["offset"].new_tensor([len(rows)])
    return {"grid_coord": half, "feat": data["feat"][rows], "offset": count}


def batch_dependence(model: torch.nn.Module, data: dict, alone: torch.Tensor) -> float:
    """How far the official features move when a second cloud shares the batch.

    The scene is batched with its own left half; both are compared with the features
    the official model gives each of them alone.
    """
    half = left_half(data)
    both = {
        "grid_coord": torch.cat([data["grid_coord"], half["grid_coord"]]),
        "feat": torch.cat([data["feat"], half["feat"]]),
        "offset": torch.cat([data["offset"], data["offset"] + half["offset"]]),
    }
    together = model(both)
    separate = torch.cat([alone, model(half)])
    return float((together - separate).abs().max())


def record_oacnns(args: argparse.Namespace) -> None:
    module = pointcept_module(args.upstream, "oacnns.oacnns_v1m1_base")
    if args.config is None:
        config = dict(OACNNS_ANALYTIC)
        stages = len(config["enc_channels"])
        official = {key: value for key, value in config.items() if key != "out_channels"}
        # The official constructor requires these three; none of them enters the
        # computation (``groups`` only has to divide every width).
        official.update(groups=[1] * stages, enc_num_ref=[16] * stages, dec_depth=[2] * stages)
        official["num_classes"] = config["out_channels"]
    else:
        config = {}  # the defaults of the GraspPanda backbone have to be this file
        official = published_backbone(args.config)
        official.pop("type")
    record_pointcept(args, module.OACNNs(**official), config, official)


def record_spunet(args: argparse.Namespace) -> None:
    module = pointcept_module(args.upstream, "sparse_unet.spconv_unet_v1m1_base")
    if args.config is None:
        config = dict(SPUNET_ANALYTIC)
        official = {key: value for key, value in config.items() if key != "out_channels"}
        official["num_classes"] = config["out_channels"] or 0  # 0: no final layer
    else:
        config = {}  # the defaults of the GraspPanda backbone have to be this file
        official = published_backbone(args.config)
        official.pop("type")
    record_pointcept(args, module.SpUNetBase(**official), config, official)


def record_pointcept(
    args: argparse.Namespace, model: torch.nn.Module, config: dict, official: dict
) -> None:
    """Run an official Pointcept sparse CNN under a closed-form state and store its output."""
    notch = getattr(args, "notch", False)
    (hashed_state if args.calibrate else analytic_state)(model)
    model = model.cuda().eval()
    cloud = notch_cloud() if notch else scene()
    data, inverse = sparse_inputs(cloud, model.in_channels, torch.device("cuda"))
    extra = {"cloud": cloud} if notch else {}
    with torch.no_grad():
        if args.calibrate:
            calibrate_norms(model, lambda: model(data))
        feat = model(data)
        if not notch:  # half of the notch cloud is too thin for the official OA-CNNs
            extra["batch_dependence"] = batch_dependence(model, data, feat)
    save_dense(
        args.name,
        feat.cpu().numpy()[inverse],
        keep=128,
        grid_size=PTV3_GRID,
        config=json.dumps(config),
        official_config=json.dumps(official),
        weights="hashed, batch norm calibrated" if args.calibrate else "analytic",
        calibrated=args.calibrate,
        state_keys=state_keys(model),
        voxels=len(feat),
        extent=(data["grid_coord"].amax(dim=0) + 1).cpu().numpy(),
        parameters=sum(p.numel() for p in model.parameters()),
        upstream_commit=args.commit or commit_of(args.upstream),
        **extra,
    )


# ------------------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, function: object, weights: str = "no") -> argparse.ArgumentParser:
        sub = commands.add_parser(name)
        sub.add_argument("--upstream", type=Path, required=True, help="official checkout")
        if weights != "no":
            sub.add_argument(
                "--weights", type=Path, required=weights == "required", help="released checkpoint"
            )
        sub.set_defaults(run=function)
        return sub

    commands.add_parser("scene").set_defaults(run=record_scene)
    command("serialization", record_serialization)
    command("pointnet2-graspnet", record_pointnet2_graspnet, weights="required")
    command("ptv3", record_ptv3, weights="optional").add_argument("--name", default=None)
    sonata = command("ptv3-sonata", record_ptv3_sonata, weights="optional")
    sonata.add_argument("--package", required=True, choices=("sonata", "concerto", "utonia"))
    sonata.add_argument("--name", required=True, help="fixture name")
    litept = command("litept", record_litept, weights="optional")
    litept.add_argument("--config", type=Path, help="published config of the released weights")
    litept.add_argument("--name", required=True, help="fixture name; selects the analytic model")
    litept.add_argument("--commit", help="upstream commit, where the checkout has no .git")
    pointnext = command("pointnext", record_pointnext)
    pointnext.add_argument("--config", type=Path, required=True, help="published model YAML")
    pointnext.add_argument("--name", required=True, help="fixture name")
    for name, function in (("oacnns", record_oacnns), ("spunet", record_spunet)):
        pointcept = command(name, function)
        pointcept.add_argument("--config", type=Path, default=None, help="published config file")
        pointcept.add_argument("--name", required=True, help="fixture name")
        pointcept.add_argument("--commit", default=None, help="commit of a checkout without .git")
        pointcept.add_argument(
            "--calibrate", action="store_true", help="see hashed_state, calibrate_norms"
        )
        if name == "oacnns":
            pointcept.add_argument("--notch", action="store_true", help="run on notch_cloud()")
    args = parser.parse_args()
    # Record in strict float32. On Ampere and newer GPUs cuDNN otherwise runs
    # convolutions in TF32 (10-bit mantissa), which alone moves outputs by about 1e-3
    # and would hide real differences.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    args.run(args)


if __name__ == "__main__":
    main()
