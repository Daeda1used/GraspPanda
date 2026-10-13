"""The sparse facade: layer semantics against brute force and against recorded
MinkowskiEngine layers, plus the errors a wrong use must raise."""

from __future__ import annotations

import copy
import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from grasppanda.ops import sparse
from grasppanda.utils.imports import MissingDependencyError

requires_spconv = pytest.mark.skipif(
    importlib.util.find_spec("spconv") is None, reason="needs the spconv extra"
)
FIXTURE = Path(__file__).parent / "fixtures" / "minkowski_layers.npz"


def random_voxels(count: int, batch: int = 2, seed: int = 0) -> torch.Tensor:
    """Unique ``[batch, x, y, z]`` rows with negative and positive coordinates."""
    generator = torch.Generator().manual_seed(seed)
    cells = torch.randint(-7, 9, (count, 3), generator=generator)
    sample = torch.randint(0, batch, (count, 1), generator=generator)
    return torch.unique(torch.cat([sample, cells], dim=1), dim=0)


def rows_by_voxel(tensor, origin: torch.Tensor, stride: int = 1) -> dict[tuple[int, ...], int]:
    """Absolute voxel coordinate (in units of the level's own grid) → row of the tensor."""
    index = sparse.indices(tensor).long()
    absolute = torch.cat([index[:, :1], index[:, 1:] + origin // stride], dim=1)
    return {tuple(row): n for n, row in enumerate(absolute.tolist())}


# ------------------------------------------------------- brute-force definitions


def subm_oracle(voxels, feats, weight):
    """out[v] = Σ W[:, i, j, k, :] in[v + (i - r, j - r, k - r)] over occupied neighbours."""
    table = {tuple(v): n for n, v in enumerate(voxels.tolist())}
    size = weight.shape[1]
    radius = size // 2
    out = torch.zeros(len(voxels), weight.shape[0], dtype=torch.float64)
    for row, (b, x, y, z) in enumerate(voxels.tolist()):
        for i, j, k in itertools.product(range(size), repeat=3):
            source = table.get((b, x + i - radius, y + j - radius, z + k - radius))
            if source is not None:
                out[row] += weight[:, i, j, k, :].double() @ feats[source].double()
    return out


def down_oracle(voxels, feats, weight, stride):
    """Coarse voxel u = floor(v / stride); out[u] = Σ W[:, i, j, k, :] in[stride u + (i, j, k)]."""
    out: dict[tuple[int, ...], torch.Tensor] = {}
    for row, (b, x, y, z) in enumerate(voxels.tolist()):
        coarse = (b, x // stride, y // stride, z // stride)  # Python floors negatives too
        i, j, k = x % stride, y % stride, z % stride
        value = weight[:, i, j, k, :].double() @ feats[row].double()
        out[coarse] = out.get(coarse, 0) + value
    return out


def up_oracle(fine_voxels, coarse, weight, stride):
    """out[v] = W[:, i, j, k, :] in[u] with v = stride u + (i, j, k)."""
    rows = []
    for b, x, y, z in fine_voxels.tolist():
        parent = coarse[(b, x // stride, y // stride, z // stride)]
        rows.append(weight[:, x % stride, y % stride, z % stride, :].double() @ parent.double())
    return torch.stack(rows)


# ------------------------------------------------------------------------ tensors


@requires_spconv
@pytest.mark.parametrize("align", [1, 4, 16])
def test_make_tensor_shifts_by_a_multiple_of_align_and_keeps_row_order(align):
    voxels = random_voxels(80)
    feats = torch.randn(len(voxels), 3)
    tensor, origin = sparse.make_tensor(feats, voxels, 2, align=align)
    assert (origin % align == 0).all() and (origin <= voxels[:, 1:].amin(0)).all()
    index = sparse.indices(tensor).long()
    assert torch.equal(index[:, 1:] + origin, voxels[:, 1:]) and torch.equal(
        index[:, 0], voxels[:, 0]
    )
    assert (index[:, 1:] >= 0).all()
    assert torch.equal(sparse.features(tensor), feats)
    inverse = torch.tensor([3, 3, 0, 7])
    assert torch.equal(sparse.features_at(tensor, inverse), feats[inverse])


@pytest.mark.parametrize(
    ("voxels", "feats", "message"),
    [
        (torch.zeros(4, 3, dtype=torch.long), torch.zeros(4, 2), r"integer \[batch, x, y, z\]"),
        (torch.zeros(4, 4), torch.zeros(4, 2), r"integer \[batch, x, y, z\]"),
        (torch.zeros(4, 4, dtype=torch.long), torch.zeros(3, 2), "one feature row per voxel"),
        (torch.zeros(0, 4, dtype=torch.long), torch.zeros(0, 2), "without voxels"),
    ],
)
def test_make_tensor_rejects_malformed_input(voxels, feats, message):
    with pytest.raises(ValueError, match=message):
        sparse.make_tensor(feats, voxels, 1)


# ------------------------------------------------------------------------- layers


@requires_spconv
@pytest.mark.parametrize("kernel_size", [3, 5])
def test_submanifold_convolution_matches_its_definition(kernel_size):
    torch.manual_seed(kernel_size)
    voxels = random_voxels(90)
    feats = torch.randn(len(voxels), 3)
    layer = sparse.SubMConv3d(3, 4, kernel_size, key="a")
    tensor, _ = sparse.make_tensor(feats, voxels, 2)
    out = layer(tensor)
    assert torch.equal(sparse.indices(out), sparse.indices(tensor))  # same voxels, same order
    expected = subm_oracle(voxels, feats, layer.weight.detach())
    assert torch.allclose(sparse.features(out).double(), expected, atol=1e-5)


@requires_spconv
def test_down_and_up_sampling_match_their_definitions():
    torch.manual_seed(0)
    voxels = random_voxels(120)
    feats = torch.randn(len(voxels), 3)
    down = sparse.SparseConv3d(3, 5, 2, 2, key="down")
    up = sparse.SparseInverseConv3d(5, 4, 2, key="down")
    tensor, origin = sparse.make_tensor(feats, voxels, 2, align=2)
    coarse = down(tensor)

    expected = down_oracle(voxels, feats, down.weight.detach(), 2)
    rows = rows_by_voxel(coarse, origin, stride=2)
    assert set(rows) == set(expected)
    produced = sparse.features(coarse).double()
    for voxel, value in expected.items():
        assert torch.allclose(produced[rows[voxel]], value, atol=1e-5)

    fine = up(coarse)
    assert torch.equal(sparse.indices(fine), sparse.indices(tensor))  # rows of the input, in order
    coarse_by_voxel = {voxel: produced[row] for voxel, row in rows.items()}
    expected_up = up_oracle(voxels, coarse_by_voxel, up.weight.detach(), 2)
    assert torch.allclose(sparse.features(fine).double(), expected_up, atol=1e-5)


@requires_spconv
def test_layers_are_plain_modules_of_the_facade_type():
    layer = sparse.SubMConv3d(3, 4, 3, bias=True, key="a")
    assert isinstance(layer, sparse.SubMConv3d) and isinstance(layer, torch.nn.Module)
    assert sorted(layer.state_dict()) == ["bias", "weight"]  # no wrapper level in checkpoints
    assert tuple(layer.weight.shape) == (4, 3, 3, 3, 3)
    clone = copy.deepcopy(layer)
    assert type(clone) is type(layer) and torch.equal(clone.weight, layer.weight)
    assert isinstance(sparse.SparseConv3d(3, 3, 2, 2), sparse.SparseConv3d)
    assert not isinstance(layer, sparse.SparseConv3d)


@requires_spconv
def test_feature_layers_and_backward():
    torch.manual_seed(1)
    voxels = random_voxels(60)
    feats = torch.randn(len(voxels), 3, requires_grad=True)
    tensor, _ = sparse.make_tensor(feats, voxels, 2)
    linear, norm, act = sparse.Linear(3, 6), sparse.BatchNorm(6), sparse.ReLU()
    out = act(norm(linear(tensor)))
    reference = torch.relu(
        torch.nn.functional.batch_norm(
            feats @ linear.weight.T + linear.bias, None, None, norm.weight, norm.bias, True
        )
    )
    assert torch.allclose(sparse.features(out), reference, atol=1e-6)
    joined = sparse.cat([out, tensor])
    assert sparse.features(joined).shape == (len(voxels), 9)
    assert torch.equal(sparse.features(joined)[:, 6:], feats)
    conv = sparse.SubMConv3d(9, 2, 3)
    sparse.features(conv(joined)).sum().backward()
    assert feats.grad is not None and feats.grad.abs().sum() > 0
    assert conv.weight.grad is not None and linear.weight.grad.abs().sum() > 0


@requires_spconv
def test_a_wrong_cpu_convolution_is_an_error_not_a_wrong_feature_map(monkeypatch):
    """The CPU-only spconv wheel miscomputes with several OpenMP threads; that must not pass."""
    from grasppanda.ops.sparse import _spconv

    _spconv.check_cpu_convolution(force=True)  # this process convolves correctly
    assert _spconv._dense_mismatch() < 1e-4
    monkeypatch.setattr(_spconv, "_cpu_check", None)
    monkeypatch.setattr(_spconv, "_dense_mismatch", lambda: 0.3)
    voxels = random_voxels(50)
    tensor, _ = sparse.make_tensor(torch.randn(len(voxels), 4), voxels, 2)
    with pytest.raises(sparse.SparseBackendError, match="OMP_NUM_THREADS=1.*CUDA build of spconv"):
        sparse.SubMConv3d(4, 4, 3)(tensor)
    with pytest.raises(sparse.SparseBackendError, match="differs from the dense one by 0.3"):
        _spconv.check_cpu_convolution()  # the verdict of a process is remembered


@requires_spconv
def test_backward_runs_on_a_machine_without_cuda(monkeypatch):
    """spconv asks for a CUDA stream it never uses for CPU tensors; without CUDA that raised."""
    import spconv.pytorch.cppcore as cppcore
    import spconv.pytorch.ops as ops

    from grasppanda.ops.sparse import _spconv

    def gradients():
        torch.manual_seed(3)
        voxels = random_voxels(80)
        feats = torch.randn(len(voxels), 4, requires_grad=True)
        conv = sparse.SubMConv3d(4, 5, 3)
        tensor, _ = sparse.make_tensor(feats, voxels, 2)
        sparse.features(conv(tensor)).pow(2).sum().backward()
        return feats.grad, conv.weight.grad

    reference = gradients() if torch.cuda.is_available() else None
    for module in (ops, cppcore):  # restored when the test ends
        monkeypatch.setattr(module, "get_current_stream", module.get_current_stream)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _spconv._allow_cpu_backward()
    assert ops.get_current_stream() == 0
    patched = gradients()
    assert all(torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in patched)
    if reference is not None:  # the stream played no part in the result
        assert all(torch.equal(one, two) for one, two in zip(reference, patched))


@requires_spconv
def test_cat_refuses_tensors_on_different_voxels():
    voxels = random_voxels(40)
    first, _ = sparse.make_tensor(torch.randn(len(voxels), 2), voxels, 2)
    second, _ = sparse.make_tensor(torch.randn(len(voxels) - 1, 2), voxels[:-1], 2)
    with pytest.raises(ValueError, match="different voxel sets"):
        sparse.cat([first, second])


@requires_spconv
def test_layers_reject_what_the_facade_does_not_define():
    with pytest.raises(ValueError, match="odd kernel size"):
        sparse.SubMConv3d(3, 4, 2)
    with pytest.raises(ValueError, match="kernel_size == stride"):
        sparse.SparseConv3d(3, 4, 3, 2)


# ------------------------------------------------------------------------ backend


def test_unknown_backend_is_rejected_by_name():
    assert sparse.get_backend() == "spconv" and "spconv" in sparse.BACKENDS
    with pytest.raises(ValueError, match="unknown sparse backend 'minkowski'. Available: spconv"):
        sparse.set_backend("minkowski")
    assert sparse.get_backend() == "spconv"


def test_importing_the_facade_does_not_import_the_backend_and_a_missing_one_is_explained(
    monkeypatch,
):
    for name in [n for n in sys.modules if n == "spconv" or n.startswith("spconv.")]:
        monkeypatch.setitem(sys.modules, name, None)
    for name in ("spconv", "spconv.pytorch", "spconv.core"):
        monkeypatch.setitem(sys.modules, name, None)  # makes `import spconv` fail
    monkeypatch.delitem(sys.modules, "grasppanda.ops.sparse._spconv", raising=False)
    assert sparse.BatchNorm(4).num_features == 4  # backend-independent parts still work
    with pytest.raises(MissingDependencyError, match=r'pip install "grasppanda\[spconv-cu118\]"'):
        sparse.SubMConv3d(3, 4, 3)
    with pytest.raises(MissingDependencyError, match="spconv"):
        sparse.make_tensor(torch.zeros(1, 2), torch.zeros(1, 4, dtype=torch.long), 1)


# ------------------------------------------------------ MinkowskiEngine kernels


def test_minkowski_kernel_layout():
    size, cin, cout = 3, 2, 5
    kernel = torch.randn(size**3, cin, cout)
    weight = sparse.convert_minkowski_kernel(kernel, size)
    assert tuple(weight.shape) == (cout, size, size, size, cin)
    for i, j, k in [(0, 0, 0), (2, 0, 1), (1, 2, 0), (2, 2, 2)]:
        cell = i + size * j + size * size * k  # first spatial axis varies fastest
        assert torch.equal(weight[:, i, j, k, :], kernel[cell].T)
    point = torch.randn(cin, cout)
    assert torch.equal(sparse.convert_minkowski_kernel(point, 1), point.T)
    assert torch.equal(sparse.convert_minkowski_kernel(point[None], 1), point.T)
    with pytest.raises(ValueError, match=r"must be \(27, in, out\)"):
        sparse.convert_minkowski_kernel(torch.randn(8, cin, cout), 3)
    with pytest.raises(ValueError, match="1x1 kernel"):
        sparse.convert_minkowski_kernel(torch.randn(3, cin, cout), 1)


@requires_spconv
def test_converted_kernels_reproduce_recorded_minkowski_layers():
    """Single MinkowskiEngine 0.5.4 layers recorded by tools/make_gsnet_fixtures.py."""
    record = np.load(FIXTURE)
    plan = [  # name, facade layer, kernel size, tensor stride after the layer
        ("stem", sparse.SubMConv3d(3, 4, 5, key="stem"), 5, 1),
        ("down1", sparse.SparseConv3d(4, 4, 2, 2, key="down1"), 2, 2),
        ("mid1", sparse.SubMConv3d(4, 6, 3, key="level1"), 3, 2),
        ("down2", sparse.SparseConv3d(6, 6, 2, 2, key="down2"), 2, 4),
        ("mid2", sparse.SubMConv3d(6, 5, 3, key="level2"), 3, 4),
        ("up2", sparse.SparseInverseConv3d(5, 6, 2, key="down2"), 2, 2),
        ("up1", sparse.SparseInverseConv3d(6, 4, 2, key="down1"), 2, 1),
        ("point", sparse.Linear(4, 2), 1, 1),
    ]
    tensor, origin = sparse.make_tensor(
        torch.from_numpy(record["feats"]), torch.from_numpy(record["coords"]).long(), 2, align=4
    )
    with torch.no_grad():
        for name, layer, kernel_size, stride in plan:
            kernel = torch.from_numpy(record[f"kernel_{name}"])
            layer.weight.copy_(sparse.convert_minkowski_kernel(kernel, kernel_size))
            if f"bias_{name}" in record:
                layer.bias.copy_(torch.from_numpy(record[f"bias_{name}"]).reshape(-1))
            tensor = layer(tensor)
            # MinkowskiEngine coordinates of a level are multiples of its tensor stride.
            index = sparse.indices(tensor).long()
            absolute = torch.cat([index[:, :1], index[:, 1:] * stride + origin], dim=1)
            rows = {tuple(row): n for n, row in enumerate(absolute.tolist())}
            reference_coords = record[f"out_coords_{name}"].tolist()
            assert set(rows) == {tuple(c) for c in reference_coords}, name
            order = torch.tensor([rows[tuple(c)] for c in reference_coords])
            reference = torch.from_numpy(record[f"out_feats_{name}"])
            produced = sparse.features(tensor)[order]
            scale = reference.abs().max()
            # float32 round-off of a different summation order; a wrong layout is off by O(1)
            assert (produced - reference).abs().max() <= 1e-5 * scale, name
