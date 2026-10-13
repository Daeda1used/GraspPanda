"""Space-filling-curve codes, checked against bit-by-bit oracles and curve properties."""

from __future__ import annotations

import itertools

import pytest
import torch

from grasppanda.ops import hilbert_code, serialization_code, z_order_code


def morton_oracle(x: int, y: int, z: int, depth: int) -> int:
    code = 0
    for bit in range(depth):
        code |= ((x >> bit) & 1) << (3 * bit + 2)
        code |= ((y >> bit) & 1) << (3 * bit + 1)
        code |= ((z >> bit) & 1) << (3 * bit)
    return code


def hilbert_oracle(point: tuple[int, int, int], depth: int) -> int:
    """Skilling's AxesToTranspose (AIP Conf. Proc. 707, 2004), scalar, then bit interleaving."""
    x = list(point)
    top = 1 << (depth - 1)
    q = top
    while q > 1:  # inverse undo
        p = q - 1
        for i in range(3):
            if x[i] & q:
                x[0] ^= p
            else:
                t = (x[0] ^ x[i]) & p
                x[0] ^= t
                x[i] ^= t
        q >>= 1
    for i in range(1, 3):  # Gray encode
        x[i] ^= x[i - 1]
    t, q = 0, top
    while q > 1:
        if x[2] & q:
            t ^= q - 1
        q >>= 1
    x = [value ^ t for value in x]
    code = 0
    for bit in range(depth):
        for dim in range(3):
            code |= ((x[dim] >> bit) & 1) << (3 * bit + 2 - dim)
    return code


def full_grid(depth: int) -> torch.Tensor:
    side = range(1 << depth)
    return torch.tensor(list(itertools.product(side, side, side)))


@pytest.mark.parametrize("depth", [1, 2, 5, 16])
def test_z_order_interleaves_bits_with_x_most_significant(depth):
    grid = torch.randint(0, 1 << depth, (200, 3), generator=torch.Generator().manual_seed(depth))
    expected = [morton_oracle(*row, depth) for row in grid.tolist()]
    assert z_order_code(grid, depth).tolist() == expected
    assert z_order_code(torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 1).tolist() == [4, 2, 1]


@pytest.mark.parametrize("depth", [1, 2, 3, 7, 16])
def test_hilbert_matches_skillings_algorithm(depth):
    grid = torch.randint(0, 1 << depth, (200, 3), generator=torch.Generator().manual_seed(depth))
    expected = [hilbert_oracle(tuple(row), depth) for row in grid.tolist()]
    assert hilbert_code(grid, depth).tolist() == expected


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_hilbert_visits_every_cell_once_through_face_neighbours(depth):
    grid = full_grid(depth)
    code = hilbert_code(grid, depth)
    assert sorted(code.tolist()) == list(range(8**depth))
    path = grid[torch.argsort(code)]
    assert ((path[1:] - path[:-1]).abs().sum(-1) == 1).all()
    # Morton order is a bijection too, but it jumps: that is why PTv3 uses both curves.
    morton = grid[torch.argsort(z_order_code(grid, depth))]
    assert sorted(z_order_code(grid, depth).tolist()) == list(range(8**depth))
    if depth > 1:
        assert ((morton[1:] - morton[:-1]).abs().sum(-1) > 1).any()


def test_dropping_three_bits_gives_the_code_of_the_parent_cell():
    grid = torch.randint(0, 64, (300, 3), generator=torch.Generator().manual_seed(0))
    for encode in (z_order_code, hilbert_code):
        assert torch.equal(encode(grid, 6) >> 3, encode(grid >> 1, 5))


def test_trans_orders_swap_x_and_y_and_batches_stay_contiguous():
    grid = torch.randint(0, 32, (100, 3), generator=torch.Generator().manual_seed(1))
    swapped = grid[:, [1, 0, 2]]
    assert torch.equal(serialization_code(grid, 5, "z-trans"), z_order_code(swapped, 5))
    assert torch.equal(serialization_code(grid, 5, "hilbert-trans"), hilbert_code(swapped, 5))
    batch = torch.arange(100) % 3
    code = serialization_code(grid, 5, "hilbert", batch)
    assert torch.equal(code >> 15, batch)
    assert torch.equal(code & ((1 << 15) - 1), hilbert_code(grid, 5))
    assert (batch[torch.argsort(code)].diff() >= 0).all()


def test_serialization_rejects_bad_input():
    grid = torch.tensor([[0, 1, 8]])
    with pytest.raises(ValueError, match=r"must lie in \[0, 2\*\*3\)"):
        z_order_code(grid, 3)
    with pytest.raises(ValueError, match="must be integers"):
        hilbert_code(torch.rand(4, 3), 3)
    with pytest.raises(ValueError, match="depth must be in 1..21"):
        z_order_code(grid, 22)
    with pytest.raises(ValueError, match="unknown serialisation order 'peano'"):
        serialization_code(grid, 4, "peano")
    with pytest.raises(ValueError, match="do not fit a 64-bit code"):
        serialization_code(grid, 20, "z", torch.tensor([9]))
