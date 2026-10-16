"""Proposals with a swappable sampler: scored seeds and plainly sampled seeds.

``GraspnessFPS`` hard-wires one choice of each part of a proposal, as its reference
implementation does. The two proposals here make the parts options:

* :class:`ScoredSeeds` keeps the scorer of ``GraspnessFPS`` (objectness and graspness
  per scene point, the same targets and losses) and opens *which score decides*, *how
  candidates are cut*, *which sampler picks among them* and *what happens when too few
  points pass*.
* :class:`SampledSeeds` predicts nothing: a sampler picks the seeds among all scene
  points of a dense encoder. It is the dense counterpart of ``SubsampledSeeds`` and
  fits detectors whose view stage owns objectness.

Samplers are components of their own (:mod:`grasppanda.models.samplers`).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from itertools import pairwise
from typing import Any, ClassVar, Literal

import torch
from torch import Tensor, nn

from grasppanda.models.base import Loss, Proposal
from grasppanda.models.samplers.base import Sampler
from grasppanda.registry import INJECTED, PROPOSALS, SAMPLERS, Cfg
from grasppanda.structures import (
    ContractError,
    Observation,
    PointFeatures,
    SeedSet,
    Supervision,
)

from .graspness import GraspnessFPS, PointScores, _offsets

__all__ = ["SampledSeeds", "ScoredSeeds"]

_SELECT_BY = ("graspness_and_objectness", "graspness", "objectness", "product", "graspness_only")
_TOO_FEW = ("pad", "best_rest")


def _sampler_problems(cls: type, node: dict[str, Any]) -> Iterator[str]:
    """Refuse, before anything is built, a sampler that reads what the proposal lacks."""
    sampler = node.get("sampler")
    if not (isinstance(sampler, dict) and sampler.get("type") in SAMPLERS):
        return
    sampler_cls = SAMPLERS.get(sampler["type"])
    missing = [name for name in sampler_cls.needs if name not in cls.sampler_inputs]
    if missing:
        fitting = [
            name
            for name in SAMPLERS.names()
            if set(SAMPLERS.get(name).needs) <= set(cls.sampler_inputs)
        ]
        yield (
            f"{sampler_cls.__name__} needs {' and '.join(missing)} per candidate point, but "
            f"{cls.__name__} predicts no score to give it. Use a sampler that reads only "
            f"{' and '.join(('coordinates', *cls.sampler_inputs))} ({', '.join(fitting)}), or a "
            "proposal that scores the points (ScoredSeeds)."
        )


def _fill(picks: Tensor, num_seeds: int, pool: Tensor, fallback: int) -> tuple[Tensor, Tensor]:
    """Bring the picks of one frame to ``num_seeds``.

    ``pool`` holds further real points in order of preference; what is still missing
    after them is filled with repeats of the first pick (``fallback`` for a frame
    without any) and marked invalid.
    """
    picks = picks[:num_seeds]
    if len(picks) < num_seeds and len(pool):
        picks = torch.cat([picks, pool[: num_seeds - len(picks)]])
    valid = torch.ones(num_seeds, dtype=torch.bool, device=picks.device)
    missing = num_seeds - len(picks)
    if missing:
        valid[len(picks) :] = False
        repeat = picks[:1] if len(picks) else picks.new_full((1,), fallback)
        picks = torch.cat([picks, repeat.expand(missing)])
    return picks, valid


@PROPOSALS.register()
class ScoredSeeds(GraspnessFPS):
    """Objectness and graspness per point; the selecting score, the cut and the sampler are options.

    The scorer, the targets and the losses are those of ``GraspnessFPS``: one linear
    layer predicts two objectness logits and a graspness value per scene point,
    supervised with ``point_labels['objectness']`` and ``['graspness']``. The defaults
    select as GSNet does (object by arg-max, graspness above 0.1, furthest point
    sampling), without the two quirks of the reference sampling kernel that
    ``GraspnessFPS`` reproduces for checkpoint parity.

    Args:
        num_seeds: Seeds per frame.
        select_by: The score that decides, called ``s`` below, and the candidate rule.
            ``graspness_and_objectness``: ``s`` = graspness; candidates have
            ``P(object) > objectness_threshold`` and ``s > graspness_threshold`` (GSNet).
            ``graspness``: ``s`` = graspness alone, objectness is only supervised.
            ``objectness``: ``s`` = ``P(object)``, candidates are the predicted
            foreground, graspness is only supervised (foreground sampling).
            ``product``: ``s`` = ``P(object) × graspness > graspness_threshold``
            (MultiGraspNet's rule).
            ``graspness_only``: ``s`` = graspness, candidates have ``s >
            graspness_threshold``, and objectness is neither predicted nor supervised
            here: the stage leaves it to a view stage that owns it (ObjectnessViewStage),
            whose objectness then also rejects background seeds at inference.
        graspness_threshold: See ``select_by``. It belongs to the label definition:
            0.1 suits graspness normalised to ``[0, 1]`` per frame.
        objectness_threshold: Probability above which a point counts as object; 0.5 is
            the arg-max of the two logits.
        top_fraction: Instead of the thresholds, the candidates are this share of the
            frame's points with the highest ``s`` (DexGraspNet 2.0 describes the top
            1 %); ``None`` uses the thresholds.
        sampler: Picks the seeds among the candidates; it receives their coordinates,
            ``s`` (never negative) and the scene features. ``None`` selects ``FPS``.
        too_few: A frame with fewer picks than ``num_seeds``. ``pad``: repeat the first
            seed and mark the repeats invalid, so they never become grasps (GSNet).
            ``best_rest``: take the best-scored points that were not candidates, as
            real seeds, so that the later stages are trained from the first iteration
            even while the scorer passes nothing.
        objectness_loss: As in ``GraspnessFPS``; ``None`` selects ``CrossEntropy``.
        graspness_loss: As in ``GraspnessFPS``; ``None`` selects ``SmoothL1`` × 10.
        in_channels: Channels of the scene features; injected by the detector.
    """

    requires = "dense"
    status = "tested"
    source = (
        "GSNet (Wang et al., ICCV 2021) scorer with the selection rules of MultiGraspNet "
        "(product), DexGraspNet 2.0 (top fraction) and a swappable sampler"
    )
    seed_scores = ("graspness", "objectness")
    #: What a sampler inside this proposal can be given besides coordinates.
    sampler_inputs: ClassVar[tuple[str, ...]] = ("scores", "feats")

    @classmethod
    def seed_scores_for(cls, options: Mapping[str, Any]) -> tuple[str, ...]:
        """Graspness only with ``select_by: graspness_only``, else both scores."""
        if options.get("select_by") == "graspness_only":
            return ("graspness",)
        return cls.seed_scores

    @classmethod
    def loss_terms_for(cls, options: Mapping[str, Any]) -> dict[str, str]:
        """No objectness term when the stage predicts graspness only."""
        if options.get("select_by") == "graspness_only":
            return {"graspness_loss": cls.loss_terms["graspness_loss"]}
        return dict(cls.loss_terms)

    @classmethod
    def without_seed_score(
        cls, options: Mapping[str, Any], score: str
    ) -> Mapping[str, Any] | str | None:
        """``select_by: graspness_only`` where the selection does not read objectness."""
        select_by = options.get("select_by", "graspness_and_objectness")
        if score != "objectness" or select_by == "graspness_only":
            return None
        if select_by in ("objectness", "product"):
            return (
                f"{cls.__name__} selects its seeds by {select_by} (`select_by: {select_by}`), "
                "which reads the objectness it predicts, so it cannot leave objectness to the "
                "view stage."
            )
        kept = {key: value for key, value in options.items() if key != "objectness_loss"}
        return {**kept, "type": cls.__name__, "select_by": "graspness_only"}

    @classmethod
    def option_problems(cls, options: Mapping[str, Any]) -> list[str]:
        """An objectness loss on a stage that predicts no objectness would be ignored."""
        if options.get("select_by") == "graspness_only" and options.get("objectness_loss"):
            return [
                f"{cls.__name__} has `select_by: graspness_only`, so it predicts no "
                "objectness, but `objectness_loss` is set. Remove it: the view stage that owns "
                "objectness has its own objectness loss."
            ]
        return []

    def __init__(
        self,
        num_seeds: int = 1024,
        select_by: Literal[
            "graspness_and_objectness", "graspness", "objectness", "product", "graspness_only"
        ] = "graspness_and_objectness",
        graspness_threshold: float = 0.1,
        objectness_threshold: float = 0.5,
        top_fraction: float | None = None,
        sampler: Cfg[Sampler] | None = None,
        too_few: Literal["pad", "best_rest"] = "pad",
        objectness_loss: Cfg[Loss] | None = None,
        graspness_loss: Cfg[Loss] | None = None,
        in_channels: int = INJECTED,
    ) -> None:
        super().__init__(
            num_seeds=num_seeds,
            graspness_threshold=graspness_threshold,
            fps_skip_near_origin=False,
            fps_tie_break="lowest_index",
            objectness_loss=objectness_loss,
            graspness_loss=graspness_loss,
            in_channels=in_channels,
        )
        if select_by not in _SELECT_BY:
            raise ValueError(
                f"ScoredSeeds: unknown select_by '{select_by}'; choose from {_SELECT_BY}"
            )
        if too_few not in _TOO_FEW:
            raise ValueError(f"ScoredSeeds: unknown too_few '{too_few}'; choose from {_TOO_FEW}")
        if top_fraction is not None and not 0.0 < top_fraction <= 1.0:
            raise ValueError("ScoredSeeds: top_fraction must lie in (0, 1]")
        self.select_by = select_by
        self.objectness_threshold = objectness_threshold
        self.top_fraction = top_fraction
        self.too_few = too_few
        self.sampler = SAMPLERS.build(sampler if sampler is not None else {"type": "FPS"})
        if select_by == "graspness_only":  # one output, the graspness; no objectness loss
            self.predicts_objectness = False
            self.score = nn.Linear(in_channels, 1)
            self.objectness_loss = None

    @classmethod
    def check_config(cls, node: dict[str, Any]) -> Iterator[str]:
        yield from _sampler_problems(cls, node)

    def selection_score(self, scores: PointScores) -> Tensor:
        """``(N,)`` the score that decides; never negative, higher is better."""
        graspness = scores.graspness.detach().clamp(min=0.0)
        if self.select_by in ("graspness_and_objectness", "graspness", "graspness_only"):
            return graspness
        is_object = scores.objectness.detach().softmax(dim=-1)[:, 1]
        return is_object if self.select_by == "objectness" else is_object * graspness

    def candidates(self, scores: PointScores) -> Tensor:
        """``(N,)`` mask of the scene points that may become seeds."""
        decides = self.selection_score(scores)
        if self.top_fraction is not None:
            mask = torch.zeros_like(decides, dtype=torch.bool)
            for start, stop in pairwise(_offsets(scores.points)):
                keep = max(1, round((stop - start) * self.top_fraction))
                mask[start + decides[start:stop].topk(min(keep, stop - start)).indices] = True
            return mask
        if scores.objectness is None:  # graspness only
            return decides > self.graspness_threshold
        is_object = scores.objectness.detach().softmax(dim=-1)[:, 1] > self.objectness_threshold
        if self.select_by == "objectness":
            return is_object
        mask = decides > self.graspness_threshold
        return mask & is_object if self.select_by == "graspness_and_objectness" else mask

    def forward(
        self,
        feats: PointFeatures,
        observation: Observation,
        seed_index: Tensor | None = None,
    ) -> tuple[SeedSet, PointScores]:
        """Score the scene points and select the seeds; see ``GraspnessFPS.forward``."""
        name = type(self).__name__
        points = observation.points
        if points is None:
            raise ContractError(f"{name} needs observation.points")
        feats.expect(points, user=f"{name} (requires '{self.requires}' scene features)")
        if feats.channels != self.in_channels:
            raise ContractError(
                f"{name} was built for {self.in_channels}-channel features, got {feats.channels}"
            )
        scores = self._point_scores(feats, points)
        if seed_index is None:
            index, valid, scores.num_candidates = self._pick(scores, feats.feats.detach())
        else:
            index, valid = self._replay(points, seed_index)
        flat = index.reshape(-1)
        offsets = torch.arange(index.shape[0] + 1, device=flat.device) * self.num_seeds
        seeds = SeedSet(
            points=points.subset(flat, offsets),
            feats=feats.feats[flat],
            scores=self._seed_scores(scores, flat, scores.graspness.detach()),
            valid=valid.reshape(-1),
            source="scored_seeds" if seed_index is None else "replayed",
        )
        return seeds, scores

    @torch.no_grad()
    def _pick(self, scores: PointScores, feats: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        coords = scores.points.coords
        decides = self.selection_score(scores)
        mask = self.candidates(scores)
        index, valid, counts = [], [], []
        for start, stop in pairwise(_offsets(scores.points)):
            if stop == start:
                raise ContractError(f"{type(self).__name__} received a frame without points")
            members = torch.nonzero(mask[start:stop]).flatten() + start
            counts.append(len(members))
            picks = members[:0]
            if len(members):
                chosen = self.sampler.sample(
                    coords[members], self.num_seeds, scores=decides[members], feats=feats[members]
                )
                picks = members[chosen.to(members.device)]
            pool = picks[:0]
            if self.too_few == "best_rest" and len(picks) < self.num_seeds:
                rest = decides[start:stop].clone()
                rest[picks - start] = -1.0
                order = torch.sort(rest, descending=True, stable=True).indices
                pool = order[: stop - start - len(picks)] + start
            frame_index, frame_valid = _fill(picks, self.num_seeds, pool, start)
            index.append(frame_index)
            valid.append(frame_valid)
        return torch.stack(index), torch.stack(valid), torch.tensor(counts, device=coords.device)

    def extra_repr(self) -> str:
        return (
            f"num_seeds={self.num_seeds}, select_by={self.select_by}, sampler={self.sampler!r}, "
            f"too_few={self.too_few}"
        )


@PROPOSALS.register()
class SampledSeeds(Proposal):
    """Seeds picked by a sampler among all points of a dense encoder; nothing is scored.

    The dense counterpart of ``SubsampledSeeds``: where a sub-sampling encoder decides
    the seeds inside its own layers, here the encoder returns one feature per scene
    point and the sampler of this stage decides. There is no score, no target and no
    loss, so it fits detectors whose view stage owns objectness (GraspNet-baseline,
    Scale-Balanced-Grasp) as well as the graspness family, whose decoders then run
    without a graspness filter on the seeds.

    Args:
        num_seeds: Seeds per frame. A frame with fewer points, or a sampler that returns
            fewer (a voxel grid), is filled with repeats that are marked invalid.
        sampler: Picks the seeds; it receives the coordinates and the scene features of
            every point of a frame. ``None`` selects ``FPS``. A sampler that needs scores
            is refused.
        in_channels: Channels of the scene features; injected by the detector.
    """

    requires = "dense"
    status = "tested"
    source = (
        "Dense-feature seed sampling as in GraspNet-baseline (Fang et al., CVPR 2020) with the "
        "samplers of 3DSSD (Yang et al., CVPR 2020) and others as options"
    )
    #: What a sampler inside this proposal can be given besides coordinates.
    sampler_inputs: ClassVar[tuple[str, ...]] = ("feats",)

    def __init__(
        self,
        num_seeds: int = 1024,
        sampler: Cfg[Sampler] | None = None,
        in_channels: int = INJECTED,
    ) -> None:
        super().__init__()
        if num_seeds < 1:
            raise ValueError(f"num_seeds must be positive, got {num_seeds}")
        self.num_seeds = num_seeds
        self.in_channels = in_channels
        self.sampler = SAMPLERS.build(sampler if sampler is not None else {"type": "FPS"})
        missing = [name for name in self.sampler.needs if name not in self.sampler_inputs]
        if missing:
            raise ValueError(
                f"SampledSeeds predicts no score, but {type(self.sampler).__name__} needs "
                f"{' and '.join(missing)}; use ScoredSeeds for score-driven samplers"
            )

    @classmethod
    def check_config(cls, node: dict[str, Any]) -> Iterator[str]:
        yield from _sampler_problems(cls, node)

    def forward(self, feats: PointFeatures, observation: Observation) -> tuple[SeedSet, None]:
        """Sample the seeds of every frame.

        Args:
            feats: Scene features. They must live on ``observation.points`` itself.
            observation: The batch being processed.

        Returns:
            The seeds, and ``None``: there are no per-point scores to supervise.
        """
        name = type(self).__name__
        points = observation.points
        if points is None:
            raise ContractError(f"{name} needs observation.points")
        feats.expect(points, user=f"{name} (requires '{self.requires}' scene features)")
        index, valid = self._pick(points.coords, feats.feats.detach(), _offsets(points))
        offsets = torch.arange(len(index) + 1, device=index.device) * self.num_seeds
        flat = index.reshape(-1)
        seeds = SeedSet(
            points=points.subset(flat, offsets),
            feats=feats.feats[flat],
            valid=valid.reshape(-1),
            source="sampled_seeds",
        )
        return seeds, None

    @torch.no_grad()
    def _pick(self, coords: Tensor, feats: Tensor, offsets: list[int]) -> tuple[Tensor, Tensor]:
        index, valid = [], []
        for start, stop in pairwise(offsets):
            if stop == start:
                raise ContractError(f"{type(self).__name__} received a frame without points")
            chosen = self.sampler.sample(
                coords[start:stop], self.num_seeds, feats=feats[start:stop]
            )
            picks = chosen.to(coords.device) + start
            frame_index, frame_valid = _fill(picks, self.num_seeds, picks[:0], start)
            index.append(frame_index)
            valid.append(frame_valid)
        return torch.stack(index), torch.stack(valid)

    def loss(self, scores: None, supervision: Supervision) -> dict[str, Tensor]:
        """Nothing is predicted here, so nothing is supervised."""
        return {}

    def extra_repr(self) -> str:
        return f"num_seeds={self.num_seeds}, sampler={self.sampler!r}"
