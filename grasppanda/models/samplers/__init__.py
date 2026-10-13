"""Samplers: which of the candidate points of a frame become seeds.

A sampler is a part of a proposal (``model.proposal.sampler``). It sees one frame at a
time: the coordinates of the candidate points, optionally one score per candidate and
the scene features, and returns the indices of the points it picks.
"""
