"""Combining components: presets, composing a config from them, probing that it runs.

* :mod:`~grasppanda.combine.presets`: what to write for a component on table-top depth
  clouds, with the reasons;
* :mod:`~grasppanda.combine.compose`: a base config plus ``slot=choice`` pairs → a
  checked child config (``grasppanda compose``);
* :mod:`~grasppanda.combine.probe`: build, train a few iterations on real frames,
  predict, one verdict (``grasppanda probe``);
* :mod:`~grasppanda.combine.grid`: the combinations that are probed for the measured
  compatibility page.

Nothing is imported here: ``compose`` needs the registries and ``probe`` the engine.
"""
