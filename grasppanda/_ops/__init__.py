"""Optional compiled kernels behind :mod:`grasppanda.ops`.

Nothing in here is needed to use GraspPanda: every operator has a pure-PyTorch
reference implementation, and that implementation defines the result. This package
holds the C++/OpenMP and CUDA sources of faster kernels (``src/``), builds them on
first use with PyTorch's extension tooling (:mod:`.build`) and exposes them to the
dispatch layer (:mod:`.library`). Without a compiler, or when a build fails, the
operators silently keep using the reference; ``grasppanda env`` says which one runs.

The kernels were written for this repository from the operator definitions in
:mod:`grasppanda.ops`. User code never imports this package: backends are selected
with ``runtime.ops`` (``docs/advanced_guides/operators.md``).
"""
