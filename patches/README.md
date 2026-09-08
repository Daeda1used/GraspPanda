# Compatibility patches

The installer applies these patches to local build copies or overlays. Original source checkouts stay unchanged; experiment provenance records the patch hashes.

| Component | Purpose |
|---|---|
| CenterGrasp | Python 3.11 dataclass defaults, headless execution and shared dependency compatibility |
| LauncherTemplate | Linux process invocation, folder links and platform-specific imports |
| GFLA | Shared-runtime compatibility for scene completion and grasp generation |
| scikit-geometry | pybind11 property lifetimes and consistent CGAL geometry types |

The scikit-geometry patch uses the library's exact-construction kernel consistently in its skeleton bindings. This can change numerical behavior relative to the original mixed-kernel binding.

Patches retain the underlying components' [licenses](../docs/THIRD_PARTY.md).
