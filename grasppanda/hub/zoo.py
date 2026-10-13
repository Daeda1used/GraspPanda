"""Looking a config up in the model zoo.

The zoo is the set of ``configs/<method>/metafile.yaml`` files. A model entry may list
weights in one of two states, told apart by ``weights.source``:

``grasppanda``
    Trained and released by this project. They load into the config as they are, so
    :func:`zoo_weights` downloads them (once, checked against their sha256).
``author``
    The checkpoint the authors of the method released, under the authors' licence. It
    is in the authors' format and has to be converted with
    ``tools/convert_checkpoint.py`` first; nothing is downloaded on the user's behalf.
"""

from __future__ import annotations

from pathlib import Path

from grasppanda.config import ConfigError, configs_root

from .metafile import Metafile, ModelEntry, load_metafile
from .weights import fetch

__all__ = ["find_model", "zoo_weights"]


def find_model(
    config: str | Path, root: str | Path | None = None
) -> tuple[Metafile, ModelEntry] | None:
    """The zoo entry of a config file, or ``None`` when no metafile lists it.

    Args:
        config: Path of a config file.
        root: Directory searched for ``<method>/metafile.yaml``; default: the shipped
            configs. A metafile names its configs relative to the shipped configs.
    """
    base = Path(root) if root else configs_root()
    target = Path(config).resolve()
    for path in sorted(base.glob("*/metafile.yaml")):
        try:
            metafile = load_metafile(path)
        except ConfigError:
            continue  # a broken metafile is reported by the zoo checks, not by a lookup
        for model in metafile.models:
            if (configs_root() / model.config).resolve() == target:
                return metafile, model
    return None


def zoo_weights(
    config: str | Path, root: str | Path | None = None, cache_dir: str | Path | None = None
) -> tuple[Path | None, str]:
    """Weights of the zoo that load into ``config`` as they are, and a note for the user.

    Returns:
        ``(path, note)``. ``path`` is a verified local file when the zoo lists weights
        released by this project for the config, else ``None``; ``note`` then says what
        the zoo knows instead (where the authors' checkpoint is and how to convert it).
    """
    found = find_model(config, root)
    if found is None:
        return None, "The model zoo does not list this config."
    metafile, model = found
    readme = metafile.path.parent / "README.md"
    weights = model.weights
    if weights is not None and weights.source == "grasppanda":
        return fetch(weights.url, weights.sha256, cache_dir=cache_dir), ""
    if weights is not None:
        return None, (
            f"The model zoo has no weights trained by GraspPanda for '{model.name}' yet. The "
            f"authors' checkpoint of {metafile.method} is at {weights.url} (licence: "
            f"{weights.license}); it is not downloaded for you. Convert it with "
            f"`python -m tools.convert_checkpoint` as described in {readme} and pass the result."
        )
    return None, (
        f"The model zoo has no weights for '{model.name}' yet; see {readme} for where the "
        "authors' checkpoint is and how to convert it."
    )
