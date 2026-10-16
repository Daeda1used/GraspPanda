"""Reading the files a camera or a point-cloud tool writes, without extra dependencies.

Depth and colour images go through Pillow; point clouds are read from ``.npy``,
``.ply`` and ``.pcd`` by the small parsers below. Every reader returns plain NumPy arrays
and raises :class:`~grasppanda.apis.inferencer.InputError` with the reason when a file
is not what its name says.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .inferencer import InputError, plain_folder_advice

__all__ = ["read_depth", "read_intrinsics", "read_mask", "read_points", "read_rgb"]

_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}  # fmt: skip


def _existing(path: str | Path, what: str) -> Path:
    path = Path(path).expanduser()
    if path.is_dir():
        raise InputError(f"{what} {path} is a directory; give one file. {plain_folder_advice()}")
    if not path.is_file():
        raise InputError(f"{what} not found: {path}")
    return path


def read_depth(path: str | Path) -> np.ndarray:
    """A depth image as stored: ``(H, W)`` integers (PNG, TIFF) or whatever a ``.npy`` holds."""
    path = _existing(path, "depth image")
    if path.suffix.lower() == ".npy":
        return np.load(path)
    from PIL import Image

    with Image.open(path) as image:
        depth = np.array(image)
    if depth.ndim != 2:
        raise InputError(
            f"{path} has {depth.shape[2]} channels, so it is a picture of depth, not a depth "
            "image. A depth image is single-channel, usually 16-bit PNG in millimetres."
        )
    return depth


def read_rgb(path: str | Path) -> np.ndarray:
    """A colour image as ``(H, W, 3)`` uint8."""
    from PIL import Image

    with Image.open(_existing(path, "colour image")) as image:
        return np.array(image.convert("RGB"))


def read_mask(path: str | Path) -> np.ndarray:
    """An object mask as ``(H, W)``; non-zero pixels are objects."""
    path = _existing(path, "mask")
    if path.suffix.lower() == ".npy":
        return np.load(path)
    from PIL import Image

    with Image.open(path) as image:
        mask = np.array(image)
    return mask if mask.ndim == 2 else mask.max(axis=2)


def read_intrinsics(text: str) -> np.ndarray:
    """``fx,fy,cx,cy``, or a file holding those four numbers or a 3 x 3 camera matrix."""
    path = Path(text).expanduser()
    if path.is_file():
        if path.suffix.lower() == ".npy":
            values = np.load(path).astype(np.float64).reshape(-1)
        else:
            content = path.read_text().replace(",", " ").replace("[", " ").replace("]", " ")
            try:
                values = np.array([float(token) for token in content.split()])
            except ValueError:
                raise InputError(f"{path} does not hold numbers only") from None
        where = str(path)
    else:
        try:
            values = np.array([float(token) for token in text.replace(",", " ").split()])
        except ValueError:
            raise InputError(
                f"--intrinsics '{text}' is neither a file nor numbers; write fx,fy,cx,cy in "
                "pixels, for example 925,925,640,360"
            ) from None
        where = f"'{text}'"
    if values.size == 4:
        return values
    if values.size == 9:
        return values.reshape(3, 3)
    raise InputError(
        f"intrinsics {where} hold {values.size} numbers; expected fx,fy,cx,cy or a 3 x 3 matrix"
    )


def _colours(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size and values.max() <= 1.0:
        values = values * 255.0
    return np.clip(np.round(values), 0, 255).astype(np.uint8)


def _read_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    data = path.read_bytes()
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise InputError(f"{path} is not a PLY file")
    header = data[:end].decode("ascii", "replace").splitlines()
    body = data[data.index(b"\n", end) + 1 :]
    layout, count, fields, element = None, 0, [], None
    for line in header:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "format":
            layout = tokens[1]
        elif tokens[0] == "element":
            if element == "vertex":
                break  # the vertices come first; later elements (faces) are not needed
            element = tokens[1]
            if element != "vertex":
                raise InputError(f"{path}: the first element is '{element}', expected 'vertex'")
            count = int(tokens[2])
        elif tokens[0] == "property" and element == "vertex":
            if tokens[1] == "list" or tokens[1] not in _PLY_TYPES:
                raise InputError(f"{path}: unsupported vertex property '{line.strip()}'")
            fields.append((tokens[2], _PLY_TYPES[tokens[1]]))
    names = [name for name, _ in fields]
    if not {"x", "y", "z"} <= set(names):
        raise InputError(f"{path}: the vertices have no x, y, z properties")
    if layout == "ascii":
        rows = body.decode("ascii", "replace").split("\n")[:count]
        table = np.array([row.split()[: len(names)] for row in rows if row.strip()], dtype=float)
        column = {name: table[:, index] for index, name in enumerate(names)}
    elif layout in ("binary_little_endian", "binary_big_endian"):
        order = "<" if layout == "binary_little_endian" else ">"
        dtype = np.dtype([(name, order + kind) for name, kind in fields])
        if len(body) < count * dtype.itemsize:
            raise InputError(f"{path} is truncated: {count} vertices declared")
        column = np.frombuffer(body, dtype=dtype, count=count)
    else:
        raise InputError(f"{path}: unknown PLY format '{layout}'")
    points = np.stack([column["x"], column["y"], column["z"]], axis=1).astype(np.float64)
    colours = None
    if {"red", "green", "blue"} <= set(names):
        colours = _colours(np.stack([column["red"], column["green"], column["blue"]], axis=1))
    return points, colours


def _read_pcd(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    data = path.read_bytes()
    header: dict[str, list[str]] = {}
    offset = 0
    while True:
        end = data.find(b"\n", offset)
        if end < 0:
            raise InputError(f"{path} is not a PCD file (no DATA line)")
        line = data[offset:end].decode("ascii", "replace").strip()
        offset = end + 1
        if line and not line.startswith("#"):
            key, *values = line.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                break
    try:
        names = header["FIELDS"]
        sizes = [int(v) for v in header["SIZE"]]
        kinds = header["TYPE"]
        counts = [int(v) for v in header.get("COUNT", ["1"] * len(names))]
        total = int(header["POINTS"][0])
    except (KeyError, ValueError, IndexError):
        raise InputError(f"{path}: incomplete PCD header") from None
    if not {"x", "y", "z"} <= set(names):
        raise InputError(f"{path}: the cloud has no x, y, z fields")
    layout = header["DATA"][0].lower()
    kind_code = {"F": "f", "I": "i", "U": "u"}
    if layout == "ascii":
        table = np.loadtxt(data[offset:].decode("ascii", "replace").splitlines(), ndmin=2)
        starts = np.cumsum([0, *counts[:-1]])
        column = {name: table[:total, start] for name, start in zip(names, starts)}
    elif layout == "binary":
        dtype = np.dtype(
            [
                (name, f"<{kind_code[kind]}{size}", (count,) if count > 1 else ())
                for name, size, kind, count in zip(names, sizes, kinds, counts)
            ]
        )
        if len(data) - offset < total * dtype.itemsize:
            raise InputError(f"{path} is truncated: {total} points declared")
        column = np.frombuffer(data, dtype=dtype, count=total, offset=offset)
    else:
        raise InputError(
            f"{path} stores its points as '{layout}', which is not supported here. Save it as "
            "binary or ascii (for example with `pcl_convert_pcd_ascii_binary`), or as .ply."
        )
    points = np.stack([column["x"], column["y"], column["z"]], axis=1).astype(np.float64)
    colours = None
    packed = next((name for name in ("rgb", "rgba") if name in names), None)
    if packed is not None:  # one 32-bit word per point: 0x00RRGGBB
        raw = np.ascontiguousarray(column[packed])
        as_float = kinds[names.index(packed)] == "F"  # PCL stores the word in a float field
        word = raw.astype(np.float32).view(np.uint32) if as_float else raw.astype(np.uint32)
        colours = np.stack([(word >> 16) & 255, (word >> 8) & 255, word & 255], axis=1)
        colours = colours.astype(np.uint8)
    return points, colours


def read_points(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """A point cloud file as ``(N, 3)`` float64 and, when the file has them, ``(N, 3)`` uint8 colours.

    ``.npy`` holds ``(N, 3)`` points or ``(N, 6)`` points and colours; ``.ply`` and ``.pcd``
    may be ASCII or binary. Units and frame are the file's own: the inferencer expects
    metres in the camera frame.
    """
    path = _existing(path, "point cloud")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
        if array.ndim == 3 and array.shape[2] >= 3:  # an organised cloud (H, W, 3)
            array = array.reshape(-1, array.shape[2])
        if array.ndim != 2 or array.shape[1] not in (3, 6):
            raise InputError(
                f"{path} holds an array of shape {array.shape}; expected (N, 3) points or "
                "(N, 6) points and colours"
            )
        colours = _colours(array[:, 3:6]) if array.shape[1] == 6 else None
        return array[:, :3].astype(np.float64), colours
    if suffix == ".ply":
        return _read_ply(path)
    if suffix == ".pcd":
        return _read_pcd(path)
    raise InputError(f"unknown point cloud format '{suffix}' ({path}); use .npy, .ply or .pcd")
