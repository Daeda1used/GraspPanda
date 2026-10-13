"""The combination explorer, served from this machine with the real compose check.

The explorer of the documentation answers a selection from the measured records, and says
"not measured" for everything else. Served by :func:`serve` it also asks this
installation: every selection is composed by :func:`grasppanda.combine.compose.compose`,
so a combination nobody measured gets the verdict of the contract rules and of the config
check, the conflict and the fix when it is refused, and the composed config, which can be
saved. Nothing is trained: the page prints the probe command and never runs it.

Only the standard library is used. The server binds to the loopback address, answers
requests addressed to it by a loopback name only, and writes nowhere but into the
directory it was started in, one ``*.yaml`` file per explicit request.
"""

from __future__ import annotations

import json
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

__all__ = ["HOST", "check", "explorer_root", "page", "save", "serve", "server"]

#: The only address the server binds to.
HOST = "127.0.0.1"
DOCS = "https://daeda1used.github.io/GraspPanda/"
#: Files of the explorer the server hands out, with their content types.
STATIC = {
    "explorer.css": "text/css; charset=utf-8",
    "explorer.js": "text/javascript; charset=utf-8",
    "explorer_logic.js": "text/javascript; charset=utf-8",
    "explorer.json": "application/json; charset=utf-8",
}
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+@-]{0,120}\.yaml$")
_LOCK = threading.Lock()


def explorer_root() -> Path:
    """Directory of the explorer's files: packaged in a wheel, ``docs/`` in a checkout."""
    package = Path(__file__).resolve().parents[1]
    packaged = package / "_explorer"
    return packaged if packaged.is_dir() else package.parent / "docs" / "explorer"


def page() -> str:
    """The page of the live explorer: the files of the documentation page, on their own."""
    return f"""<!doctype html>
<html lang="en" dir="ltr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GraspPanda combination explorer (live)</title>
<link rel="stylesheet" href="explorer.css">
</head>
<body class="gx-standalone">
<main>
<h1>Combination explorer <span class="gx-live-tag">live</span></h1>
<p>Swap any stage of a grasp detector. Marks and verdicts come from the measured records; the
<b>contract check</b> of exactly your selection is run now, by this installation. Nothing is
trained here.</p>
<div id="gp-explorer" class="gpx" data-src="explorer.json" data-live="api/" data-docs="{DOCS}">
<noscript><p>The explorer needs JavaScript.</p></noscript>
</div>
</main>
<script src="explorer_logic.js"></script>
<script src="explorer.js"></script>
</body>
</html>
"""


def _methods() -> dict[str, str]:
    from .grid import FAMILIES, METHODS

    return {**METHODS, **FAMILIES}


def _compose(request: dict[str, Any]) -> Any:
    from .compose import compose

    method, sets = request.get("method"), request.get("sets")
    configs = _methods()
    if method not in configs:
        raise ValueError(f"unknown method {method!r}. Methods: {', '.join(configs)}.")
    if not isinstance(sets, list) or not all(isinstance(item, str) for item in sets):
        raise ValueError("`sets` must be a list of `slot=option` strings")
    if len(sets) > 32 or any(len(item) > 200 or "=" not in item for item in sets):
        raise ValueError("`sets` holds something that is no `slot=option`")
    with _LOCK:
        return compose(configs[method], sets)


def check(request: dict[str, Any]) -> dict[str, Any]:
    """Compose the selection ``{"method": ..., "sets": [...]}`` and report the verdict.

    The report is what ``grasppanda compose --json`` prints: the stages, what compose did,
    the contract rules group by group, the issues of the config check and, when the
    combination is accepted, the config.
    """
    composition = _compose(request)
    return {
        "command": composition.command,
        "accepted": composition.accepted,
        "adapter": composition.adapter,
        "stages": [
            {"slot": slot, "component": component, "origin": composition.origins.get(slot, "")}
            for slot, component in composition.stages.items()
        ],
        "actions": list(composition.actions),
        "contracts": [
            {"slots": list(verdict.slots), "problems": list(verdict.problems)}
            for verdict in composition.verdicts
        ],
        "problems": list(composition.problems),
        "issues": [{"path": issue.path, "message": issue.message} for issue in composition.issues],
        "config_hash": composition.config_hash,
        "config": composition.text() if composition.accepted else "",
    }


def save(request: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Write the composed config of an accepted selection into ``directory``.

    ``request["name"]`` is a plain file name ending in ``.yaml``; an existing file is kept
    unless ``request["overwrite"]`` is true. A refused selection writes nothing.
    """
    name = request.get("name")
    if not isinstance(name, str) or not _NAME.match(name):
        raise ValueError("the name must be a plain file name that ends in .yaml")
    target = directory / name
    if target.exists() and request.get("overwrite") is not True:
        return {"written": "", "exists": True, "name": name}
    composition = _compose(request)
    if not composition.accepted:
        raise ValueError("refused by the contract rules or the config check: nothing was written")
    target.write_text(composition.text(target), encoding="utf-8")
    return {"written": name, "exists": False, "name": name, "next": f"grasppanda probe {name}"}


class _Handler(BaseHTTPRequestHandler):
    server_version = "GraspPandaExplorer"
    directory: Path = Path.cwd()

    def log_message(self, format: str, *args: Any) -> None:  # the terminal stays quiet
        return

    # A page of another site must not reach this server through the visitor's browser:
    # requests are answered only when they are addressed to a loopback name, and a write
    # only when it comes from this page.
    def _local(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in ("127.0.0.1", "localhost", "::1")

    def _same_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        return origin.split("://", 1)[-1] == (self.headers.get("Host") or "")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if not self._local():
            self._json(HTTPStatus.FORBIDDEN, {"error": "this server answers localhost only"})
            return
        name = self.path.split("?", 1)[0].lstrip("/")
        if name in ("", "index.html"):
            self._send(HTTPStatus.OK, page().encode("utf-8"), "text/html; charset=utf-8")
        elif name in STATIC and (explorer_root() / name).is_file():
            self._send(HTTPStatus.OK, (explorer_root() / name).read_bytes(), STATIC[name])
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": f"no such file: {name}"})

    def do_POST(self) -> None:
        if not self._local() or not self._same_origin():
            self._json(HTTPStatus.FORBIDDEN, {"error": "this server answers its own page only"})
            return
        if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
            self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "send application/json"})
            return
        name = self.path.split("?", 1)[0].strip("/")
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= 65536:
                raise ValueError("the request body is missing or too large")
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("the request must be a JSON object")
            if name == "api/compose":
                self._json(HTTPStatus.OK, check(request))
            elif name == "api/save":
                self._json(HTTPStatus.OK, save(request, self.directory))
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": f"no such action: {name}"})
        except Exception as error:  # the request is wrong, or compose says why it cannot
            self._json(HTTPStatus.BAD_REQUEST, {"error": f"{type(error).__name__}: {error}"})


def server(port: int = 0, directory: Path | None = None) -> ThreadingHTTPServer:
    """A server of the live explorer on the loopback address; ``port`` 0 picks a free one.

    Args:
        port: The port to listen on.
        directory: Where composed configs are saved; the working directory by default.
    """
    if not (explorer_root() / "explorer.json").is_file():
        raise FileNotFoundError(
            f"the files of the explorer are missing from {explorer_root()}; "
            "in a checkout, run `python -m tools.make_explorer`"
        )
    handler = type("Handler", (_Handler,), {"directory": (directory or Path.cwd()).resolve()})
    httpd = ThreadingHTTPServer((HOST, port), handler)
    httpd.daemon_threads = True
    return httpd


def serve(port: int = 8765, directory: Path | None = None, open_browser: bool = True) -> int:
    """Serve the live explorer until interrupted; returns the exit status."""
    httpd = server(port, directory)
    url = f"http://{HOST}:{httpd.server_address[1]}/"
    print(f"GraspPanda combination explorer (live): {url}")
    print("  the contract check of every selection is run by this installation;")
    print(f"  composed configs are saved into {Path(httpd.RequestHandlerClass.directory)}")
    print("  nothing is trained. Stop with Ctrl+C.")
    if open_browser:
        import webbrowser

        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        httpd.server_close()
    return 0
