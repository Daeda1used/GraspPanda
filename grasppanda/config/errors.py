"""Config errors that point at a path and explain how to fix it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["REFUSAL_KINDS", "ConfigError", "Issue", "Refusal", "kind_of"]

#: Why a check refuses a config, kind by kind. Every rule of the stage contracts and of
#: the config checker states the kind of what it refuses, so that a refusal says whether
#: it is a property of the components or of what the toolbox implements today.
REFUSAL_KINDS: dict[str, str] = {
    "structural": (
        "a genuine incompatibility: the components cannot work together whatever their "
        "options (two label sets, two owners of one score and no option to drop one, a "
        "classification loss on a regression term)"
    ),
    "not-implemented": (
        "the combination makes sense, but the code that joins the two parts does not exist "
        "yet; the message names what would lift the refusal"
    ),
    "fixed-by-compose": (
        "an adapter or an option lifts it, and `grasppanda compose` inserts or sets it; the "
        "config was written without it (or composed with `--no-adapter`)"
    ),
    "not-a-combination": (
        "the components fit, but this config does not state them fully: a component lacks "
        "a required option, or two options disagree (a size, a grid, a range, a section)"
    ),
    "policy": (
        "refused on purpose until the config says otherwise (code a model hub would run on "
        "this machine)"
    ),
}


class Refusal(str):
    """A refusal message that knows its kind (:data:`REFUSAL_KINDS`) and its rule.

    It is the message itself, a ``str``, so everything that reads messages keeps working;
    ``kind`` and ``rule`` travel with it into the :class:`Issue` of the config check and
    into the report of ``grasppanda compose``.
    """

    kind: str
    rule: str

    def __new__(cls, message: str, kind: str, rule: str = "") -> Refusal:
        if kind not in REFUSAL_KINDS:
            raise ValueError(f"unknown refusal kind '{kind}'; kinds: {', '.join(REFUSAL_KINDS)}")
        made = super().__new__(cls, message)
        made.kind, made.rule = kind, rule
        return made

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (str(self), self.kind, self.rule))


def kind_of(message: Any, default: str | None = None) -> str | None:
    """The kind a message carries, or ``default`` for a plain string."""
    return getattr(message, "kind", default)


@dataclass(frozen=True)
class Issue:
    """One problem found in a config.

    Attributes:
        path: Dotted config path (``model.head.num_angle``); empty for file-level problems.
        message: What is wrong and, where possible, what to write instead.
        source: File (or ``<command line>``) that set the offending value.
        kind: Why it is refused (:data:`REFUSAL_KINDS`); ``None`` for a rule of a project
            that does not say.
    """

    path: str
    message: str
    source: str | None = None
    kind: str | None = None


class ConfigError(Exception):
    """Raised with every problem found in one validation pass, not just the first."""

    def __init__(self, issues: list[Issue], title: str | None = None) -> None:
        self.issues = list(issues)
        self.title = title
        super().__init__(self.render())

    def render(self) -> str:
        count = len(self.issues)
        noun = "problem" if count == 1 else "problems"
        head = f"{count} config {noun}"
        if self.title:
            head += f" in {self.title}"
        lines = [head]
        for issue in self.issues:
            lines.append("")
            lines.append(f"  {issue.path or '<file>'}")
            lines.extend(f"    {line}" for line in issue.message.splitlines())
            if issue.source:
                lines.append(f"    set in: {issue.source}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()
