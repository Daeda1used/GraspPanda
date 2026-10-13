"""Config errors that point at a path and explain how to fix it."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ConfigError", "Issue"]


@dataclass(frozen=True)
class Issue:
    """One problem found in a config.

    Attributes:
        path: Dotted config path (``model.head.num_angle``); empty for file-level problems.
        message: What is wrong and, where possible, what to write instead.
        source: File (or ``<command line>``) that set the offending value.
    """

    path: str
    message: str
    source: str | None = None


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
