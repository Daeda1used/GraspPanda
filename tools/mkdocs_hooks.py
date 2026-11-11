"""Hooks of the documentation build (``hooks`` in mkdocs.yml).

One rule: a short identifier in running text or in a table stays in one piece. The theme
lets inline code break at a hyphen and draws each half as a chip of its own, so
``pointnet2-transformer`` at the end of a line reads as two options.
"""

from __future__ import annotations

import re
from typing import Any

#: Inline code without white space, up to this many characters, does not break. Longer
#: code (paths, config names with their folder) may, or it would overflow a phone.
LIMIT = 34
_SHORT = re.compile(rf"<code>([^<\s]{{1,{LIMIT}}})</code>")


def keep_short_code_together(html: str) -> str:
    return _SHORT.sub(r'<code class="gp-nowrap">\1</code>', html)


def on_page_content(html: str, **kwargs: Any) -> str:
    return keep_short_code_together(html)
