"""Public surface for the LaTeX table and TikZ plot builders.

Exports resolve lazily (PEP 562): ``from gerrytools.latex import TexTable`` imports only the
table stack, and the TikZ plot classes load on first access.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gerrytools.latex._text import latex_escape
    from gerrytools.latex.document import TexDocument
    from gerrytools.latex.paintball import UNSET, PaintballPlot, Unset
    from gerrytools.latex.seatsvotes import SeatsVotesPlot
    from gerrytools.latex.table import TexTable
    from gerrytools.latex.tikz_table import TikzTable

__all__ = [
    "TexDocument",
    "TexTable",
    "TikzTable",
    "PaintballPlot",
    "SeatsVotesPlot",
    "UNSET",
    "Unset",
    "latex_escape",
]

_LAZY_EXPORTS = {
    "TexDocument": "gerrytools.latex.document",
    "TexTable": "gerrytools.latex.table",
    "TikzTable": "gerrytools.latex.tikz_table",
    "PaintballPlot": "gerrytools.latex.paintball",
    "SeatsVotesPlot": "gerrytools.latex.seatsvotes",
    "UNSET": "gerrytools.latex.paintball",
    "Unset": "gerrytools.latex.paintball",
    "latex_escape": "gerrytools.latex._text",
}


def __getattr__(name: str) -> object:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    attribute = getattr(importlib.import_module(module_name), name)
    globals()[name] = attribute  # cache so later lookups bypass __getattr__
    return attribute


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
