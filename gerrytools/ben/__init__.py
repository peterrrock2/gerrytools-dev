"""Record GerryChain runs as self-describing BENDL files.

``RecordedChain`` wraps ``gerrychain.MarkovChain``: iterate it as usual and every step's
assignment streams into a BENDL bundle alongside the dual graph and optional metadata, published
atomically on clean completion. A clean run exposes a ``RecordedRun`` reader on
``chain.recording`` for reading the file back.
"""

from .recorded_chain import (
    GraphOrder,
    GraphOrderName,
    RecordedChain,
    RecordedRun,
    RunIterator,
    Variant,
)

__all__ = [
    "GraphOrder",
    "GraphOrderName",
    "RecordedChain",
    "RecordedRun",
    "RunIterator",
    "Variant",
]
