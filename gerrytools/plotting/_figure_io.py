from __future__ import annotations

from io import BytesIO
from typing import Any, cast

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcsetup
from matplotlib.figure import Figure

from gerrytools.typing import MplKwargs


def show_figure(
    fig: Figure,
    *,
    non_gui_filename: str,
    non_gui_prefix: str,
) -> None:
    """Display a figure inline in notebooks or via GUI backend in scripts.

    If no GUI backend is available, the figure is written to ``non_gui_filename``
    and a short message is printed.

    Args:
        fig (Figure): Matplotlib figure to display.
        non_gui_filename (str): Filepath used when no GUI backend is available.
        non_gui_prefix (str): Prefix used in the printed non-GUI status message.
    """
    # Notebook: display PNG inline
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if (
            ip is not None and getattr(ip, "kernel", None) is not None
        ):  # pragma: no cover — only reachable inside a live Jupyter kernel
            from IPython.display import Image, display

            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=fig.dpi)
            buf.seek(0)
            display(Image(data=buf.getvalue()))
            return
    except Exception:
        pass

    backend = matplotlib.get_backend()
    if backend not in rcsetup.interactive_bk:
        fig.savefig(non_gui_filename, bbox_inches="tight", dpi=fig.dpi)
        print(f"[{non_gui_prefix}] Non-GUI backend ({backend}); saved to {non_gui_filename}")
        return

    # pragma: no cover — only reachable when an interactive GUI backend (e.g. TkAgg) is
    # active; the test suite always uses the Agg (non-interactive) backend.
    plt.figure(fig.number)  # pragma: no cover
    plt.show(block=True)  # pragma: no cover


def save_figure(fig: Figure, filepath: str, **kwargs: object) -> None:
    """Save a Matplotlib figure with consistent defaults.

    Args:
        fig (Figure): Matplotlib figure to save.
        filepath (str): Output file path.
        **kwargs (object): Additional ``Figure.savefig`` keyword arguments. If omitted,
            defaults are ``bbox_inches="tight"`` and ``dpi=fig.dpi``.
    """
    savefig_kwargs: MplKwargs = dict(kwargs)
    savefig_kwargs.setdefault("bbox_inches", "tight")
    savefig_kwargs.setdefault("dpi", fig.dpi)
    # Cast to satisfy the type-checker: Matplotlib accepts a broad dynamic kwargs
    # surface here, but static stubs are narrower.
    fig.savefig(filepath, **cast(dict[str, Any], savefig_kwargs))
