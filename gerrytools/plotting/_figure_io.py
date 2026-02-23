from __future__ import annotations

from io import BytesIO
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcsetup
from matplotlib.figure import Figure


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
        if ip is not None and getattr(ip, "kernel", None) is not None:
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

    plt.figure(fig.number)
    plt.show(block=True)


def save_figure(fig: Figure, filepath: str, **kwargs: Any) -> None:
    """Save a Matplotlib figure with consistent defaults.

    Args:
        fig (Figure): Matplotlib figure to save.
        filepath (str): Output file path.
        **kwargs (Any): Additional ``Figure.savefig`` keyword arguments. If omitted,
            defaults are ``bbox_inches="tight"`` and ``dpi=fig.dpi``.
    """
    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("dpi", fig.dpi)
    fig.savefig(filepath, **kwargs)
