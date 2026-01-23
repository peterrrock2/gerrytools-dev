from matplotlib.axes import Axes

from gerrytools.colors import DEFAULT_GREY


def arrow(ax, text, orientation="horizontal", color=DEFAULT_GREY, padding=0.1) -> Axes:
    """
    For some partisan metrics, we want to draw an arrow showing where the POV-party's
    advantage is. Depending on the orientation of the scores (histograms have
    scores arranged horizontally, violinplots have scores arranged vertically),
    we either place the arrow at the bottom left, pointing rightward, or in the
    middle of the y-axis, pointing up.

    Args:
        ax (Axes): `Axes` object onto which the arrow's plotted.
        text (str): String plotted on top of the arrow.
        orientation (str, optional): Direction the arrow's pointing; acceptable
            values are `"horizontal"` and `"vertical"`. Defaults to `"horizontal"`.
        color (str, optional): Color of the arrow.
        padding (float, optional): Spacing between the arrow and its axis. Defaults
            to `0.1`.

    Returns:
        matplotlib `Axes`.
    """

    if orientation == "horizontal":
        x = ax.get_xlim()[0]
        y = ax.get_ylim()[0] - padding * ax.get_ylim()[1]
        horizontal_align = "left"
        rotation = 0
    elif orientation == "vertical":
        x = ax.get_xlim()[0] - padding * (sum(map(lambda x: abs(x), ax.get_xlim())))
        y = sum(ax.get_ylim()) / 2
        horizontal_align = "center"
        rotation = 90

    ax.text(
        x,
        y,
        text,
        ha=horizontal_align,
        va="center",
        color="white",
        rotation=rotation,
        size=10,
        bbox=dict(
            boxstyle="rarrow,pad=0.3",
            fc=color,
            alpha=1,
            ec="black",
        ),
    )

    return ax
