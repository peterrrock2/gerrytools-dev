from random import choice
from string import hexdigits as hex
import math


def hexshift(color) -> str:
    """
    Randomly modifies the provided hexadecimal color.

    Args:
        color (str): A hexadecimal color string; e.g. `"#FFFF00"`.

    Returns:
        A hexadecimal color string.
    """
    # Choose a hexidecimal digit, first paring down the digits we'll use.
    h = hex.upper()[:-6]
    sub = choice(h)
    char = choice(color[1:])

    # Find the character we're going to replace that's *not* the same character
    # as the one we got from the hexadecimal string.
    while sub == char:
        sub = choice(h)

    # Return the subbed string.
    return color.replace(char, sub)


def districtr(N):
    colors = [
        "#0099cd",
        "#ffca5d",
        "#00cd99",
        "#99cd00",
        "#cd0099",
        "#9900cd",
        "#8dd3c7",
        "#bebada",
        "#fb8072",
        "#80b1d3",
        "#fdb462",
        "#b3de69",
        "#fccde5",
        "#bc80bd",
        "#ccebc5",
        "#ffed6f",
        "#ffffb3",
        "#a6cee3",
        "#1f78b4",
        "#b2df8a",
        "#33a02c",
        "#fb9a99",
        "#e31a1c",
        "#fdbf6f",
        "#ff7f00",
        "#cab2d6",
        "#6a3d9a",
        "#b15928",
        "#64ffda",
        "#00B8D4",
        "#A1887F",
        "#76FF03",
        "#DCE775",
        "#B388FF",
        "#FF80AB",
        "#D81B60",
        "#26A69A",
        "#FFEA00",
        "#6200EA",
    ]

    repeats = math.ceil(N / len(colors))
    tail = [hexshift(c) for c in colors * (repeats - 1)]
    return (colors + (tail if tail else []))[:N]
