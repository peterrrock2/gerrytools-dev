import math
import random
from string import hexdigits as hex

from gerrytools.typing import HexColor

DISTRICTR_COLOR_DICT = {
    "tombblue": "#0099cd",
    "nacho": "#ffca5d",
    "carribeangreen": "#00cd99",
    "viricgreen": "#99cd00",
    "indianlake": "#cd0099",
    "violetink": "#9900cd",
    "greendaze": "#8dd3c7",
    "playfulpurple": "#bebada",
    "smokedsalmon": "#fb8072",
    "meadowblossomblue": "#80b1d3",
    "mangocreamsicles": "#fdb462",
    "lastoflettuce": "#b3de69",
    "classicrose": "#fccde5",
    "lavendersweater": "#bc80bd",
    "mintfrappe": "#ccebc5",
    "yellowsand": "#ffed6f",
    "vic20creme": "#ffffb3",
    "bluecalico": "#a6cee3",
    "bumanguesblue": "#1f78b4",
    "sagesensation": "#b2df8a",
    "dryhighlightergreen": "#33a02c",
    "rubberradish": "#fb9a99",
    "akirared": "#e31a1c",
    "cinnamonbuff": "#fdbf6f",
    "orangejuice": "#ff7f00",
    "elfinherb": "#cab2d6",
    "poppypompadour": "#6a3d9a",
    "stirlandbattlemire": "#b15928",
    "spindrift": "#64ffda",
    "maldives": "#00b8d4",
    "velvetychestnut": "#a1887f",
    "radium": "#76ff03",
    "mindaro": "#dce775",
    "lilacgeode": "#b388ff",
    "informativepink": "#ff80ab",
    "exoticliras": "#d81b60",
    "tropicalhideaway": "#26a69a",
    "middleyellow": "#ffea00",
    "gonzoviolet": "#6200ea",
}


def hexshift(color: str, *, seed: int = 42) -> str:
    """Randomly modifies the provided hexadecimal color.

    Args:
        color (str): A hexadecimal color string; e.g. `"#FFFF00"`.

    Returns:
        str: A hexadecimal color string.
    """
    rng = random.Random(seed)

    # Choose a hexidecimal digit, first paring down the digits we'll use.
    h = hex.lower()[:-6]
    sub = rng.choice(h)
    char = rng.choice(color[1:])

    # Find the character we're going to replace that's *not* the same character
    # as the one we got from the hexadecimal string.
    while sub == char:
        sub = rng.choice(h)

    # Return the subbed string.
    return color.replace(char, sub)


def districtr(N: int) -> list[HexColor]:
    """Returns a list of N hex colors from the districtr palette.

    When ``N`` exceeds the number of base palette colors, the palette is
    extended by appending hex-shifted variants of the base colors.

    Args:
        N (int): The number of colors to return.

    Returns:
        list[HexColor]: A list of ``N`` hex color strings from the districtr
        palette.
    """

    colors = list(DISTRICTR_COLOR_DICT.values())

    repeats = math.ceil(N / len(colors))
    tail = [hexshift(c) for c in colors * (repeats - 1)]
    return (colors + (tail if tail else []))[:N]
