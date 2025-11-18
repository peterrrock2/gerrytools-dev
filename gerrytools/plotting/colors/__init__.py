from gerrytools.plotting.colors.districtr import districtr
from gerrytools.plotting.colors.seaborn import flare, purples, redbluecmap, greens
from gerrytools.plotting.colors.latex import latex
from gerrytools.plotting.colors.latex_full import latex_full
from gerrytools.plotting.colors.utils import compare_palettes, preview_palette


DEFAULT_GREY = "#5c676f"
"""
Default grey plotting color; used in histograms, violin plots, and arrows.
"""

CITIZEN_BLUE = "#4693b3"
"""
Citizen ensemble blue color; used in histograms, violin plots, and arrows. (Aka
Citizen Kane).
"""

OVERLAYS = ("gainsboro", "silver", "darkgray", "gray", "dimgrey")
"""
Overlay colors for choropleth maps.
"""


ENSEMBLE_COLORS = {
    "smc": "#ffca5d",
    "forest": "#00cd99",
    "rrc": "#0099cd",
    "revrecom": "#0099cd",
    "recoma": "#99cd00",
    "recomb": "#cd0099",
    "recomc": "#9900cd",
    "recomd": "#8dd3c7",
}
"""
A dictionary mapping ensemble abbreviations to their corresponding standard colors.

These were the colors used in the final version of the RRC paper.
"""


__all__ = [
    "districtr",
    "redbluecmap",
    "flare",
    "purples",
    "greens",
    "latex",
    "latex_full",
    "CITIZEN_BLUE",
    "OVERLAYS",
    "ENSEMBLE_COLORS",
    "compare_palettes",
    "preview_palette",
]
