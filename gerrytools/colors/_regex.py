import re

VALID_COLOR_HEX_RE = re.compile(
    r"^#?(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$"
)
"""Matches 3/4/6/8-digit hex color strings, with or without a leading ``#``."""
