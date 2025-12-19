import re

HEX8_PATTERN = re.compile(r"^#[0-9A-Fa-f]{8}$")
"""A compiled regular expression pattern to match 8-digit hexadecimal color strings."""
HEX8_OR_NONE_PATTERN = re.compile(r"^(#[0-9A-Fa-f]{8}|none)$", re.IGNORECASE)
"""A compiled regular expression pattern to match 8-digit hexadecimal color strings or "none"."""

VALID_COLOR_HEX_RE = re.compile(
    r"^#?(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$"
)
