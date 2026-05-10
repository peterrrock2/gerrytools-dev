"""Tests for the named color source registry.

The registry replaces the cascade of four parallel dicts (gerrytools, districtr,
latex, matplotlib) with an ordered tuple of `NamedColorSource` instances. These
tests exercise the resolver, the precedence ordering, and the `which_color_source`
provenance helper.
"""

import pytest

from gerrytools.colors import which_color_source
from gerrytools.colors._sources import (
    _OVERRIDES_SOURCE,
    _REGISTRY,
    NamedColorSource,
    _resolve_named_color,
    get_all_supported_colors_dict,
)

# =====================================
# == NamedColorSource lookup contract ==
# =====================================


class TestNamedColorSourceLookup:
    def test_lookup_returns_hex_for_known_exact_name(self):
        source = NamedColorSource(name="example", mapping={"foo": "#abcdef"})
        assert source.lookup("foo") == "#abcdef"

    def test_lookup_returns_hex_for_lowercased_name(self):
        source = NamedColorSource(name="example", mapping={"foo": "#abcdef"})
        assert source.lookup("FOO") == "#abcdef"
        assert source.lookup("Foo") == "#abcdef"

    def test_lookup_returns_none_for_unknown_name(self):
        source = NamedColorSource(name="example", mapping={"foo": "#abcdef"})
        assert source.lookup("bar") is None

    def test_lookup_prefers_exact_over_lowercase_when_both_exist(self):
        source = NamedColorSource(
            name="example",
            mapping={"FOO": "#aaaaaa", "foo": "#bbbbbb"},
        )
        # Exact "FOO" hits the upper-case key; "foo" hits its own entry.
        assert source.lookup("FOO") == "#aaaaaa"
        assert source.lookup("foo") == "#bbbbbb"

    def test_lowercase_index_is_built_once_at_construction(self):
        # The lowercase index is a private cached field; mutating the original
        # mapping post-construction must NOT affect lookups.
        original_mapping = {"foo": "#abcdef"}
        source = NamedColorSource(name="example", mapping=original_mapping)
        assert source.lookup("FOO") == "#abcdef"
        original_mapping["bar"] = "#000000"  # mutate after construction
        assert source.lookup("bar") == "#000000"  # exact still scans live mapping
        assert source.lookup("BAR") is None  # lowercase index is frozen

    def test_named_color_source_is_frozen(self):
        source = NamedColorSource(name="example", mapping={"foo": "#abcdef"})
        with pytest.raises((AttributeError, TypeError)):
            source.name = "renamed"  # type: ignore[misc]


# ============================
# == REGISTRY RESOLUTION ==
# ============================


class TestRegistryResolver:
    def test_resolve_overrides_source_wins_for_green(self):
        # The overrides source comes first in the registry; "green" resolves
        # to bright #00ff00 even though matplotlib also defines it.
        assert _resolve_named_color("green") == "#00ff00"
        assert _resolve_named_color("Green") == "#00ff00"
        assert _resolve_named_color("GREEN") == "#00ff00"

    def test_resolve_gerrytools_aliases(self):
        assert _resolve_named_color("citizen_blue") == "#4693b3"
        assert _resolve_named_color("default_grey") == "#5c676f"
        assert _resolve_named_color("default_gray") == "#5c676f"

    def test_resolve_color_corrected_palette(self):
        # cc:* names previously raised KeyError; with Candidate 2 they resolve.
        assert _resolve_named_color("cc:applegreen") == "#73b900"
        assert _resolve_named_color("cc:denim") == "#0064bd"
        # Case-insensitive
        assert _resolve_named_color("CC:APPLEGREEN") == "#73b900"

    def test_resolve_districtr_palette(self):
        # "tombblue" is in DISTRICTR_COLOR_DICT
        assert _resolve_named_color("tombblue") == "#0099cd"
        assert _resolve_named_color("ToMbBlUe") == "#0099cd"

    def test_resolve_latex_palette(self):
        assert _resolve_named_color("red") == "#ff0000"

    def test_resolve_matplotlib_palette(self):
        # "tab:blue" only lives in matplotlib's own mapping
        result = _resolve_named_color("tab:blue")
        assert result.startswith("#")

    def test_resolve_unknown_name_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown color name"):
            _resolve_named_color("absolutely-not-a-color")


# ===============================
# == PROVENANCE — which_color_source ==
# ===============================


class TestWhichColorSource:
    def test_overrides_source_owns_green(self):
        assert which_color_source("green") == "overrides"
        assert which_color_source("GREEN") == "overrides"

    def test_gerrytools_source_owns_citizen_blue(self):
        assert which_color_source("citizen_blue") == "gerrytools"
        assert which_color_source("default_grey") == "gerrytools"

    def test_color_corrected_source_owns_cc_names(self):
        assert which_color_source("cc:applegreen") == "color-corrected"

    def test_districtr_source_owns_districtr_names(self):
        assert which_color_source("tombblue") == "districtr"

    def test_latex_source_owns_red(self):
        # "red" is in matplotlib too, but the registry order puts latex first.
        assert which_color_source("red") == "latex"

    def test_matplotlib_source_owns_matplotlib_only_names(self):
        assert which_color_source("tab:blue") == "matplotlib"

    def test_unknown_name_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown color name"):
            which_color_source("definitely-not-a-color")


# =====================
# == REGISTRY SHAPE ==
# =====================


class TestRegistryShape:
    def test_overrides_source_is_first_in_registry(self):
        assert _REGISTRY[0] is _OVERRIDES_SOURCE

    def test_registry_source_names_are_unique(self):
        names = [source.name for source in _REGISTRY]
        assert len(names) == len(set(names))

    def test_registry_includes_all_six_named_sources(self):
        expected_names = {
            "overrides",
            "gerrytools",
            "color-corrected",
            "districtr",
            "latex",
            "matplotlib",
        }
        assert {source.name for source in _REGISTRY} == expected_names


# ==================================
# == get_all_supported_colors_dict ==
# ==================================


class TestGetAllSupportedColorsDict:
    def test_includes_overrides_value_for_green(self):
        # overrides has higher precedence than matplotlib; the composed dict
        # must reflect that.
        all_colors = get_all_supported_colors_dict()
        assert all_colors["green"] == "#00ff00"

    def test_includes_gerrytools_aliases(self):
        all_colors = get_all_supported_colors_dict()
        assert all_colors["citizen_blue"] == "#4693b3"

    def test_includes_color_corrected_names(self):
        all_colors = get_all_supported_colors_dict()
        assert all_colors["cc:applegreen"] == "#73b900"

    def test_includes_none_sentinel(self):
        all_colors = get_all_supported_colors_dict()
        assert all_colors["none"] == "none"

    def test_higher_precedence_overrides_lower(self):
        # For any name resolvable through the registry, the value in the
        # composed dict must equal what _resolve_named_color returns.
        all_colors = get_all_supported_colors_dict()
        for sample_name in ("green", "citizen_blue", "cc:applegreen", "tombblue", "red"):
            assert all_colors[sample_name] == _resolve_named_color(sample_name)
