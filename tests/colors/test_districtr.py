from string import hexdigits

from gerrytools.colors.districtr import DISTRICTR_COLOR_DICT, districtr, hexshift

# =================
# == HEX SHIFTING ==
# =================


class TestHexshift:
    def test_hexshift_is_deterministic_for_a_seed(self):
        shifted_once = hexshift("#ffca5d", seed=7)
        shifted_twice = hexshift("#ffca5d", seed=7)
        assert shifted_once == shifted_twice

    def test_hexshift_preserves_hex_shape_but_changes_value(self):
        shifted = hexshift("#ffca5d", seed=11)
        assert shifted.startswith("#")
        assert len(shifted) == 7
        assert all(char in hexdigits for char in shifted[1:])
        assert shifted != "#ffca5d"

    def test_hexshift_produces_different_result_for_different_seeds(self):
        a = hexshift("#ffca5d", seed=1)
        b = hexshift("#ffca5d", seed=2)
        # Different seeds should (almost certainly) produce different outputs
        assert a != b


# =======================
# == DISTRICTR PALETTE ==
# =======================


class TestDistrictrPalette:
    def test_districtr_returns_requested_prefix_for_small_n(self):
        base_colors = list(DISTRICTR_COLOR_DICT.values())
        assert districtr(3) == base_colors[:3]

    def test_districtr_returns_empty_list_for_zero(self):
        assert districtr(0) == []

    def test_districtr_returns_full_base_palette_at_exact_boundary(self):
        # N == len(base) — tail is empty, no hexshifting occurs
        base_colors = list(DISTRICTR_COLOR_DICT.values())
        result = districtr(len(base_colors))
        assert result == base_colors

    def test_districtr_extends_palette_with_shifted_tail(self):
        base_colors = list(DISTRICTR_COLOR_DICT.values())
        extended = districtr(len(base_colors) + 2)
        assert len(extended) == len(base_colors) + 2
        assert extended[: len(base_colors)] == base_colors
        assert extended[len(base_colors)] != base_colors[0]
        assert extended[len(base_colors) + 1] != base_colors[1]
