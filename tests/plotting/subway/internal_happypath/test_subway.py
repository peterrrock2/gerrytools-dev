"""Tests for subway sign plot."""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import pytest

from gerrytools.plotting.other.subway import (
    SubwaySignOptions,
    _validate_subway_settings,
    subway_signs,
)

# ================
# == Validation ==
# ================


class TestSubwayValidation:
    """Cover _validate_subway_settings error paths."""

    def _default_options(self):
        return SubwaySignOptions()

    def test_invalid_orientation_raises(self):
        with pytest.raises(ValueError, match="orientation"):
            _validate_subway_settings(
                colors=["red"],
                labels=["A"],
                orientation="diagonal",  # type: ignore[arg-type]
                n_bands=None,
                max_items_per_band=None,
                sign_options=self._default_options(),
            )

    def test_both_n_bands_and_max_items_raises(self):
        with pytest.raises(ValueError, match="Only one"):
            _validate_subway_settings(
                colors=["red", "blue"],
                labels=["A", "B"],
                orientation="horizontal",
                n_bands=2,
                max_items_per_band=1,
                sign_options=self._default_options(),
            )

    def test_mismatched_label_color_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            _validate_subway_settings(
                colors=["red", "blue"],
                labels=["A"],
                orientation="horizontal",
                n_bands=None,
                max_items_per_band=None,
                sign_options=self._default_options(),
            )

    def test_empty_labels_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _validate_subway_settings(
                colors=[],
                labels=[],
                orientation="horizontal",
                n_bands=None,
                max_items_per_band=None,
                sign_options=self._default_options(),
            )

    def test_bad_radius_raises(self):
        bad_options = SubwaySignOptions(radius=-1.0)
        with pytest.raises(ValueError, match="radius"):
            _validate_subway_settings(
                colors=["red"],
                labels=["A"],
                orientation="horizontal",
                n_bands=None,
                max_items_per_band=None,
                sign_options=bad_options,
            )

    def test_zero_n_bands_raises(self):
        with pytest.raises(ValueError, match="n_bands"):
            _validate_subway_settings(
                colors=["red"],
                labels=["A"],
                orientation="horizontal",
                n_bands=0,
                max_items_per_band=None,
                sign_options=self._default_options(),
            )

    def test_zero_max_items_per_band_raises(self):
        with pytest.raises(ValueError, match="max_items_per_band"):
            _validate_subway_settings(
                colors=["red"],
                labels=["A"],
                orientation="horizontal",
                n_bands=None,
                max_items_per_band=0,
                sign_options=self._default_options(),
            )


# =====================
# == Basic rendering ==
# =====================


class TestSubwayBasicRendering:
    def test_simple_call_returns_without_error(self):
        """Three signs should render without any error."""
        subway_signs(
            colors=["red", "blue", "green"],
            labels=["A", "B", "C"],
        )

    def test_save_path_creates_file(self, tmp_path):
        """save_path should produce a file on disk."""
        out = str(tmp_path / "subway.png")
        subway_signs(
            colors=["red", "blue", "green"],
            labels=["A", "B", "C"],
            save_path=out,
        )
        assert Path(out).exists()

    def test_single_sign(self, tmp_path):
        """A single sign should render without error."""
        out = str(tmp_path / "single.png")
        subway_signs(colors=["#3366cc"], labels=["1"], save_path=out)
        assert Path(out).exists()


# ====================
# == Layout options ==
# ====================


class TestSubwayLayoutOptions:
    def test_max_items_per_band(self, tmp_path):
        """max_items_per_band=2 distributes 5 items across multiple rows."""
        out = str(tmp_path / "max2.png")
        subway_signs(
            colors=["red", "blue", "green", "orange", "purple"],
            labels=["1", "2", "3", "4", "5"],
            max_items_per_band=2,
            save_path=out,
        )
        assert Path(out).exists()

    def test_n_bands(self, tmp_path):
        """n_bands=2 distributes 6 items across 2 rows."""
        out = str(tmp_path / "nbands2.png")
        subway_signs(
            colors=["red", "blue", "green", "orange", "purple", "yellow"],
            labels=["1", "2", "3", "4", "5", "6"],
            n_bands=2,
            save_path=out,
        )
        assert Path(out).exists()

    def test_default_single_band_horizontal(self, tmp_path):
        """Default layout should place all signs in one horizontal band."""
        out = str(tmp_path / "single_band.png")
        subway_signs(
            colors=["red", "blue", "green"],
            labels=["A", "B", "C"],
            orientation="horizontal",
            save_path=out,
        )
        assert Path(out).exists()

    def test_vertical_orientation_default(self, tmp_path):
        """Default vertical layout places all signs in one column."""
        out = str(tmp_path / "vertical.png")
        subway_signs(
            colors=["red", "blue", "green"],
            labels=["A", "B", "C"],
            orientation="vertical",
            save_path=out,
        )
        assert Path(out).exists()

    def test_vertical_with_max_items_per_band(self, tmp_path):
        """Vertical with max_items_per_band should create multiple columns."""
        out = str(tmp_path / "vertical_max.png")
        subway_signs(
            colors=["red", "blue", "green", "orange", "purple"],
            labels=["1", "2", "3", "4", "5"],
            orientation="vertical",
            max_items_per_band=3,
            save_path=out,
        )
        assert Path(out).exists()

    def test_horizontal_with_n_bands(self, tmp_path):
        """Horizontal with n_bands distributes into rows."""
        out = str(tmp_path / "horiz_nbands.png")
        subway_signs(
            colors=["red", "blue", "green", "orange"],
            labels=["A", "B", "C", "D"],
            orientation="horizontal",
            n_bands=2,
            save_path=out,
        )
        assert Path(out).exists()


# =================
# == Ragged edge ==
# =================


class TestSubwayRaggedEdge:
    def test_ragged_edge_last(self, tmp_path):
        """7 signs in 3-column layout with raggededge='last'."""
        out = str(tmp_path / "ragged_last.png")
        subway_signs(
            colors=["red", "blue", "green", "orange", "purple", "cyan", "magenta"],
            labels=["1", "2", "3", "4", "5", "6", "7"],
            max_items_per_band=3,
            sign_options=SubwaySignOptions(raggededge="last"),
            save_path=out,
        )
        assert Path(out).exists()

    def test_ragged_edge_first(self, tmp_path):
        """7 signs in 3-column layout with raggededge='first'."""
        out = str(tmp_path / "ragged_first.png")
        subway_signs(
            colors=["red", "blue", "green", "orange", "purple", "cyan", "magenta"],
            labels=["1", "2", "3", "4", "5", "6", "7"],
            max_items_per_band=3,
            sign_options=SubwaySignOptions(raggededge="first"),
            save_path=out,
        )
        assert Path(out).exists()

    def test_ragged_edge_vertical_last(self, tmp_path):
        """7 signs vertical 3-row layout with raggededge='last'."""
        out = str(tmp_path / "ragged_vert_last.png")
        subway_signs(
            colors=["red", "blue", "green", "orange", "purple", "cyan", "magenta"],
            labels=["1", "2", "3", "4", "5", "6", "7"],
            orientation="vertical",
            max_items_per_band=3,
            sign_options=SubwaySignOptions(raggededge="last"),
            save_path=out,
        )
        assert Path(out).exists()

    def test_ragged_edge_vertical_first(self, tmp_path):
        """7 signs vertical 3-row layout with raggededge='first'."""
        out = str(tmp_path / "ragged_vert_first.png")
        subway_signs(
            colors=["red", "blue", "green", "orange", "purple", "cyan", "magenta"],
            labels=["1", "2", "3", "4", "5", "6", "7"],
            orientation="vertical",
            max_items_per_band=3,
            sign_options=SubwaySignOptions(raggededge="first"),
            save_path=out,
        )
        assert Path(out).exists()


# ===========================
# == Reverse display order ==
# ===========================


class TestSubwayReverseOrder:
    def test_reverse_display_order_horizontal(self, tmp_path):
        """reverse_display_order=True with horizontal layout."""
        out = str(tmp_path / "reverse_horiz.png")
        subway_signs(
            colors=["red", "blue", "green", "orange", "purple", "cyan"],
            labels=["1", "2", "3", "4", "5", "6"],
            reverse_display_order=True,
            save_path=out,
        )
        assert Path(out).exists()

    def test_reverse_display_order_vertical(self, tmp_path):
        """reverse_display_order=True with vertical layout."""
        out = str(tmp_path / "reverse_vert.png")
        subway_signs(
            colors=["red", "blue", "green", "orange", "purple", "cyan"],
            labels=["1", "2", "3", "4", "5", "6"],
            orientation="vertical",
            reverse_display_order=True,
            save_path=out,
        )
        assert Path(out).exists()

    def test_reverse_display_order_ragged_last(self, tmp_path):
        """reverse_display_order with ragged last edge."""
        out = str(tmp_path / "reverse_ragged_last.png")
        subway_signs(
            colors=["red", "blue", "green", "orange", "purple", "cyan", "magenta"],
            labels=["1", "2", "3", "4", "5", "6", "7"],
            max_items_per_band=3,
            reverse_display_order=True,
            sign_options=SubwaySignOptions(raggededge="last"),
            save_path=out,
        )
        assert Path(out).exists()

    def test_reverse_display_order_ragged_first(self, tmp_path):
        """reverse_display_order with ragged first edge."""
        out = str(tmp_path / "reverse_ragged_first.png")
        subway_signs(
            colors=["red", "blue", "green", "orange", "purple", "cyan", "magenta"],
            labels=["1", "2", "3", "4", "5", "6", "7"],
            max_items_per_band=3,
            reverse_display_order=True,
            sign_options=SubwaySignOptions(raggededge="first"),
            save_path=out,
        )
        assert Path(out).exists()


# ======================
# == Missing coverage ==
# ======================


class TestSubwayCoverageMissing:
    """Cover remaining uncovered branches: n_bands+vertical, invalid raggededge, bad color."""

    def test_n_bands_vertical_layout(self, tmp_path):
        """Vertical layouts also support `n_bands`."""
        out = str(tmp_path / "nbands_vert.png")
        subway_signs(
            colors=["red", "blue", "green", "orange", "purple", "yellow"],
            labels=["1", "2", "3", "4", "5", "6"],
            orientation="vertical",
            n_bands=2,
            save_path=out,
        )
        assert Path(out).exists()

    def test_invalid_raggededge_raises(self):
        """Invalid ragged-edge values raise `ValueError`."""
        import dataclasses

        from gerrytools.plotting.other.subway import _determine_grid_position

        opts = dataclasses.replace(SubwaySignOptions(), raggededge="invalid")
        with pytest.raises(ValueError, match="raggededge"):
            _determine_grid_position(
                linear_index=0,
                sign_options=opts,
                orientation="horizontal",
                item_count=3,
                row_count=1,
                column_count=3,
            )

    def test_invalid_color_raises(self):
        """Unconvertible colors raise `ValueError`."""
        with pytest.raises((ValueError, Exception)):
            subway_signs(
                colors=["not_a_real_color_xyz"],
                labels=["A"],
            )
