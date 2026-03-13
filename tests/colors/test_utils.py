import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gerrytools.colors.utils import compare_palettes, preview_palette

# =======================
# == PREVIEW PALETTES ==
# =======================


class TestPreviewPalette:
    def test_preview_palette_uses_existing_axes_when_provided(self):
        fig, ax = plt.subplots()
        out_fig, out_ax = preview_palette(["#ff0000", "#0000ff"], ax=ax)
        assert out_fig is fig
        assert out_ax is ax
        assert len(out_ax.patches) == 2
        plt.close(fig)

    def test_preview_palette_adds_annotations_when_both_index_and_hex_requested(self):
        fig, ax = preview_palette(["#ffffff", "#000000"], show_indices=True, show_hex=True)
        assert len(ax.texts) == 2
        assert ax.texts[0].get_text() == "0\n#ffffff"
        assert list(ax.get_xticks()) == []
        assert list(ax.get_yticks()) == []
        assert all(not spine.get_visible() for spine in ax.spines.values())
        plt.close(fig)

    def test_preview_palette_adds_only_hex_when_show_hex_without_indices(self):
        # show_hex=True, show_indices=False — no index prefix, no newline
        fig, ax = preview_palette(["#aabbcc"], show_hex=True, show_indices=False)
        assert len(ax.texts) == 1
        assert ax.texts[0].get_text() == "#aabbcc"
        plt.close(fig)

    def test_preview_palette_empty_list_produces_no_patches(self):
        fig, ax = preview_palette([])
        assert len(ax.patches) == 0
        plt.close(fig)


# =======================
# == COMPARE PALETTES ==
# =======================


class TestComparePalettes:
    def test_compare_palettes_accepts_mappings(self):
        fig, ax = compare_palettes({"warm": ["#ff0000", "#ffaa00"], "cool": ["#0000ff"]})
        assert len(ax.patches) == 3
        labels = {text.get_text() for text in ax.texts}
        assert "warm" in labels
        assert "cool" in labels
        plt.close(fig)

    def test_compare_palettes_accepts_sequences_and_hex_labels(self):
        fig, ax = compare_palettes([["#ffffff"], ["#000000", "#ff0000"]], show_hex=True)
        text_values = {text.get_text() for text in ax.texts}
        assert "0" in text_values
        assert "1" in text_values
        assert "#ffffff" in text_values
        assert "#000000" in text_values
        plt.close(fig)

    def test_compare_palettes_accepts_explicit_figsize(self):
        fig, ax = compare_palettes({"a": ["#ff0000"]}, figsize=(8, 4))
        assert fig.get_size_inches() == pytest.approx([8.0, 4.0])
        plt.close(fig)

    def test_compare_palettes_rejects_empty_input(self):
        with pytest.raises(ValueError, match="No palettes provided"):
            compare_palettes([])
