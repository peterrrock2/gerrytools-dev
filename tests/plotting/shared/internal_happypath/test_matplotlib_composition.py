"""End-to-end coverage of gerrytools composing with raw matplotlib.

Once a user has an :class:`~matplotlib.axes.Axes` — either because they
passed one to a gerrytools plot via ``ax=`` or pulled one out via
``plot.ax`` — that axes is a shared matplotlib surface. Most-recent-wins
per setting: whichever side (gerrytools or the user) touched a unit last
should be respected, and anything the user drew directly on the axes must
survive future gerrytools rebuilds.

These tests exercise the five concrete patterns we care about:

- subplot-grid embedding — the plot fills exactly its target cell;
- overlay on existing content (e.g. ``ax.imshow(backdrop)``) — the backdrop
  survives the gerrytools render;
- post-render customization — annotations added via ``ax.text(...)`` between
  renders survive the next rebuild;
- post-render axes-state mutation — ``ax.set_xlim(...)`` after ``.ax``
  survives the next rebuild;
- pre-render axes-state configuration — ``ax.set_xlim(...)`` *before*
  ``Histogram(ax=ax)`` survives the first render.

Both a data plot (``Histogram``) and a geometry plot (``ColoredGeoPlot``)
are exercised where the scenario is meaningful for both families.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from gerrytools.plotting.data.histogram import Histogram  # noqa: E402
from gerrytools.plotting.geometry.coloredgeoplot import ColoredGeoPlot  # noqa: E402

# ---------------------------------------------------------------------------
# Subplot grid embedding
# ---------------------------------------------------------------------------


class TestSubplotGridEmbedding:
    def test_histogram_renders_in_one_subplot_cell_only(self):
        fig, axes = plt.subplots(2, 2)
        hist = Histogram(ax=axes[0, 0])
        hist.add_histogram([1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0])
        hist.ax  # triggers render

        # Target cell has artists; sibling cells were untouched.
        assert len(axes[0, 0].patches) > 0
        for row, col in [(0, 1), (1, 0), (1, 1)]:
            assert len(axes[row, col].patches) == 0
            assert len(axes[row, col].lines) == 0

    def test_geoplot_renders_in_one_subplot_cell_only(self, testing_gdf):
        fig, axes = plt.subplots(2, 2)
        plot = ColoredGeoPlot(testing_gdf, ax=axes[0, 0])
        plot.add_outline_layer()
        plot.ax

        assert (len(axes[0, 0].patches) + len(axes[0, 0].lines) + len(axes[0, 0].collections)) > 0
        for row, col in [(0, 1), (1, 0), (1, 1)]:
            assert (
                len(axes[row, col].patches)
                + len(axes[row, col].lines)
                + len(axes[row, col].collections)
            ) == 0


# ---------------------------------------------------------------------------
# Overlay on existing content
# ---------------------------------------------------------------------------


class TestOverlayOnExistingContent:
    def test_imshow_backdrop_survives_histogram_render(self):
        fig, ax = plt.subplots()
        ax.imshow([[0.1, 0.2], [0.3, 0.4]])
        images_before = len(ax.images)
        hist = Histogram(ax=ax)
        hist.add_histogram([1.0, 2.0, 3.0])
        hist.ax
        assert len(ax.images) == images_before
        # And re-rendering preserves it too.
        hist.ax
        assert len(ax.images) == images_before

    def test_imshow_backdrop_survives_geoplot_render(self, testing_gdf):
        fig, ax = plt.subplots()
        ax.imshow([[0.1, 0.2], [0.3, 0.4]])
        images_before = len(ax.images)
        plot = ColoredGeoPlot(testing_gdf, ax=ax)
        plot.add_outline_layer()
        plot.ax
        assert len(ax.images) == images_before


# ---------------------------------------------------------------------------
# Post-render customization (external annotations)
# ---------------------------------------------------------------------------


class TestPostRenderCustomization:
    def test_external_text_survives_subsequent_histogram_rebuild(self):
        hist = Histogram()
        hist.add_histogram([1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0])
        ax = hist.ax
        ax.text(2.0, 0.5, "Note", color="red")
        hist.ax  # second render
        matching = [t for t in ax.texts if t.get_text() == "Note"]
        assert len(matching) == 1

    def test_external_text_survives_subsequent_geoplot_rebuild(self, testing_gdf):
        plot = ColoredGeoPlot(testing_gdf)
        plot.add_outline_layer()
        ax = plot.ax
        ax.text(0.5, 0.5, "Note", transform=ax.transAxes)
        plot.ax  # second render
        matching = [t for t in ax.texts if t.get_text() == "Note"]
        assert len(matching) == 1


# ---------------------------------------------------------------------------
# Post-render axes-state mutation
# ---------------------------------------------------------------------------


class TestPostRenderAxesStateMutation:
    def test_external_xlim_after_render_survives_rebuild_histogram(self):
        hist = Histogram()
        hist.add_histogram([1.0, 2.0, 3.0, 4.0, 5.0])
        ax = hist.ax
        ax.set_xlim(0.0, 100.0)
        hist.ax
        assert hist._ax.get_xlim() == (0.0, 100.0)

    def test_external_ylim_after_render_survives_rebuild_histogram(self):
        hist = Histogram()
        hist.add_histogram([1.0, 2.0, 3.0, 4.0, 5.0])
        ax = hist.ax
        ax.set_ylim(0.0, 999.0)
        hist.ax
        assert hist._ax.get_ylim() == (0.0, 999.0)

    def test_external_xlim_after_render_survives_rebuild_geoplot(self, testing_gdf):
        plot = ColoredGeoPlot(testing_gdf)
        plot.add_outline_layer()
        ax = plot.ax
        ax.set_xlim(-1000.0, 1000.0)
        plot.ax
        assert plot._ax.get_xlim() == (-1000.0, 1000.0)


# ---------------------------------------------------------------------------
# Pre-render axes-state configuration
# ---------------------------------------------------------------------------


class TestPreConfiguredAxes:
    def test_pre_set_xlim_preserved_after_histogram_first_render(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0.0, 50.0)
        hist = Histogram(ax=ax)
        hist.add_histogram([1.0, 2.0, 3.0])
        hist.ax
        assert ax.get_xlim() == (0.0, 50.0)

    def test_pre_set_title_preserved_after_histogram_first_render(self):
        fig, ax = plt.subplots()
        ax.set_title("user title")
        hist = Histogram(ax=ax)  # title=None default = "no opinion"
        hist.add_histogram([1.0, 2.0, 3.0])
        hist.ax
        assert ax.get_title() == "user title"

    def test_pre_set_xlim_preserved_after_geoplot_first_render(self, testing_gdf):
        fig, ax = plt.subplots()
        ax.set_xlim(0.0, 50.0)
        plot = ColoredGeoPlot(testing_gdf, ax=ax)
        plot.add_outline_layer()
        plot.ax
        assert ax.get_xlim() == (0.0, 50.0)


# ---------------------------------------------------------------------------
# Chained: most-recent-wins across multiple touches
# ---------------------------------------------------------------------------


class TestChainedTouches:
    def test_external_xlim_then_explicit_gerrytools_setter_wins(self):
        """After an external ``ax.set_xlim`` yields the unit to external
        state, a subsequent gerrytools ``set_xlim`` reclaims and wins."""
        hist = Histogram()
        hist.add_histogram([1.0, 2.0, 3.0, 4.0, 5.0])
        ax = hist.ax
        ax.set_xlim(0.0, 100.0)
        hist.ax
        assert hist._ax.get_xlim() == (0.0, 100.0)
        hist.set_xlim(-10.0, 10.0)
        hist.ax
        assert hist._ax.get_xlim() == (-10.0, 10.0)

    def test_imshow_backdrop_and_post_render_annotation_both_survive(self):
        """Backdrop imshow AND a post-render annotation should both
        survive the next rebuild."""
        fig, ax = plt.subplots()
        ax.imshow([[0.1, 0.2], [0.3, 0.4]])
        hist = Histogram(ax=ax)
        hist.add_histogram([1.0, 2.0, 3.0])
        hist.ax  # first render
        ax.text(1.0, 0.0, "Annotation", color="red")
        hist.ax  # second render
        assert len(ax.images) == 1
        assert any(t.get_text() == "Annotation" for t in ax.texts)


# ---------------------------------------------------------------------------
# matplotlib environment canary: annotation handles must stay removable
# ---------------------------------------------------------------------------


class TestAnnotationHandlesAreRemovable:
    """Tripwire for the artist registry's core assumption.

    The registry replaces ``ax.clear()`` by calling ``.remove()`` on each
    artist gerrytools created during a render. That only works while
    matplotlib's annotation APIs keep returning a live, removable handle.
    If a future matplotlib version changes any of these to return ``None``
    or an object whose ``.remove()`` is a no-op, the registry would
    silently leak artists across rebuilds (the registry swallows removal
    errors by design, so the failure would be invisible in normal use).

    These run once and fail loudly the moment that contract breaks, so a
    matplotlib upgrade surfaces the regression here rather than as drifting
    artist counts in unrelated plot tests.
    """

    def test_axvline_returns_removable_handle(self):
        self._assert_removable(lambda ax: ax.axvline(0.5))

    def test_axhline_returns_removable_handle(self):
        self._assert_removable(lambda ax: ax.axhline(0.5))

    def test_axvspan_returns_removable_handle(self):
        self._assert_removable(lambda ax: ax.axvspan(0.1, 0.2))

    def test_axhspan_returns_removable_handle(self):
        self._assert_removable(lambda ax: ax.axhspan(0.1, 0.2))

    def test_annotate_returns_removable_handle(self):
        self._assert_removable(lambda ax: ax.annotate("note", (0.5, 0.5)))

    @staticmethod
    def _assert_removable(draw):
        """Draw via ``draw(ax)`` and assert the handle is live and removable.

        Membership is checked against ``ax.get_children()`` rather than a
        specific container (``ax.lines`` / ``ax.patches`` / ``ax.texts``) so
        the canary does not care which list a given matplotlib version files
        the artist under — only that the returned handle is attached and that
        ``.remove()`` actually detaches it.
        """
        fig, ax = plt.subplots()
        try:
            handle = draw(ax)
            assert handle is not None, "matplotlib returned no handle"
            assert hasattr(handle, "remove"), f"{type(handle)!r} is not removable"
            assert handle in ax.get_children(), "handle was not attached to the axes"
            handle.remove()
            assert handle not in ax.get_children(), "handle survived .remove()"
        finally:
            plt.close(fig)
