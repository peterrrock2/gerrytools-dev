"""Tests for non-visual LaTeX document behavior."""

from __future__ import annotations

import gc
from pathlib import Path
from types import SimpleNamespace

import pytest

import gerrytools.latex.document as document_module
from gerrytools.latex.document import TexDocument


def _fake_compile_pdf(
    self: TexDocument,
    preferred_engine: str | None = None,
) -> None:
    del preferred_engine
    self._pdf_path.write_bytes(b"%PDF-1.4\n% fake pdf\n")


def _fake_render_to_temp_png(
    self: TexDocument,
    preferred_engine: str | None = None,
    dpi: int = 250,
) -> None:
    del preferred_engine
    del dpi
    self._pdf_path.write_bytes(b"%PDF-1.4\n% fake pdf\n")
    self._png_path.write_bytes(b"\x89PNG\r\n\x1a\n")


# =========================
# == CONSTRUCTION & I/O ==
# =========================
class TestTexDocumentConstruction:
    def test_document_init_can_render_minimal_document_string(self):
        doc = TexDocument()
        doc.package_list = []
        doc.color_dict = {}

        expected = (
            r"\documentclass[border=2pt]{standalone}"
            "\n"
            r"\begin{document}"
            "\n"
            "\n"
            r"\end{document}"
        )
        assert str(doc) == expected

    def test_document_cleanup_removes_workdir_after_collection(self):
        doc = TexDocument()

        folder = doc._workdir
        assert folder.exists()

        del doc
        gc.collect()

        assert not folder.exists()

    def test_document_add_packages_deduplicates_plain_and_optioned_packages(self):
        doc = TexDocument()

        doc.add_packages("tikz")
        doc.add_packages(["tikz", "hyperref"])
        doc.add_package_with_options("geometry", options="margin=1in")
        doc.add_package_with_options("geometry", options=["margin=1in"])
        doc.add_package_with_options(
            "hyperref",
            options=["colorlinks", "linkcolor=blue"],
        )
        doc.add_package_with_options(
            "hyperref",
            options=["colorlinks", "linkcolor=blue"],
        )

        assert doc.package_list.count("tikz") == 1
        assert doc.package_list.count("hyperref") == 0
        assert doc.extra_package_commands.count(r"\usepackage[margin=1in]{geometry}") == 1
        assert (
            doc.extra_package_commands.count(r"\usepackage[colorlinks,linkcolor=blue]{hyperref}")
            == 1
        )

    def test_document_add_packages_skips_packages_already_loaded_with_options(self):
        doc = TexDocument()

        doc.add_package_with_options("geometry", options="margin=1in")
        doc.add_packages("geometry")

        assert "geometry" not in doc.package_list
        assert doc.extra_package_commands.count(r"\usepackage[margin=1in]{geometry}") == 1

    def test_document_reregistering_optioned_package_replaces_options(self):
        doc = TexDocument()

        doc.add_package_with_options("geometry", options="margin=1in")
        doc.add_package_with_options("geometry", options="margin=2cm")

        assert doc.extra_package_commands == [r"\usepackage[margin=2cm]{geometry}"]

    def test_document_add_command_appends_custom_latex(self):
        doc = TexDocument()

        command = r"\newcommand{\example}[1]{\textbf{#1}}"
        doc.add_command(command)

        assert command in str(doc)


# ====================
# == COLOR HANDLING ==
# ====================
class TestTexDocumentColors:
    def test_document_add_color_supports_rgb_hex_names_and_xcolor(self):
        doc = TexDocument()

        doc.add_color("myblue", (0, 0, 255))
        doc.add_color("myred", (1.0, 0, 0))
        doc.add_color("mygreen", "#00FF00")
        doc.add_color("mygreenname", "green")
        doc.add_color("mygray", "AAAAAA")
        doc.add_color("myxcolor", "red!60!black")
        doc.add_color("mytabblue", "tab:blue")

        preamble = doc.preamble
        assert r"\definecolor{myblue}{RGB}{0,0,255}" in preamble
        assert r"\definecolor{myred}{rgb}{1.00,0.00,0.00}" in preamble
        assert r"\definecolor{mygreen}{HTML}{00ff00}" in preamble
        assert r"\definecolor{mygreenname}{HTML}{00ff00}" in preamble
        assert r"\definecolor{mygray}{HTML}{aaaaaa}" in preamble
        assert r"\colorlet{myxcolor}{red!60!black}" in preamble
        assert r"\definecolor{mytabblue}{HTML}{1f77b4}" in preamble

    def test_resolve_color_reuses_existing_auto_colors(self):
        doc = TexDocument()

        initial_color_count = len(doc.color_dict)
        first = doc.resolve_color("#123456", prefix="auto")
        second = doc.resolve_color("#123456", prefix="ignored")

        assert first == second
        assert len(doc.color_dict) == initial_color_count + 1
        assert doc.color_dict[first] == ("HTML", "123456")

    def test_resolve_color_returns_existing_name_for_xcolor_expression(self):
        doc = TexDocument()

        color_name = doc.resolve_color("denim!25!amber", prefix="auto")

        assert color_name == "denim!25!amber"
        assert "denim!25!amber" not in doc.color_dict

    def test_resolve_color_reuses_rgb_auto_colors(self):
        doc = TexDocument()

        first = doc.resolve_color((0.1, 0.2, 0.3), prefix="auto")
        second = doc.resolve_color((0.1, 0.2, 0.3), prefix="ignored")

        assert first == second
        assert doc.color_dict[first] == ("rgb", (0.1, 0.2, 0.3))

    def test_resolve_color_reuses_RGB_auto_colors(self):
        doc = TexDocument()

        first = doc.resolve_color((10, 20, 30), prefix="auto")
        second = doc.resolve_color((10, 20, 30), prefix="ignored")

        assert first == second
        assert doc.color_dict[first] == ("RGB", (10, 20, 30))

    def test_document_bad_color_raises_for_invalid_inputs(self):
        doc = TexDocument()

        with pytest.raises(ValueError, match="Color must be an xcolor expression"):
            doc.add_color("badcolor1", "zzzzzz")

        with pytest.raises(ValueError, match="Color must be an xcolor expression"):
            doc.add_color("badcolor2", 2)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="Color must be an xcolor expression"):
            doc.add_color("badcolor3", (256, 0, 0))

        with pytest.raises(ValueError, match="cannot be registered"):
            doc.add_color("badcolor4", "none")

    def test_preamble_raises_for_unsupported_color_type(self):
        doc = TexDocument()
        doc.color_dict["bad"] = ("XYZ", "value")

        with pytest.raises(ValueError, match="Unsupported color type"):
            _ = doc.preamble


# =======================
# == SAVE OPERATIONS ==
# =======================
class TestTexDocumentSaveOperations:
    def test_save_pdf_validates_path_type_extension_and_parent(self, tmp_path: Path):
        doc = TexDocument()

        with pytest.raises(TypeError, match="Path must be a string or Path object"):
            doc.save_pdf(123)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match=r"File extension must be '\.pdf'"):
            doc.save_pdf(tmp_path / "out.png")

        with pytest.raises(FileNotFoundError, match="does not exist"):
            doc.save_pdf(tmp_path / "missing" / "out.pdf")

    def test_save_png_validates_path_type_extension_and_parent(self, tmp_path: Path):
        doc = TexDocument()

        with pytest.raises(TypeError, match="Path must be a string or Path object"):
            doc.save_png(123)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match=r"File extension must be '\.png'"):
            doc.save_png(tmp_path / "out.pdf")

        with pytest.raises(FileNotFoundError, match="does not exist"):
            doc.save_png(tmp_path / "missing" / "out.png")

    def test_save_pdf_uses_rendered_temp_pdf_without_latex_engine(
        self,
        monkeypatch,
        tmp_path: Path,
    ):
        monkeypatch.setattr(
            TexDocument,
            "_compile_pdf",
            _fake_compile_pdf,
        )
        doc = TexDocument()

        output_path = tmp_path / "document.pdf"
        doc.save_pdf(output_path)

        assert output_path.read_bytes().startswith(b"%PDF-1.4")

    def test_save_png_uses_rendered_temp_png_without_latex_engine(
        self,
        monkeypatch,
        tmp_path: Path,
    ):
        monkeypatch.setattr(
            TexDocument,
            "_render_to_temp_png",
            _fake_render_to_temp_png,
        )
        doc = TexDocument()

        output_path = tmp_path / "document.png"
        doc.save_png(output_path)

        assert output_path.read_bytes() == b"\x89PNG\r\n\x1a\n"


# ==========================
# == RENDERER HELPERS ==
# ==========================
class TestDocumentRendererHelpers:
    def test_render_pdf_to_png_raises_when_no_renderer_is_available(
        self, monkeypatch, tmp_path: Path
    ):
        monkeypatch.setattr(document_module, "_which_any", lambda names: None)

        with pytest.raises(RuntimeError, match="No PDF renderer found"):
            document_module._render_pdf_to_png(
                tmp_path / "input.pdf",
                tmp_path / "output.png",
            )

    def test_render_pdf_to_png_uses_pdftoppm_and_moves_produced_file(
        self,
        monkeypatch,
        tmp_path: Path,
    ):
        pdf_path = tmp_path / "input.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")
        output_path = tmp_path / "output.rendered"
        produced_path = tmp_path / "output.png"
        calls: list[list[str]] = []

        def fake_run(cmd, capture_output, text):
            del capture_output
            del text
            calls.append(cmd)
            produced_path.write_bytes(b"png-bytes")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(document_module, "_which_any", lambda names: "pdftoppm")
        monkeypatch.setattr(document_module.subprocess, "run", fake_run)

        document_module._render_pdf_to_png(pdf_path, output_path, dpi=300)

        assert calls[0][0] == "pdftoppm"
        assert calls[0][1:4] == ["-png", "-r", "300"]
        assert output_path.read_bytes() == b"png-bytes"
        assert not produced_path.exists()

    def test_render_pdf_to_png_uses_pdftocairo_renderer(self, monkeypatch, tmp_path: Path):
        pdf_path = tmp_path / "input.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")
        output_path = tmp_path / "output.png"
        calls: list[list[str]] = []

        def fake_run(cmd, capture_output, text):
            del capture_output
            del text
            calls.append(cmd)
            output_path.write_bytes(b"cairo-png")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(document_module, "_which_any", lambda names: "pdftocairo")
        monkeypatch.setattr(document_module.subprocess, "run", fake_run)

        document_module._render_pdf_to_png(pdf_path, output_path, dpi=180)

        assert calls[0][0] == "pdftocairo"
        assert calls[0][1:4] == ["-png", "-r", "180"]
        assert output_path.read_bytes() == b"cairo-png"

    def test_render_pdf_to_png_uses_ghostscript_renderer(self, monkeypatch, tmp_path: Path):
        pdf_path = tmp_path / "input.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")
        output_path = tmp_path / "output.png"
        calls: list[list[str]] = []

        def fake_run(cmd, capture_output, text):
            del capture_output
            del text
            calls.append(cmd)
            for arg in cmd:
                if arg.startswith("-sOutputFile="):
                    Path(arg.removeprefix("-sOutputFile=")).write_bytes(b"gs-png")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(document_module, "_which_any", lambda names: "gs")
        monkeypatch.setattr(document_module.subprocess, "run", fake_run)

        document_module._render_pdf_to_png(pdf_path, output_path)

        assert calls[0][0] == "gs"
        assert output_path.read_bytes() == b"gs-png"

    def test_render_pdf_to_png_uses_imagemagick_renderer(self, monkeypatch, tmp_path: Path):
        pdf_path = tmp_path / "input.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")
        output_path = tmp_path / "output.png"
        calls: list[list[str]] = []

        def fake_run(cmd, capture_output, text):
            del capture_output
            del text
            calls.append(cmd)
            Path(cmd[-1]).write_bytes(b"convert-png")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(document_module, "_which_any", lambda names: "convert")
        monkeypatch.setattr(document_module.subprocess, "run", fake_run)

        document_module._render_pdf_to_png(pdf_path, output_path)

        assert calls[0][0] == "convert"
        assert output_path.read_bytes() == b"convert-png"

    def test_render_pdf_to_png_raises_with_renderer_log_on_failure(
        self, monkeypatch, tmp_path: Path
    ):
        pdf_path = tmp_path / "input.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        def fake_run(cmd, capture_output, text):
            del cmd
            del capture_output
            del text
            return SimpleNamespace(returncode=1, stdout="out", stderr="err")

        monkeypatch.setattr(document_module, "_which_any", lambda names: "convert")
        monkeypatch.setattr(document_module.subprocess, "run", fake_run)

        with pytest.raises(RuntimeError, match="PDF->PNG render failed using convert"):
            document_module._render_pdf_to_png(pdf_path, tmp_path / "output.png")

    def test_render_pdf_to_png_raises_when_output_is_missing(self, monkeypatch, tmp_path: Path):
        pdf_path = tmp_path / "input.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        def fake_run(cmd, capture_output, text):
            del cmd
            del capture_output
            del text
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(document_module, "_which_any", lambda names: "gs")
        monkeypatch.setattr(document_module.subprocess, "run", fake_run)

        with pytest.raises(RuntimeError, match="renderer reported success but"):
            document_module._render_pdf_to_png(pdf_path, tmp_path / "output.png")

    def test_render_to_temp_png_raises_when_no_engine_is_found(self, monkeypatch):
        doc = TexDocument()
        monkeypatch.setattr(document_module, "_which_any", lambda names: None)

        with pytest.raises(RuntimeError, match="No TeX engine found"):
            doc._render_to_temp_png()

    def test_render_to_temp_png_raises_when_preferred_engine_is_missing(self, monkeypatch):
        doc = TexDocument()
        monkeypatch.setattr(document_module, "_which_any", lambda names: None)

        with pytest.raises(RuntimeError, match="TeX engine xelatex not found"):
            doc._render_to_temp_png(preferred_engine="xelatex")

    def test_render_to_temp_png_uses_standard_engine_and_calls_pdf_renderer(
        self,
        monkeypatch,
    ):
        doc = TexDocument()
        doc.body_string = "body"
        calls: dict[str, object] = {}

        def fake_run(cmd, capture_output, text):
            del capture_output
            del text
            calls["cmd"] = cmd
            doc._pdf_path.write_bytes(b"%PDF-1.4\n")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        def fake_render(pdf_path, png_path, dpi):
            calls["render"] = (pdf_path, png_path, dpi)
            png_path.write_bytes(b"png")

        monkeypatch.setattr(document_module, "_which_any", lambda names: "pdflatex")
        monkeypatch.setattr(document_module.subprocess, "run", fake_run)
        monkeypatch.setattr(document_module, "_render_pdf_to_png", fake_render)

        doc._render_to_temp_png(dpi=111)

        assert calls["cmd"] == [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "-output-directory",
            str(doc._workdir),
            str(doc._tex_path),
        ]
        assert calls["render"] == (doc._pdf_path, doc._png_path, 111)

    def test_render_to_temp_png_uses_tectonic_command_shape(self, monkeypatch):
        doc = TexDocument()
        calls: dict[str, object] = {}

        def fake_run(cmd, capture_output, text):
            del capture_output
            del text
            calls["cmd"] = cmd
            doc._pdf_path.write_bytes(b"%PDF-1.4\n")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        def fake_render(pdf_path, png_path, dpi):
            calls["render"] = (pdf_path, png_path, dpi)
            png_path.write_bytes(b"png")

        monkeypatch.setattr(document_module, "_which_any", lambda names: "tectonic")
        monkeypatch.setattr(document_module.subprocess, "run", fake_run)
        monkeypatch.setattr(document_module, "_render_pdf_to_png", fake_render)

        doc._render_to_temp_png()

        assert calls["cmd"] == ["tectonic", str(doc._tex_path), "--outdir", str(doc._workdir)]

    def test_render_to_temp_png_raises_when_compile_fails(self, monkeypatch):
        doc = TexDocument()

        def fake_run(cmd, capture_output, text):
            del cmd
            del capture_output
            del text
            return SimpleNamespace(returncode=1, stdout="stdout", stderr="stderr")

        monkeypatch.setattr(document_module, "_which_any", lambda names: "pdflatex")
        monkeypatch.setattr(document_module.subprocess, "run", fake_run)

        with pytest.raises(RuntimeError, match="LaTeX compile failed with pdflatex"):
            doc._render_to_temp_png()
