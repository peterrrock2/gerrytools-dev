import logging
import shutil
import subprocess
import tempfile
import uuid
import weakref
from pathlib import Path
from typing import Iterable, Optional

from gerrytools.latex._colors import to_latex_color_spec
from gerrytools.logging import get_logger
from gerrytools.typing import Color

logger = get_logger(__name__)


def _render_pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 250) -> None:
    """
    Render the first page of a PDF to a PNG without PyMuPDF.

    Preference order:
      1) pdftocairo (poppler)
      2) pdftoppm   (poppler)
      3) gs         (ghostscript)
      4) magick/convert (imagemagick)  [least preferred]
    """
    renderer = _which_any(["pdftocairo", "pdftoppm", "gs", "magick", "convert"])
    if renderer is None:
        raise RuntimeError(
            "No PDF renderer found. Install one of:\n"
            "  - poppler-utils (pdftocairo / pdftoppm)\n"
            "  - ghostscript (gs)\n"
            "  - imagemagick (magick/convert)\n"
        )

    out_base = png_path.with_suffix("")  # e.g. /tmp/xyz -> renderer appends .png

    if renderer == "pdftocairo":
        # Produces exactly <out_base>.png
        cmd = [
            "pdftocairo",
            "-png",
            "-r",
            str(dpi),
            "-f",
            "1",
            "-l",
            "1",
            "-singlefile",
            str(pdf_path),
            str(out_base),
        ]

    elif renderer == "pdftoppm":
        # Produces exactly <out_base>.png with -singlefile
        cmd = [
            "pdftoppm",
            "-png",
            "-r",
            str(dpi),
            "-f",
            "1",
            "-singlefile",
            str(pdf_path),
            str(out_base),
        ]

    elif renderer == "gs":
        # Safe-ish ghostscript invocation: first page only, transparent background
        cmd = [
            "gs",
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=pngalpha",
            f"-r{dpi}",
            "-dFirstPage=1",
            "-dLastPage=1",
            f"-sOutputFile={str(png_path)}",
            str(pdf_path),
        ]

    else:  # imagemagick convert/magick
        # Note: some distros lock down PDF conversion in ImageMagick policy.xml
        cmd = [
            renderer,
            "-density",
            str(dpi),
            f"{str(pdf_path)}[0]",
            "-quality",
            "100",
            str(png_path),
        ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        raise RuntimeError(f"PDF->PNG render failed using {renderer}.\n\nLOG:\n{log}")

    # Poppler outputs <out_base>.png; ensure final path exists where we expect.
    if renderer in {"pdftocairo", "pdftoppm"}:
        produced = out_base.with_suffix(".png")
        if produced != png_path:
            png_path.unlink(missing_ok=True)
            produced.replace(png_path)

    if not png_path.exists():
        raise RuntimeError(f"PDF->PNG renderer reported success but {png_path} not found.")


def _which_any(names: Iterable[str]) -> Optional[str]:  # pragma: no cover
    """Checks for the first available executable in the provided list.

    Args:
        names (Iterable[str]): List of executable names to check.

    Returns:
        Optional[str]: The name of the first found executable, or None if none are found.
    """
    for n in names:
        if shutil.which(n):
            return n
    return None


def _in_jupyter_kernel() -> bool:  # pragma: no cover
    """Checks if the current environment is a IPython (Jupyter) kernel.

    Returns:
        bool: True if running in a IPython (Jupyter) kernel, False otherwise.
    """
    try:
        # Import here this doesn't need to be installed to run package
        from IPython.core.getipython import get_ipython

        ip = get_ipython()
        return ip is not None and "IPKernelApp" in getattr(ip, "config", {})
    except Exception:
        return False


class TexDocument:
    """Class for creating and previewing LaTeX documents.

    Allows adding packages, commands, and colors, and can render
    the LaTeX content to a PNG for display in Jupyter or a Qt window.

    Args:
        tex_string (str): The LaTeX content to be previewed.

    Attributes:
        body_string (str): The LaTeX document body content.
        package_list (list[str]): List of LaTeX packages to include.
        extra_package_commands (list[str]): Additional LaTeX package commands.
        command_list (list[str]): List of custom LaTeX commands.
        color_dict (dict[str, tuple[str, Color]]): Dictionary of color definitions.
        engine_preference_order (tuple[str, ...]): Preferred order of TeX engines to use.

    Methods:
        preview(): Renders and displays the LaTeX content.
    """

    def __init__(self) -> None:
        self._uuid = uuid.uuid4().hex
        self._workdir: Path = Path(tempfile.mkdtemp(prefix="latex-preview-"))
        self._tex_path = self._workdir / f"{self._uuid}.tex"
        self._pdf_path = self._workdir / f"{self._uuid}.pdf"
        self._png_path = self._workdir / f"{self._uuid}.png"
        self.body_string: str = ""
        self.package_list: list[str] = [
            "amsmath",
            "amssymb",
            "graphicx",
            "booktabs",
            "array",
            "latexcolors",
            "siunitx",
            "xfp",
        ]
        self.extra_package_commands: list[str] = []
        self.command_list: list[str] = []
        self.color_dict: dict[
            str, tuple[str, str | tuple[float, float, float] | tuple[int, int, int]]
        ] = {
            "snsgreen": ("rgb", (0.16, 0.51, 0.25)),
            "snspurple": ("rgb", (0.5, 0.24, 0.55)),
        }
        self._auto_color_count = 0
        self._auto_color_map: dict[str, str] = {}
        self.engine_preference_order = ("tectonic", "pdflatex", "xelatex", "lualatex")
        self._finalizer = weakref.finalize(
            self,
            shutil.rmtree,
            self._workdir,
            True,  # ignore_errors
        )

    def __repr__(self) -> str:  # pragma: no cover
        return self._tex_doc_string()

    def __str__(self) -> str:  # pragma: no cover
        return self._tex_doc_string()

    def add_packages(self, packages: str | list[str]) -> None:
        """Adds one or more LaTeX packages to the package list.

        Examples:
            tex_preview.add_packages("geometry")
            tex_preview.add_packages(["geometry", "hyperref"])

        Args:
            packages (str | list[str]): A single package name or a list of package names to add.

        Returns:
            None
        """
        if isinstance(packages, str):
            package_list = [packages]
        else:
            package_list = packages
        for package_name in package_list:
            if package_name not in self.package_list:
                self.package_list.append(package_name)

    def add_package_with_options(self, package_name: str, options: str | list[str]) -> None:
        """Adds a LaTeX package with options to the extra package commands.

        Examples:
            tex_preview.add_package_with_options("geometry", "margin=1in")
            tex_preview.add_package_with_options("hyperref", ["colorlinks", "linkcolor=blue"])

        Args:
            package_name (str): The name of the LaTeX package to add.
            options (str | list[str]): A single option string or a list of option strings.

        Returns:
            None
        """
        options_list = [options] if isinstance(options, str) else options
        options_str = ",".join(options_list)
        pkg_cmd = rf"\usepackage[{options_str}]{{{package_name}}}"
        if pkg_cmd not in self.extra_package_commands:
            self.extra_package_commands.append(pkg_cmd)

    def add_command(self, command: str) -> None:
        r"""Adds a custom LaTeX command to the document.

        Examples:
            tex_preview.add_command(r"\newcommand{\R}{\mathbb{R}}")

        Args:
            command (str): The LaTeX command to add.

        Returns:
            None
        """
        self.command_list.append(command)

    def add_color(self, color_name: str, color: Color) -> None:
        """Adds a custom color definition to the document.

        Examples:
            tex_preview.add_color("myblue", (0.0, 0.0, 1.0))

        Args:
            color_name (str): The name of the color to define.
            color (Color): HEX string or RGB tuple. RGB tuple values must all be in
                ``[0, 1]`` or all be in ``[0, 255]``.

        Returns:
            None

        Raises:
            ValueError: If ``color`` is not a valid HEX string or RGB tuple.
        """
        try:
            color_type, color_value = to_latex_color_spec(color)
        except ValueError as exc:
            if not isinstance(color, tuple) or len(color) != 3:
                raise ValueError("Color must be a HEX string or an RGB tuple of length 3.") from exc
            raise ValueError(
                "Color values must be in the range [0, 1] or in the range [0, 255]."
            ) from exc
        if color_type == "NAME":
            raise ValueError(
                "Color string must be a HEX string in the format '#RRGGBB' or 'RRGGBB'."
            )
        if color_type == "HTML":
            assert isinstance(color_value, str)
            self.color_dict[color_name] = ("HTML", color_value)
            return
        if color_type == "rgb":
            assert isinstance(color_value, tuple)
            self.color_dict[color_name] = (
                "rgb",
                (round(color_value[0], 2), round(color_value[1], 2), round(color_value[2], 2)),
            )
            return
        assert isinstance(color_value, tuple)
        self.color_dict[color_name] = (
            "RGB",
            (int(color_value[0]), int(color_value[1]), int(color_value[2])),
        )

    def _next_auto_color_name(self, prefix: str) -> str:
        """Generate the next unique auto-color name.

        Args:
            prefix (str): Prefix to prepend to the generated color name.

        Returns:
            str: Unique color name within this document instance.
        """
        self._auto_color_count += 1
        return f"{prefix}{self._auto_color_count}"

    def resolve_color(self, color: Color, *, prefix: str) -> str:
        """Resolve a ``Color`` value to a LaTeX-usable color name.

        Args:
            color (Color): Color value represented as a name, HEX string, or RGB tuple.
            prefix (str): Prefix used when defining new auto-generated color names.

        Returns:
            str: Existing or newly-defined LaTeX color name.
        """
        color_type, color_value = to_latex_color_spec(color)
        if color_type == "NAME":
            assert isinstance(color_value, str)
            return color_value

        if color_type == "HTML":
            assert isinstance(color_value, str)
            key = f"HTML:{color_value}"
        elif color_type == "rgb":
            assert isinstance(color_value, tuple)
            key = f"rgb:{color_value[0]:0.6f},{color_value[1]:0.6f},{color_value[2]:0.6f}"
        else:
            assert isinstance(color_value, tuple)
            key = f"RGB:{color_value[0]},{color_value[1]},{color_value[2]}"

        if key in self._auto_color_map:
            return self._auto_color_map[key]

        color_name = self._next_auto_color_name(prefix)
        if color_type == "HTML":
            assert isinstance(color_value, str)
            self.add_color(color_name, color_value)
        else:
            assert isinstance(color_value, tuple)
            self.add_color(color_name, color_value)
        self._auto_color_map[key] = color_name
        return color_name

    @property
    def preamble(self) -> str:
        lines = [r"\documentclass[border=2pt]{standalone}"]
        lines += [rf"\usepackage{{{pkg}}}" for pkg in self.package_list]
        lines.extend(self.extra_package_commands)
        for color_name, (color_type, color_val) in self.color_dict.items():
            match color_type:
                case "rgb":
                    assert isinstance(color_val, tuple), "Invalid color value for rgb."
                    lines.append(
                        rf"\definecolor{{{color_name}}}{{rgb}}{{{color_val[0]:0.2f},"
                        rf"{color_val[1]:0.2f},{color_val[2]:0.2f}}}"
                    )
                case "RGB":
                    assert isinstance(color_val, tuple), "Invalid color value for RGB."
                    lines.append(
                        rf"\definecolor{{{color_name}}}{{RGB}}{{{int(round(color_val[0]))},"
                        rf"{int(round(color_val[1]))},{int(round(color_val[2]))}}}"
                    )
                case "HTML":
                    lines.append(rf"\definecolor{{{color_name}}}{{HTML}}{{{color_val}}}")
                case "NONE":  # pragma: no coverj
                    pass
                case _:  # pragma: no cover
                    raise ValueError(
                        f"Unsupported color type: {color_type} found in color dictionary. "
                        "Only 'rgb', 'RGB', and 'HTML' are supported."
                    )
        return "\n".join(lines)

    def _tex_doc_string(self) -> str:
        """Generates the complete LaTeX document string."""
        lines = [self.preamble]
        lines.extend(self.command_list)
        lines += [r"\begin{document}", self.body_string, r"\end{document}"]
        output_string = "\n".join(lines).lstrip()
        logger.log(logging.DEBUG, "Generated LaTeX document:\n%s", output_string, stacklevel=2)
        return output_string

    def _render_to_temp_png(self, preferred_engine: Optional[str] = None, dpi: int = 250) -> None:
        """Renders the LaTeX document to a temporary PNG file."""
        if preferred_engine is None:
            engine = _which_any(self.engine_preference_order)
        else:  # pragma: no cover
            engine = _which_any([preferred_engine])

        if engine is None and preferred_engine is not None:  # pragma: no cover
            raise RuntimeError(f"TeX engine {preferred_engine} not found.")
        elif engine is None:  # pragma: no cover
            raise RuntimeError(
                f"No TeX engine found. Please install one of: [{', '.join(self.engine_preference_order)}]"
            )

        self._tex_path.write_text(self._tex_doc_string(), encoding="utf-8")

        if engine == "tectonic":  # pragma: no cover
            cmd = [engine, str(self._tex_path), "--outdir", str(self._workdir)]
        else:
            cmd = [
                engine,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                "-output-directory",
                str(self._workdir),
                str(self._tex_path),
            ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0 or not self._pdf_path.exists():  # pragma: no cover
            raise RuntimeError(f"LaTeX compile failed with {engine}.\n\nLOG:\n{log}")

        _render_pdf_to_png(self._pdf_path, self._png_path, dpi=dpi)

    def _show_png_jupyter(self) -> None:  # pragma: no cover
        """Displays the rendered PNG in a Jupyter notebook."""
        from IPython.display import Image, display

        display(Image(filename=str(self._png_path)))

    def _show_png_qt(
        self, *, title: str = "LaTeX Preview", max_size=(1200, 800)
    ) -> None:  # pragma: no cover
        """Displays the rendered PNG in a Qt window."""
        try:
            from PyQt6 import QtCore, QtGui, QtWidgets
        except ImportError as exc:
            raise RuntimeError(
                "PyQt6 is required for non-Jupyter preview. Install PyQt6 or use save_png/save_pdf."
            ) from exc

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        pix = QtGui.QPixmap(str(self._png_path))
        if pix.isNull():
            raise RuntimeError("Failed to load PNG into QPixmap.")

        max_w, max_h = max_size
        scale = min(1.0, max_w / max(1, pix.width()), max_h / max(1, pix.height()))
        shown = (
            pix
            if scale >= 1.0
            else pix.scaled(
                int(pix.width() * scale),
                int(pix.height() * scale),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        )

        label = QtWidgets.QLabel()
        label.setPixmap(shown)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(label)
        scroll.setWidgetResizable(True)

        win = QtWidgets.QMainWindow()
        win.setWindowTitle(title)
        win.setCentralWidget(scroll)
        win.resize(min(max_w, shown.width() + 30), min(max_h, shown.height() + 50))
        win.show()

        # In notebooks, calling app.exec() can hang; only do it when NOT in Jupyter.
        if not _in_jupyter_kernel():
            app.exec()

    def preview(self) -> None:  # pragma: no cover
        """Displays the rendered LaTeX document as a PNG image."""
        self._render_to_temp_png()
        if _in_jupyter_kernel():
            self._show_png_jupyter()
        else:
            self._show_png_qt(title="LaTeX Preview", max_size=(1200, 800))

    def save_pdf(self, path: str | Path) -> None:  # pragma: no cover
        """Saves the rendered LaTeX document as a PDF file.

        Args:
            path (str | Path): The file path to save the PDF to.
        Returns:
            None
        """
        if not isinstance(path, (str, Path)):
            raise ValueError("Path must be a string or Path object.")

        if not str(path).endswith(".pdf"):
            raise ValueError("File extension must be '.pdf'")

        full_path = Path(path).resolve()

        if not full_path.parent.exists():
            raise FileNotFoundError(f"The directory {full_path.parent} does not exist.")

        self._render_to_temp_png()
        shutil.copy2(self._pdf_path, full_path) if path else None

    def save_png(self, path: str | Path) -> None:  # pragma: no cover
        """Saves the rendered LaTeX document as a PNG file.

        Args:
            path (str | Path): The file path to save the PNG to.
        Returns:
            None
        """
        if not isinstance(path, (str, Path)):
            raise ValueError("Path must be a string or Path object.")

        if not str(path).endswith(".png"):
            raise ValueError("File extension must be '.png'")

        full_path = Path(path).resolve()

        if not full_path.parent.exists():
            raise FileNotFoundError(f"The directory {full_path.parent} does not exist.")

        self._render_to_temp_png()
        shutil.copy2(self._png_path, full_path) if path else None
