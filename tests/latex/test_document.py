import pytest

from gerrytools.latex.document import TexDocument


def test_document_init():
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


def test_document_cleanup():
    import gc
    import os

    doc = TexDocument()
    doc._render_to_temp_png()

    folder = doc._workdir
    del doc

    gc.collect()

    assert not os.path.exists(folder)


def test_document_add_package():
    doc = TexDocument()
    doc.add_packages("lipsum")
    doc.add_packages(["tikz", "mathscinet"])
    doc.add_package_with_options("geometry", options="margin=1in")
    doc.add_package_with_options("hyperref", options=["colorlinks", "linkcolor=blue"])

    assert r"\usepackage{lipsum}" in str(doc)
    assert r"\usepackage{tikz}" in str(doc)
    assert r"\usepackage{mathscinet}" in str(doc)


def test_document_add_color():
    doc = TexDocument()
    doc.add_color("myblue", (0, 0, 255))
    doc.add_color("myred", (1.0, 0, 0))
    doc.add_color("mygreen", "#00FF00")
    doc.add_color("mygreenname", "green")
    doc.add_color("mygray", "AAAAAA")
    doc.add_color("myxcolor", "red!60!black")
    doc.add_color("mytabblue", "tab:blue")

    output = str(doc)
    assert r"\definecolor{myblue}{RGB}{0,0,255}" in output
    assert r"\definecolor{myred}{rgb}{1.00,0.00,0.00}" in output
    assert r"\definecolor{mygreen}{HTML}{00ff00}" in output
    assert r"\definecolor{mygreenname}{HTML}{00ff00}" in output
    assert r"\colorlet{myxcolor}{red!60!black}" in output
    assert r"\definecolor{mytabblue}{HTML}{1f77b4}" in output


def test_document_bad_color_raises():
    doc = TexDocument()

    with pytest.raises(ValueError, match="Color must be an xcolor expression"):
        doc.add_color("badcolor1", "zzzzzz")

    with pytest.raises(ValueError, match="Color must be an xcolor expression"):
        doc.add_color("badcolor1", 2)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Color must be an xcolor expression"):
        doc.add_color("badcolor2", (256, 0, 0))

    with pytest.raises(ValueError, match="cannot be registered"):
        doc.add_color("badcolor3", "none")


def test_document_add_command():
    doc = TexDocument()

    cmd1 = r"\newcommand{\example}[1]{\textbf{#1}}"
    doc.add_command(cmd1)

    assert cmd1 in str(doc)
