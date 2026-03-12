import re

import pytest

OPT_IN_MARKERS = ("latex", "snapshot")
_MARKEXPR_TOKEN_RE = re.compile(r"[A-Za-z_]+")


def pytest_addoption(parser):
    group = parser.getgroup("gerrytools")
    group.addoption(
        "--with-latex",
        action="store_true",
        default=False,
        help="Include tests marked 'latex'.",
    )
    group.addoption(
        "--with-snapshot",
        action="store_true",
        default=False,
        help="Include tests marked 'snapshot'.",
    )
    group.addoption(
        "--marked-only",
        action="store_true",
        default=False,
        help="When using additive gerrytools markers, run only the marked tests.",
    )


def _opt_in_markers_from_expression(markexpr: str) -> set[str]:
    tokens = _MARKEXPR_TOKEN_RE.findall(markexpr)
    if len(tokens) == 0 or "not" in tokens:
        return set()

    non_operators = {token for token in tokens if token not in {"and", "or"}}
    if len(non_operators) == 0:
        return set()
    if not non_operators.issubset(set(OPT_IN_MARKERS)):
        return set()

    return non_operators


def pytest_configure(config):
    markexpr = (config.option.markexpr or "").strip()
    opt_in_markers = _opt_in_markers_from_expression(markexpr)
    if len(opt_in_markers) == 0:
        return

    if "latex" in opt_in_markers:
        config.option.with_latex = True
    if "snapshot" in opt_in_markers:
        config.option.with_snapshot = True

    if config.getoption("--marked-only"):
        return

    baseline_expr = " and ".join(f"not {marker}" for marker in OPT_IN_MARKERS)
    config.option.markexpr = f"({baseline_expr}) or ({markexpr})"


def pytest_collection_modifyitems(config, items):
    include_latex = config.getoption("--with-latex")
    include_snapshot = config.getoption("--with-snapshot")

    skip_latex = pytest.mark.skip(reason="need --with-latex to run")
    skip_snapshot = pytest.mark.skip(reason="need --with-snapshot or --with-latex to run")

    for item in items:
        has_snapshot_marker = item.get_closest_marker("snapshot") is not None
        has_latex_marker = item.get_closest_marker("latex") is not None

        if has_snapshot_marker:
            if not (include_snapshot or include_latex):
                item.add_marker(skip_snapshot)
            continue

        if has_latex_marker and not include_latex:
            item.add_marker(skip_latex)
