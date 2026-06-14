from collections.abc import Callable
from dataclasses import dataclass

import httpx


@dataclass
class _Route:
    """A single matcher plus the responder to use when it matches."""

    predicate: Callable[[httpx.Request], bool]
    responder: Callable[[httpx.Request], httpx.Response]


class MockHTTP:
    """Records issued requests and routes them to registered responses.

    Routes are tried in registration order; the first whose predicate matches wins. A request
    matching no route gets a 404 so missing routes surface as loud test failures rather than silent
    passes.
    """

    def __init__(self) -> None:
        self._routes: list[_Route] = []
        self.requests: list[httpx.Request] = []

    def route(
        self,
        *,
        url_contains: str | None = None,
        responder: Callable[[httpx.Request], httpx.Response] | None = None,
        status_code: int = 200,
        json: object | None = None,
        content: bytes | None = None,
        text: str | None = None,
    ) -> "MockHTTP":
        """Register a response for requests whose URL contains ``url_contains``.

        Pass ``responder`` for full control (it receives the request and returns a response);
        otherwise the ``status_code``/``json``/``content``/ ``text`` are used to build a static
        response. Returns ``self`` so calls can be chained.
        """

        if responder is None:

            def static_responder(request: httpx.Request) -> httpx.Response:
                kwargs: dict = {}
                if json is not None:
                    kwargs["json"] = json
                if content is not None:
                    kwargs["content"] = content
                if text is not None:
                    kwargs["text"] = text
                return httpx.Response(status_code, **kwargs)

            responder = static_responder

        def predicate(request: httpx.Request) -> bool:
            return url_contains is None or url_contains in str(request.url)

        self._routes.append(_Route(predicate, responder))
        return self

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Transport entry point: record the request and route it."""

        self.requests.append(request)
        for route in self._routes:
            if route.predicate(request):
                return route.responder(request)
        return httpx.Response(404, text=f"no mock route for {request.url}")

    @property
    def urls(self) -> list[str]:
        """The URLs of every request issued so far, in order."""

        return [str(request.url) for request in self.requests]


def acs_api_payload(tables, geoid_values: dict[str, object]) -> list[list[str]]:
    """Build a Census-style list-of-lists ACS response for ``tables``.

    The header carries every estimate (``E``) and margin (``M``) variable column for all
    ``tables``, so a request for any single table/suffix can select its columns out of the one
    response. Each geography in ``geoid_values`` gets that scalar in every value column; the
    GEO_ID stub prefix is synthetic since only the substring after ``US`` is consumed.

    Args:
        tables: ACSTableInfo instances whose variables should appear.
        geoid_values: Mapping from GEOID to the value placed in every value column for that row.

    Returns:
        list[list[str]]: ``[header, *rows]`` with all cells as strings.
    """

    estimate_cols: list[str] = []
    margin_cols: list[str] = []
    for table in tables:
        estimate_cols += list(table.construct_long_names(suffix="E").keys())
        margin_cols += list(table.construct_long_names(suffix="M").keys())

    header = ["GEO_ID"] + estimate_cols + margin_cols
    width = len(estimate_cols) + len(margin_cols)
    rows = [[f"0500000US{geoid}"] + [str(value)] * width for geoid, value in geoid_values.items()]
    return [header] + rows


def geo_id_payload(geoids: list[str]) -> list[list[str]]:
    """Build a GEO_ID-only Census response (for ACS5 completeness checks)."""

    return [["GEO_ID"]] + [[f"0500000US{geoid}"] for geoid in geoids]
