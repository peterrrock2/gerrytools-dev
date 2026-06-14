import httpx
import pytest

from tests.data._helpers import MockHTTP


@pytest.fixture
def mock_http(monkeypatch: pytest.MonkeyPatch) -> MockHTTP:
    """Patch ``httpx.Client`` so all clients use a recording mock transport.

    Every ``httpx.Client(...)`` built while this fixture is active is given an
    ``httpx.MockTransport`` bound to the returned ``MockHTTP``, so no request escapes to the
    network and all are recorded for assertions.
    """

    mock = MockHTTP()
    real_client = httpx.Client

    def client_factory(*args, **kwargs) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(mock.handle)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    return mock
