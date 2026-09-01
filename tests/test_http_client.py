"""HTTP hardening guards for the MLB Stats API client.

`response_code` used to be a bare `requests.get(url)` with no timeout, no retry
and no status check: a hung MLB endpoint blocked the whole pipeline forever, and
a 500 or a 404 surfaced as a JSON decode error somewhere far downstream instead
of as an HTTP failure at the call site.

The retry test drives a real loopback HTTP server rather than a monkeypatched
`get`, because the retry lives in the urllib3 adapter under `Session.send` --
patching `session.get` would bypass the very thing under test and pass no matter
what. No network access is required.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests

from Simulator import get_game_information as ggi


class _ScriptedHandler(BaseHTTPRequestHandler):
    """Replays `server.script` (a list of status codes), one entry per request."""

    def do_GET(self):  # noqa: N802  (BaseHTTPRequestHandler's naming)
        self.server.request_count += 1
        status = (self.server.script.pop(0) if self.server.script
                  else self.server.default_status)
        body = json.dumps({"ok": True, "hits": self.server.request_count}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # keep pytest output clean


@pytest.fixture
def scripted_server():
    """Factory: give it a list of status codes, get back (base_url, server)."""
    servers = []

    def _start(script, default_status=200):
        server = HTTPServer(("127.0.0.1", 0), _ScriptedHandler)
        server.script = list(script)
        server.default_status = default_status
        server.request_count = 0
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        host, port = server.server_address
        return f"http://{host}:{port}/", server

    yield _start

    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_a_503_is_retried_and_the_200_body_is_returned(scripted_server):
    """The first attempt fails with a retryable status; the client must not give up."""
    base_url, server = scripted_server([503, 200])

    payload = ggi.response_code(base_url, "v1", "schedule")

    assert payload == {"ok": True, "hits": 2}
    assert server.request_count == 2, "the 503 was not retried"


def test_a_404_raises_an_http_error(scripted_server):
    """A non-retryable failure must raise at the call site, not decode as JSON."""
    base_url, server = scripted_server([404])

    with pytest.raises(requests.HTTPError):
        ggi.response_code(base_url, "v1", "game/1/feed/live")

    assert server.request_count == 1, "a 404 must not be retried"


def test_the_request_passes_a_timeout(monkeypatch):
    """No timeout means a hung endpoint blocks the pipeline forever."""
    captured = {}

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    def _fake_get(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _FakeResponse()

    monkeypatch.setattr(ggi._session, "get", _fake_get)

    assert ggi.response_code("https://example.test/", "v1", "teams") == {"ok": True}
    assert captured["url"] == "https://example.test/v1/teams"
    assert captured["kwargs"].get("timeout") is not None, "no timeout was passed"
