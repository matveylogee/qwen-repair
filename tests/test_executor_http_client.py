import json
import os
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from qwen_repair.executors.bundles import registry_from_fixtures
from qwen_repair.executors.docker import DockerConfig
from qwen_repair.executors.http_client import HttpExecutor, HttpTransportError, validate_endpoint
from qwen_repair.executors.http_server import make_server
from qwen_repair.schemas import TestResult


TOKEN = "local-test-token-" + "x" * 32


class ProtocolExecutor:
    """Only HTTP behavior is tested here; Docker isolation has a separate integration test."""

    synthetic_demo = False
    config = DockerConfig(image="python@sha256:" + "a" * 64)

    def __init__(self):
        self.preflight_report = {"passed": True}
        self.registry = registry_from_fixtures()
        self.calls = []
        self.cancelled = threading.Event()

    def run(self, candidate, bundle_id):
        self.calls.append((candidate, bundle_id))
        return TestResult(status="passed", tests_total=1, tests_passed=1, synthetic_demo=False)

    def cancel_current(self):
        self.cancelled.set()


@contextmanager
def serving(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def new_client(endpoint, **options):
    return HttpExecutor(
        endpoint, registry_from_fixtures(), run_id="http-unit-run", policy_version="unit-policy",
        task_id="demo-sum-positive", limits=ProtocolExecutor.config.model_dump(), poll_interval=0.01,
        **options,
    )


@pytest.fixture
def http_pair(monkeypatch):
    monkeypatch.setenv("QREPAIR_EXECUTOR_TOKEN", TOKEN)
    engine = ProtocolExecutor()
    with serving(make_server(engine, registry_from_fixtures(), TOKEN)) as endpoint:
        yield new_client(endpoint), engine


def wait_for_result(client, request):
    until = time.monotonic() + 3
    while time.monotonic() < until:
        result = client.receive(request)
        if result is not None:
            return result
        time.sleep(0.01)
    pytest.fail("HTTP test response did not arrive")


def test_live_http_roundtrip_duplicate_submission_and_revoked_preflight(http_pair):
    client, engine = http_pair
    assert client.preflight()["passed"]
    request = client.submit("def sum_positive(values): return 1", "sum_positive.public.v1")
    client.redeliver(request)
    assert wait_for_result(client, request).status == "passed"
    client.redeliver(request)
    assert wait_for_result(client, request).status == "passed"
    assert len(engine.calls) == 1
    engine.preflight_report["passed"] = False
    assert not client.preflight()["passed"]
    assert not client.preflight_report["passed"]
    assert client.run("unused", "sum_positive.public.v1").status == "infra_error"
    assert len(engine.calls) == 1


def test_lost_submission_ack_reuses_request_id_without_reexecution(http_pair, monkeypatch):
    client, engine = http_pair
    exchange = client._exchange
    submissions = []

    def lose_first_ack(method, path, payload=None, **kwargs):
        result = exchange(method, path, payload, **kwargs)
        if method == "POST" and path == "/v1/requests":
            submissions.append(payload["request"]["request_id"])
            if len(submissions) == 1:
                raise HttpTransportError("Simulated dropped acknowledgement", retryable=True)
        return result

    monkeypatch.setattr(client, "_exchange", lose_first_ack)
    assert client.run("def sum_positive(values): return 1", "sum_positive.public.v1").status == "passed"
    assert len(submissions) == 2 and submissions[0] == submissions[1]
    assert len(engine.calls) == 1
    assert not client.pending


def test_wrong_token_is_infrastructure_failure_without_candidate_execution(http_pair, monkeypatch):
    original, engine = http_pair
    monkeypatch.setenv("QREPAIR_EXECUTOR_TOKEN", "different-token-" + "z" * 32)
    client = new_client(original.endpoint)
    assert not client.preflight()["passed"]
    assert client.run("unused", "sum_positive.public.v1").status == "infra_error"
    assert not engine.calls


@pytest.mark.parametrize("mutation", ["session", "identity", "synthetic"])
def test_invalid_remote_results_never_become_rewards(http_pair, monkeypatch, mutation):
    client, _ = http_pair
    assert client.preflight()["passed"]
    request = client.submit("def sum_positive(values): return 1", "sum_positive.public.v1")
    wait_for_result(client, request)
    exchange = client._exchange

    def tamper(method, path, payload=None, **kwargs):
        status, result = exchange(method, path, payload, **kwargs)
        if method == "GET" and path.startswith("/v1/requests/") and status == 200:
            if mutation == "session":
                result["worker_id"] = "0" * 32
            elif mutation == "identity":
                result["response"]["policy_version"] = "previous-policy"
            else:
                result["response"]["result"]["synthetic_demo"] = True
        return status, result

    monkeypatch.setattr(client, "_exchange", tamper)
    with pytest.raises(HttpTransportError):
        client.receive(request)


def test_worker_restart_requires_a_new_client(http_pair, monkeypatch):
    client, _ = http_pair
    assert client.preflight()["passed"]
    exchange = client._exchange

    def restarted(method, path, payload=None, **kwargs):
        status, result = exchange(method, path, payload, **kwargs)
        if path == "/v1/preflight":
            result["worker_id"] = "b" * 32
        return status, result

    monkeypatch.setattr(client, "_exchange", restarted)
    assert not client.preflight()["passed"]


def test_deadline_cancels_remote_execution_and_remains_infra(monkeypatch):
    class WaitingExecutor(ProtocolExecutor):
        def run(self, candidate, bundle_id):
            self.calls.append((candidate, bundle_id))
            self.cancelled.wait(3)
            return TestResult(status="infra_error", synthetic_demo=False)

    monkeypatch.setenv("QREPAIR_EXECUTOR_TOKEN", TOKEN)
    engine = WaitingExecutor()
    with serving(make_server(engine, registry_from_fixtures(), TOKEN)) as endpoint:
        client = new_client(endpoint, deadline_seconds=0.15)
        assert client.run("unused", "sum_positive.public.v1").status == "infra_error"
        assert engine.cancelled.wait(1)
        assert len(engine.calls) == 1
        assert not client.pending


def test_redirect_does_not_forward_authorization(monkeypatch):
    reached = []

    class Target(BaseHTTPRequestHandler):
        def do_GET(self):
            reached.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    with serving(ThreadingHTTPServer(("127.0.0.1", 0), Target)) as target:
        class Redirect(Target):
            def do_GET(self):
                self.send_response(302)
                self.send_header("Location", target + "/stolen")
                self.send_header("Content-Length", "0")
                self.end_headers()

        monkeypatch.setenv("QREPAIR_EXECUTOR_TOKEN", TOKEN)
        with serving(ThreadingHTTPServer(("127.0.0.1", 0), Redirect)) as endpoint:
            client = new_client(endpoint)
            assert not client.preflight()["passed"]
    assert reached == []


@pytest.mark.parametrize("endpoint", [
    "http://example.com", "http://127.0.0.1.evil.test", "http://0.0.0.0:8765",
    "https://user:secret@example.com", "https://example.com?token=secret",
    "https://example.com/#secret", "https://example.com/path", "https://example.com:0",
    "https://example.com\n", "file:///etc/passwd",
])
def test_unsafe_endpoints_are_rejected_without_echoing_input(endpoint):
    with pytest.raises(ValueError) as caught:
        validate_endpoint(endpoint)
    assert endpoint not in str(caught.value)
    assert "secret" not in str(caught.value)


def test_https_and_loopback_origins_are_supported():
    assert validate_endpoint("https://temporary.example.com/") == "https://temporary.example.com"
    assert validate_endpoint("http://127.0.0.1:8765/") == "http://127.0.0.1:8765"
    assert validate_endpoint("http://[::1]:8765") == "http://[::1]:8765"


def test_missing_credentials_and_arbitrary_server_error_are_redacted(monkeypatch):
    monkeypatch.delenv("QREPAIR_EXECUTOR_TOKEN", raising=False)
    with pytest.raises(ValueError, match="missing or invalid"):
        new_client("http://127.0.0.1:8765")
    secret = "sentinel-private-server-response"

    class ErrorResponse(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"error": secret, "token": TOKEN}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    monkeypatch.setenv("QREPAIR_EXECUTOR_TOKEN", TOKEN)
    with serving(ThreadingHTTPServer(("127.0.0.1", 0), ErrorResponse)) as endpoint:
        client = new_client(endpoint)
        report = json.dumps(client.preflight())
        assert secret not in report and TOKEN not in report
        with pytest.raises(HttpTransportError) as caught:
            client._exchange("GET", "/v1/preflight")
        assert secret not in str(caught.value) and TOKEN not in str(caught.value)


@pytest.mark.docker
def test_http_real_docker_roundtrip(monkeypatch):
    image = os.environ.get("QREPAIR_TEST_DOCKER_IMAGE")
    if not image:
        pytest.skip("Set QREPAIR_TEST_DOCKER_IMAGE to verify real HTTP/Docker execution")
    from qwen_repair.data.fixtures import load_fixture
    from qwen_repair.executors.docker import DockerExecutor

    registry = registry_from_fixtures()
    engine = DockerExecutor(registry, DockerConfig(image=image))
    assert engine.preflight()["passed"]
    monkeypatch.setenv("QREPAIR_EXECUTOR_TOKEN", TOKEN)
    fixture = load_fixture("sum_positive")
    with serving(make_server(engine, registry, TOKEN)) as endpoint:
        client = HttpExecutor(
            endpoint, registry, run_id="real-http-integration", policy_version="reference-probe",
            task_id=fixture.task.task_id, limits=engine.config.model_dump(), poll_interval=0.05,
        )
        result = client.run(fixture.corrected_code, fixture.task.public_tests_ref)
        assert result.status == "passed" and result.tests_passed == 2
        assert result.synthetic_demo is False
        result = client.run(fixture.task.buggy_code, fixture.task.public_tests_ref)
        assert result.status == "failed" and result.synthetic_demo is False
