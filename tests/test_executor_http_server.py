import http.client
import json
import socket
import threading
import time
from contextlib import contextmanager
from hashlib import sha256
from uuid import uuid4

import pytest

from qwen_repair.executors.bundles import canonical_json, registry_from_fixtures
from qwen_repair.executors.docker import DockerConfig
from qwen_repair.executors.filequeue import QueueRequest, _identity
from qwen_repair.executors.http_server import make_server
from qwen_repair.schemas import TestResult


TOKEN = "protocol-test-token-" + "x" * 32


class ProtocolExecutor:
    """HTTP-only test backend. No candidate code is evaluated by these tests."""

    synthetic_demo = False

    def __init__(self, block=False):
        self.registry = registry_from_fixtures()
        self.config = DockerConfig(image="python@sha256:" + "a" * 64)
        self.preflight_report = {"passed": True}
        self.calls = []
        self.started = threading.Event()
        self.cancelled = threading.Event()
        self.release = threading.Event()
        self.block = block

    def run(self, candidate, bundle_id):
        self.cancelled.clear()
        self.calls.append((candidate, bundle_id))
        self.started.set()
        if self.block:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not self.release.wait(0.01):
                if self.cancelled.is_set():
                    return TestResult(status="infra_error", synthetic_demo=False)
        return TestResult(status="passed", tests_total=1, tests_passed=1, synthetic_demo=False)

    def cancel_current(self):
        self.cancelled.set()


@contextmanager
def serving(executor=None):
    executor = executor or ProtocolExecutor()
    server = make_server(executor, executor.registry, TOKEN)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield server, executor
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()
        assert not thread.is_alive()
        assert not server._executor_thread.is_alive()


def request(server, method, path, payload=None, *, token=TOKEN, extra_headers=None):
    connection = http.client.HTTPConnection(*server.server_address, timeout=3)
    headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    body = canonical_json(payload) if payload is not None else None
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read()), response.headers
    finally:
        connection.close()


def submission(server, *, deadline_seconds=5):
    source = "def sum_positive(values): return 1"
    bundle = server.registry.get("sum_positive.public.v1")
    job = QueueRequest(
        request_id=uuid4().hex, run_id="http-test", policy_version="step-0",
        task_id="demo-sum-positive", candidate=source,
        candidate_hash=sha256(source.encode()).hexdigest(),
        test_bundle_id=bundle.bundle_id, test_bundle_hash=bundle.bundle_hash,
        limits=server.limits, limits_hash=server.limits_hash,
        deadline=time.time() + deadline_seconds,
    )
    return {"worker_id": server.worker_id, "request": job.model_dump(mode="json")}


def completed(server, request_id):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        status, payload, _ = request(
            server, "GET", f"/v1/requests/{request_id}",
            extra_headers={"X-Qrepair-Worker-Id": server.worker_id},
        )
        if status == 200:
            return payload["response"]
        assert status == 202
        time.sleep(0.01)
    pytest.fail("HTTP result did not complete")


def test_server_is_loopback_only_and_requires_real_preflight():
    backend = ProtocolExecutor()
    backend.preflight_report = None
    with pytest.raises(ValueError, match="preflight"):
        make_server(backend, backend.registry, TOKEN)
    backend.preflight_report = {"passed": True}
    with pytest.raises(ValueError, match="token"):
        make_server(backend, backend.registry, "short")
    with serving(backend) as (server, _):
        assert server.server_address[0] == "127.0.0.1"


@pytest.mark.parametrize("method,path", [
    ("GET", "/v1/preflight"), ("POST", "/v1/requests"),
    ("GET", "/v1/requests/" + "a" * 32),
    ("POST", "/v1/requests/" + "a" * 32 + "/cancel"),
    ("OPTIONS", "/v1/preflight"),
])
def test_all_routes_require_bearer_auth(method, path):
    with serving() as (server, executor):
        status, payload, _ = request(server, method, path, {}, token="incorrect")
        assert status == 401
        assert payload == {"error": "unauthorized"}
        assert not executor.calls


def test_preflight_and_idempotent_execution_round_trip():
    with serving() as (server, executor):
        status, capabilities, headers = request(server, "GET", "/v1/preflight")
        assert status == 200
        assert capabilities["protocol"] == "http-json-v1"
        assert capabilities["passed"] and capabilities["limits_hash"] == server.limits_hash
        assert "Access-Control-Allow-Origin" not in headers
        payload = submission(server)
        request_id = payload["request"]["request_id"]
        for _ in range(2):
            status, ack, _ = request(server, "POST", "/v1/requests", payload)
            assert status == 202 and ack["request_id"] == request_id
        response = completed(server, request_id)
        assert response["result"]["status"] == "passed"
        assert response["result"]["synthetic_demo"] is False
        assert response["candidate_hash"] == payload["request"]["candidate_hash"]
        assert len(executor.calls) == 1
        status, _, _ = request(server, "POST", "/v1/requests", payload)
        assert status == 202 and len(executor.calls) == 1
        assert server._jobs[request_id].request is None


def test_request_collision_and_restart_session_mismatch_fail_closed():
    with serving() as (server, executor):
        payload = submission(server)
        assert request(server, "POST", "/v1/requests", payload)[0] == 202
        changed = json.loads(json.dumps(payload))
        changed["request"]["policy_version"] = "another-step"
        assert request(server, "POST", "/v1/requests", changed)[0] == 409
        changed = {**payload, "worker_id": uuid4().hex}
        assert request(server, "POST", "/v1/requests", changed)[0] == 409
        status, _, _ = request(server, "GET", f"/v1/requests/{payload['request']['request_id']}",
                               extra_headers={"X-Qrepair-Worker-Id": "previous-session"})
        assert status == 409
        assert len(executor.calls) <= 1


def test_cancel_active_and_queued_jobs_preserves_serial_execution():
    backend = ProtocolExecutor(block=True)
    with serving(backend) as (server, executor):
        first = submission(server)
        assert request(server, "POST", "/v1/requests", first)[0] == 202
        assert executor.started.wait(1)
        second = submission(server)
        assert request(server, "POST", "/v1/requests", second)[0] == 202
        for payload in (second, first):
            identity = _identity(QueueRequest.model_validate(payload["request"]))
            status, _, _ = request(server, "POST", f"/v1/requests/{identity['request_id']}/cancel",
                                   {"worker_id": server.worker_id, "identity": identity})
            assert status == 202
        assert completed(server, first["request"]["request_id"])["result"]["status"] == "infra_error"
        assert completed(server, second["request"]["request_id"])["result"]["status"] == "infra_error"
        assert len(executor.calls) == 1


def test_deadline_cancels_active_job_and_expired_submission_is_rejected():
    with serving(ProtocolExecutor(block=True)) as (server, executor):
        expired = submission(server, deadline_seconds=-1)
        assert request(server, "POST", "/v1/requests", expired)[0] == 410
        payload = submission(server, deadline_seconds=0.15)
        assert request(server, "POST", "/v1/requests", payload)[0] == 202
        result = completed(server, payload["request"]["request_id"])
        assert result["result"]["status"] == "infra_error"
        assert executor.cancelled.is_set()


@pytest.mark.parametrize("field,value,status", [
    ("candidate_hash", "0" * 64, 422),
    ("test_bundle_hash", "0" * 64, 409),
    ("limits_hash", "0" * 64, 409),
    ("test_bundle_id", "not-registered", 422),
])
def test_untrusted_request_cannot_select_code_hash_tests_or_limits(field, value, status):
    with serving() as (server, executor):
        payload = submission(server)
        payload["request"][field] = value
        assert request(server, "POST", "/v1/requests", payload)[0] == status
        assert not executor.calls


def test_pending_queue_is_bounded(monkeypatch):
    monkeypatch.setattr("qwen_repair.executors.http_server.MAX_ACTIVE_REQUESTS", 2)
    with serving(ProtocolExecutor(block=True)) as (server, executor):
        assert request(server, "POST", "/v1/requests", submission(server))[0] == 202
        assert executor.started.wait(1)
        assert request(server, "POST", "/v1/requests", submission(server))[0] == 202
        assert request(server, "POST", "/v1/requests", submission(server))[0] == 429
        assert len(executor.calls) == 1


def test_header_and_body_limits_are_checked_before_parsing_candidate():
    with serving() as (server, executor):
        status, _, _ = request(server, "POST", "/v1/requests", {},
                               extra_headers={"Content-Length": str(3 * 1024 * 1024)})
        assert status == 413
        status, _, _ = request(server, "GET", "/v1/preflight", extra_headers={"X-Fill": "x" * 20000})
        assert status == 431
        status, _, _ = request(server, "POST", "/v1/requests", {}, extra_headers={"Transfer-Encoding": "chunked"})
        assert status == 400
        assert not executor.calls


def test_duplicate_json_keys_and_unknown_routes_are_rejected():
    with serving() as (server, executor):
        connection = http.client.HTTPConnection(*server.server_address, timeout=3)
        connection.request("POST", "/v1/requests", body=b'{"worker_id":"a","worker_id":"b"}',
                           headers={"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 400
        assert json.loads(response.read())["error"] == "invalid_json"
        connection.close()
        assert request(server, "GET", "/../../etc/passwd")[0] == 404
        assert not executor.calls


def test_shutdown_cancels_active_job_and_joins_executor_thread():
    backend = ProtocolExecutor(block=True)
    with serving(backend) as (server, executor):
        assert request(server, "POST", "/v1/requests", submission(server))[0] == 202
        assert executor.started.wait(1)
        server.shutdown()
        assert executor.cancelled.is_set()
    assert not server._executor_thread.is_alive()


def test_preflight_failure_after_start_stops_new_jobs():
    with serving() as (server, executor):
        executor.preflight_report["passed"] = False
        assert request(server, "GET", "/v1/preflight")[1]["passed"] is False
        assert request(server, "POST", "/v1/requests", submission(server))[0] == 503
        assert not executor.calls


def test_shutdown_before_serving_cannot_reopen_service():
    backend = ProtocolExecutor()
    server = make_server(backend, backend.registry, TOKEN)
    try:
        server.shutdown()
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert not server.ready()
    finally:
        server.server_close()
