import http.client
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from hashlib import sha256
from uuid import uuid4

from qwen_repair.schemas import TestResult

from .bundles import canonical_json
from .fake import code_hash
from .filequeue import MAX_MESSAGE_BYTES, QueueIdentity, QueueRequest, QueueResponse
from .http_auth import validate_token


PROTOCOL = "http-json-v1"
RETRYABLE_STATUS = {408, 429, 502, 503, 504}


def validate_endpoint(value: str) -> str:
    message = "HTTP executor requires an HTTPS origin, or HTTP on an explicit loopback host"
    if not isinstance(value, str) or not value or any(ord(char) <= 32 or ord(char) >= 127 for char in value):
        raise ValueError(message)
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
        host = parsed.hostname
        if (
            not host or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or parsed.path not in {"", "/"}
            or "?" in value or "#" in value or "\\" in value or "%" in parsed.netloc
            or parsed.scheme not in {"http", "https"}
            or (port is not None and not 1 <= port <= 65535)
            or (parsed.scheme == "http" and host not in {"127.0.0.1", "localhost", "::1"})
        ):
            raise ValueError(message)
    except (ValueError, TypeError):
        raise ValueError(message) from None
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


class HttpTransportError(ValueError):
    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HttpExecutor:
    synthetic_demo = False
    runner_version = PROTOCOL

    def __init__(
        self, endpoint, registry, *, run_id, policy_version, task_id, limits,
        token_env="QREPAIR_EXECUTOR_TOKEN", deadline_seconds=120.0, poll_interval=0.5,
    ):
        self.endpoint = validate_endpoint(endpoint)
        if not isinstance(token_env, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", token_env) is None:
            raise ValueError("HTTP executor requires a valid token environment variable name")
        try:
            token = validate_token(os.environ.get(token_env, ""))
        except ValueError:
            raise ValueError("HTTP executor token is missing or invalid in the configured environment variable") from None
        if not all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in (deadline_seconds, poll_interval)):
            raise ValueError("HTTP executor timing limits must be positive and finite")
        self._token = token
        self.registry = registry
        self.run_id = run_id
        self.policy_version = policy_version
        self.task_id = task_id
        self.limits = json.loads(canonical_json(limits))
        self.limits_hash = sha256(canonical_json(self.limits)).hexdigest()
        self.deadline_seconds = deadline_seconds
        self.poll_interval = poll_interval
        self._worker_id = None
        self.pending: dict[str, QueueRequest] = {}
        # Never forward a bearer token to a redirect target or an ambient HTTP proxy.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def _exchange(self, method, path, payload=None, *, timeout=5.0):
        headers = {"Authorization": "Bearer " + self._token, "Accept": "application/json"}
        if self._worker_id is not None:
            headers["X-Qrepair-Worker-Id"] = self._worker_id
        body = canonical_json(payload) if payload is not None else None
        if body is not None:
            if len(body) > MAX_MESSAGE_BYTES:
                raise HttpTransportError("HTTP executor request exceeds the size limit")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.endpoint + path, data=body, headers=headers, method=method)
        try:
            with self._opener.open(request, timeout=max(0.01, min(5.0, timeout))) as response:
                if response.status not in {200, 202}:
                    raise HttpTransportError("Unexpected HTTP executor status")
                if response.headers.get_content_type() != "application/json":
                    raise HttpTransportError("HTTP executor returned a non-JSON response")
                length = response.headers.get("Content-Length")
                if length is not None and (not length.isdecimal() or int(length) > MAX_MESSAGE_BYTES):
                    raise HttpTransportError("HTTP executor response exceeds the size limit")
                raw = response.read(MAX_MESSAGE_BYTES + 1)
                if len(raw) > MAX_MESSAGE_BYTES:
                    raise HttpTransportError("HTTP executor response exceeds the size limit")
                try:
                    decoded = json.loads(raw)
                except (ValueError, UnicodeError, RecursionError):
                    raise HttpTransportError("HTTP executor returned invalid JSON") from None
                if not isinstance(decoded, dict):
                    raise HttpTransportError("HTTP executor returned an invalid message")
                return response.status, decoded
        except urllib.error.HTTPError as error:
            status = error.code
            error.close()
            raise HttpTransportError(
                f"HTTP executor rejected the request (status {status})",
                retryable=status in RETRYABLE_STATUS, status=status,
            ) from None
        except (OSError, urllib.error.URLError, http.client.HTTPException):
            raise HttpTransportError("HTTP executor connection failed", retryable=True) from None

    def preflight(self):
        report = {
            "schema_version": "1.0", "backend": PROTOCOL, "synthetic_demo": False,
            "passed": False, "authenticated": False, "workers": [],
            "reason": "No compatible authenticated HTTP worker with successful Docker preflight.",
        }
        self.preflight_report = report
        try:
            status, payload = self._exchange("GET", "/v1/preflight")
            worker_id = payload.get("worker_id")
            if (
                status != 200 or payload.get("protocol") != PROTOCOL
                or payload.get("synthetic_demo") is not False
                or payload.get("passed") is not True
                or not isinstance(worker_id, str) or re.fullmatch(r"[0-9a-f]{32}", worker_id) is None
                or (self._worker_id is not None and self._worker_id != worker_id)
                or payload.get("limits_hash") != self.limits_hash
                or canonical_json(payload.get("limits")) != canonical_json(self.limits)
                or not isinstance(payload.get("bundle_hashes"), dict)
                or any(payload["bundle_hashes"].get(key) != value for key, value in self.registry.hashes().items())
            ):
                return report
            self._worker_id = worker_id
            report.update(
                passed=True, authenticated=True, workers=[worker_id], reason="ready",
                limits_hash=self.limits_hash, bundle_hashes=self.registry.hashes(),
            )
        except (HttpTransportError, TypeError, ValueError):
            pass
        self.preflight_report = report
        return report

    def _build_request(self, candidate, bundle_id):
        bundle = self.registry.get(bundle_id)
        return QueueRequest(
            request_id=uuid4().hex, run_id=self.run_id, policy_version=self.policy_version,
            task_id=self.task_id, candidate=candidate, candidate_hash=code_hash(candidate),
            test_bundle_id=bundle_id, test_bundle_hash=bundle.bundle_hash,
            limits=self.limits, limits_hash=self.limits_hash, deadline=time.time() + self.deadline_seconds,
        )

    def _check_ack(self, payload, request):
        if payload.get("worker_id") != self._worker_id or payload.get("request_id") != request.request_id:
            raise HttpTransportError("HTTP executor acknowledgement identity mismatch")

    def redeliver(self, request, *, timeout=5.0):
        status, payload = self._exchange(
            "POST", "/v1/requests", {"worker_id": self._worker_id, "request": request.model_dump(mode="json")},
            timeout=timeout,
        )
        if status != 202:
            raise HttpTransportError("HTTP executor did not acknowledge the request")
        self._check_ack(payload, request)

    def submit(self, candidate, test_bundle_id):
        if self._worker_id is None and not self.preflight()["passed"]:
            raise HttpTransportError("HTTP worker preflight failed")
        request = self._build_request(candidate, test_bundle_id)
        self.pending[request.request_id] = request
        self.redeliver(request)
        return request

    def receive(self, request, *, timeout=5.0):
        status, payload = self._exchange("GET", "/v1/requests/" + request.request_id, timeout=timeout)
        if payload.get("worker_id") != self._worker_id:
            raise HttpTransportError("HTTP executor worker identity mismatch")
        if time.time() >= request.deadline:
            raise HttpTransportError("HTTP executor response arrived after the deadline")
        if status == 202:
            self._check_ack(payload, request)
            return None
        try:
            response = QueueResponse.model_validate(payload.get("response"))
        except (ValueError, TypeError):
            raise HttpTransportError("HTTP executor returned an invalid result") from None
        identity_fields = set(QueueIdentity.model_fields)
        if response.model_dump(include=identity_fields) != request.model_dump(include=identity_fields):
            raise HttpTransportError("HTTP executor response identity mismatch")
        if response.result.synthetic_demo:
            raise HttpTransportError("A synthetic result cannot satisfy an HTTP request")
        return response.result

    def cancel(self, request_id):
        request = self.pending.get(request_id)
        if request is None:
            return
        status, payload = self._exchange("POST", f"/v1/requests/{request_id}/cancel", {
            "worker_id": self._worker_id,
            "identity": request.model_dump(include=set(QueueIdentity.model_fields)),
        })
        if status != 202:
            raise HttpTransportError("HTTP executor did not acknowledge cancellation")
        self._check_ack(payload, request)

    def cancel_all(self):
        for request_id in tuple(self.pending):
            try:
                self.cancel(request_id)
            except HttpTransportError:
                pass

    def run(self, candidate, test_bundle_id):
        if not self.preflight()["passed"]:
            return TestResult(status="infra_error", feedback="HTTP worker preflight failed.", synthetic_demo=False)
        request = None
        received = False
        try:
            request = self._build_request(candidate, test_bundle_id)
            self.pending[request.request_id] = request
            deadline = time.monotonic() + self.deadline_seconds
            acknowledged = False
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                try:
                    if not acknowledged:
                        self.redeliver(request, timeout=remaining)
                        acknowledged = True
                    result = self.receive(request, timeout=deadline - time.monotonic())
                    if result is not None:
                        received = True
                        return result
                except HttpTransportError as error:
                    if not error.retryable:
                        raise
                time.sleep(min(self.poll_interval, max(0, deadline - time.monotonic())))
            return TestResult(status="infra_error", feedback="HTTP executor deadline exceeded.", synthetic_demo=False)
        except (HttpTransportError, ValueError, TypeError):
            return TestResult(status="infra_error", feedback="HTTP executor transport or identity validation failed.", synthetic_demo=False)
        finally:
            if request is not None:
                if not received:
                    try:
                        self.cancel(request.request_id)
                    except HttpTransportError:
                        pass
                self.pending.pop(request.request_id, None)
