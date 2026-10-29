import json
import re
import threading
import time
from collections import deque
from hashlib import sha256
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from qwen_repair.schemas import TestResult

from .bundles import BundleRegistry, canonical_json
from .fake import code_hash

MAX_MESSAGE_BYTES = 2 * 1024 * 1024
MAX_TRACKED_REQUESTS = 1024
HEARTBEAT_LIFETIME_SECONDS = 120
HEARTBEAT_NAME = re.compile(r"^(?P<timestamp_ms>[0-9]{13,16})-(?P<worker_id>[0-9a-f]{32})$")


class QueueProtocolError(ValueError):
    pass


class QueueIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    run_id: str = Field(min_length=1, max_length=128)
    policy_version: str = Field(min_length=1, max_length=256)
    task_id: str = Field(min_length=1, max_length=256)
    candidate_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_bundle_id: str = Field(min_length=1, max_length=256)
    test_bundle_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    limits_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    deadline: float = Field(gt=0, allow_inf_nan=False)


class QueueRequest(QueueIdentity):
    candidate: str = Field(max_length=1048576)
    limits: dict[str, JsonValue]


class QueueResponse(QueueIdentity):
    result: TestResult


def _write_once(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as output:
            output.write(payload)
            output.flush()
            import os

            os.fsync(output.fileno())
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != payload:
            raise QueueProtocolError("An immutable queue file changed")


def publish_message(directory: Path, payload: dict) -> None:
    body = canonical_json(payload)
    if len(body) > MAX_MESSAGE_BYTES:
        raise QueueProtocolError("Queue message exceeds its byte limit")
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise QueueProtocolError("Queue message directory cannot be a symlink")
    digest = sha256(body).hexdigest().encode("ascii")
    _write_once(directory / "payload.json", body)
    _write_once(directory / "checksum.sha256", digest)
    _write_once(directory / "READY", digest)


def read_message(directory: Path) -> dict | None:
    paths = [directory / name for name in ("payload.json", "checksum.sha256", "READY")]
    if directory.is_symlink() or any(path.is_symlink() for path in paths):
        raise QueueProtocolError("Queue message files cannot be symlinks")
    if not all(path.is_file() for path in paths):
        return None
    if paths[0].stat().st_size > MAX_MESSAGE_BYTES or any(path.stat().st_size > 64 for path in paths[1:]):
        raise QueueProtocolError("Queue message exceeds its byte limit")
    body = paths[0].read_bytes()
    digest = sha256(body).hexdigest().encode("ascii")
    if paths[1].read_bytes() != digest or paths[2].read_bytes() != digest:
        raise QueueProtocolError("Queue checksum or ready marker does not match")
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise QueueProtocolError("Expected a JSON object")
    return payload


def _identity(request: QueueRequest) -> dict:
    return request.model_dump(include=set(QueueIdentity.model_fields))


def _infra(message: str) -> TestResult:
    return TestResult(status="infra_error", feedback=message, synthetic_demo=False)


class FileQueueExecutor:
    synthetic_demo = False
    runner_version = "filequeue-v1"

    def __init__(
        self, root: str | Path, registry: BundleRegistry, *, run_id: str,
        policy_version: str, task_id: str, limits: dict,
        deadline_seconds: float = 120.0, poll_interval: float = 0.25,
    ):
        if deadline_seconds <= 0 or poll_interval <= 0:
            raise ValueError("Queue timing limits must be positive")
        self.root = Path(root)
        self.registry = registry
        self.run_id = run_id
        self.policy_version = policy_version
        self.task_id = task_id
        self.limits = json.loads(canonical_json(limits))
        self.deadline_seconds = deadline_seconds
        self.poll_interval = poll_interval
        self.pending: set[str] = set()

    def preflight(self) -> dict:
        matches = []
        limits_hash = sha256(canonical_json(self.limits)).hexdigest()
        scan_time = time.time()
        newest: dict[str, list[tuple[int, Path]]] = {}
        legacy_ignored = 0
        for directory in (self.root / "heartbeats").glob("*"):
            name = HEARTBEAT_NAME.fullmatch(directory.name)
            if name is None:
                legacy_ignored += 1
                continue
            timestamp_ms = int(name["timestamp_ms"])
            if not scan_time - HEARTBEAT_LIFETIME_SECONDS < timestamp_ms / 1000 <= scan_time + 5:
                continue
            worker_id = name["worker_id"]
            newest.setdefault(worker_id, []).append((timestamp_ms, directory))
            newest[worker_id] = sorted(newest[worker_id], reverse=True)[:2]
        for worker_id, candidates in newest.items():
            try:
                if read_message(self.root / "worker_stops" / worker_id) is not None:
                    continue
                for timestamp_ms, directory in candidates:
                    payload = read_message(directory)
                    if payload is None:
                        continue
                    now = time.time()
                    created = payload.get("created_at", float("inf"))
                    valid_until = payload.get("valid_until", 0)
                    if (
                        payload.get("protocol") == self.runner_version
                        and payload.get("worker_id") == worker_id
                        and payload.get("preflight_passed") is True
                        and payload.get("synthetic_demo") is False
                        and payload.get("limits_hash") == limits_hash
                        and abs(created - timestamp_ms / 1000) < 0.002
                        and created <= now + 5
                        and now < valid_until <= created + HEARTBEAT_LIFETIME_SECONDS
                        and all(payload.get("bundle_hashes", {}).get(key) == value for key, value in self.registry.hashes().items())
                    ):
                        matches.append((worker_id, valid_until))
                    break
            except (OSError, ValueError, KeyError, TypeError):
                continue
        checked_at = time.time()
        workers = sorted({worker_id for worker_id, valid_until in matches if valid_until > checked_at})
        return {
            "schema_version": "1.0", "backend": self.runner_version,
            "synthetic_demo": False, "passed": len(workers) == 1,
            "workers": workers, "checked_at": checked_at,
            "legacy_heartbeats_ignored": legacy_ignored,
            "reason": "ready" if len(workers) == 1 else "Exactly one fresh compatible worker is required.",
        }

    def submit(self, candidate: str, test_bundle_id: str) -> QueueRequest:
        bundle = self.registry.get(test_bundle_id)
        request = QueueRequest(
            request_id=uuid4().hex, run_id=self.run_id, policy_version=self.policy_version,
            task_id=self.task_id, candidate=candidate, candidate_hash=code_hash(candidate),
            test_bundle_id=test_bundle_id, test_bundle_hash=bundle.bundle_hash,
            limits=self.limits, limits_hash=sha256(canonical_json(self.limits)).hexdigest(),
            deadline=time.time() + self.deadline_seconds,
        )
        self.redeliver(request)
        self.pending.add(request.request_id)
        return request

    def redeliver(self, request: QueueRequest) -> None:
        publish_message(self.root / "requests" / request.request_id, request.model_dump(mode="json"))

    def cancel(self, request_id: str) -> None:
        if re.fullmatch(r"[0-9a-f]{32}", request_id) is None:
            raise ValueError("Invalid request ID")
        publish_message(self.root / "cancellations" / request_id, {"request_id": request_id, "cancelled": True})

    def cancel_all(self) -> None:
        for request_id in tuple(self.pending):
            self.cancel(request_id)

    def receive(self, request: QueueRequest) -> TestResult | None:
        payload = read_message(self.root / "responses" / request.request_id)
        if payload is None:
            return None
        response = QueueResponse.model_validate(payload)
        if response.model_dump(include=set(QueueIdentity.model_fields)) != _identity(request):
            raise QueueProtocolError("Response identity does not match the request")
        if time.time() > request.deadline:
            raise QueueProtocolError("Response arrived after the request deadline")
        if response.result.synthetic_demo:
            raise QueueProtocolError("A synthetic result cannot satisfy a real queue request")
        return response.result

    def run(self, candidate: str, test_bundle_id: str) -> TestResult:
        if not self.preflight()["passed"]:
            return _infra("No single fresh compatible worker with successful Docker preflight.")
        request = None
        try:
            request = self.submit(candidate, test_bundle_id)
            deadline = time.monotonic() + self.deadline_seconds
            retry_at = time.monotonic() + 5
            while time.monotonic() < deadline:
                result = self.receive(request)
                if result is not None:
                    return result
                if time.monotonic() >= retry_at:
                    self.redeliver(request)
                    retry_at = time.monotonic() + 5
                time.sleep(min(self.poll_interval, max(0, deadline - time.monotonic())))
            self.cancel(request.request_id)
            return _infra("Queue deadline exceeded; request cancelled.")
        except (OSError, ValueError, TypeError):
            if request is not None:
                try:
                    self.cancel(request.request_id)
                except (OSError, ValueError):
                    pass
            return _infra("Queue transport or identity validation failed.")
        finally:
            if request is not None:
                self.pending.discard(request.request_id)


class QueueWorker:
    def __init__(self, root: str | Path, registry: BundleRegistry, executor):
        report = getattr(executor, "preflight_report", None)
        if getattr(executor, "synthetic_demo", True) or not report or not report.get("passed"):
            raise ValueError("Worker requires a real executor with successful preflight")
        self.root = Path(root)
        self.registry = registry
        self.executor = executor
        self.worker_id = uuid4().hex
        self.limits = executor.config.model_dump(mode="json")
        self.limits_hash = sha256(canonical_json(self.limits)).hexdigest()
        self._last_heartbeat = 0.0
        self._closed = False
        self._completed_responses: dict[str, dict] = {}
        self._published_responses: set[str] = set()
        self._observed_responses: set[str] = set()
        self._request_deadlines: dict[str, float] = {}
        self._retired_requests: set[str] = set()
        self._retired_order: deque[str] = deque()
        self.transport_issues: dict[str, str] = {}
        self.heartbeat()

    def heartbeat(self) -> None:
        now = time.time()
        if self._closed or now - self._last_heartbeat < 10:
            return
        directory = self.root / "heartbeats" / f"{int(now * 1000)}-{self.worker_id}"
        publish_message(directory, {
            "schema_version": "1.0", "protocol": "filequeue-v1",
            "worker_id": self.worker_id, "created_at": now, "valid_until": now + HEARTBEAT_LIFETIME_SECONDS,
            "preflight_passed": bool(self.executor.preflight_report.get("passed")),
            "synthetic_demo": False, "limits_hash": self.limits_hash,
            "bundle_hashes": self.registry.hashes(),
        })
        self._last_heartbeat = now

    def close(self) -> None:
        self._closed = True
        self.executor.cancel_current()
        publish_message(self.root / "worker_stops" / self.worker_id, {
            "worker_id": self.worker_id, "stopped": True,
        })

    def _record_transport_issue(self, request_id: str, reason: str) -> None:
        if request_id not in self.transport_issues and len(self.transport_issues) >= MAX_TRACKED_REQUESTS:
            self.transport_issues.pop(next(iter(self.transport_issues)))
        self.transport_issues[request_id] = reason

    def _refresh_heartbeat(self) -> None:
        try:
            self.heartbeat()
        except (OSError, ValueError):
            self._record_transport_issue("heartbeat", "heartbeat_publication_failed")

    def _prune_request_cache(self) -> None:
        now = time.time()
        for request_id, deadline in tuple(self._request_deadlines.items()):
            if deadline <= now:
                self._request_deadlines.pop(request_id)
                self._completed_responses.pop(request_id, None)
                self._published_responses.discard(request_id)
                self._observed_responses.discard(request_id)
                self.transport_issues.pop(request_id, None)
                self._retire_request(request_id)

    def _retire_request(self, request_id: str) -> None:
        if request_id in self._retired_requests:
            return
        if len(self._retired_order) >= MAX_TRACKED_REQUESTS:
            self._retired_requests.discard(self._retired_order.popleft())
        self._retired_order.append(request_id)
        self._retired_requests.add(request_id)

    def _cancelled(self, request: QueueRequest) -> bool:
        if time.time() >= request.deadline:
            return True
        marker = read_message(self.root / "cancellations" / request.request_id)
        return marker is not None

    def _execute(self, request: QueueRequest) -> TestResult:
        if code_hash(request.candidate) != request.candidate_hash:
            return _infra("Candidate checksum does not match.")
        if request.limits_hash != self.limits_hash or canonical_json(request.limits) != canonical_json(self.limits):
            return _infra("Requested resource limits differ from the trusted worker profile.")
        try:
            bundle = self.registry.get(request.test_bundle_id)
        except ValueError:
            return _infra("Unknown local test bundle.")
        if bundle.bundle_hash != request.test_bundle_hash:
            return _infra("Test bundle checksum does not match.")
        if self._cancelled(request):
            return _infra("Request expired or was cancelled.")
        done = threading.Event()

        def watch_cancel() -> None:
            while not done.wait(0.1):
                try:
                    cancelled = self._cancelled(request)
                except (OSError, ValueError):
                    cancelled = True
                if cancelled:
                    self.executor.cancel_current()
                    return

        watcher = threading.Thread(target=watch_cancel, daemon=True)
        watcher.start()
        try:
            result = self.executor.run(request.candidate, request.test_bundle_id)
            if self._cancelled(request):
                return _infra("Request expired or was cancelled during execution.")
            if result.synthetic_demo:
                return _infra("Worker returned a synthetic outcome.")
            return result
        finally:
            done.set()
            watcher.join(timeout=1)

    def process_once(self) -> int:
        if self._closed:
            raise RuntimeError("Queue worker is closed")
        self._prune_request_cache()
        self._refresh_heartbeat()
        completed = 0
        for directory in sorted((self.root / "requests").glob("*")):
            if re.fullmatch(r"[0-9a-f]{32}", directory.name) is None:
                continue
            if directory.name in self._published_responses or directory.name in self._observed_responses or directory.name in self._retired_requests:
                continue
            try:
                payload = read_message(directory)
                if payload is None:
                    continue
                request = QueueRequest.model_validate(payload)
                if request.request_id != directory.name:
                    continue
            except (OSError, ValueError, TypeError):
                self._record_transport_issue(directory.name, "request_unreadable_or_invalid")
                continue

            request_id = request.request_id
            if request.deadline <= time.time():
                self._retire_request(request_id)
                continue
            if request_id not in self._request_deadlines:
                if len(self._request_deadlines) >= MAX_TRACKED_REQUESTS:
                    self._record_transport_issue("capacity", "active_request_cache_full")
                    continue
                self._request_deadlines[request_id] = request.deadline
            response_directory = self.root / "responses" / request_id
            if request_id in self._published_responses:
                continue
            response = self._completed_responses.get(request_id)
            if response is None:
                try:
                    response_exists = response_directory.exists() or response_directory.is_symlink()
                except OSError:
                    self._record_transport_issue(request_id, "response_directory_unreadable")
                    continue
                if response_exists or request_id in self._observed_responses:
                    self._observed_responses.add(request_id)
                    # A prior worker may have finished execution before its files synced.
                    # Without its exact result bytes, rerunning would break idempotence.
                    try:
                        existing = read_message(response_directory)
                        if existing is None:
                            reason = "existing_response_incomplete"
                        else:
                            parsed = QueueResponse.model_validate(existing)
                            if parsed.model_dump(include=set(QueueIdentity.model_fields)) != _identity(request) or parsed.result.synthetic_demo:
                                raise QueueProtocolError("Existing response identity is invalid")
                            reason = "existing_response_complete"
                    except (OSError, ValueError, TypeError):
                        reason = "existing_response_unreadable_or_invalid"
                    self._record_transport_issue(request_id, reason)
                    continue
                try:
                    result = self._execute(request)
                except (OSError, ValueError, TypeError):
                    result = _infra("Worker rejected an invalid request or transport state.")
                response = QueueResponse(**_identity(request), result=result).model_dump(mode="json")
                self._completed_responses[request_id] = response
            try:
                publish_message(response_directory, response)
            except (OSError, ValueError):
                self._record_transport_issue(request_id, "response_publication_pending")
                continue
            self._published_responses.add(request_id)
            self.transport_issues.pop(request_id, None)
            completed += 1
            self._refresh_heartbeat()
        return completed
