import hashlib
import http.client
import json
import queue
import re
import secrets
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from .bundles import BundleRegistry, canonical_json
from .filequeue import QueueIdentity, QueueRequest, QueueResponse, _identity, _infra


PROTOCOL = "http-json-v1"
MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_HEADER_BYTES = 16 * 1024
MAX_ACTIVE_REQUESTS = 16
MAX_CACHED_REQUESTS = 1024
RESULT_RETENTION_SECONDS = 60
MAX_DEADLINE_SECONDS = 3600
REQUEST_ID = re.compile(r"[0-9a-f]{32}")
TOKEN = re.compile(r"[A-Za-z0-9_-]{32,256}")


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: str
    request: QueueRequest


class Cancellation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: str
    identity: QueueIdentity


class HTTPProtocolError(Exception):
    def __init__(self, status: int, code: str):
        self.status = status
        self.code = code


@dataclass
class Job:
    request: QueueRequest | None
    identity: dict
    payload_hash: str
    monotonic_deadline: float
    expires_at: float
    state: str = "queued"
    response: dict | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    cancel_reason: str = "Request cancelled."


class _BoundedHeaders:
    def __init__(self, stream):
        self.stream = stream
        self.remaining = MAX_HEADER_BYTES

    def readline(self, limit=-1):
        size = self.remaining + 1 if limit < 0 else min(limit, self.remaining + 1)
        line = self.stream.readline(size)
        self.remaining -= len(line)
        if self.remaining < 0:
            raise http.client.LineTooLong("request headers")
        return line


class ExecutorHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = False
    request_queue_size = 16

    def __init__(self, executor, registry: BundleRegistry, token: str, port: int):
        if TOKEN.fullmatch(token) is None:
            raise ValueError("HTTP token must contain 32 to 256 URL-safe ASCII characters")
        if not 0 <= port <= 65535:
            raise ValueError("Invalid loopback port")
        if getattr(executor, "synthetic_demo", True) or not (getattr(executor, "preflight_report", None) or {}).get("passed"):
            raise ValueError("HTTP server requires successful real Docker preflight")
        if executor.registry.hashes() != registry.hashes():
            raise ValueError("HTTP and Docker registries must match")
        self.executor = executor
        self.registry = registry
        self.worker_id = uuid4().hex
        self.limits = executor.config.model_dump(mode="json")
        self.limits_hash = hashlib.sha256(canonical_json(self.limits)).hexdigest()
        self._authorization = ("Bearer " + token).encode("ascii")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._pending: queue.Queue[str] = queue.Queue(maxsize=MAX_ACTIVE_REQUESTS)
        self._stopping = threading.Event()
        self._serving = threading.Event()
        self._handlers = threading.BoundedSemaphore(32)
        self._active_request: str | None = None
        self._executor_thread: threading.Thread | None = None
        super().__init__(("127.0.0.1", port), ExecutorRequestHandler)
        self._executor_thread = threading.Thread(
            target=self._run_jobs, name="qrepair-http-executor", daemon=False
        )
        self._executor_thread.start()

    def ready(self) -> bool:
        report = getattr(self.executor, "preflight_report", None)
        return not self._stopping.is_set() and bool(report and report.get("passed"))

    def capability(self) -> dict:
        return {
            "protocol": PROTOCOL, "worker_id": self.worker_id,
            "synthetic_demo": False, "passed": self.ready(),
            "bundle_hashes": self.registry.hashes(),
            "limits": self.limits, "limits_hash": self.limits_hash,
        }

    def check_session(self, worker_id: str | None) -> None:
        if worker_id != self.worker_id:
            raise HTTPProtocolError(409, "worker_session_changed")

    def _prune(self) -> None:
        now = time.time()
        for request_id, job in tuple(self._jobs.items()):
            if job.response is not None and job.expires_at <= now:
                del self._jobs[request_id]

    def _ack(self, request_id: str, job: Job) -> dict:
        return {"worker_id": self.worker_id, "request_id": request_id, "state": job.state}

    def submit(self, submission: Submission) -> dict:
        self.check_session(submission.worker_id)
        request = submission.request
        payload_hash = hashlib.sha256(canonical_json(request.model_dump(mode="json"))).hexdigest()
        with self._lock:
            self._prune()
            existing = self._jobs.get(request.request_id)
            if existing is not None and existing.payload_hash != payload_hash:
                raise HTTPProtocolError(409, "request_id_collision")
            remaining = request.deadline - time.time()
            if remaining <= 0:
                raise HTTPProtocolError(410, "deadline_expired")
            if remaining > MAX_DEADLINE_SECONDS:
                raise HTTPProtocolError(422, "deadline_too_far")
            if existing is not None:
                return self._ack(request.request_id, existing)
            if not self.ready():
                raise HTTPProtocolError(503, "executor_not_ready")
            if len(request.candidate.encode("utf-8")) > self.limits["max_candidate_bytes"]:
                raise HTTPProtocolError(413, "candidate_too_large")
            if hashlib.sha256(request.candidate.encode("utf-8")).hexdigest() != request.candidate_hash:
                raise HTTPProtocolError(422, "candidate_hash_mismatch")
            if request.limits_hash != self.limits_hash or canonical_json(request.limits) != canonical_json(self.limits):
                raise HTTPProtocolError(409, "executor_limits_mismatch")
            try:
                bundle = self.registry.get(request.test_bundle_id)
            except ValueError as error:
                raise HTTPProtocolError(422, "unknown_test_bundle") from error
            if request.test_bundle_hash != bundle.bundle_hash:
                raise HTTPProtocolError(409, "test_bundle_hash_mismatch")
            active_count = sum(job.response is None for job in self._jobs.values())
            if active_count >= MAX_ACTIVE_REQUESTS or len(self._jobs) >= MAX_CACHED_REQUESTS:
                raise HTTPProtocolError(429, "executor_capacity_reached")
            job = Job(
                request=request, identity=_identity(request), payload_hash=payload_hash,
                monotonic_deadline=time.monotonic() + remaining,
                expires_at=request.deadline + RESULT_RETENTION_SECONDS,
            )
            self._jobs[request.request_id] = job
            try:
                self._pending.put_nowait(request.request_id)
            except queue.Full as error:
                del self._jobs[request.request_id]
                raise HTTPProtocolError(429, "executor_capacity_reached") from error
            return self._ack(request.request_id, job)

    def result(self, worker_id: str | None, request_id: str) -> tuple[int, dict]:
        self.check_session(worker_id)
        with self._lock:
            self._prune()
            job = self._jobs.get(request_id)
            if job is None:
                raise HTTPProtocolError(404, "unknown_request")
            if job.response is None and time.monotonic() >= job.monotonic_deadline:
                job.cancel_reason = "Request deadline exceeded."
                job.cancel_event.set()
                if job.state == "queued":
                    self._finish(job, _infra(job.cancel_reason))
            if job.response is None:
                return 202, self._ack(request_id, job)
            return 200, {"worker_id": self.worker_id, "response": job.response}

    def cancel(self, request_id: str, cancellation: Cancellation) -> dict:
        self.check_session(cancellation.worker_id)
        with self._lock:
            job = self._jobs.get(request_id)
            if job is None:
                raise HTTPProtocolError(404, "unknown_request")
            if cancellation.identity.model_dump() != job.identity:
                raise HTTPProtocolError(409, "request_identity_mismatch")
            if job.response is None:
                job.cancel_event.set()
                if job.state == "queued":
                    self._finish(job, _infra(job.cancel_reason))
                else:
                    job.state = "cancelling"
            return self._ack(request_id, job)

    def _finish(self, job: Job, result) -> None:
        if job.response is not None:
            return
        job.response = QueueResponse(**job.identity, result=result).model_dump(mode="json")
        job.request = None
        job.state = "completed"

    def _run_jobs(self) -> None:
        while not self._stopping.is_set():
            try:
                request_id = self._pending.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                with self._lock:
                    job = self._jobs.get(request_id)
                    if job is None or job.response is not None:
                        continue
                    if self._stopping.is_set() or job.cancel_event.is_set() or time.monotonic() >= job.monotonic_deadline:
                        self._finish(job, _infra("Request cancelled or expired before execution."))
                        continue
                    if not self.ready():
                        self._finish(job, _infra("Executor preflight is no longer valid."))
                        continue
                    request = job.request
                    job.state = "running"
                    self._active_request = request_id
                done = threading.Event()

                def watch_cancel():
                    while not done.wait(0.05):
                        if time.monotonic() >= job.monotonic_deadline:
                            job.cancel_reason = "Request deadline exceeded."
                            job.cancel_event.set()
                        if self._stopping.is_set() or job.cancel_event.is_set():
                            # Repeat until cleanup completes; run() may clear its cancellation flag at entry.
                            self.executor.cancel_current()

                watcher = threading.Thread(target=watch_cancel, name="qrepair-http-cancel", daemon=True)
                watcher.start()
                try:
                    result = self.executor.run(request.candidate, request.test_bundle_id)
                    if result.synthetic_demo:
                        result = _infra("A synthetic result cannot satisfy a real HTTP request.")
                except Exception:
                    result = _infra("Executor failed while processing the request.")
                finally:
                    done.set()
                    watcher.join(timeout=1)
                if watcher.is_alive():
                    self._cancel_jobs()
                    result = _infra("Cancellation supervisor did not stop.")
                with self._lock:
                    if self._stopping.is_set() or job.cancel_event.is_set() or time.monotonic() >= job.monotonic_deadline:
                        result = _infra(job.cancel_reason)
                    self._finish(job, result)
                    self._active_request = None
            finally:
                self._pending.task_done()

    def process_request(self, request, client_address):
        if not self._handlers.acquire(blocking=False):
            try:
                request.settimeout(1.0)
                body = b'{"error":"http_capacity_reached"}'
                request.sendall(
                    b"HTTP/1.0 503 Service Unavailable\r\nContent-Type: application/json\r\n"
                    b"Connection: close\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
                )
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._handlers.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._handlers.release()

    def handle_error(self, request, client_address):
        pass

    def serve_forever(self, poll_interval=0.1):
        if self._stopping.is_set():
            return
        self._serving.set()
        try:
            super().serve_forever(poll_interval=poll_interval)
        finally:
            self._serving.clear()

    def _cancel_jobs(self):
        self._stopping.set()
        with self._lock:
            for job in self._jobs.values():
                if job.response is None:
                    job.cancel_reason = "HTTP executor is shutting down."
                    job.cancel_event.set()
                    if job.state == "queued":
                        self._finish(job, _infra(job.cancel_reason))
        self.executor.cancel_current()

    def shutdown(self):
        self._cancel_jobs()
        if self._serving.is_set():
            super().shutdown()

    def server_close(self):
        self._cancel_jobs()
        super().server_close()
        if self._executor_thread is None:
            return
        self._executor_thread.join(timeout=2 * self.limits["startup_timeout_seconds"] + self.limits["timeout_seconds"] + 5)
        if self._executor_thread.is_alive():
            raise RuntimeError("Executor did not stop within its cleanup deadline")


class ExecutorRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "qrepair"
    sys_version = ""

    def setup(self):
        self.request.settimeout(2.0)
        super().setup()
        self._connection_deadline = threading.Timer(10.0, self._expire_connection)
        self._connection_deadline.daemon = True
        self._connection_deadline.start()

    def _expire_connection(self):
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def finish(self):
        self._connection_deadline.cancel()
        super().finish()

    def parse_request(self):
        if len(self.raw_requestline) > 4096:
            self.requestline = ""
            self.request_version = "HTTP/1.0"
            self.command = None
            self.send_error(414)
            return False
        stream = self.rfile
        self.rfile = _BoundedHeaders(stream)
        try:
            return super().parse_request()
        finally:
            self.rfile = stream

    def log_message(self, format, *args):
        pass

    def send_error(self, code, message=None, explain=None):
        self._json(code, {"error": "invalid_http_request"})

    def _json(self, status, payload):
        body = canonical_json(payload)
        if len(body) > MAX_BODY_BYTES:
            status, body = 500, b'{"error":"response_too_large"}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        if status in (202, 429, 503):
            self.send_header("Retry-After", "1")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(body)

    def _authenticated(self):
        headers = self.headers.get_all("Authorization", [])
        if len(headers) != 1:
            raise HTTPProtocolError(401, "unauthorized")
        try:
            presented = headers[0].encode("ascii")
        except UnicodeEncodeError as error:
            raise HTTPProtocolError(401, "unauthorized") from error
        if not secrets.compare_digest(presented, self.server._authorization):
            raise HTTPProtocolError(401, "unauthorized")
        if self.headers.get_all("Transfer-Encoding"):
            raise HTTPProtocolError(400, "transfer_encoding_not_supported")
        if self.headers.get("Content-Encoding", "identity") != "identity":
            raise HTTPProtocolError(415, "content_encoding_not_supported")
        if self.headers.get("Expect"):
            raise HTTPProtocolError(417, "expectation_not_supported")

    def _body(self) -> dict:
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not re.fullmatch(r"[0-9]+", lengths[0]):
            raise HTTPProtocolError(411, "one_content_length_required")
        length = int(lengths[0])
        if length > MAX_BODY_BYTES:
            raise HTTPProtocolError(413, "request_too_large")
        if self.headers.get_content_type() != "application/json":
            raise HTTPProtocolError(415, "json_content_type_required")
        body = bytearray()
        deadline = time.monotonic() + 5
        while len(body) < length:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HTTPProtocolError(408, "request_body_timeout")
            self.connection.settimeout(min(2, remaining))
            chunk = self.rfile.read1(min(65536, length - len(body)))
            if not chunk:
                raise HTTPProtocolError(400, "incomplete_request_body")
            body.extend(chunk)

        def reject_constant(value):
            raise ValueError("Nonfinite JSON number")

        def unique_object(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("Duplicate JSON key")
                result[key] = value
            return result

        try:
            payload = json.loads(body, parse_constant=reject_constant, object_pairs_hook=unique_object)
        except (ValueError, UnicodeError, RecursionError) as error:
            raise HTTPProtocolError(400, "invalid_json") from error
        if not isinstance(payload, dict):
            raise HTTPProtocolError(400, "json_object_required")
        return payload

    def _dispatch(self):
        self._authenticated()
        path = urlsplit(self.path)
        if path.query or path.fragment or path.scheme or path.netloc:
            raise HTTPProtocolError(404, "unknown_route")
        if self.command == "GET" and path.path == "/v1/preflight":
            return 200, self.server.capability()
        if self.command == "POST" and path.path == "/v1/requests":
            return 202, self.server.submit(Submission.model_validate(self._body()))
        match = re.fullmatch(r"/v1/requests/([0-9a-f]{32})(/cancel)?", path.path)
        if match:
            request_id, cancellation = match.groups()
            if self.command == "GET" and cancellation is None:
                sessions = self.headers.get_all("X-Qrepair-Worker-Id", [])
                if len(sessions) != 1:
                    raise HTTPProtocolError(409, "worker_session_required")
                return self.server.result(sessions[0], request_id)
            if self.command == "POST" and cancellation:
                return 202, self.server.cancel(request_id, Cancellation.model_validate(self._body()))
        raise HTTPProtocolError(404, "unknown_route")

    def _handle(self):
        try:
            status, payload = self._dispatch()
        except HTTPProtocolError as error:
            status, payload = error.status, {"error": error.code}
            if error.status != 401:
                payload["worker_id"] = self.server.worker_id
        except (ValueError, TypeError, RecursionError):
            status, payload = 422, {"error": "invalid_protocol_payload"}
        except (TimeoutError, socket.timeout):
            status, payload = 408, {"error": "request_timeout"}
        try:
            self._json(status, payload)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    do_GET = _handle
    do_POST = _handle
    do_PUT = _handle
    do_PATCH = _handle
    do_DELETE = _handle
    do_OPTIONS = _handle
    do_HEAD = _handle


def make_server(executor, registry: BundleRegistry, token: str, port: int = 0) -> ExecutorHTTPServer:
    return ExecutorHTTPServer(executor, registry, token, port)
