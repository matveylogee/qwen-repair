import json
import time

import pytest

from qwen_repair.executors.bundles import registry_from_fixtures
from qwen_repair.executors.docker import DockerConfig
from qwen_repair.executors.filequeue import (
    FileQueueExecutor, QueueProtocolError, QueueWorker, publish_message, read_message,
)
from qwen_repair.schemas import TestResult


class TransportTestExecutor:
    """Protocol-only stand-in. These tests do not establish Docker isolation."""

    synthetic_demo = False
    preflight_report = {"passed": True}
    config = DockerConfig(image="python@sha256:" + "a" * 64)

    def __init__(self):
        self.calls = []

    def run(self, candidate, bundle_id):
        self.calls.append((candidate, bundle_id))
        return TestResult(status="passed", tests_passed=1, tests_total=1, synthetic_demo=False)

    def cancel_current(self):
        pass


def queue_pair(tmp_path):
    registry = registry_from_fixtures()
    executor = TransportTestExecutor()
    worker = QueueWorker(tmp_path, registry, executor)
    client = FileQueueExecutor(
        tmp_path, registry, run_id="unit-run", policy_version="unit-policy",
        task_id="demo-sum-positive", limits=executor.config.model_dump(),
    )
    return client, worker, executor


def test_message_needs_payload_checksum_and_ready(tmp_path):
    directory = tmp_path / "message"
    directory.mkdir()
    (directory / "payload.json").write_text('{"value": 1}')
    assert read_message(directory) is None
    (directory / "READY").write_text("wrong")
    assert read_message(directory) is None
    (directory / "checksum.sha256").write_text("wrong")
    with pytest.raises(QueueProtocolError, match="checksum"):
        read_message(directory)


def test_publish_is_idempotent_and_rejects_changes(tmp_path):
    publish_message(tmp_path / "message", {"value": 1})
    publish_message(tmp_path / "message", {"value": 1})
    with pytest.raises(QueueProtocolError, match="immutable"):
        publish_message(tmp_path / "message", {"value": 2})


def test_request_round_trip_redelivery_and_worker_idempotence(tmp_path):
    client, worker, executor = queue_pair(tmp_path)
    assert client.preflight()["passed"]
    request = client.submit("def sum_positive(values): return 1", "sum_positive.public.v1")
    client.redeliver(request)
    assert worker.process_once() == 1
    assert worker.process_once() == 0
    assert len(executor.calls) == 1
    result = client.receive(request)
    assert result.status == "passed"
    assert not result.synthetic_demo
    response = read_message(tmp_path / "responses" / request.request_id)
    for name in ("run_id", "policy_version", "task_id", "candidate_hash", "test_bundle_hash", "limits_hash"):
        assert response[name] == getattr(request, name)


@pytest.mark.parametrize("field", ["run_id", "policy_version", "task_id", "candidate_hash", "test_bundle_hash"])
def test_stale_or_wrong_response_identity_rejected(tmp_path, field):
    client, worker, _ = queue_pair(tmp_path)
    request = client.submit("def sum_positive(values): return 1", "sum_positive.public.v1")
    worker.process_once()
    changed = request.model_copy(update={field: "b" * 64 if "hash" in field else "different"})
    with pytest.raises(QueueProtocolError, match="identity"):
        client.receive(changed)


def test_expiry_and_cancellation_do_not_execute_candidate(tmp_path):
    client, worker, executor = queue_pair(tmp_path)
    request = client.submit("def sum_positive(values): return 1", "sum_positive.public.v1")
    client.cancel(request.request_id)
    assert worker.process_once() == 1
    assert client.receive(request).status == "infra_error"
    assert not executor.calls


def test_changed_resource_limits_rejected_by_worker(tmp_path):
    client, worker, executor = queue_pair(tmp_path)
    client.limits["memory_mb"] = 4096
    assert not client.preflight()["passed"]
    request = client.submit("def sum_positive(values): return 1", "sum_positive.public.v1")
    worker.process_once()
    assert client.receive(request).status == "infra_error"
    assert not executor.calls


def test_multiple_workers_and_expired_heartbeats_fail_closed(tmp_path, monkeypatch):
    client, worker, executor = queue_pair(tmp_path)
    QueueWorker(tmp_path, worker.registry, executor)
    assert not client.preflight()["passed"]
    now = time.time()
    monkeypatch.setattr("qwen_repair.executors.filequeue.time.time", lambda: now + 121)
    assert not client.preflight()["passed"]
    assert client.run("candidate", "sum_positive.public.v1").status == "infra_error"


def test_corrupt_response_checksum_rejected(tmp_path):
    client, worker, _ = queue_pair(tmp_path)
    request = client.submit("def sum_positive(values): return 1", "sum_positive.public.v1")
    worker.process_once()
    (tmp_path / "responses" / request.request_id / "payload.json").write_text(json.dumps({"forged": True}))
    with pytest.raises(QueueProtocolError, match="checksum"):
        client.receive(request)


def test_graceful_worker_shutdown_revokes_heartbeat(tmp_path):
    client, worker, _ = queue_pair(tmp_path)
    assert client.preflight()["passed"]
    worker.close()
    assert not client.preflight()["passed"]
    with pytest.raises(RuntimeError, match="closed"):
        worker.process_once()


@pytest.mark.parametrize("response_state", ["partial", "corrupt"])
def test_worker_skips_existing_response_without_rerun_or_overwrite(tmp_path, response_state):
    client, worker, executor = queue_pair(tmp_path)
    request = client.submit("def sum_positive(values): return 1", "sum_positive.public.v1")
    response_directory = tmp_path / "responses" / request.request_id
    response_directory.mkdir(parents=True)
    original = b'{"previous_worker_payload":true}'
    (response_directory / "payload.json").write_bytes(original)
    if response_state == "corrupt":
        (response_directory / "checksum.sha256").write_bytes(b"0" * 64)
        (response_directory / "READY").write_bytes(b"0" * 64)
    for _ in range(3):
        assert worker.process_once() == 0
    assert executor.calls == []
    assert (response_directory / "payload.json").read_bytes() == original
    assert request.request_id in worker.transport_issues
    if response_state == "corrupt":
        with pytest.raises(QueueProtocolError, match="checksum"):
            client.receive(request)
    else:
        assert client.receive(request) is None
    fresh = client.submit("def sum_positive(values): return 2", "sum_positive.public.v1")
    assert worker.process_once() == 1
    assert len(executor.calls) == 1
    assert client.receive(fresh).status == "passed"


def test_published_response_disappearing_or_corrupting_never_reexecutes(tmp_path):
    client, worker, executor = queue_pair(tmp_path)
    request = client.submit("def sum_positive(values): return 1", "sum_positive.public.v1")
    assert worker.process_once() == 1
    response_directory = tmp_path / "responses" / request.request_id
    (response_directory / "READY").unlink()
    assert worker.process_once() == 0
    (response_directory / "checksum.sha256").write_bytes(b"0" * 64)
    assert worker.process_once() == 0
    assert len(executor.calls) == 1


def test_interrupted_publication_retries_identical_cached_result(tmp_path, monkeypatch):
    from qwen_repair.executors import filequeue

    client, worker, executor = queue_pair(tmp_path)
    request = client.submit("def sum_positive(values): return 1", "sum_positive.public.v1")
    original_publish = filequeue.publish_message
    payloads = []

    def interrupted(directory, payload):
        if directory.parent.name == "responses":
            payloads.append(json.dumps(payload, sort_keys=True))
            if len(payloads) == 1:
                directory.mkdir(parents=True)
                from qwen_repair.executors.bundles import canonical_json

                (directory / "payload.json").write_bytes(canonical_json(payload))
                raise OSError("Simulated interrupted cloud write")
        return original_publish(directory, payload)

    monkeypatch.setattr(filequeue, "publish_message", interrupted)
    assert worker.process_once() == 0
    assert len(executor.calls) == 1
    assert worker.process_once() == 1
    assert len(executor.calls) == 1
    assert payloads[0] == payloads[1]
    assert client.receive(request).status == "passed"


def test_restart_waits_for_partial_response_instead_of_executing(tmp_path):
    client, worker, executor = queue_pair(tmp_path)
    request = client.submit("def sum_positive(values): return 1", "sum_positive.public.v1")
    assert worker.process_once() == 1
    response_directory = tmp_path / "responses" / request.request_id
    ready = (response_directory / "READY").read_bytes()
    (response_directory / "READY").unlink()
    worker.close()
    restarted = QueueWorker(tmp_path, worker.registry, executor)
    assert restarted.process_once() == 0
    assert len(executor.calls) == 1
    (response_directory / "READY").write_bytes(ready)
    assert restarted.process_once() == 0
    assert client.receive(request).status == "passed"


def test_expired_cached_request_is_forgotten_without_reexecution(tmp_path, monkeypatch):
    client, worker, executor = queue_pair(tmp_path)
    request = client.submit("def sum_positive(values): return 1", "sum_positive.public.v1")
    assert worker.process_once() == 1
    assert request.request_id in worker._completed_responses
    monkeypatch.setattr("qwen_repair.executors.filequeue.time.time", lambda: request.deadline + 1)
    assert worker.process_once() == 0
    assert not worker._request_deadlines
    assert not worker._completed_responses
    assert not worker._published_responses
    assert len(executor.calls) == 1
    with pytest.raises(QueueProtocolError, match="deadline"):
        client.receive(request)


def test_active_request_cache_has_bounded_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr("qwen_repair.executors.filequeue.MAX_TRACKED_REQUESTS", 2)
    client, worker, executor = queue_pair(tmp_path)
    for _ in range(3):
        client.submit("def sum_positive(values): return 1", "sum_positive.public.v1")
    assert worker.process_once() == 2
    assert len(executor.calls) == 2
    assert len(worker._request_deadlines) == 2
    assert len(worker._completed_responses) == 2
    assert worker.transport_issues["capacity"] == "active_request_cache_full"


def test_cancelled_active_execution_and_heartbeat_failure_do_not_kill_worker(tmp_path, monkeypatch):
    client, worker, executor = queue_pair(tmp_path)
    request = client.submit("def sum_positive(values): return 1", "sum_positive.public.v1")
    original_run = executor.run

    def cancel_during_execution(candidate, bundle_id):
        client.cancel(request.request_id)
        return original_run(candidate, bundle_id)

    monkeypatch.setattr(executor, "run", cancel_during_execution)
    monkeypatch.setattr(worker, "heartbeat", lambda: (_ for _ in ()).throw(OSError("Transient sync failure")))
    assert worker.process_once() == 1
    assert client.receive(request).status == "infra_error"
    assert worker.transport_issues["heartbeat"] == "heartbeat_publication_failed"


def test_preflight_skips_stale_and_legacy_heartbeats_without_reading(tmp_path, monkeypatch):
    from qwen_repair.executors import filequeue

    client, worker, _ = queue_pair(tmp_path)
    heartbeat_root = tmp_path / "heartbeats"
    (heartbeat_root / ("f" * 32)).mkdir()
    (heartbeat_root / f"{int((time.time() - 121) * 1000)}-{'e' * 32}").mkdir()
    original_read = filequeue.read_message
    reads = []

    def track_read(directory):
        reads.append(directory)
        return original_read(directory)

    monkeypatch.setattr(filequeue, "read_message", track_read)
    report = client.preflight()
    assert report["passed"]
    assert report["legacy_heartbeats_ignored"] == 1
    assert len([path for path in reads if path.parent.name == "heartbeats"]) == 1


def test_preflight_checks_freshness_after_slow_io(tmp_path, monkeypatch):
    from qwen_repair.executors import filequeue

    client, worker, _ = queue_pair(tmp_path)
    original_read = filequeue.read_message
    clock = [time.time()]

    def delayed_read(directory):
        payload = original_read(directory)
        if directory.parent.name == "heartbeats":
            clock[0] += 121
        return payload

    monkeypatch.setattr(filequeue.time, "time", lambda: clock[0])
    monkeypatch.setattr(filequeue, "read_message", delayed_read)
    assert not client.preflight()["passed"]


def test_newest_partial_heartbeat_can_fall_back_to_previous_fresh_one(tmp_path, monkeypatch):
    client, worker, _ = queue_pair(tmp_path)
    later = time.time() + 10
    monkeypatch.setattr("qwen_repair.executors.filequeue.time.time", lambda: later)
    directory = tmp_path / "heartbeats" / f"{int(later * 1000)}-{worker.worker_id}"
    directory.mkdir()
    assert client.preflight()["passed"]


def test_finished_requests_are_skipped_before_cloud_payload_reads(tmp_path, monkeypatch):
    from qwen_repair.executors import filequeue

    client, worker, _ = queue_pair(tmp_path)
    request = client.submit("def sum_positive(values): return 1", "sum_positive.public.v1")
    worker.process_once()
    original_read = filequeue.read_message

    def reject_request_read(directory):
        assert directory.parent.name != "requests", "Finished request must not hydrate again"
        return original_read(directory)

    monkeypatch.setattr(filequeue, "read_message", reject_request_read)
    assert worker.process_once() == 0
    monkeypatch.setattr(filequeue.time, "time", lambda: request.deadline + 1)
    assert worker.process_once() == 0
