from types import SimpleNamespace

import pytest

from qwen_repair.executors.batch import run_checks
from qwen_repair.executors.bundles import registry_from_fixtures
from qwen_repair.executors.docker import DockerConfig
from qwen_repair.executors.filequeue import FileQueueExecutor, read_message
from qwen_repair.schemas import TestResult


def queue_client(tmp_path, monkeypatch):
    backend = FileQueueExecutor(
        tmp_path, registry_from_fixtures(), run_id="batch-unit", policy_version="unit",
        task_id="sum_positive", limits=DockerConfig(image="python@sha256:" + "a" * 64).model_dump(),
    )
    monkeypatch.setattr(backend, "preflight", lambda: {"passed": True})
    return backend


def outcome(status):
    return TestResult(status=status, tests_passed=int(status == "passed"),
                      tests_failed=int(status == "failed"), tests_total=int(status in {"passed", "failed"}),
                      synthetic_demo=False)


def test_batch_publishes_every_request_before_waiting_and_preserves_input_order(tmp_path, monkeypatch):
    first = queue_client(tmp_path, monkeypatch)
    second = queue_client(tmp_path, monkeypatch)
    statuses = {"first": "passed", "second": "failed", "third": "syntax_error"}
    preflights = []
    for backend in (first, second):
        def preflight(backend=backend):
            preflights.append(backend)
            assert not (tmp_path / "requests").exists()
            return {"passed": True}

        monkeypatch.setattr(backend, "preflight", preflight)

        def receive(request):
            directories = list((tmp_path / "requests").iterdir())
            assert len(directories) == 3, "A wait started before all requests were published"
            assert all(read_message(path) is not None for path in directories)
            return outcome(statuses[request.candidate])

        monkeypatch.setattr(backend, "receive", receive)
    results = run_checks([(first, "first", "sum_positive.public.v1"),
                          (second, "second", "sum_positive.private.v1"),
                          (first, "third", "sum_positive.public.v1")])
    assert [result.status for result in results] == list(statuses.values())
    assert preflights == [first, second]
    assert not first.pending and not second.pending
    assert not (tmp_path / "cancellations").exists()


@pytest.mark.parametrize("failure", ["infra", "receive_error", "publication_error", "keyboard_wait", "keyboard_publish"])
def test_batch_failure_cancels_every_outstanding_request(tmp_path, monkeypatch, failure):
    backend = queue_client(tmp_path, monkeypatch)
    old_request = backend.submit("preexisting", "sum_positive.public.v1")
    published = []
    submit = backend.submit

    def tracked_submit(candidate, bundle):
        request = submit(candidate, bundle)
        published.append(request)
        if len(published) == 2 and failure == "keyboard_publish":
            raise KeyboardInterrupt
        if len(published) == 2 and failure == "publication_error":
            raise OSError("Transport write interrupted")
        return request

    def receive(request):
        if failure == "keyboard_wait":
            raise KeyboardInterrupt
        if failure == "receive_error":
            raise OSError("Transport read failed")
        return outcome("infra_error")

    monkeypatch.setattr(backend, "submit", tracked_submit)
    monkeypatch.setattr(backend, "receive", receive)
    checks = [(backend, str(index), "sum_positive.public.v1") for index in range(3)]
    if failure.startswith("keyboard"):
        with pytest.raises(KeyboardInterrupt):
            run_checks(checks)
    else:
        assert [result.status for result in run_checks(checks)] == ["infra_error"] * 3
    assert backend.pending == {old_request.request_id}
    assert not (tmp_path / "cancellations" / old_request.request_id).exists()
    for request in published:
        if failure == "infra" and request is published[0]:
            continue
        assert read_message(tmp_path / "cancellations" / request.request_id)["cancelled"] is True


def test_failed_preflight_prevents_the_entire_batch_from_publishing(tmp_path, monkeypatch):
    ready = queue_client(tmp_path, monkeypatch)
    unavailable = queue_client(tmp_path, monkeypatch)
    monkeypatch.setattr(unavailable, "preflight", lambda: {"passed": False})
    results = run_checks([(ready, "first", "sum_positive.public.v1"),
                          (unavailable, "second", "sum_positive.public.v1")])
    assert [result.status for result in results] == ["infra_error", "infra_error"]
    assert not (tmp_path / "requests").exists()


def test_direct_backends_remain_sequential_and_infrastructure_failure_stops_the_batch():
    calls = []

    def run(candidate, bundle):
        calls.append(candidate)
        return outcome("infra_error" if candidate == "stop" else "passed")

    backend = SimpleNamespace(run=run)
    results = run_checks([(backend, candidate, "public") for candidate in ("first", "stop", "unstarted")])
    assert calls == ["first", "stop"]
    assert [result.status for result in results] == ["passed", "infra_error", "infra_error"]


def test_batch_cap_rejects_work_before_any_publication(tmp_path, monkeypatch):
    backend = queue_client(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="32"):
        run_checks([(backend, "candidate", "sum_positive.public.v1")] * 33)
    assert not (tmp_path / "requests").exists()
    assert run_checks([]) == []
