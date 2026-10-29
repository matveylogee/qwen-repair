import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_repair.executors.bundles import (
    BundleRegistry, TestBundle, TestCase, export_registry, load_registry,
    registry_from_fixtures,
)
from qwen_repair.executors.docker import ContainerOutput, DockerConfig, DockerExecutor
from qwen_repair.schemas import TestResult

IMAGE = "python@sha256:" + "a" * 64


def executor_for(expected=7):
    bundle = TestBundle(
        bundle_id="public", entry_point="solve", visibility="public",
        cases=[TestCase(args=[1], expected=expected)],
    )
    return DockerExecutor(BundleRegistry({"public": bundle}), DockerConfig(image=IMAGE))


def test_image_must_be_digest_pinned():
    with pytest.raises(ValueError, match="digest"):
        DockerConfig(image="python:latest")


def test_missing_docker_fails_closed(monkeypatch):
    executor = executor_for()
    monkeypatch.setattr("qwen_repair.executors.docker.shutil.which", lambda _: None)
    report = executor.preflight()
    assert report["passed"] is False
    assert report["checks"] == {"docker_available": False}
    result = executor.run("def solve(x): return x", "public")
    assert result.status == "infra_error"
    assert result.synthetic_demo is False


def test_container_limits_and_only_minimal_readonly_mount(tmp_path):
    command = executor_for()._create_command("unit-container", tmp_path)
    joined = " ".join(command)
    for flag in ("--read-only", "--cap-drop", "--security-opt", "--pids-limit", "--cpus", "--memory", "--memory-swap", "--tmpfs", "--log-driver"):
        assert flag in command
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--user") + 1] == "65532:65532"
    assert command[command.index("--pull") + 1] == "never"
    assert command.count("--mount") == 1
    assert f"source={tmp_path},target=/input,readonly" in joined
    assert "docker.sock" not in joined
    assert "--privileged" not in command
    assert "--gpus" not in command


def test_expected_values_never_enter_container_and_cleanup_runs(monkeypatch):
    executor = executor_for(expected="PRIVATE_EXPECTED_SENTINEL")
    observed = {}

    def run(command, **kwargs):
        if command[1] == "create":
            mount = command[command.index("--mount") + 1]
            directory = Path(mount.split("source=", 1)[1].split(",target=", 1)[0])
            observed["directory"] = directory
            assert {path.name for path in directory.iterdir()} == {"runner.py", "solution.py"}
            assert "PRIVATE_EXPECTED_SENTINEL" not in "".join(path.read_text() for path in directory.iterdir())
        elif command[1] == "rm":
            observed["removed"] = True
        return SimpleNamespace(returncode=0)

    def capture(command, payload):
        observed["payload"] = json.loads(payload)
        return ContainerOutput("timeout")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(executor, "_capture", capture)
    output = executor._execute_case("def solve(x): return x", "solve", TestCase(args=[1], expected="PRIVATE_EXPECTED_SENTINEL"))
    assert output.status == "timeout"
    assert observed["payload"] == {"entry_point": "solve", "args": [1], "kwargs": {}}
    assert observed["removed"]
    assert not observed["directory"].exists()


@pytest.mark.parametrize(
    ("stdout", "expected_status"),
    [(b'{"value":7}', "passed"), (b'{"value":6}', "failed"),
     (b'{"passed":true}', "runtime_error"), (b'{"value":7}\n{"passed":true}', "runtime_error"),
     (b'{"value":NaN}', "runtime_error")],
)
def test_host_judge_compares_values_never_trusts_stdout_verdict(monkeypatch, stdout, expected_status):
    executor = executor_for()
    executor._preflight_passed = True
    monkeypatch.setattr(executor, "_execute_case", lambda *args: ContainerOutput("ok", stdout))
    result = executor.run("def solve(x): return x", "public")
    assert result.status == expected_status
    assert result.synthetic_demo is False


def test_json_comparison_preserves_python_numeric_equality(monkeypatch):
    executor = executor_for(expected=1)
    executor._preflight_passed = True
    monkeypatch.setattr(executor, "_execute_case", lambda *args: ContainerOutput("ok", b'{"value":1.0}'))
    assert executor.run("def solve(x): return x", "public").status == "passed"


def test_cleanup_failure_revokes_preflight(monkeypatch):
    executor = executor_for()
    executor._preflight_passed = True
    executor.preflight_report = {"passed": True}
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(executor, "_capture", lambda *args: ContainerOutput("ok", b'{"value":7}'))
    monkeypatch.setattr(executor, "_container_state", lambda _: {"Status": "exited", "ExitCode": 0, "OOMKilled": False, "Error": ""})
    monkeypatch.setattr(executor, "_remove_container", lambda _: False)
    result = executor.run("def solve(x): return x", "public")
    assert result.status == "infra_error"
    assert not executor.preflight_report["passed"]
    assert not executor._preflight_passed


def test_preflight_must_pass_every_probe(monkeypatch):
    executor = executor_for()
    monkeypatch.setattr("qwen_repair.executors.docker.shutil.which", lambda _: "/mock/docker")
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: SimpleNamespace(returncode=0, stdout="linux" if command[1] == "info" else "sha256:abc"))

    def probe(candidate, bundle):
        name = bundle.bundle_id.split(".")[-1]
        if name == "network":
            return TestResult(status="failed", tests_failed=1, tests_total=1, synthetic_demo=False)
        status = {"timeout": "timeout", "output_limit": "resource_limit"}.get(name, "passed")
        return TestResult(status=status, tests_passed=int(status == "passed"), tests_total=1, synthetic_demo=False)

    monkeypatch.setattr(executor, "_run_bundle", probe)
    report = executor.preflight()
    assert not report["passed"]
    assert not report["checks"]["network"]
    assert {"host_files", "process_limit", "output_limit", "timeout", "privileges_readonly"} <= set(report["checks"])


def test_registry_round_trip_and_hash_changes(tmp_path):
    registry = registry_from_fixtures()
    destination = tmp_path / "registry.json"
    export_registry(registry, destination)
    assert load_registry(destination).hashes() == registry.hashes()
    assert len(registry.hashes()) == 8
    bundle = registry.get("sum_positive.public.v1")
    before = bundle.bundle_hash
    bundle.cases[0].args.append(999)
    assert bundle.bundle_hash != before
    assert registry.get(bundle.bundle_id).bundle_hash == before


def test_successful_preflight_is_cached_only_on_this_instance(monkeypatch):
    executor = executor_for()
    executor._preflight_passed = True
    executor.preflight_report = {"passed": True, "checks": {"normal": True}}
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("Cached preflight must not start Docker"))
    report = executor.preflight()
    report["checks"]["normal"] = False
    assert executor.preflight_report["checks"]["normal"]
    assert not executor_for()._preflight_passed


@pytest.mark.skipif(not os.environ.get("QREPAIR_TEST_DOCKER_IMAGE"), reason="Set QREPAIR_TEST_DOCKER_IMAGE to run real container preflight")
def test_real_docker_preflight_and_authored_reference():
    from qwen_repair.data.fixtures import load_fixture

    executor = DockerExecutor(registry_from_fixtures(), DockerConfig(image=os.environ["QREPAIR_TEST_DOCKER_IMAGE"]))
    assert executor.preflight()["passed"]
    fixture = load_fixture("sum_positive")
    assert executor.run(fixture.reference_code, fixture.task.public_tests_ref).status == "passed"
    assert executor.run(fixture.task.buggy_code, fixture.task.private_tests_ref).status == "failed"
