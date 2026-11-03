import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from qwen_repair.data.fixtures import load_fixture
from qwen_repair.executors.bundles import registry_from_fixtures
from qwen_repair.executors.docker import DockerConfig
from qwen_repair.m1 import executor_factory
from qwen_repair.m1_config import ExecutorSettings, InferenceConfig
from qwen_repair.training.common import executor_provenance, record_executors


IMAGE = "python@sha256:" + "a" * 64
PRIVATE_VALUE = "never_serialize_this_private_value_0123456789"


@pytest.mark.parametrize("extra", [{"private_note": PRIVATE_VALUE}, {"credentials_path": f"/private/{PRIVATE_VALUE}"}])
def test_executor_config_rejects_unknown_fields_without_echoing_values(extra):
    with pytest.raises(ValidationError) as error:
        ExecutorSettings(backend="docker", image=IMAGE, **extra)
    assert PRIVATE_VALUE not in str(error.value)


def test_executor_artifact_config_keeps_only_the_selected_transport_fields():
    local = ExecutorSettings(backend="docker", image=IMAGE, queue_dir="unused-local-value")
    assert "queue_dir" not in local.artifact_config()
    remote = ExecutorSettings(backend="file_queue", image=IMAGE, queue_dir="exchange/session-1")
    assert remote.artifact_config()["queue_dir"] == "exchange/session-1"
    with pytest.raises(ValidationError, match="requires queue_dir"):
        ExecutorSettings(backend="file_queue", image=IMAGE)


def test_queue_factory_binds_task_policy_limits_and_fails_closed(monkeypatch, tmp_path):
    import qwen_repair.m1 as m1

    instances = []

    class StubClient:
        def __init__(self, root, registry, **kwargs):
            self.root, self.registry, self.kwargs = root, registry, kwargs
            instances.append(self)

        def preflight(self):
            return {"passed": len(instances) == 1, "reason": PRIVATE_VALUE}

    monkeypatch.setattr(m1, "FileQueueExecutor", StubClient)
    settings = ExecutorSettings(backend="file_queue", image=IMAGE, queue_dir=str(tmp_path / "exchange"))
    registry = registry_from_fixtures()
    factory = executor_factory(settings, registry, run_id="run-1", policy_version="step-3")
    task = load_fixture("sum_positive").task
    backend = factory(task)
    assert backend.root == settings.queue_dir
    assert backend.kwargs["run_id"] == "run-1"
    assert backend.kwargs["policy_version"] == "step-3"
    assert backend.kwargs["task_id"] == task.task_id
    assert backend.kwargs["limits"] == DockerConfig(image=IMAGE).model_dump()
    assert backend.preflight_report["passed"] is True
    with pytest.raises(RuntimeError, match="No compatible queue worker") as error:
        factory(task)
    assert PRIVATE_VALUE not in str(error.value)


def backend_for_provenance(kind="file_queue", **updates):
    profile = {**DockerConfig(image=IMAGE).model_dump(), "private_note": PRIVATE_VALUE,
               "credentials_path": "/private/" + PRIVATE_VALUE}
    backend = SimpleNamespace(
        runner_version="filequeue-v1" if kind == "file_queue" else "docker-json-v1",
        limits=profile, registry=registry_from_fixtures(), run_id="transport-1", private_note=PRIVATE_VALUE,
        _repair_training_preflight={"passed": True, "workers": ["old-worker"],
                                    "private_note": PRIVATE_VALUE, "config": profile},
    )
    if kind == "docker":
        backend.config = SimpleNamespace(model_dump=lambda: profile)
    for key, value in updates.items():
        setattr(backend, key, value)
    return backend


@pytest.mark.parametrize("kind", ["docker", "file_queue"])
def test_training_provenance_strips_private_fields_and_resume_allows_worker_rotation(tmp_path, kind):
    fixture = load_fixture("sum_positive")
    backend = backend_for_provenance(kind)
    provenance = executor_provenance(backend)
    assert PRIVATE_VALUE not in json.dumps(provenance)
    assert provenance["profile"] == DockerConfig(image=IMAGE).model_dump()
    assert provenance["test_bundle_hashes"] == backend.registry.hashes()
    previous = {"task_id": fixture.task.task_id, **provenance}
    (tmp_path / "manifest.json").write_text(json.dumps({"resumed_executor_profiles": [previous]}))
    (tmp_path / "environment.json").write_text("{}")
    changed = backend_for_provenance(kind, run_id="transport-2",
                                     _repair_training_preflight={"passed": True, "workers": ["new-worker"]})
    record_executors(tmp_path, [fixture], [changed])
    payload = json.loads((tmp_path / "manifest.json").read_text())
    assert payload["executors"][0]["transport_run_id"] == "transport-2"
    assert payload["executors"][0]["preflight"]["workers"] == ["new-worker"]
    assert PRIVATE_VALUE not in json.dumps(payload)


@pytest.mark.parametrize("kind", ["docker", "file_queue"])
@pytest.mark.parametrize("change", ["image", "limits", "registry", "runner"])
def test_training_resume_rejects_changed_execution_contract(tmp_path, kind, change):
    fixture, backend = load_fixture("sum_positive"), backend_for_provenance(kind)
    previous = {"task_id": fixture.task.task_id, **executor_provenance(backend)}
    (tmp_path / "manifest.json").write_text(json.dumps({"resumed_executor_profiles": [previous]}))
    (tmp_path / "environment.json").write_text("{}")
    if change == "image":
        backend.limits["image"] = "python@sha256:" + "b" * 64
    elif change == "limits":
        backend.limits["memory_mb"] = 1024
    elif change == "registry":
        backend.registry = SimpleNamespace(hashes=lambda: {"different": "0" * 64})
    else:
        backend.runner_version = "changed-runner-version"
    with pytest.raises(ValueError, match="profile or test bundles changed"):
        record_executors(tmp_path, [fixture], [backend])


def test_inference_manifest_records_actual_queue_runner_without_unrelated_private_values(tmp_path, monkeypatch):
    import qwen_repair.m1 as m1
    from qwen_repair.schemas import Generation, TestResult

    fixture = load_fixture("sum_positive")

    class StubPolicy:
        policy_version = "test-policy"
        synthetic_demo = False
        token_count_method = "tokenizer"

        def count_messages(self, messages):
            return len(messages)

        def generate(self, messages, config):
            response = f"```python\n{fixture.corrected_code}\n```"
            return Generation(text=response, input_tokens=len(messages), output_tokens=1,
                              policy_version=self.policy_version, prompt_token_ids=[1] * len(messages),
                              completion_token_ids=[2], token_count_method="tokenizer", synthetic_demo=False)

    backend = SimpleNamespace(
        runner_version="filequeue-v1", synthetic_demo=False, policy_version="preload",
        preflight_report={"passed": True, "backend": "filequeue-v1"}, private_note=PRIVATE_VALUE,
        run=lambda *args: TestResult(status="passed", tests_total=1, tests_passed=1, synthetic_demo=False),
    )
    monkeypatch.setattr(m1, "executor_factory", lambda *args, **kwargs: lambda task: backend)
    monkeypatch.setattr(m1.TransformersPolicy, "from_pretrained", lambda **kwargs: StubPolicy())
    monkeypatch.setattr(m1, "inspect_environment", lambda config: {})
    monkeypatch.setenv("UNRELATED_PRIVATE_VALUE", PRIVATE_VALUE)
    settings = ExecutorSettings(backend="file_queue", image=IMAGE, queue_dir=str(tmp_path / "exchange"))
    output = tmp_path / "inference"
    m1.run_inference(InferenceConfig(device="cpu", fixture_names=["sum_positive"]), settings, output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["runner_version"] == "filequeue-v1"
    assert manifest["executor_config"]["queue_dir"] == settings.queue_dir
    assert manifest["status"] == "completed"
    assert backend.policy_version == StubPolicy.policy_version
    for path in output.iterdir():
        assert PRIVATE_VALUE not in path.read_text()
