import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from qwen_repair.data.fixtures import load_fixture
from qwen_repair.executors.bundles import registry_from_fixtures
from qwen_repair.executors.docker import DockerConfig
from qwen_repair.executors.http_auth import load_or_create_token, validate_token
from qwen_repair.m1 import executor_factory
from qwen_repair.m1_config import ExecutorSettings, InferenceConfig
from qwen_repair.training.common import executor_provenance, record_executors


IMAGE = "python@sha256:" + "a" * 64
SECRET = "never_serialize_this_secret_token_0123456789"


@pytest.mark.parametrize("endpoint", ["https://worker.example", "https://worker.example:443/", "http://127.0.0.1:8765", "http://localhost:8765", "http://[::1]:8765"])
def test_http_config_accepts_origin_and_stores_no_auth_value(endpoint):
    settings = ExecutorSettings(backend="http", image=IMAGE, endpoint=endpoint)
    assert settings.endpoint == endpoint.rstrip("/")
    assert settings.token_env == "QREPAIR_EXECUTOR_TOKEN"
    assert settings.artifact_config()["endpoint"] == settings.endpoint
    assert "token_env" not in settings.artifact_config()


@pytest.mark.parametrize("endpoint", [
    "http://worker.example", "http://127.0.0.2", "http://2130706433", "http://localhost.evil.example",
    f"https://{SECRET}@worker.example", f"https://user:{SECRET}@worker.example",
    f"https://worker.example/{SECRET}", f"https://worker.example?token={SECRET}",
    f"https://worker.example#{SECRET}", "https://worker.example:99999", "",
])
def test_http_config_rejects_invalid_origin_without_echoing_input(endpoint):
    with pytest.raises(ValidationError) as error:
        ExecutorSettings(backend="http", image=IMAGE, endpoint=endpoint)
    assert SECRET not in str(error.value)


@pytest.mark.parametrize("extra", [{"token": SECRET}, {"token_file": f"/private/{SECRET}"}, {"token_env": f"Bearer {SECRET}"}])
def test_executor_config_rejects_auth_value_and_path_without_echoing_them(extra):
    with pytest.raises(ValidationError) as error:
        ExecutorSettings(backend="http", image=IMAGE, endpoint="https://worker.example", **extra)
    assert SECRET not in str(error.value)


def test_token_file_creation_reuses_owned_mode_0600_file(tmp_path):
    path = tmp_path / "nested" / "token.txt"
    token = load_or_create_token(path)
    assert validate_token(token) == token
    assert path.stat().st_mode & 0o777 == 0o600
    assert load_or_create_token(path) == token
    assert path.read_text() == token + "\n"


@pytest.mark.parametrize("content", ["short", SECRET + "\r\n", SECRET + "\n\n", "a" * 257, "я" * 40])
def test_invalid_token_file_does_not_echo_contents_or_path(tmp_path, content):
    path = tmp_path / SECRET
    path.write_text(content)
    path.chmod(0o600)
    with pytest.raises(ValueError) as error:
        load_or_create_token(path)
    assert SECRET not in str(error.value)


def test_token_file_rejects_group_readable_file_and_symlink(tmp_path):
    path = tmp_path / "token"
    path.write_text(SECRET)
    path.chmod(0o640)
    with pytest.raises(ValueError, match="0600"):
        load_or_create_token(path)
    path.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="secure"):
        load_or_create_token(link)


def test_http_factory_binds_task_policy_limits_and_fails_closed(monkeypatch):
    import qwen_repair.executors.http_client as client

    instances = []

    class StubClient:
        def __init__(self, endpoint, registry, **kwargs):
            self.endpoint, self.registry, self.kwargs = endpoint, registry, kwargs
            instances.append(self)

        def preflight(self):
            return {"passed": len(instances) == 1, "reason": SECRET}

    monkeypatch.setattr(client, "HttpExecutor", StubClient)
    settings = ExecutorSettings(backend="http", image=IMAGE, endpoint="https://worker.example")
    registry = registry_from_fixtures()
    factory = executor_factory(settings, registry, run_id="run-1", policy_version="step-3")
    task = load_fixture("sum_positive").task
    backend = factory(task)
    assert backend.kwargs["run_id"] == "run-1"
    assert backend.kwargs["policy_version"] == "step-3"
    assert backend.kwargs["task_id"] == task.task_id
    assert backend.kwargs["limits"] == DockerConfig(image=IMAGE).model_dump()
    assert backend.kwargs["token_env"] == settings.token_env
    assert backend.preflight_report["passed"] is True
    with pytest.raises(RuntimeError, match="HTTP executor preflight") as error:
        factory(task)
    assert SECRET not in str(error.value)


def backend_for_provenance(**updates):
    profile = {**DockerConfig(image=IMAGE).model_dump(), "token": SECRET, "token_file": "/private/" + SECRET}
    backend = SimpleNamespace(
        runner_version="http-json-v1", endpoint="https://first.example", limits=profile,
        registry=registry_from_fixtures(), run_id="transport-1", _token=SECRET,
        _repair_training_preflight={"passed": True, "authenticated": True, "workers": ["old-worker"],
                                    "token": SECRET, "token_file": "/private/" + SECRET, "config": profile},
    )
    for key, value in updates.items():
        setattr(backend, key, value)
    return backend


def test_training_provenance_strips_auth_and_resume_allows_transport_rotation(tmp_path):
    fixture = load_fixture("sum_positive")
    backend = backend_for_provenance()
    provenance = executor_provenance(backend)
    assert SECRET not in json.dumps(provenance)
    assert provenance["profile"] == DockerConfig(image=IMAGE).model_dump()
    assert provenance["test_bundle_hashes"] == backend.registry.hashes()
    previous = {"task_id": fixture.task.task_id, **provenance}
    (tmp_path / "manifest.json").write_text(json.dumps({"resumed_executor_profiles": [previous]}))
    (tmp_path / "environment.json").write_text("{}")
    changed = backend_for_provenance(endpoint="https://new.example", run_id="transport-2",
                                     _repair_training_preflight={"passed": True, "workers": ["new-worker"]})
    record_executors(tmp_path, [fixture], [changed])
    payload = json.loads((tmp_path / "manifest.json").read_text())
    assert payload["executors"][0]["endpoint"] == "https://new.example"
    assert SECRET not in json.dumps(payload)
    changed.limits["memory_mb"] = 1024
    with pytest.raises(ValueError, match="profile or test bundles changed"):
        record_executors(tmp_path, [fixture], [changed])


@pytest.mark.parametrize("change", ["image", "registry", "runner"])
def test_training_resume_still_rejects_changed_execution_contract(tmp_path, change):
    fixture, backend = load_fixture("sum_positive"), backend_for_provenance()
    previous = {"task_id": fixture.task.task_id, **executor_provenance(backend)}
    (tmp_path / "manifest.json").write_text(json.dumps({"resumed_executor_profiles": [previous]}))
    (tmp_path / "environment.json").write_text("{}")
    if change == "image":
        backend.limits["image"] = "python@sha256:" + "b" * 64
    elif change == "registry":
        backend.registry = SimpleNamespace(hashes=lambda: {"different": "0" * 64})
    else:
        backend.runner_version = "http-json-v2"
    with pytest.raises(ValueError, match="profile or test bundles changed"):
        record_executors(tmp_path, [fixture], [backend])


def test_inference_manifest_uses_http_metadata_without_auth(tmp_path, monkeypatch):
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
        runner_version="http-json-v1", synthetic_demo=False, policy_version="preload",
        preflight_report={"passed": True, "backend": "http-json-v1"}, _token=SECRET,
        run=lambda *args: TestResult(status="passed", tests_total=1, tests_passed=1, synthetic_demo=False),
    )
    monkeypatch.setattr(m1, "executor_factory", lambda *args, **kwargs: lambda task: backend)
    monkeypatch.setattr(m1.TransformersPolicy, "from_pretrained", lambda **kwargs: StubPolicy())
    monkeypatch.setattr(m1, "inspect_environment", lambda config: {})
    monkeypatch.setenv("QREPAIR_EXECUTOR_TOKEN", SECRET)
    settings = ExecutorSettings(backend="http", image=IMAGE, endpoint="https://worker.example")
    output = tmp_path / "inference"
    m1.run_inference(InferenceConfig(device="cpu", fixture_names=["sum_positive"]), settings, output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["runner_version"] == "http-json-v1"
    assert manifest["executor_config"]["endpoint"] == settings.endpoint
    assert manifest["status"] == "completed"
    assert backend.policy_version == StubPolicy.policy_version
    for path in output.iterdir():
        assert SECRET not in path.read_text()
        assert "token_env" not in path.read_text()
