import json
import shutil
import sys
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest

from qwen_repair.artifacts import write_json
from qwen_repair.data.fixtures import fixture_hashes, load_fixture
from qwen_repair.executors.bundles import BundleRegistry, registry_from_fixtures
from qwen_repair.executors.docker import DockerConfig
from qwen_repair.schemas import TestResult
from qwen_repair.training import common, validation
from qwen_repair.training.config import GRPOConfig, ML_VERSIONS, SFTConfig


class AuthoredContractJudge:
    synthetic_demo = False
    runner_version = "authored-contract-only"
    _repair_training_preflight = {"passed": True}

    def __init__(self, fixture):
        self.fixture = fixture
        self.registry = registry_from_fixtures()
        self.config = DockerConfig(image="python@sha256:" + "a" * 64)
        self.calls = []
        self.override = None

    def run(self, code, bundle):
        self.calls.append((code, bundle))
        if self.override is not None:
            return self.override
        passed = code in {self.fixture.reference_code, self.fixture.corrected_code}
        total = len(self.registry.get(bundle).cases)
        return TestResult(status="passed" if passed else "failed", tests_total=total,
                          tests_passed=total if passed else 0, tests_failed=0 if passed else total, synthetic_demo=False,
                          duration_seconds=123.0, feedback="Prepared contract result; no execution.")


@pytest.fixture
def authored_run(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "source_hashes", lambda: {"contract.py": "test-source"})
    monkeypatch.setattr(common, "check_ml_versions", lambda: dict(ML_VERSIONS))
    fixtures = [load_fixture(name) for name in ("sum_positive", "last_or_none")]
    judges = [AuthoredContractJudge(fixture) for fixture in fixtures]
    config = SFTConfig(device="cpu")
    output = tmp_path / "run"
    output.mkdir()
    manifest = {
        "run_id": "a" * 32, "kind": "training", "method": "sft",
        "source_hashes": common.source_hashes(), "fixture_hashes": fixture_hashes(),
        "versions": dict(ML_VERSIONS),
        "executors": [{"task_id": fixture.task.task_id, **common.executor_provenance(judge)}
                      for fixture, judge in zip(fixtures, judges, strict=True)],
    }
    write_json(output / "manifest.json", manifest)
    return fixtures, judges, config, output


def checkpoint(output, config, copy_cache=True):
    directory = output / "checkpoint-1"
    directory.mkdir()
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json", "adapter_model.safetensors"):
        (directory / name).write_bytes(b"CPU contract only; never deserialize")
    write_json(directory / "trainer_state.json", {"global_step": 1})
    if copy_cache:
        shutil.copyfile(output / validation.CACHE_FILENAME, directory / validation.CACHE_FILENAME)
    manifest = json.loads((output / "manifest.json").read_text())
    write_json(directory / "checkpoint_complete.json", {
        "run_id": manifest["run_id"], "global_step": 1, "scaler_required": False, "resolved_dtype": "float32",
        "config": config.model_dump(exclude={"stop_after_steps"}), "files": common.file_hashes(directory),
        "provenance": {key: manifest[key] for key in
                       ("source_hashes", "fixture_hashes", "versions", "executors", "authored_validation")},
    })
    return directory


def rewrite_cache_and_seal(directory, mutate):
    path = directory / validation.CACHE_FILENAME
    payload = json.loads(path.read_text())
    mutate(payload)
    write_json(path, payload)
    marker_path = directory / "checkpoint_complete.json"
    marker = json.loads(marker_path.read_text())
    marker["files"] = common.file_hashes(directory)
    marker["provenance"]["authored_validation"]["sha256"] = sha256(path.read_bytes()).hexdigest()
    write_json(marker_path, marker)


@pytest.mark.parametrize("method,expected", [("sft", 10), ("grpo_fixed_two_turn", 4)])
def test_all_authored_checks_use_one_bounded_batch(authored_run, monkeypatch, method, expected):
    fixtures, judges, config, output = authored_run
    if method != "sft":
        config = GRPOConfig(device="cpu", sft_adapter="unused")
    batches = []
    original = validation.run_checks

    def observed(checks):
        batches.append(list(checks))
        return original(checks)

    monkeypatch.setattr(validation, "run_checks", observed)
    results = validation.validate_authored(fixtures, judges, config, output, method)
    assert len(results) == len(batches[0]) == expected
    assert len(batches) == 1
    assert sum(len(judge.calls) for judge in judges) == expected
    manifest = json.loads((output / "manifest.json").read_text())
    metrics = manifest["authored_validation_metrics"]
    assert metrics["checks_executed"] == expected and metrics["checks_reused"] == 0
    assert metrics["reused_from_checkpoint"] is None
    assert 0 <= metrics["wall_seconds"] < 123.0


def test_resume_uses_only_verified_checkpoint_cache_without_executor_calls(authored_run, monkeypatch):
    fixtures, judges, config, output = authored_run
    original = validation.validate_authored(fixtures, judges, config, output, "sft")
    saved = checkpoint(output, config)
    (output / validation.CACHE_FILENAME).write_text("untrusted live cache")
    for judge in judges:
        judge.calls.clear()
    monkeypatch.setattr(validation, "run_checks", lambda checks: pytest.fail("Resume must not execute static checks"))
    resumed = validation.validate_authored(fixtures, judges, config.model_copy(update={"stop_after_steps": 2}), output, "sft", saved)
    assert resumed == original
    assert not any(judge.calls for judge in judges)
    metrics = json.loads((output / "manifest.json").read_text())["authored_validation_metrics"]
    assert metrics["checks_reused"] == 10 and metrics["checks_executed"] == 0
    assert metrics["reused_from_checkpoint"] == str(saved)
    assert 0 <= metrics["wall_seconds"] < 123.0


def test_resume_rejects_tampered_checkpoint_cache_before_execution(authored_run):
    fixtures, judges, config, output = authored_run
    validation.validate_authored(fixtures, judges, config, output, "sft")
    saved = checkpoint(output, config)
    (saved / validation.CACHE_FILENAME).write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        validation.validate_authored(fixtures, judges, config, output, "sft", saved)
    assert sum(len(judge.calls) for judge in judges) == 10


@pytest.mark.parametrize("changed", ["code", "bundle", "runner", "limits", "run_id"])
def test_resume_requires_exact_authored_identity(authored_run, changed):
    fixtures, judges, config, output = authored_run
    validation.validate_authored(fixtures, judges, config, output, "sft")
    saved = checkpoint(output, config)
    if changed == "code":
        fixtures[0] = replace(fixtures[0], corrected_code=fixtures[0].corrected_code + "\n# changed")
    elif changed == "bundle":
        bundles = {key: judges[0].registry.get(key) for key in judges[0].registry.hashes()}
        key = fixtures[0].task.public_tests_ref
        bundles[key] = bundles[key].model_copy(update={"entry_point": "different_function"})
        judges[0].registry = BundleRegistry(bundles)
    elif changed == "runner":
        judges[0].runner_version = "different-runner"
    elif changed == "limits":
        judges[0].config = judges[0].config.model_copy(update={"memory_mb": judges[0].config.memory_mb + 1})
    else:
        manifest = json.loads((output / "manifest.json").read_text())
        manifest["run_id"] = "b" * 32
        write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="changed|same verified training run"):
        validation.validate_authored(fixtures, judges, config, output, "sft", saved)
    assert sum(len(judge.calls) for judge in judges) == 10


@pytest.mark.parametrize("field", ["method", "source_hashes", "fixture_hashes", "config", "run_id"])
def test_even_resealed_cache_must_match_run_context(authored_run, field):
    fixtures, judges, config, output = authored_run
    validation.validate_authored(fixtures, judges, config, output, "sft")
    saved = checkpoint(output, config)
    rewrite_cache_and_seal(saved, lambda payload: payload.update({field: "different"}))
    with pytest.raises(ValueError, match="cache context"):
        validation.validate_authored(fixtures, judges, config, output, "sft", saved)


@pytest.mark.parametrize("result", [
    TestResult(status="infra_error", synthetic_demo=False),
    TestResult(status="passed", tests_total=1, tests_passed=1, synthetic_demo=True),
])
@pytest.mark.parametrize("cached", [False, True])
def test_infrastructure_or_synthetic_results_never_enter_cache(authored_run, result, cached):
    fixtures, judges, config, output = authored_run
    saved = None
    if cached:
        validation.validate_authored(fixtures, judges, config, output, "sft")
        saved = checkpoint(output, config)
        rewrite_cache_and_seal(saved, lambda payload: payload["checks"][0].update(result=result.model_dump()))
    else:
        judges[0].override = result
    with pytest.raises(common.TrainingInfrastructureError):
        validation.validate_authored(fixtures, judges, config, output, "sft", saved)
    assert (output / validation.CACHE_FILENAME).exists() is cached


@pytest.mark.parametrize("cached", [False, True])
def test_static_validation_rejects_incomplete_test_counts(authored_run, cached):
    fixtures, judges, config, output = authored_run
    saved = None
    result = TestResult(status="passed", tests_total=1, tests_passed=1, synthetic_demo=False)
    assert len(judges[0].registry.get(fixtures[0].task.public_tests_ref).cases) > result.tests_total
    if cached:
        validation.validate_authored(fixtures, judges, config, output, "sft")
        saved = checkpoint(output, config)
        rewrite_cache_and_seal(saved, lambda payload: payload["checks"][0].update(result=result.model_dump()))
    else:
        judges[0].override = result
    with pytest.raises(ValueError, match="trusted test bundle"):
        validation.validate_authored(fixtures, judges, config, output, "sft", saved)


def test_checkpoint_binds_cache_before_inventory_and_detects_live_changes(authored_run, monkeypatch):
    fixtures, judges, config, output = authored_run
    validation.validate_authored(fixtures, judges, config, output, "sft")
    saved = checkpoint(output, config, copy_cache=False)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(TrainerCallback=object))
    callback = common.checkpoint_callback(config)
    callback.summary = lambda: {}
    args = SimpleNamespace(output_dir=str(output), fp16=False, bf16=False)
    callback.on_save(args, SimpleNamespace(global_step=1), None)
    assert (saved / validation.CACHE_FILENAME).read_bytes() == (output / validation.CACHE_FILENAME).read_bytes()
    assert validation.CACHE_FILENAME in common.verify_checkpoint(saved, config)["files"]
    before = common.file_hashes(saved)
    (output / validation.CACHE_FILENAME).write_text("modified after validation")
    with pytest.raises(ValueError, match="changed before checkpoint"):
        callback.on_save(args, SimpleNamespace(global_step=1), None)
    assert common.file_hashes(saved) == before
