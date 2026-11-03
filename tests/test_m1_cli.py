import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_repair.cli import main


ROOT = Path(__file__).resolve().parents[1]


def test_gpu_doctor_missing_dependencies_reports_failure_without_importing_torch(monkeypatch, capsys):
    import qwen_repair.runtime as runtime

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(runtime.importlib.metadata, "version", missing)
    assert main(["doctor", "--gpu"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is False
    assert report["checks"]["dependency_versions"] is False
    assert report["gpu"]["status"] == "torch_missing"
    assert report["synthetic_demo"] is False


def test_m1_notebook_is_empty_valid_and_delegates_model_operations_to_cli():
    notebook = json.loads((ROOT / "notebooks/01_m1_feasibility.ipynb").read_text())
    assert notebook["nbformat"] == 4
    sources = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        source = "".join(cell["source"])
        compile(source, "01_m1_feasibility.ipynb", "exec")
        sources.append(source)
    source = "\n".join(sources)
    for command in ("doctor", "executor-preflight", "executor-probe", "infer", "sft-smoke", "grpo-smoke", "--resume-from"):
        assert command in source
    assert "drive.mount" in source
    assert "SESSION_NAME" in source
    assert "RUN_INFERENCE = False" in source
    assert "RUN_TRAINING = False" in source
    for forbidden in ("AutoModel", "AutoTokenizer", "GRPOTrainer", "SFTTrainer", "EpisodeController", "exec(", "eval("):
        assert forbidden not in source


@pytest.mark.parametrize("command", ["executor-probe", "queue-probe"])
def test_executor_probe_keeps_queue_alias(tmp_path, monkeypatch, capsys, command):
    import qwen_repair.m1 as m1
    from qwen_repair.schemas import TestResult

    executor = SimpleNamespace(runner_version="docker-json-v1", run=lambda *args: TestResult(
        status="passed", tests_total=1, tests_passed=1, synthetic_demo=False,
    ))
    monkeypatch.setattr(m1, "executor_factory", lambda *args, **kwargs: lambda task: executor)
    output = tmp_path / "probe.json"
    assert main([command, "--config", str(ROOT / "configs/docker_local.yaml"), "--output", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True
    assert report["runner_version"] == executor.runner_version
    assert json.loads(output.read_text()) == report


def test_invalid_executor_config_cli_never_echoes_unknown_field_values(tmp_path, capsys):
    sentinel = "DO_NOT_PRINT_THIS_TOKEN"
    config = tmp_path / "executor.yaml"
    config.write_text(f"backend: file_queue\nimage: python@sha256:{'a' * 64}\nqueue_dir: queue\nextra_field: {sentinel}\n")
    assert main(["executor-preflight", "--config", str(config)]) == 2
    output = capsys.readouterr()
    assert sentinel not in output.out + output.err
    assert "extra_field" in output.err
