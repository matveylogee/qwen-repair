import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/01_m1_feasibility.ipynb"


def notebook_harness(tmp_path, monkeypatch, *, inference=False, training=False, failed_command=None):
    cells = ["".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text())["cells"]
             if cell["cell_type"] == "code"]
    namespace = {}
    exec(compile(cells[0], str(NOTEBOOK), "exec"), namespace)
    assert namespace["RUN_INFERENCE"] is False
    assert namespace["RUN_TRAINING"] is False
    namespace.update(COMMIT_SHA="a" * 40, SESSION_NAME="test-session",
                     CHECKOUT_DIR=str(tmp_path / "checkout"), ENV_DIR=str(tmp_path / "env"),
                     DRIVE_MOUNT=str(tmp_path / "drive"), RUN_INFERENCE=inference, RUN_TRAINING=training)
    calls, downloads = [], []

    def mount(path):
        assert path == namespace["DRIVE_MOUNT"]
        (Path(path) / "MyDrive/qwen-repair-exchange/sessions/test-session").mkdir(parents=True)

    google = ModuleType("google")
    colab = ModuleType("google.colab")
    colab.drive = SimpleNamespace(mount=mount)
    colab.files = SimpleNamespace(download=downloads.append)
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)

    def run(command, **kwargs):
        assert kwargs.get("check") is True
        calls.append(command)
        if failed_command in command:
            raise subprocess.CalledProcessError(1, command)
        if command[:2] == ["git", "clone"]:
            Path(command[-1]).mkdir()
        if "--output" in command:
            output = Path(command[command.index("--output") + 1])
            if output.suffix == ".json":
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({"passed": True}))
            else:
                output.mkdir(parents=True, exist_ok=True)
                (output / "training_result.json").write_text("{}")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "a" * 40 + "\n")
    return cells[1:], namespace, calls, downloads


def run_cells(cells, namespace):
    for source in cells:
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)


def test_default_run_all_only_calls_cpu_and_transport_commands(tmp_path, monkeypatch):
    cells, namespace, calls, downloads = notebook_harness(tmp_path, monkeypatch)
    run_cells(cells, namespace)
    cli_commands = [call[3] for call in calls if call[1:3] == ["-m", "qwen_repair"]]
    assert cli_commands == ["doctor", "executor-preflight", "executor-probe"]
    assert all("--gpu" not in call and ".[gpu]" not in call for call in calls)
    assert namespace["transport_ready"] is True
    assert namespace["gpu_ready"] is False
    assert len(downloads) == 1 and Path(downloads[0]).is_file()


@pytest.mark.parametrize("failure", ["executor-preflight", "executor-probe"])
def test_transport_failure_prevents_even_opted_in_gpu_stages(tmp_path, monkeypatch, failure):
    cells, namespace, calls, _ = notebook_harness(
        tmp_path, monkeypatch, inference=True, training=True, failed_command=failure,
    )
    with pytest.raises(subprocess.CalledProcessError):
        run_cells(cells, namespace)
    assert namespace["transport_ready"] is False
    assert all(".[gpu]" not in call and "infer" not in call and "sft-smoke" not in call for call in calls)
    gpu_cell = next(source for source in cells if '".[gpu]"' in source)
    with pytest.raises(RuntimeError, match="executor preflight"):
        run_cells([gpu_cell], namespace)
    assert all(".[gpu]" not in call for call in calls)


def test_inference_opt_in_keeps_training_disabled(tmp_path, monkeypatch):
    cells, namespace, calls, _ = notebook_harness(tmp_path, monkeypatch, inference=True)
    run_cells(cells, namespace)
    assert any("infer" in call for call in calls)
    assert all("sft-smoke" not in call and "grpo-smoke" not in call for call in calls)
    probe_index = next(index for index, call in enumerate(calls) if "executor-probe" in call)
    gpu_index = next(index for index, call in enumerate(calls) if ".[gpu]" in call)
    assert probe_index < gpu_index


def test_training_opt_in_uses_separate_stop_and_resume_calls(tmp_path, monkeypatch):
    cells, namespace, calls, _ = notebook_harness(tmp_path, monkeypatch, training=True)
    run_cells(cells, namespace)
    for command in ("sft-smoke", "grpo-smoke"):
        stages = [call for call in calls if command in call]
        assert len(stages) == 2
        assert stages[0][-2:] == ["--stop-after", "1"]
        assert "--resume-from" in stages[1]
    assert all("infer" not in call for call in calls)


@pytest.mark.parametrize("session", ["", "../other", "/absolute", "with/slash"])
def test_invalid_session_is_rejected_before_setup(tmp_path, monkeypatch, session):
    cells, namespace, calls, _ = notebook_harness(tmp_path, monkeypatch)
    namespace["SESSION_NAME"] = session
    with pytest.raises(ValueError, match="SESSION_NAME"):
        run_cells(cells, namespace)
    assert not calls
