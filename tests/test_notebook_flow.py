import ast
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/01_m1_feasibility.ipynb"


def load_output_helper():
    notebook = json.loads(NOTEBOOK.read_text())
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code" and "def run_visible(" in "".join(cell["source"]))
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
                 or isinstance(node, ast.FunctionDef) and node.name == "run_visible"]
    namespace = {}
    exec(compile(tree, str(NOTEBOOK), "exec"), namespace)
    return namespace


def test_subprocess_output_is_visible_before_child_exits(tmp_path):
    namespace = load_output_helper()
    release = tmp_path / "release"
    observed = []

    def show(value, **kwargs):
        observed.append(value)
        if "waiting for visible output" in value:
            release.touch()

    namespace["print"] = show
    child = """
import pathlib, sys, time
release = pathlib.Path(sys.argv[1])
print('waiting for visible output', flush=True)
deadline = time.monotonic() + 5
while not release.exists():
    if time.monotonic() > deadline:
        raise SystemExit(17)
    time.sleep(0.02)
print('finished', flush=True)
"""
    result = namespace["run_visible"]([sys.executable, "-u", "-c", child, str(release)], check=True, timeout=10)
    assert result.returncode == 0
    assert result.stdout == "waiting for visible output\nfinished\n"
    assert "".join(observed) == result.stdout


@pytest.mark.parametrize("check", [False, True])
def test_subprocess_failure_preserves_visible_and_captured_output(capsys, check):
    run_visible = load_output_helper()["run_visible"]
    command = [sys.executable, "-u", "-c", "print('failure detail', flush=True); raise SystemExit(7)"]
    if check:
        with pytest.raises(subprocess.CalledProcessError) as error:
            run_visible(command, check=True)
        assert error.value.returncode == 7
        assert error.value.stdout == "failure detail\n"
    else:
        result = run_visible(command, check=False)
        assert result.returncode == 7
        assert result.stdout == "failure detail\n"
    assert capsys.readouterr().out == "failure detail\n"


def test_subprocess_timeout_preserves_output_and_returns_promptly(capsys):
    run_visible = load_output_helper()["run_visible"]
    command = [sys.executable, "-u", "-c", "import time; print('started', flush=True); time.sleep(30)"]
    with pytest.raises(subprocess.TimeoutExpired) as error:
        run_visible(command, check=True, timeout=1)
    assert error.value.stdout == "started\n"
    assert capsys.readouterr().out == "started\n"


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
            if hasattr(kwargs.get("stdout"), "write"):
                kwargs["stdout"].write("setup diagnostic: subprocess failed\n")
                kwargs["stdout"].flush()
            raise subprocess.CalledProcessError(1, command, output="setup diagnostic: subprocess failed\n")
        if command[1:3] == ["-m", "venv"] and "--without-pip" not in command:
            raise subprocess.CalledProcessError(1, command, output="ensurepip is unavailable\n")
        if command[1:3] == ["-m", "pip"]:
            assert command[0] == sys.executable
            assert command[3:5] == ["--python", str(Path(namespace["ENV_DIR"]) / "bin/python")]
            assert hasattr(kwargs["stdout"], "write")
            assert kwargs["stderr"] == subprocess.STDOUT
            assert kwargs["text"] is True
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


@pytest.mark.parametrize("failure", ["venv", "pip"])
def test_setup_failure_shows_subprocess_output_and_stops(tmp_path, monkeypatch, capsys, failure):
    cells, namespace, calls, _ = notebook_harness(tmp_path, monkeypatch, failed_command=failure)
    with pytest.raises(subprocess.CalledProcessError):
        run_cells(cells, namespace)
    assert "setup diagnostic: subprocess failed" in capsys.readouterr().out
    assert namespace["transport_ready"] is False
    assert namespace["gpu_ready"] is False
    assert all("doctor" not in call and "executor-probe" not in call for call in calls)


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
