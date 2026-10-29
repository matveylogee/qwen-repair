import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from qwen_repair.schemas import PrivateEvaluation, Trajectory


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/local_demo.yaml"


def cli(*args, env=None):
    return subprocess.run(
        [sys.executable, "-m", "qwen_repair", *map(str, args)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def read_jsonl(path, model):
    return [model.model_validate_json(line) for line in path.read_text().splitlines()]


@pytest.fixture
def demo_run(tmp_path):
    run = tmp_path / "demo"
    completed = cli("demo", "--config", CONFIG, "--output", run)
    assert completed.returncode == 0, completed.stderr
    return run


def test_help_does_not_import_gpu_libraries():
    script = """
import runpy
import sys
sys.argv = ['qrepair', '--help']
try:
    runpy.run_module('qwen_repair', run_name='__main__')
except SystemExit as error:
    assert error.code == 0
assert not {'torch', 'transformers', 'peft', 'trl', 'vllm'} & sys.modules.keys()
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    for command in ("doctor", "demo", "summarize"):
        assert command in result.stdout


def test_doctor_does_not_expose_environment_secrets():
    secret = "DO_NOT_PRINT_TEST_ACCESS_TOKEN_9127"
    env = {**os.environ, "HF_TOKEN": secret, "AWS_SECRET_ACCESS_KEY": secret}
    result = cli("doctor", "--config", CONFIG, env=env)
    assert result.returncode == 0, result.stderr
    assert secret not in result.stdout + result.stderr
    assert json.loads(result.stdout)


def test_malformed_yaml_does_not_dump_source_or_traceback(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("scenarios: [DO_NOT_PRINT_CONFIG_SECRET\n", encoding="utf-8")
    result = cli("doctor", "--config", config)
    assert result.returncode == 2
    assert "Invalid YAML syntax" in result.stderr
    assert "DO_NOT_PRINT_CONFIG_SECRET" not in result.stdout + result.stderr
    assert "Traceback" not in result.stderr


def test_demo_metrics_and_summarize_agree(demo_run):
    metrics = json.loads((demo_run / "metrics.json").read_text())
    result = cli("summarize", "--run", demo_run)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == metrics
    assert metrics["synthetic_demo"] is True
    assert metrics["total_episodes"] == 5
    assert metrics["eligible_episodes"] == 4
    assert metrics["excluded_episodes"] == metrics["infrastructure_error_episodes"] == 1
    assert metrics["final_success_rate"] == 0.5
    assert metrics["recovery_rate"] == pytest.approx(2 / 3)
    assert metrics["regression_rate"] == 1
    assert metrics["recovery_denominator"] == 3
    assert metrics["regression_denominator"] == 1
    assert metrics["attempt_success"] == {"1": 0.25, "2": 0.5}
    assert metrics["solved_by_attempt"] == {"1": 0.25, "2": 0.75}


def test_demo_bundle_roundtrip_and_infra_exclusion(demo_run):
    for filename in ("manifest.json", "config.yaml", "metrics.json", "trajectories.jsonl",
                     "private_evaluation.jsonl", "environment.json", "checksums.json"):
        assert (demo_run / filename).is_file()
    manifest = json.loads((demo_run / "manifest.json").read_text())
    trajectories = read_jsonl(demo_run / "trajectories.jsonl", Trajectory)
    evaluations = read_jsonl(demo_run / "private_evaluation.jsonl", PrivateEvaluation)
    assert len(trajectories) == len(evaluations) == 5
    by_id = {evaluation.episode_id: evaluation for evaluation in evaluations}
    for trajectory in trajectories:
        evaluation = by_id[trajectory.episode_id]
        assert trajectory.run_id == evaluation.run_id == manifest["run_id"]
        assert trajectory.task_id == evaluation.task_id
        assert trajectory.schema_version == evaluation.schema_version == manifest["schema_version"] == "1.0"
        if trajectory.stop_reason == "infra_error":
            assert evaluation.terminal_reward is None
            assert evaluation.excluded_reason is not None
        else:
            assert evaluation.terminal_reward in (0, 1)
            assert evaluation.excluded_reason is None
    public_history = (demo_run / "trajectories.jsonl").read_text()
    assert "PRIVATE_SENTINEL" not in public_history
    assert "REFERENCE_SENTINEL" not in public_history


def test_existing_run_directory_is_preserved(demo_run):
    before = {path.name: path.read_bytes() for path in demo_run.iterdir() if path.is_file()}
    result = cli("demo", "--config", CONFIG, "--output", demo_run)
    assert result.returncode != 0
    after = {path.name: path.read_bytes() for path in demo_run.iterdir() if path.is_file()}
    assert before == after


def test_same_seed_reproduces_meaningful_events(demo_run, tmp_path):
    other = tmp_path / "repeat"
    result = cli("demo", "--config", CONFIG, "--output", other)
    assert result.returncode == 0, result.stderr

    def stable(value):
        if isinstance(value, dict):
            return {key: stable(item) for key, item in value.items()
                    if key not in {"run_id", "episode_id", "duration_seconds"}}
        if isinstance(value, list):
            return [stable(item) for item in value]
        return value

    for name in ("trajectories.jsonl", "private_evaluation.jsonl"):
        first = [stable(json.loads(line)) for line in (demo_run / name).read_text().splitlines()]
        second = [stable(json.loads(line)) for line in (other / name).read_text().splitlines()]
        assert first == second
    assert json.loads((demo_run / "manifest.json").read_text())["run_id"] != json.loads(
        (other / "manifest.json").read_text()
    )["run_id"]


def test_notebook_is_valid_thin_and_has_no_outputs():
    notebook = json.loads((ROOT / "notebooks/00_preflight.ipynb").read_text())
    assert notebook["nbformat"] == 4
    sources = []
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
            compile(source, "00_preflight.ipynb", "exec")
            sources.append(source)
    all_code = "\n".join(sources)
    for expected in ("REPO_URL", "COMMIT_SHA", '"doctor"', '"demo"', '"summarize"'):
        assert expected in all_code
    for forbidden in ("EpisodeController", "ScriptedPolicy", "FakeExecutor", "exec(", "eval("):
        assert forbidden not in all_code
