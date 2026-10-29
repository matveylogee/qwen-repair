import importlib.metadata
import json
from pathlib import Path

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
    for command in ("doctor", "queue-probe", "infer", "sft-smoke", "grpo-smoke", "--resume-from"):
        assert command in source
    for forbidden in ("AutoModel", "AutoTokenizer", "GRPOTrainer", "SFTTrainer", "EpisodeController", "exec(", "eval("):
        assert forbidden not in source
