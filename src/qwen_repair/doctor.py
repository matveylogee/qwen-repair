import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from qwen_repair import __version__
from qwen_repair.config import DemoConfig


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_state(directory: Path) -> dict:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=directory, capture_output=True,
            text=True, timeout=5, check=False,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=directory, capture_output=True,
            text=True, timeout=5, check=False,
        )
        return {
            "git_sha": sha.stdout.strip() if sha.returncode == 0 else None,
            "git_dirty": bool(dirty.stdout) if dirty.returncode == 0 else None,
        }
    except (OSError, subprocess.TimeoutExpired):
        return {"git_sha": None, "git_dirty": None}


def inspect_environment(config: DemoConfig) -> dict:
    return {
        "schema_version": "1.0",
        "synthetic_demo": True,
        "python": platform.python_version(),
        "python_supported": sys.version_info >= (3, 11),
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "package": {"name": "qwen-repair", "version": __version__},
        "dependencies": {
            name: package_version(name) for name in ("pydantic", "PyYAML", "pytest")
        },
        "optional_gpu_libraries": {
            name: package_version(name)
            for name in ("torch", "transformers", "peft", "trl", "vllm", "verl")
        },
        "gpu": {"status": "not_probed", "reason": "M0 does not import GPU libraries"},
        "docker": {
            "cli_available": shutil.which("docker") is not None,
            "daemon_checked": False,
            "isolation_verified": False,
        },
        "executors": {"fake": True, "real_isolated": False},
        "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
        "config": config.model_dump(mode="json"),
        "token_count_method": "utf8_bytes_proxy",
    }
