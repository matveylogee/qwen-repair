import importlib.metadata
import platform
import shutil
from pathlib import Path
from time import perf_counter

from qwen_repair import __version__
from qwen_repair.training.config import ML_VERSIONS


def gpu_preflight() -> dict:
    report = {
        "schema_version": "1.0", "synthetic_demo": False,
        "python": platform.python_version(), "package_version": __version__,
        "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
        "dependencies": {}, "gpu": {"status": "not_probed"},
        "checks": {}, "passed": False,
    }
    for name, expected in ML_VERSIONS.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        matches = actual is not None and actual.split("+")[0] == expected
        report["dependencies"][name] = {"expected": expected, "installed": actual, "matches": matches}
    report["checks"]["dependency_versions"] = all(item["matches"] for item in report["dependencies"].values())
    if report["dependencies"]["torch"]["installed"] is None:
        report["gpu"] = {"status": "torch_missing"}
        return report
    try:
        import torch

        available = torch.cuda.is_available()
        report["checks"]["cuda_available"] = available
        report["gpu"] = {"status": "available" if available else "unavailable", "cuda_version": torch.version.cuda}
        if not available:
            return report
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        bf16 = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16 else torch.float16
        report["gpu"].update({"device": device, "name": props.name,
                              "vram_bytes": props.total_memory,
                              "compute_capability": [props.major, props.minor],
                              "bf16_supported": bf16, "suggested_dtype": "bfloat16" if bf16 else "float16"})
        started = perf_counter()
        tensor = torch.ones((16, 16), device="cuda", dtype=dtype)
        result = tensor @ tensor
        torch.cuda.synchronize()
        report["checks"]["cuda_matmul"] = bool(torch.all(result == 16).item())
        report["gpu"]["probe_seconds"] = perf_counter() - started
        del tensor, result
        report["passed"] = all(report["checks"].values())
    except Exception as error:
        report["gpu"]["status"] = "probe_failed"
        report["error_type"] = type(error).__name__
    return report
