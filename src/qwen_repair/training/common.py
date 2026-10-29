import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from qwen_repair.artifacts import write_json

from .config import ML_VERSIONS


class TrainingInfrastructureError(RuntimeError):
    pass


def check_ml_versions() -> dict[str, str]:
    found = {}
    for name, expected in ML_VERSIONS.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError("Install the pinned GPU extra before training") from error
        if actual.split("+")[0] != expected:
            raise RuntimeError(f"Expected {name}=={expected}, found {actual}")
        found[name] = actual
    return found


def device_dtype(config):
    import torch

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a GPU runtime before loading Qwen")
    selected = config.dtype
    if selected == "auto":
        selected = (
            "bfloat16" if config.device == "cuda" and torch.cuda.is_bf16_supported()
            else "float16" if config.device == "cuda" else "float32"
        )
    if selected == "bfloat16" and config.device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support the requested bfloat16 dtype")
    if config.device == "cpu" and selected != "float32":
        raise ValueError("The bounded CPU training profile requires float32")
    return getattr(torch, selected), selected


def load_base(config):
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    check_ml_versions()
    dtype, dtype_name = device_dtype(config)
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id, revision=config.model_revision, trust_remote_code=False
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, revision=config.model_revision, trust_remote_code=False,
        dtype=dtype, attn_implementation="sdpa",
    ).to(config.device)
    return model, tokenizer, dtype_name


def add_lora(model, rank: int):
    from peft import LoraConfig, get_peft_model

    return get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    ))


def checked_executor(executor, task):
    backend = executor(task) if callable(executor) else executor
    if backend is None or getattr(backend, "synthetic_demo", True):
        raise TrainingInfrastructureError("Real training requires a non-synthetic isolated executor")
    preflight = getattr(backend, "preflight", None)
    if preflight is None:
        raise TrainingInfrastructureError("The executor has no isolation preflight")
    report = getattr(backend, "preflight_report", None)
    if not report or report.get("passed") is not True:
        report = preflight()
    if report.get("passed") is not True:
        raise TrainingInfrastructureError("Executor isolation preflight did not pass")
    backend._repair_training_preflight = report
    return backend


def require_real_result(result):
    if result.synthetic_demo:
        raise TrainingInfrastructureError("Synthetic outcomes cannot supervise a real training run")
    if result.status == "infra_error":
        raise TrainingInfrastructureError("Executor infrastructure failure; training group cancelled")
    return result


def file_hashes(directory: Path, ignored=frozenset({"checkpoint_complete.json"})) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*")) if path.is_file() and path.name not in ignored
    }


def source_hashes() -> dict[str, str]:
    package_root = Path(__file__).parents[1]
    return {
        path.relative_to(package_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package_root.rglob("*.py"))
    }


def verify_checkpoint(directory: str | Path, config=None, *, allow_test_checkpoint=False, expected_dtype=None) -> dict:
    directory = Path(directory)
    payload = json.loads((directory / "checkpoint_complete.json").read_text())
    required = {
        "trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
        "adapter_config.json", "adapter_model.safetensors",
    }
    if payload.get("scaler_required") or (
        config is not None and config.device == "cuda" and config.dtype == "float16"
    ) or expected_dtype == "float16":
        required.add("scaler.pt")
    if not required.issubset(payload["files"]):
        raise ValueError("Checkpoint is missing optimizer, scheduler, RNG or trainer state")
    if file_hashes(directory) != payload["files"]:
        raise ValueError("Checkpoint file inventory or checksum mismatch")
    provenance = payload.get("provenance")
    if not allow_test_checkpoint:
        required_provenance = {"source_hashes", "fixture_hashes", "versions", "executors"}
        if not isinstance(provenance, dict) or any(not provenance.get(key) for key in required_provenance):
            raise ValueError("Production checkpoint requires complete provenance")
        if not payload.get("run_id") or not isinstance(payload.get("scaler_required"), bool):
            raise ValueError("Production checkpoint requires run identity and scaler metadata")
        if payload.get("resolved_dtype") not in {"float32", "bfloat16", "float16"}:
            raise ValueError("Production checkpoint requires its resolved dtype")
        if config is not None and hasattr(config, "sft_adapter") and not provenance.get("initial_sft_adapter_hashes"):
            raise ValueError("GRPO checkpoint requires the initial SFT adapter hashes")
    if expected_dtype is not None and payload.get("resolved_dtype") != expected_dtype:
        raise ValueError("Resolved dtype differs from the checkpoint")
    if config is not None:
        expected = config.model_dump(exclude={"stop_after_steps"})
        if payload["config"] != expected:
            raise ValueError("Resume requires the same resolved training config and planned step horizon")
        if provenance:
            from qwen_repair.data.fixtures import fixture_hashes

            if provenance["source_hashes"] != source_hashes() or provenance["fixture_hashes"] != fixture_hashes():
                raise ValueError("Source code or fixture hashes changed since the checkpoint")
            if provenance["versions"] != check_ml_versions():
                raise ValueError("ML runtime versions changed since the checkpoint")
            if "initial_sft_adapter_hashes" in provenance and provenance["initial_sft_adapter_hashes"] != file_hashes(
                Path(config.sft_adapter), ignored=frozenset()
            ):
                raise ValueError("The initial SFT adapter changed since the GRPO checkpoint")
    return payload


def checkpoint_callback(config, *, allow_test_checkpoint=False):
    from transformers import TrainerCallback

    class CompletedCheckpoint(TrainerCallback):
        scaler_getter = None

        def on_step_end(self, args, state, control, **kwargs):
            if config.stop_after_steps is not None and state.global_step >= config.stop_after_steps:
                control.should_training_stop = True
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            directory = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            manifest_path = Path(args.output_dir) / "manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            run_id = manifest.get("run_id")
            scaler_required = self.scaler_getter() if self.scaler_getter is not None else bool(args.fp16)
            atomic_json(directory / "checkpoint_complete.json", {
                "schema_version": "1.0", "global_step": state.global_step,
                "run_id": run_id,
                "scaler_required": scaler_required,
                "resolved_dtype": "float16" if args.fp16 else "bfloat16" if args.bf16 else "float32",
                "provenance": {key: manifest[key] for key in (
                    "source_hashes", "fixture_hashes", "versions", "initial_sft_adapter_hashes", "executors"
                ) if key in manifest},
                "config": config.model_dump(exclude={"stop_after_steps"}),
                "files": file_hashes(directory),
            })
            verify_checkpoint(directory, config, allow_test_checkpoint=allow_test_checkpoint)
            write_json(Path(args.output_dir) / "latest_checkpoint.json", {"path": directory.name})
            return control

    return CompletedCheckpoint()


def begin_run(config, output: Path, resume_from, method: str):
    import torch
    import yaml

    from qwen_repair import __version__
    from qwen_repair.data.fixtures import fixture_hashes
    from qwen_repair.doctor import git_state, inspect_environment
    from qwen_repair.harness.prompts import PROMPT_VERSION

    versions = check_ml_versions()
    _, dtype_name = device_dtype(config)
    if config.device == "cuda":
        import torch
        torch.cuda.reset_peak_memory_stats()
    if resume_from:
        restored = verify_checkpoint(resume_from, config, expected_dtype=dtype_name)
        if output.exists():
            manifest_path = output / "manifest.json"
            if not manifest_path.exists() or json.loads(manifest_path.read_text()).get("kind") != "training":
                raise ValueError("Existing resume output is not a training run directory")
        output.mkdir(parents=True, exist_ok=True)
    else:
        restored = {}
        output.mkdir(parents=True, exist_ok=False)
    (output / "training_complete.json").unlink(missing_ok=True)
    environment = inspect_environment(config)
    environment.update(synthetic_demo=False, dependencies=versions, token_count_method="tokenizer")
    environment["executors"] = {"fake": False, "real_isolated": False, "status": "pending_preflight"}
    environment["gpu"] = {
        "status": "available" if config.device == "cuda" else "cpu_profile",
        "cuda": torch.version.cuda,
        "name": torch.cuda.get_device_name() if config.device == "cuda" else None,
        "vram_bytes": torch.cuda.get_device_properties(0).total_memory if config.device == "cuda" else None,
    }
    (output / "config.yaml").write_text(yaml.safe_dump(config.model_dump(), sort_keys=True), encoding="utf-8")
    write_json(output / "environment.json", environment)
    write_json(output / "manifest.json", {
        "schema_version": "1.0", "kind": "training", "method": method,
        "run_id": restored.get("run_id") or uuid4().hex,
        "synthetic_demo": False, "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__, **git_state(Path.cwd()),
        "config": config.model_dump(), "seed": config.seed,
        "model_id": config.model_id, "model_revision": config.model_revision,
        "tokenizer_revision": config.model_revision, "prompt_version": PROMPT_VERSION,
        "enable_thinking": False, "token_count_method": "tokenizer", "fixture_hashes": fixture_hashes(),
        "source_hashes": source_hashes(),
        "resumed_executor_profiles": restored.get("provenance", {}).get("executors"),
        "versions": versions, "resume_from": str(resume_from) if resume_from else None,
        "segment_start_step": restored.get("global_step", 0),
    })
    write_json(output / "training_status.json", {
        "schema_version": "1.0", "method": method, "status": "running",
        "synthetic_demo": False, "config": config.model_dump(),
        "versions": versions, "resume_from": str(resume_from) if resume_from else None,
        "segment_start_step": restored.get("global_step", 0),
    })
    return perf_counter()


def finish_run(config, output, trainer, started, dtype_name, **extra):
    import torch

    trainer.save_model(str(output / "adapter"))
    trainer.processing_class.save_pretrained(str(output / "adapter"))
    write_json(output / "adapter" / "repair_training.json", {
        "model_id": config.model_id, "model_revision": config.model_revision,
        "global_step": trainer.state.global_step,
    })
    status = json.loads((output / "training_status.json").read_text())
    segment_start_step = status["segment_start_step"]
    summary = {
        "schema_version": "1.0", "synthetic_demo": False,
        "status": "paused_at_checkpoint" if trainer.state.global_step < config.max_steps else "completed",
        "global_step": trainer.state.global_step, "planned_steps": config.max_steps,
        "global_step_scope": "cumulative", "metrics_scope": "current_process_segment",
        "segment_start_step": segment_start_step,
        "optimizer_steps_in_segment": trainer.state.global_step - segment_start_step,
        "duration_seconds": perf_counter() - started, "dtype": dtype_name,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if config.device == "cuda" else None,
        "training_log": [entry for entry in trainer.state.log_history if entry.get("step", -1) > segment_start_step], **extra,
    }
    write_json(output / "training_result.json", summary)
    status.update(status=summary["status"], global_step=trainer.state.global_step)
    write_json(output / "training_status.json", status)
    return summary


def record_executors(output: Path, fixtures, executors):
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["executors"] = []
    for fixture, backend in zip(fixtures, executors, strict=True):
        profile = getattr(backend, "config", None)
        profile = profile.model_dump() if hasattr(profile, "model_dump") else getattr(backend, "limits", {})
        manifest["executors"].append({
            "task_id": fixture.task.task_id, "runner_version": backend.runner_version,
            "profile": profile, "preflight": backend._repair_training_preflight,
            "test_bundle_hashes": backend.registry.hashes(),
            "transport_run_id": getattr(backend, "run_id", None),
        })
    previous = manifest.get("resumed_executor_profiles")
    if previous:
        def stable_profiles(rows):
            return [{key: row[key] for key in ("task_id", "runner_version", "profile", "test_bundle_hashes")} for row in rows]
        if stable_profiles(previous) != stable_profiles(manifest["executors"]):
            raise ValueError("Executor profile or test bundles changed since the checkpoint")
    write_json(output / "manifest.json", manifest)
    environment_path = output / "environment.json"
    environment = json.loads(environment_path.read_text())
    environment["executors"] = {"fake": False, "real_isolated": True, "status": "preflight_passed"}
    write_json(environment_path, environment)


def atomic_json(path: Path, payload):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        write_json(temporary, payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def seal_training_run(output: Path, result):
    manifest = json.loads((output / "manifest.json").read_text())
    manifest.update(status=result["status"], completed_at=datetime.now(timezone.utc).isoformat(), performance=result)
    write_json(output / "manifest.json", manifest)
    inventory = file_hashes(output, ignored=frozenset({"training_complete.json"}))
    atomic_json(output / "training_complete.json", {
        "schema_version": "1.0", "run_id": manifest["run_id"],
        "status": result["status"], "files": inventory,
    })


def fail_run(output: Path, error: Exception):
    path = output / "training_status.json"
    if path.exists():
        payload = json.loads(path.read_text())
        payload.update(status="failed", error_type=type(error).__name__)
        write_json(path, payload)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        payload.update(status="failed", error_type=type(error).__name__)
        write_json(manifest_path, payload)
    (output / "training_complete.json").unlink(missing_ok=True)
