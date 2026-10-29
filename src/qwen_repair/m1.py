import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import yaml

from qwen_repair import __version__
from qwen_repair.artifacts import sha256, write_json, write_jsonl
from qwen_repair.config import DemoConfig
from qwen_repair.data.fixtures import fixture_hashes, load_fixture
from qwen_repair.data.pilot import load_pilot
from qwen_repair.doctor import git_state, inspect_environment
from qwen_repair.evaluation import evaluate_private, summarize
from qwen_repair.executors.bundles import load_registry, registry_from_fixtures
from qwen_repair.executors.docker import DockerConfig, DockerExecutor
from qwen_repair.executors.filequeue import FileQueueExecutor
from qwen_repair.harness.controller import EpisodeController
from qwen_repair.harness.prompts import PROMPT_VERSION
from qwen_repair.m1_config import ExecutorSettings, InferenceConfig
from qwen_repair.policies.transformers import TransformersPolicy
from qwen_repair.schemas import RunManifest, TaskRecord


def executor_factory(settings: ExecutorSettings, registry, *, run_id: str, policy_version: str):
    limits = DockerConfig(image=settings.image, timeout_seconds=settings.timeout_seconds)
    if settings.backend == "docker":
        backend = DockerExecutor(registry, limits)
        report = backend.preflight()
        if not report["passed"]:
            raise RuntimeError(f"Docker preflight failed: {json.dumps(report, ensure_ascii=False)}")
        return lambda task: backend

    def create(task):
        backend = FileQueueExecutor(
            settings.queue_dir, registry, run_id=run_id, policy_version=policy_version,
            task_id=task.task_id, limits=limits.model_dump(mode="json"),
            deadline_seconds=settings.deadline_seconds, poll_interval=settings.poll_interval,
        )
        report = backend.preflight()
        if not report["passed"]:
            raise RuntimeError("No compatible queue worker. Start the Docker worker and verify folder synchronization first.")
        backend.preflight_report = report
        return backend

    return create


def run_inference(config: InferenceConfig, settings: ExecutorSettings, output: Path) -> dict:
    if config.pilot_dir:
        directory = Path(config.pilot_dir)
        records, dataset_manifest = load_pilot(directory)
        tasks = [TaskRecord.model_validate(record["task"]) for record in records[:config.max_tasks]]
        registry = load_registry(directory / "registry.json")
        dataset_hashes = dataset_manifest["artifact_hashes"]
    else:
        if not config.fixture_names or len(set(config.fixture_names)) != len(config.fixture_names):
            raise ValueError("fixture_names must be nonempty and unique")
        tasks = [load_fixture(name).task for name in config.fixture_names[:config.max_tasks]]
        registry = registry_from_fixtures()
        dataset_hashes = fixture_hashes()
    if not tasks:
        raise ValueError("No tasks selected")
    run_id = uuid4().hex
    factory = executor_factory(settings, registry, run_id=run_id, policy_version="preload")
    factory(tasks[0])
    output.mkdir(parents=True, exist_ok=False)
    manifest = RunManifest(
        run_id=run_id, status="running", synthetic_demo=False,
        created_at=datetime.now(timezone.utc).isoformat(), package_version=__version__,
        **git_state(Path.cwd()), config=config.model_dump(mode="json"), seed=config.seed,
        policy_version=f"{config.model_id}@{config.model_revision}", prompt_version=PROMPT_VERSION,
        runner_version="docker-json-v1" if settings.backend == "docker" else "filequeue-v1",
        token_count_method="tokenizer", model_revision=config.model_revision,
        tokenizer_revision=config.tokenizer_revision, fixture_hashes={},
        source_hashes={path.relative_to(Path(__file__).parent).as_posix(): sha256(path) for path in sorted(Path(__file__).parent.rglob("*.py"))},
        dataset_hashes=dataset_hashes, executor_config=settings.model_dump(mode="json"),
    )
    write_json(output / "manifest.json", manifest)
    try:
        started = perf_counter()
        policy = TransformersPolicy.from_pretrained(**config.model_dump(include={
            "model_id", "model_revision", "tokenizer_revision", "device", "dtype", "seed", "do_sample",
            "temperature", "top_p", "top_k", "max_context_tokens", "adapter_path",
        }))
        manifest.policy_version = policy.policy_version
        manifest.performance["model_load_seconds"] = perf_counter() - started
        if config.device == "cuda":
            policy._torch.cuda.reset_peak_memory_stats()
        trajectories, evaluations, executor_reports = [], [], []
        for index, task in enumerate(tasks):
            backend = factory(task)
            executor_reports.append({"task_id": task.task_id, "report": backend.preflight_report})
            if hasattr(backend, "policy_version"):
                backend.policy_version = policy.policy_version
            trajectory = EpisodeController(config).run(
                task, policy, backend, run_id=run_id, episode_id=f"{run_id}-{index:03}", scenario="qwen_inference",
            )
            trajectories.append(trajectory)
            evaluation = evaluate_private(trajectory, task, backend)
            if trajectory.synthetic_demo or evaluation.synthetic_demo:
                raise RuntimeError("Real inference cannot include synthetic outcomes")
            evaluations.append(evaluation)
        metrics = summarize(trajectories, evaluations)
        environment = inspect_environment(DemoConfig())
        environment.update({"synthetic_demo": False, "config": config.model_dump(mode="json"), "token_count_method": "tokenizer",
                            "executors": {"fake": False, "real_isolated": True}, "executor_profile": settings.model_dump(mode="json"),
                            "executor_preflight": executor_reports})
        if settings.backend == "docker":
            environment["docker"] = {
                "execution_location": "local", "daemon_checked": True,
                "isolation_preflight_passed": all(item["report"]["passed"] for item in executor_reports),
                "image": settings.image,
            }
        else:
            environment["docker"] = {
                "execution_location": "remote_worker", "local_daemon_checked": False,
                "worker_preflight_verified": all(item["report"]["passed"] for item in executor_reports),
                "worker_image": settings.image,
            }
        if config.device == "cuda":
            environment["gpu"] = {"status": "used", "name": policy._torch.cuda.get_device_name(),
                                  "cuda_version": policy._torch.version.cuda,
                                  "peak_allocated_bytes": policy._torch.cuda.max_memory_allocated()}
            manifest.performance["peak_vram_bytes"] = environment["gpu"]["peak_allocated_bytes"]
        else:
            environment["gpu"] = {"status": "not_used", "reason": "CPU inference selected"}
        (output / "config.yaml").write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
        write_json(output / "environment.json", environment)
        write_jsonl(output / "trajectories.jsonl", trajectories)
        write_jsonl(output / "private_evaluation.jsonl", evaluations)
        write_json(output / "metrics.json", metrics)
        manifest.status = "completed_with_exclusions" if metrics["excluded_episodes"] else "completed"
        manifest.completed_at = datetime.now(timezone.utc).isoformat()
        manifest.artifact_hashes = {path.name: sha256(path) for path in output.iterdir() if path.name != "manifest.json"}
        write_json(output / "manifest.json", manifest)
        write_json(output / "checksums.json", {"schema_version": "1.0", "synthetic_demo": False,
                                               "files": {path.name: sha256(path) for path in output.iterdir()}})
        return metrics
    except Exception as error:
        manifest.status = "failed"
        manifest.error_type = type(error).__name__
        manifest.completed_at = datetime.now(timezone.utc).isoformat()
        write_json(output / "manifest.json", manifest)
        raise
