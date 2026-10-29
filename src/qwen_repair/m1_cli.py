from pathlib import Path


def add_arguments(commands):
    preflight = commands.add_parser("executor-preflight", help="Probe the selected isolated executor")
    preflight.add_argument("--config", required=True, type=Path)
    preflight.add_argument("--registry", type=Path)
    preflight.add_argument("--output", type=Path)
    registry = commands.add_parser("export-registry", help="Export trusted authored test bundles for the worker")
    registry.add_argument("--output", required=True, type=Path)
    worker = commands.add_parser("queue-worker", help="Run one trusted local Docker worker")
    worker.add_argument("--config", required=True, type=Path)
    worker.add_argument("--queue", required=True, type=Path)
    worker.add_argument("--registry", type=Path)
    worker.add_argument("--once", action="store_true")
    probe = commands.add_parser("queue-probe", help="Measure one real request round trip before rollout")
    probe.add_argument("--config", required=True, type=Path)
    probe.add_argument("--output", required=True, type=Path)
    fetch = commands.add_parser("pilot-fetch", help="Download at most 100 rows at a pinned KodCode revision")
    fetch.add_argument("--output", required=True, type=Path)
    fetch.add_argument("--limit", type=int, default=100)
    prepare = commands.add_parser("pilot-prepare", help="Statically filter a small train-only pilot")
    prepare.add_argument("--source", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--max-tasks", type=int, default=10)
    prepare.add_argument("--seed", type=int, default=42)
    validate = commands.add_parser("pilot-validate", help="Check references and mutations in real isolation")
    validate.add_argument("--source", required=True, type=Path)
    validate.add_argument("--output", required=True, type=Path)
    validate.add_argument("--executor-config", required=True, type=Path)
    for name, help_text in (
        ("infer", "Run Qwen repair episodes with a real executor"),
        ("sft-smoke", "Run bounded LoRA SFT with verified recovery feedback"),
        ("grpo-smoke", "Run bounded multi-turn TRL GRPO from an SFT adapter"),
    ):
        parser = commands.add_parser(name, help=help_text)
        parser.add_argument("--config", required=True, type=Path)
        parser.add_argument("--executor-config", required=True, type=Path)
        parser.add_argument("--output", required=True, type=Path)
        if name != "infer":
            parser.add_argument("--resume-from", type=Path)
            parser.add_argument("--stop-after", type=int, help="Stop at this step without changing the planned training horizon")


def _save_report(path, report):
    from qwen_repair.artifacts import write_json
    if path:
        if path.exists():
            raise ValueError("Report output already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)


def run_command(args):
    from time import perf_counter, sleep
    from uuid import uuid4

    from qwen_repair.data.fixtures import load_fixture
    from qwen_repair.data.pilot import fetch_pilot, prepare_pilot, validate_pilot
    from qwen_repair.executors.bundles import export_registry, load_registry, registry_from_fixtures
    from qwen_repair.executors.docker import DockerConfig, DockerExecutor
    from qwen_repair.executors.filequeue import FileQueueExecutor, QueueWorker
    from qwen_repair.m1 import executor_factory, run_inference
    from qwen_repair.m1_config import ExecutorSettings, InferenceConfig, read_yaml_model

    if args.command == "pilot-fetch":
        return fetch_pilot(args.output, limit=args.limit)
    if args.command == "pilot-prepare":
        return prepare_pilot(args.source, args.output, seed=args.seed, max_tasks=args.max_tasks)
    if args.command == "export-registry":
        if args.output.exists():
            raise ValueError("Registry output already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        registry = registry_from_fixtures()
        export_registry(registry, args.output)
        return {"output": str(args.output), "bundle_hashes": registry.hashes()}
    if args.command == "pilot-validate":
        registry = load_registry(args.source / "registry.json")
        settings = read_yaml_model(args.executor_config, ExecutorSettings)
        factory = executor_factory(settings, registry, run_id=uuid4().hex, policy_version="reference-validation")
        return validate_pilot(args.source, args.output, factory)
    if args.command in {"executor-preflight", "queue-worker", "queue-probe"}:
        settings = read_yaml_model(args.config, ExecutorSettings)
        registry_path = getattr(args, "registry", None)
        registry = load_registry(registry_path) if registry_path else registry_from_fixtures()
        limits = DockerConfig(image=settings.image, timeout_seconds=settings.timeout_seconds)
        if args.command == "executor-preflight":
            if settings.backend == "docker":
                executor = DockerExecutor(registry, limits)
            else:
                executor = FileQueueExecutor(settings.queue_dir, registry, run_id=uuid4().hex,
                                             policy_version="preflight", task_id="preflight", limits=limits.model_dump(mode="json"),
                                             deadline_seconds=settings.deadline_seconds, poll_interval=settings.poll_interval)
            report = executor.preflight()
            _save_report(args.output, report)
            return report
        if args.command == "queue-worker":
            if settings.backend != "docker":
                raise ValueError("The local worker requires a Docker executor profile")
            executor = DockerExecutor(registry, limits)
            report = executor.preflight()
            if not report["passed"]:
                return report
            worker = QueueWorker(args.queue, registry, executor)
            completed = 0
            try:
                while True:
                    completed += worker.process_once()
                    if args.once:
                        break
                    sleep(0.25)
            except KeyboardInterrupt:
                pass
            finally:
                worker.close()
            return {"status": "stopped", "completed_requests": completed}
        fixture = load_fixture("sum_positive")
        factory = executor_factory(settings, registry, run_id=uuid4().hex, policy_version="queue-probe")
        executor = factory(fixture.task)
        started = perf_counter()
        result = executor.run(fixture.corrected_code, fixture.task.public_tests_ref)
        report = {"schema_version": "1.0", "kind": "executor_roundtrip_probe", "synthetic_demo": False,
                  "passed": result.status == "passed" and not result.synthetic_demo,
                  "roundtrip_seconds": perf_counter() - started, "executor_seconds": result.duration_seconds,
                  "result": result.model_dump(mode="json")}
        _save_report(args.output, report)
        return report
    settings = read_yaml_model(args.executor_config, ExecutorSettings)
    if args.command == "infer":
        return run_inference(read_yaml_model(args.config, InferenceConfig), settings, args.output)
    from qwen_repair.training.config import GRPOConfig, SFTConfig
    model = SFTConfig if args.command == "sft-smoke" else GRPOConfig
    config = read_yaml_model(args.config, model)
    if args.stop_after is not None:
        config = model.model_validate({**config.model_dump(), "stop_after_steps": args.stop_after})
    registry = registry_from_fixtures()
    factory = executor_factory(settings, registry, run_id=uuid4().hex, policy_version=args.command)
    if args.command == "sft-smoke":
        from qwen_repair.training.sft import run_sft_smoke
        return run_sft_smoke(config, args.output, resume_from=args.resume_from, executor=factory)
    from qwen_repair.training.grpo import run_grpo_smoke
    return run_grpo_smoke(config, args.output, executor=factory, resume_from=args.resume_from)
