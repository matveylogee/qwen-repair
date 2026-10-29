import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qrepair", description="Python repair: synthetic demo and bounded isolated M1 experiments."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Inspect CPU setup without importing GPU libraries")
    doctor.add_argument("--config", type=Path, help="Validate and report a demo YAML config")
    doctor.add_argument("--gpu", action="store_true", help="Explicitly import and probe the pinned GPU stack")
    doctor.add_argument("--output", type=Path, help="Write a new JSON environment report")
    demo = commands.add_parser("demo", help="Write a synthetic run bundle")
    demo.add_argument("--config", type=Path, required=True)
    demo.add_argument("--output", type=Path, required=True, help="New run directory (must not exist)")
    summary = commands.add_parser("summarize", help="Verify a saved run and recompute its metrics")
    summary.add_argument("--run", type=Path, required=True)
    summary.add_argument("--output", type=Path, help="Optional new JSON file outside the run bundle")
    from qwen_repair.m1_cli import add_arguments
    add_arguments(commands)
    return parser


def run_demo(config, output: Path) -> dict:
    from datetime import datetime, timezone
    from uuid import uuid4

    import yaml

    from qwen_repair import __version__
    from qwen_repair.artifacts import sha256, write_json, write_jsonl
    from qwen_repair.data.fixtures import fixture_hashes, load_demo_scenario
    from qwen_repair.doctor import git_state, inspect_environment
    from qwen_repair.evaluation import evaluate_private, summarize
    from qwen_repair.harness.controller import EpisodeController
    from qwen_repair.harness.prompts import PROMPT_VERSION
    from qwen_repair.policies import ScriptedPolicy
    from qwen_repair.schemas import RunManifest

    output.mkdir(parents=True, exist_ok=False)
    package_root = Path(__file__).parent
    manifest = RunManifest(
        run_id=uuid4().hex,
        status="running",
        created_at=datetime.now(timezone.utc).isoformat(),
        package_version=__version__,
        **git_state(Path.cwd()),
        config=config.model_dump(mode="json"),
        seed=config.seed,
        policy_version="scripted-v1",
        prompt_version=PROMPT_VERSION,
        runner_version="fake-lookup-v1",
        fixture_hashes=fixture_hashes(),
        source_hashes={
            path.relative_to(package_root).as_posix(): sha256(path)
            for path in sorted(package_root.rglob("*.py"))
        },
    )
    write_json(output / "manifest.json", manifest)
    try:
        (output / "config.yaml").write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
        )
        write_json(output / "environment.json", inspect_environment(config))
        controller = EpisodeController(config)
        trajectories, evaluations = [], []
        for index, scenario_name in enumerate(config.scenarios, 1):
            scenario = load_demo_scenario(scenario_name)
            # The optional third attempt repeats the last authored response.
            responses = scenario.responses + [scenario.responses[-1]] * max(
                0, config.max_attempts - len(scenario.responses)
            )
            policy = ScriptedPolicy(responses)
            trajectory = controller.run(
                scenario.task, policy, scenario.executor,
                run_id=manifest.run_id,
                episode_id=f"demo-{config.seed}-{index:03}-{scenario_name}",
                scenario=scenario_name,
            )
            trajectories.append(trajectory)
            evaluations.append(evaluate_private(trajectory, scenario.task, scenario.executor))
        metrics = summarize(trajectories, evaluations)
        write_jsonl(output / "trajectories.jsonl", trajectories)
        write_jsonl(output / "private_evaluation.jsonl", evaluations)
        write_json(output / "metrics.json", metrics)
        manifest.status = "completed_with_exclusions" if metrics["excluded_episodes"] else "completed"
        manifest.completed_at = datetime.now(timezone.utc).isoformat()
        manifest.artifact_hashes = {
            path.name: sha256(path)
            for path in sorted(output.iterdir()) if path.name != "manifest.json"
        }
        write_json(output / "manifest.json", manifest)
        write_json(output / "checksums.json", {
            "schema_version": "1.0",
            "synthetic_demo": True,
            "files": {path.name: sha256(path) for path in sorted(output.iterdir())},
        })
        return metrics
    except Exception as error:
        manifest.status = "failed"
        manifest.completed_at = datetime.now(timezone.utc).isoformat()
        manifest.error_type = type(error).__name__
        write_json(output / "manifest.json", manifest)
        raise


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from pydantic import ValidationError

    from qwen_repair.artifacts import read_run
    from qwen_repair.config import DemoConfig, load_config
    from qwen_repair.doctor import inspect_environment
    from qwen_repair.evaluation import summarize

    try:
        if args.command == "doctor":
            if args.gpu:
                from qwen_repair.runtime import gpu_preflight
                result = gpu_preflight()
            else:
                config = load_config(args.config) if args.config else DemoConfig()
                result = inspect_environment(config)
            from qwen_repair.m1_cli import _save_report
            _save_report(args.output, result)
        elif args.command == "demo":
            result = run_demo(load_config(args.config), args.output)
        elif args.command == "summarize":
            trajectories, evaluations = read_run(args.run)
            result = summarize(trajectories, evaluations)
            if args.output:
                if args.output.resolve().is_relative_to(args.run.resolve()):
                    raise ValueError("Summary output must be outside the immutable run bundle")
                with args.output.open("x", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        else:
            from qwen_repair.m1_cli import run_command
            result = run_command(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 1 if result.get("passed") is False else 0
    except ValidationError as error:
        for detail in error.errors(include_input=False, include_url=False):
            location = ".".join(str(part) for part in detail["loc"])
            print(f"qrepair: invalid {location}: {detail['msg']}", file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"qrepair: {error}", file=sys.stderr)
        return 2
