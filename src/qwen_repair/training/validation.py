import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from qwen_repair.artifacts import write_json
from qwen_repair.executors.batch import run_checks
from qwen_repair.schemas import TestResult

from .common import atomic_json, executor_provenance, require_real_result, verify_checkpoint


CACHE_FILENAME = "authored_validation.json"


def validate_authored(fixtures, executors, config, output: Path, method: str, resume_from=None):
    started = perf_counter()
    if method not in {"sft", "grpo_fixed_two_turn"}:
        raise ValueError("Authored validation supports only SFT and GRPO fixtures")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    context = {
        "schema_version": "1.0", "kind": "authored_validation", "synthetic_demo": False,
        "run_id": manifest["run_id"], "method": method,
        "config": config.model_dump(exclude={"stop_after_steps"}),
        "source_hashes": manifest["source_hashes"], "fixture_hashes": manifest["fixture_hashes"],
    }
    checks, identities = [], []
    for fixture, backend in zip(fixtures, executors, strict=True):
        task = fixture.task
        profile = executor_provenance(backend)
        variants = [("reference", fixture.reference_code)]
        if method == "sft":
            variants.append(("target", fixture.corrected_code))
        authored = [(role, code, bundle) for bundle in (task.public_tests_ref, task.private_tests_ref)
                    for role, code in variants]
        if method == "sft" and config.include_recovery:
            authored.append(("buggy", task.buggy_code, task.public_tests_ref))
        for role, code, bundle in authored:
            checks.append((backend, code, bundle))
            identities.append({
                "task_id": task.task_id, "role": role, "test_bundle_ref": bundle,
                "code_sha256": sha256(code.encode("utf-8")).hexdigest(),
                "test_bundle_sha256": profile["test_bundle_hashes"][bundle],
                "tests_total": len(backend.registry.get(bundle).cases),
                "runner_version": profile["runner_version"], "executor_profile": profile["profile"],
            })
    if not checks or len(checks) > 32:
        raise ValueError("Authored validation requires between 1 and 32 checks")
    cache_path = output / CACHE_FILENAME
    if resume_from is not None:
        checkpoint = Path(resume_from)
        restored = verify_checkpoint(checkpoint, config)
        cache_info = restored["provenance"].get("authored_validation")
        if restored["run_id"] != context["run_id"] or not cache_info or cache_info.get("file") != CACHE_FILENAME:
            raise ValueError("Resume requires authored validation from the same verified training run")
        payload = json.loads((checkpoint / CACHE_FILENAME).read_text())
        if set(payload) != {*context, "checks"} or any(payload[key] != value for key, value in context.items()):
            raise ValueError("Authored validation cache context differs from this training run")
        rows = payload["checks"]
        if not isinstance(rows, list) or len(rows) != len(identities):
            raise ValueError("Authored validation cache does not match the required checks")
        results = []
        for row, identity in zip(rows, identities, strict=True):
            if not isinstance(row, dict) or set(row) != {*identity, "result"} or any(row[key] != value for key, value in identity.items()):
                raise ValueError("Authored validation code, tests or executor profile changed")
            results.append(TestResult.model_validate(row["result"]))
    else:
        results = run_checks(checks)
    validated = {}
    rows = []
    for identity, result in zip(identities, results, strict=True):
        result = require_real_result(result)
        if result.tests_total != identity["tests_total"]:
            raise ValueError("Authored validation result does not cover the trusted test bundle")
        if identity["role"] == "buggy":
            if result.status == "passed":
                raise ValueError("Recovery supervision requires an actually failing public attempt")
        elif result.status != "passed":
            raise ValueError("Authored reference or target does not pass its trusted tests")
        key = (identity["task_id"], identity["role"], identity["test_bundle_ref"])
        validated[key] = result
        rows.append({**identity, "result": result.model_dump(mode="json")})
    atomic_json(cache_path, {**context, "checks": rows})
    manifest["authored_validation"] = {
        "file": CACHE_FILENAME, "sha256": sha256(cache_path.read_bytes()).hexdigest(), "checks": len(rows),
    }
    manifest["authored_validation_reused"] = resume_from is not None
    manifest["authored_validation_metrics"] = {
        "checks_executed": 0 if resume_from is not None else len(rows),
        "checks_reused": len(rows) if resume_from is not None else 0,
        "reused_from_checkpoint": str(resume_from) if resume_from is not None else None,
        "wall_seconds": perf_counter() - started,
    }
    write_json(manifest_path, manifest)
    return validated
