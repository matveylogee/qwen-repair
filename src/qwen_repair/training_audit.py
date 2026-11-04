"""Read training evidence on CPU; never unpickle checkpoint state."""

import json
import math
from pathlib import Path

from qwen_repair.artifact_transfer import verify_training_seal
from qwen_repair.training.common import verify_checkpoint


def audit_training(run: Path) -> dict:
    run = Path(run)
    seal = verify_training_seal(run)
    manifest = json.loads((run / "manifest.json").read_text())
    result = json.loads((run / "training_result.json").read_text())
    segments = [json.loads(path.read_text()) for path in sorted((run / "segments").glob("*.json"))]
    segments.sort(key=lambda row: row["result"]["segment_start_step"])
    checks = {
        "sealed_inventory": True,
        "completed_plan": seal["status"] == "completed" and result["global_step"] == result["planned_steps"],
        "non_synthetic": manifest.get("synthetic_demo") is False,
        "gpu_segments": bool(segments),
        "contiguous_segments": bool(segments),
        "actual_optimizer_updates": bool(segments),
        "finite_gradients": bool(segments),
        "adapter_roundtrip": bool(segments),
        "checkpoint_integrity": bool(segments),
        "resume_parameter_match": False,
    }
    end = 0
    resumed = 0
    processes = []
    summaries = []
    for segment in segments:
        evidence = segment["result"]
        provenance = segment["manifest"]
        updates = evidence.get("optimizer_updates") or {}
        if not isinstance(updates, dict):
            raise ValueError("Optimizer evidence must be an object")
        processes.append(provenance.get("process_id"))
        start, finish = evidence["segment_start_step"], evidence["global_step"]
        checks["contiguous_segments"] &= (
            segment.get("run_id") == seal["run_id"] and provenance.get("run_id") == seal["run_id"]
            and start == end and finish > start
        )
        checks["non_synthetic"] &= segment.get("synthetic_demo") is False
        checks["gpu_segments"] &= segment["environment"].get("gpu", {}).get("status") == "available" and provenance["config"]["device"] == "cuda"
        checks["actual_optimizer_updates"] &= updates.get("applied") == finish - start and updates.get("skipped") == 0
        checks["finite_gradients"] &= updates.get("nonfinite_gradient_steps") == 0
        checks["adapter_roundtrip"] &= evidence.get("adapter_roundtrip_verified") is True and math.isfinite(
            evidence.get("adapter_roundtrip_max_abs_diff", float("nan"))
        )
        checkpoint = run / f"checkpoint-{finish}"
        try:
            marker = verify_checkpoint(checkpoint)
            checks["checkpoint_integrity"] &= marker["run_id"] == seal["run_id"] and marker["global_step"] == finish
        except (OSError, ValueError, KeyError):
            checks["checkpoint_integrity"] = False
        if start:
            resumed += 1
            try:
                previous = verify_checkpoint(run / f"checkpoint-{start}")
                matched = bool(updates.get("before_sha256")) and updates["before_sha256"] == previous.get("optimizer_updates", {}).get("after_sha256")
            except (OSError, ValueError, KeyError):
                matched = False
            checks["resume_parameter_match"] = matched if resumed == 1 else checks["resume_parameter_match"] and matched
            checks["contiguous_segments"] &= bool(provenance.get("resume_from"))
        end = finish
        summaries.append({"start_step": start, "end_step": finish,
                          "duration_seconds": evidence["duration_seconds"], "peak_vram_bytes": evidence["peak_vram_bytes"],
                          "optimizer_updates": updates,
                          "learning_signal_observed": evidence.get("learning_signal_observed")})
    checks["contiguous_segments"] &= end == result["global_step"]
    checks["new_process_resume"] = resumed > 0 and all(type(pid) is int and pid > 0 for pid in processes) and len(set(processes)) == len(processes)
    is_grpo = manifest["method"] == "grpo_fixed_two_turn"
    learning_signal = any(s["learning_signal_observed"] is True for s in summaries) if is_grpo else any(
        s["optimizer_updates"].get("trainable_parameters_changed") is True
        and s["optimizer_updates"].get("nonzero_gradient_steps", 0) > 0 for s in summaries
    )
    plumbing_passed = all(checks.values())
    checks["learning_signal_observed"] = learning_signal
    return {"schema_version": "1.0", "kind": "training_evidence_audit", "synthetic_demo": False,
            "run_id": seal["run_id"], "method": manifest["method"], "passed": all(checks.values()),
            "plumbing_passed": plumbing_passed,
            "status": "validated_smoke" if all(checks.values()) else "plumbing_only" if plumbing_passed else "incomplete_evidence",
            "checks": checks, "learning_signal_observed": learning_signal,
            "learning_signal_required_for_research_training": True,
            "research_training_ready": all(checks.values()) and learning_signal,
            "runtime_loss_recovery_verified": False,
            "scope": "two-step smoke with both boundary checkpoints retained; sealed reports and hashes, no optimizer/RNG deserialization",
            "segments": summaries}
